import torch
import torch.nn.functional as F
from auc_model.operations import *
from torch.autograd import Variable
from auc_model.genotypes import *
import numpy as np



class MixedOp(nn.Module):

    def __init__(self, d_model, pri):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        if pri == 1:
            for primitive in PRIMITIVES1:
                op = OPS1[primitive](d_model)
                self._ops.append(op)
        elif pri == 2:
            for primitive in PRIMITIVES2:
                op = OPS2[primitive](d_model)
                self._ops.append(op)
        elif pri == 3:
            for primitive in PRIMITIVES3:
                op = OPS3[primitive](d_model)
                self._ops.append(op)

    def forward(self, x1, x2, x3, x4, weights):
        return sum(w * op(x1, x2, x3, x4) for w, op in zip(weights, self._ops))


class MixedOp_fuse(nn.Module):
    def __init__(self, d_model):
        super(MixedOp_fuse, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES4:
            op = OPS4[primitive](d_model)
            self._ops.append(op)

    def forward(self, all_x, weights):
        return sum(w * op(all_x) for w, op in zip(weights, self._ops))


class Cell_s(nn.Module):

    def __init__(self, d_model, steps):
        super(Cell_s, self).__init__()
        self._steps = steps
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            op = MixedOp(d_model, 1)
            self._ops.append(op)

    def forward(self, s0, x1, x2, s2, weights):
        states = [s0]
        for i in range(self._steps):
            s = self._ops[i](states[i], x1, x2, s2, weights[i])
            states.append(s)

        return states


class Cell_x(nn.Module):
    def __init__(self, d_model, steps):
        super(Cell_x, self).__init__()
        self._steps = steps
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            op = MixedOp(d_model, 2)
            self._ops.append(op)

    def forward(self, current_x, s, other_x, s2, weights):
        states = [current_x]
        for i in range(self._steps):
            state = self._ops[i](states[i], s, other_x, s2, weights[i])
            states.append(state)

        return states


class Cell_fuse(nn.Module):
    def __init__(self, d_model, steps):
        super(Cell_fuse, self).__init__()
        self._steps = steps
        self._ops1 = nn.ModuleList()
        self._ops2 = nn.ModuleList()
        for i in range(self._steps):
            op = MixedOp_fuse(d_model)
            self._ops2.append(op)
            for j in range(i + 4):
                op = MixedOp(d_model, 3)
                self._ops1.append(op)

    def forward(self, x1, x2, x3, x4, weights1, weights2):
        states = [x1, x2, x3, x4]
        offset = 0
        for i in range(self._steps):
            s = [self._ops1[offset + j](h, None, None, None, weights1[offset + j]) for j, h in enumerate(states)]
            offset += len(states)
            s = self._ops2[i](s, weights2[i])
            states.append(s)

        return states


class Network(nn.Module):
    def __init__(self, vocab_size1, vocab_size2, vocab_size3, d_model, steps, steps2, criterion, lamb, num_classes=1):
        super(Network, self).__init__()
        self._criterion = criterion
        self._steps = steps
        self._steps2 = steps2
        self.lamb = lamb
        self._vocab_size1 = vocab_size1
        self._vocab_size2 = vocab_size2
        self._vocab_size3 = vocab_size3
        self.d_model = d_model
        self.embbedding1 = nn.Sequential(nn.Linear(vocab_size1, d_model))
        self.embbedding2 = nn.Sequential(nn.Linear(vocab_size2, d_model))
        self.linear = nn.Linear(vocab_size3, d_model)
        self.linear_combine = nn.Linear(steps2, 1, bias=False)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, num_classes)
        self.cell_s = Cell_s(d_model, steps)
        self.cell_s2 = Cell_s(d_model, steps)
        self.cell_x1 = Cell_x(d_model, steps)
        self.cell_x2 = Cell_x(d_model, steps)
        self.cell_fuse = Cell_fuse(d_model, steps2)
        self.linear_x1 = nn.Linear(d_model, d_model)
        self.linear_x2 = nn.Linear(d_model, d_model)
        self.linear_s = nn.Linear(d_model, d_model)
        self.linear_s2 = nn.Linear(d_model, d_model)
        self.linear_note = nn.Linear(768, d_model)

        self.relu = nn.ReLU()
        self.pooler = MaxPoolLayer()
        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps2) for n in range(4 + i))
        num_ops1 = len(PRIMITIVES1)
        num_ops2 = len(PRIMITIVES2)
        num_ops3 = len(PRIMITIVES3)
        num_ops4 = len(PRIMITIVES4)

        self.alphas_s = Variable(1e-3 * torch.ones(self._steps, num_ops1).cuda(), requires_grad=True)
        self.alphas_s2 = Variable(1e-3 * torch.ones(self._steps, num_ops1).cuda(), requires_grad=True)
        self.alphas_x1 = Variable(1e-3 * torch.ones(self._steps, num_ops2).cuda(), requires_grad=True)
        # self.alphas_encoder = Variable(1e-3 * torch.ones(num_ops).cuda(), requires_grad=True)
        self.alphas_fuse1 = Variable(1e-3 * torch.ones(k, num_ops3).cuda(), requires_grad=True)
        self.alphas_fuse2 = Variable(1e-3 * torch.ones(self._steps2, num_ops4).cuda(), requires_grad=True)
        self.alphas_x2 = Variable(1e-3 * torch.ones(self._steps, num_ops2).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_s,
            self.alphas_s2,
            self.alphas_x1,
            self.alphas_x2,
            # self.alphas_encoder,
            self.alphas_fuse1,
            self.alphas_fuse2
        ]

    def new(self):
        model_new = Network(self._vocab_size1, self._vocab_size2, self._vocab_size3, self.d_model, self._steps, self._steps2, self._criterion, self.lamb).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def arch_parameters(self):
        return self._arch_parameters

    def _loss(self, input, target):
        logits = self(input[0], input[1], input[2], input[3])
        # print(target.size())
        return self._criterion(logits, target)

    def val_loss(self, input, target):
        logits = self(input[0], input[1], input[2], input[3])
        selection_weights = torch.softmax(self.alphas_fuse1, dim=-1)[:, 0]
        node_weights = []
        start_index = 0
        for i in range(self._steps2):
            node_weight = selection_weights[torch.LongTensor([start_index, start_index+1, start_index+2, start_index+3])]
            node_weights.append(node_weight)
            start_index = start_index + i + 4
        bce = nn.BCELoss()
        penalty = 0.0
        for j in range(len(node_weights)):
            for k in range(len(node_weights)):
                # print(node_weights[j].size())
                penalty += bce(node_weights[j], node_weights[k].detach())
        return self._criterion(logits, target) - self.lamb * penalty

    def forward(self, x1, x2, s, text):
        x1 = self.embbedding1(x1)
        x2 = self.embbedding2(x2)
        x1 = self.dropout1(x1)
        x2 = self.dropout2(x2)
        s = self.relu(self.linear(s))
        s2 = self.relu(self.linear_note(text))
        cell_s = self.cell_s(s, x1, x2, s2, torch.softmax(self.alphas_s, dim=-1))[-1]
        cell_s2 = self.cell_s2(s2, x1, x2, s, torch.softmax(self.alphas_s2, dim=-1))[-1]
        cell_x1 = self.cell_x1(x1, s, x2, s2, torch.softmax(self.alphas_x1, dim=-1))[-1]
        cell_x2 = self.cell_x2(x2, s, x1, s2, torch.softmax(self.alphas_x2, dim=-1))[-1]
        cell_x1 = self.pooler(cell_x1)
        cell_x2 = self.pooler(cell_x2)
        cell_x1 = self.relu(self.linear_x1(cell_x1))
        cell_x2 = self.relu(self.linear_x2(cell_x2))
        cell_s = self.relu(self.linear_s(cell_s))
        cell_s2 = self.relu(self.linear_s2(cell_s2))
        final_states = self.cell_fuse(cell_s, cell_x1, cell_x2, cell_s2, torch.softmax(self.alphas_fuse1, dim=-1),
                                      torch.softmax(self.alphas_fuse2, dim=-1))
        out = self.linear_combine(torch.stack(final_states[-self._steps2:], dim=-1)).squeeze()
        return self.classifier(out).squeeze()

    def genotype(self):

        weights_s = torch.softmax(self.alphas_s, dim=-1)
        gene_s = []
        for i in range(self._steps):
            k_best = torch.argmax(weights_s[i])
            gene_s.append(PRIMITIVES1[k_best])

        weights_s2 = torch.softmax(self.alphas_s2, dim=-1)
        gene_s2 = []
        for i in range(self._steps):
            k_best = torch.argmax(weights_s2[i])
            gene_s2.append(PRIMITIVES1[k_best])

        weights_x1 = torch.softmax(self.alphas_x1, dim=-1)
        gene_x1 = []
        for i in range(self._steps):
            k_best = torch.argmax(weights_x1[i])
            gene_x1.append(PRIMITIVES2[k_best])

        weights_x2 = torch.softmax(self.alphas_x2, dim=-1)
        gene_x2 = []
        for i in range(self._steps):
            k_best = torch.argmax(weights_x2[i])
            gene_x2.append(PRIMITIVES2[k_best])

        weights_fuse1 = torch.softmax(self.alphas_fuse1, dim=-1)
        weights_fuse2 = torch.softmax(self.alphas_fuse2, dim=-1)
        gene_fuse1 = []
        n = 4
        start = 0
        for i in range(self._steps2):
            end = start + n
            W = weights_fuse1[start:end]
            for j in range(len(W)):
                k_best = torch.argmax(W[j])
                gene_fuse1.append((PRIMITIVES3[k_best], j))
            start = end
            n += 1
        gene_fuse2 = []
        for i in range(self._steps2):
            k_best = torch.argmax(weights_fuse2[i])
            gene_fuse2.append(PRIMITIVES4[k_best])

        genotype = Genotype(
            s=gene_s,
            s2=gene_s2,
            x1=gene_x1,
            x2=gene_x2,
            select=gene_fuse1,
            fuse=gene_fuse2
        )
        return genotype
