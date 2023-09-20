import numpy as np
import torch

from recall_model.model_search import *


class Network_prune(nn.Module):
    def __init__(self, model):
        super(Network_prune, self).__init__()
        self.model = model
        self._initialize_proj_weights()
        self._initialize_flags()

    def get_softmax(self):
        weights_s = F.softmax(self.proj_weights['s'], dim=-1)
        weights_s2 = F.softmax(self.proj_weights['s2'], dim=-1)
        weights_x1 = F.softmax(self.proj_weights['x1'], dim=-1)
        weights_x2 = F.softmax(self.proj_weights['x2'], dim=-1)
        weights_fuse1 = F.softmax(self.proj_weights['select'], dim=-1)
        weights_fuse2 = F.softmax(self.proj_weights['fuse'], dim=-1)

        return {'s': weights_s, 's2': weights_s2, 'x1': weights_x1, 'x2': weights_x2, 'select': weights_fuse1, 'fuse': weights_fuse2}

    def _initialize_proj_weights(self):
        self.proj_weights = {
            's': torch.from_numpy(self.model.alphas_s.data.cpu().numpy()),
            's2': torch.from_numpy(self.model.alphas_s2.data.cpu().numpy()),
            'x1': torch.from_numpy(self.model.alphas_x1.data.cpu().numpy()),
            'x2': torch.from_numpy(self.model.alphas_x2.data.cpu().numpy()),
            'select': torch.from_numpy(self.model.alphas_fuse1.data.cpu().numpy()),
            'fuse': torch.from_numpy(self.model.alphas_fuse2.data.cpu().numpy())
        }

    def _initialize_flags(self):
        k = sum(1 for i in range(self.model._steps2) for n in range(4 + i))
        num_ops1 = len(PRIMITIVES1)
        num_ops2 = len(PRIMITIVES2)
        num_ops3 = len(PRIMITIVES3)
        num_ops4 = len(PRIMITIVES4)

        self.candidate_flags_weights = {
            's': torch.ones(self.model._steps, num_ops1).cuda(),
            's2': torch.ones(self.model._steps, num_ops1).cuda(),
            'x1': torch.ones(self.model._steps, num_ops2).cuda(),
            'x2': torch.ones(self.model._steps, num_ops2).cuda(),
            'select': torch.ones(k, num_ops3).cuda(),
            'fuse': torch.ones(self.model._steps2, num_ops4).cuda(),
        }

        self.candidate_flags_cells = {
            's': True,
            's2': True,
            'x1': True,
            'x2': True,
            'select': True,
            'fuse': True
        }

    def get_projected_weights(self, cell_type):
        weights = self.get_softmax()[cell_type]

        return weights

    def project_op(self, eid, opid, cell_type):
        if torch.sum(self.candidate_flags_weights[cell_type][eid]) > 1.0:
            self.proj_weights[cell_type][eid][opid] = -np.inf
            self.candidate_flags_weights[cell_type][eid][opid] = 0
            if all([float(torch.sum(self.candidate_flags_weights[cell_type][e])) == 1.0 for e in range(self.candidate_flags_weights[cell_type].size(0))]):
                self.candidate_flags_cells[cell_type] = False
        else:
            print('invalid prune')

    def forward(self, x1, x2, s, text, weights_dict=None):
        weights_s = self.get_projected_weights('s')
        weights_s2 = self.get_projected_weights('s2')
        weights_x1 = self.get_projected_weights('x1')
        weights_x2 = self.get_projected_weights('x2')
        weights_fuse1 = self.get_projected_weights('select')
        weights_fuse2 = self.get_projected_weights('fuse')
        if weights_dict is not None:
            if 's' in weights_dict:
                weights_s = weights_dict['s']
            if 's2' in weights_dict:
                weights_s = weights_dict['s2']
            if 'x1' in weights_dict:
                weights_x1 = weights_dict['x1']
            if 'x2' in weights_dict:
                weights_x2 = weights_dict['x2']
            if 'select' in weights_dict:
                weights_fuse1 = weights_dict['select']
            if 'fuse' in weights_dict:
                weights_fuse2 = weights_dict['fuse']
        x1 = self.model.embbedding1(x1)
        x2 = self.model.embbedding2(x2)
        x1 = self.model.dropout1(x1)
        x2 = self.model.dropout2(x2)
        s = self.model.relu(self.model.linear(s))
        s2 = self.model.relu(self.model.linear_note(text))
        cell_s = self.model.cell_s(s, x1, x2, s2, weights_s)[-1]
        cell_s2 = self.model.cell_s2(s2, x1, x2, s, weights_s2)[-1]
        cell_x1 = self.model.cell_x1(x1, s, x2, s2, weights_x1)[-1]
        cell_x2 = self.model.cell_x2(x2, s, x1, s2, weights_x2)[-1]
        cell_x1 = self.model.pooler(cell_x1)
        cell_x2 = self.model.pooler(cell_x2)
        cell_x1 = self.model.relu(self.model.linear_x1(cell_x1))
        cell_x2 = self.model.relu(self.model.linear_x2(cell_x2))
        cell_s = self.model.relu(self.model.linear_s(cell_s))
        cell_s2 = self.model.relu(self.model.linear_s2(cell_s2))
        final_states = self.model.cell_fuse(cell_s, cell_x1, cell_x2, cell_s2,
                                      weights_fuse1, weights_fuse2)
        out = self.model.linear_combine(torch.stack(final_states[-self.model._steps2:], dim=-1)).squeeze()
        return torch.softmax(self.model.classifier(out).squeeze(), dim=-1)

    def genotype(self):
        weights_s = self.get_projected_weights('s')
        weights_s2 = self.get_projected_weights('s2')
        weights_x1 = self.get_projected_weights('x1')
        weights_x2 = self.get_projected_weights('x2')
        weights_fuse1 = self.get_projected_weights('select')
        weights_fuse2 = self.get_projected_weights('fuse')

        gene_s = []
        for i in range(self.model._steps):
            k_best = torch.argmax(weights_s[i])
            gene_s.append(PRIMITIVES1[k_best])

        gene_s2 = []
        for i in range(self.model._steps):
            k_best = torch.argmax(weights_s2[i])
            gene_s2.append(PRIMITIVES1[k_best])

        gene_x1 = []
        for i in range(self.model._steps):
            k_best = torch.argmax(weights_x1[i])
            gene_x1.append(PRIMITIVES2[k_best])

        gene_x2 = []
        for i in range(self.model._steps):
            k_best = torch.argmax(weights_x2[i])
            gene_x2.append(PRIMITIVES2[k_best])

        gene_fuse1 = []
        n = 3
        start = 0
        for i in range(self.model._steps2):
            end = start + n
            W = weights_fuse1[start:end]
            for j in range(len(W)):
                k_best = torch.argmax(W[j])
                gene_fuse1.append((PRIMITIVES3[k_best], j))
            start = end
            n += 1

        gene_fuse2 = []
        for i in range(self.model._steps2):
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
