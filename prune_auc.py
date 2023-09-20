import argparse
import os
import random

import torch.optim
from sklearn.metrics import roc_auc_score, \
    precision_recall_curve, auc, average_precision_score
from torch.optim import Adam
from auc_model.dataset import *
from auc_model.model_prune import *

from utils.utils import check_path, bool_flag

def eval_metric(eval_set, model, bert, loss_func, weights_dict):
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        losses = 0.0
        for i, data in enumerate(eval_set):
            x1, x2, s, ids, tokens, mask, labels = data
            text = bert(ids, tokens, mask)
            logits = model(x1, x2, s, text, weights_dict)
            loss = loss_func(logits, labels).item() * labels.size(0)
            scores = torch.sigmoid(logits)
            scores = scores.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            y_true = np.concatenate((y_true, labels))
            y_score = np.concatenate((y_score, scores))
            losses += loss
        roc_auc = roc_auc_score(y_true, y_score)
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(lr_recall, lr_precision)
        mean_loss = losses / len(eval_set)
        average_precision = average_precision_score(y_true, y_score)
    return roc_auc, pr_auc, average_precision, mean_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('-me', '--max_epochs_before_stop', default=10, type=int)
    parser.add_argument('--d_model', default=256, type=int, help='dimension of hidden layers')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of hidden layers')
    parser.add_argument('--dropout_emb', default=0.1, type=float, help='dropout rate of embedding layers')
    parser.add_argument('--num_layers', default=1, type=int, help='number of transformer layers of EHR encoder')
    parser.add_argument('--num_heads', default=4, type=int, help='number of attention heads')
    parser.add_argument('--target_disease', default='ARF', choices=['mortality', 'ARF', 'Shock'])
    parser.add_argument('--duration', default=12.0,type=float)
    parser.add_argument('-lr', '--learning_rate', default=2e-6, type=float, help='learning rate')
    parser.add_argument('-alr', '--arch_learning_rate', default=0.00001, type=float, help='learning rate')
    parser.add_argument('-lrm', '--learning_rate_min', type=float, default=0.0001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--wdecay', default=0.001, type=float)
    parser.add_argument('--arch_wdecay', default=0.001, type=float)
    parser.add_argument('--steps', default=2, type=int)
    parser.add_argument('--steps2', default=3, type=int)
    parser.add_argument('--clip', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--mode', default='train', choices=['train', 'pred', 'study'], help='run training or evaluation')
    parser.add_argument('--save_dir', default='./saved_models/', help='models output directory')
    args = parser.parse_args()
    print(args)

    if args.mode == 'train':
        model_path = os.path.join(args.save_dir, 'models.pt')
        log_path = os.path.join(args.save_dir, 'log_prune.csv')
        check_path(model_path)
        with open(log_path, 'w') as fout:
            fout.write('step,dev_roc,dev_pr,dev_avpre,test_roc,test_pr,test_avpre\n')
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
        model, old_args = torch.load(model_path)
        model.to(device)
        datamodule = DataModule(device, task=old_args.target_disease, duration=old_args.duration)
        train_dataset, dev_dataset, test_dataset = datamodule.get_data()
        train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=True)
        dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
        loss_func = nn.BCEWithLogitsLoss(reduction='mean')
        model_proj = Network_prune(model)
        model_proj.to(device)
        bert = Bert()
        bert.to(device)
        bert.eval()
        # architect = Architect(model, old_args)
        model_proj.train()
        optim = Adam(
            model_proj.parameters(),
            args.learning_rate,
            weight_decay=args.wdecay)
        i = 0
        while True:
            selected_cell, selected_edge = sample_edge(model_proj)
            model_proj = prune(model_proj, selected_cell, selected_edge, dev_dataloader, bert, loss_func)
            i = i + 1
            if i % 8 == 0:
                model_proj, d_roc_auc, d_pr_auc, d_avpre, t_roc_auc, t_pr_auc, t_avpre = fine_tune(model_proj, bert, loss_func, dev_dataloader, train_dataloader, test_dataloader,
                                       optim)
                torch.save([model_proj, old_args], os.path.join(args.save_dir, 'model-prune-{}.pt'.format(i)))
                with open(log_path, 'a') as fout:
                    fout.write('{},{},{},{},{},{},{}\n'.format(i, d_roc_auc, d_pr_auc, d_avpre, t_roc_auc, t_pr_auc, t_avpre))
            # else:
                # d_roc_auc, d_pr_auc, d_avpre, _ = eval_metric(dev_dataloader, model_proj, bert, loss_func, None)
                # t_roc_auc, t_pr_auc, t_avpre, _ = eval_metric(test_dataloader, model_proj, bert, loss_func, None)

            if not any(list(model_proj.candidate_flags_cells.values())):
                # model_proj = fine_tune(model_proj, bert, loss_func, dev_dataloader, train_dataloader, test_dataloader,
                #                        optim)
                break
        print('steps: ', i)
        #torch.save([model_proj, old_args], os.path.join(args.save_dir, 'model-prune.pt'))



def sample_edge(model):
    Cells = [cell for cell in model.candidate_flags_cells.keys() if model.candidate_flags_cells[cell] == True]
    selected_cell = random.choice(Cells)
    weights = model.get_projected_weights(selected_cell)
    candidate_edges = [i for i in range(weights.size(0)) if float(torch.sum(model.candidate_flags_weights[selected_cell][i])) > 1.0]
    selected_edge = random.choice(candidate_edges)
    return selected_cell, selected_edge


def prune(model, selected_cell, selected_edge, dev_dataloader, bert, loss_func):
    weights = model.get_projected_weights(selected_cell)
    best_opid = 0
    highest_pr = None
    prs = []
    for j in range(weights.size(1)):
        if float(model.candidate_flags_weights[selected_cell][selected_edge][j]) == 0.0:
            prs.append(0.0)
            continue
        else:
            proj_mask = torch.ones_like(weights[selected_edge])
            proj_mask[j] = 0
            weights[selected_edge] = weights[selected_edge] * proj_mask
            weights_dict = {selected_cell: weights}
            roc, pr, avpre, loss = eval_metric(dev_dataloader, model, bert, loss_func, weights_dict)
            prs.append(pr)
            if highest_pr is None or pr > highest_pr:
                highest_pr = pr
                best_opid = j
    model.project_op(selected_edge, best_opid, selected_cell)
    return model


def fine_tune(model, bert, loss_func, dev_dataloader, train_dataloader, test_dataloader, optim):
    model.train()
    for epoch_id in range(1):
        for i, data in enumerate(train_dataloader):
            x1, x2, s, ids, tokens, mask, label = data
            text = bert(ids, tokens, mask)
            optim.zero_grad()
            out = model(x1, x2, s, text)
            loss = loss_func(out, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
    model.eval()
    tr_roc_auc, tr_pr_auc, tr_avpre, tr_loss = eval_metric(train_dataloader, model, bert, loss_func, None)
    d_roc_auc, d_pr_auc, d_avpre, d_loss = eval_metric(dev_dataloader, model, bert, loss_func, None)
    t_roc_auc, t_pr_auc, t_avpre, t_loss = eval_metric(test_dataloader, model, bert, loss_func, None)
    print('-' * 71)
    print('| train_auc {:7.4f} | dev_auc {:7.4f} | test_auc {:7.4f} '.format(tr_roc_auc,
                                                                             d_roc_auc,
                                                                             t_roc_auc))
    print('| train_pr {:7.4f} | dev_pr {:7.4f} | test_pr {:7.4f} '.format(tr_pr_auc,
                                                                          d_pr_auc,
                                                                          t_pr_auc))
    print('| train_avpre {:7.4f} | dev_avpre {:7.4f} | test_avpre {:7.4f} '.format(tr_avpre,
                                                                                   d_avpre,
                                                                                   t_avpre))
    print('-' * 71)

    return model, d_roc_auc, d_pr_auc, d_avpre, t_roc_auc, t_pr_auc, t_avpre


if __name__ == '__main__':
    main()