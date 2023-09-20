import argparse
import os
import time

import torch.optim
from sklearn.metrics import roc_auc_score, \
    precision_recall_curve, auc, average_precision_score
from torch.optim import Adam
from auc_model.dataset import *
from auc_model.architect import *
from auc_model.model_search import *

from utils.utils import check_path, export_config, bool_flag


def eval_metric(eval_set, model, bert, loss_func):
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        losses = 0.0
        for i, data in enumerate(eval_set):
            x1, x2, s, ids, tokens, mask, labels = data
            text = bert(ids, tokens, mask)
            logits = model(x1, x2, s, text)
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
    parser.add_argument('-me', '--max_epochs_before_stop', default=5, type=int)
    parser.add_argument('--d_model', default=256, type=int, help='dimension of hidden layers')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of hidden layers')
    parser.add_argument('--dropout_emb', default=0.1, type=float, help='dropout rate of embedding layers')
    parser.add_argument('--num_layers', default=1, type=int, help='number of transformer layers of EHR encoder')
    parser.add_argument('--num_heads', default=4, type=int, help='number of attention heads')
    parser.add_argument('--target_disease', default='ARF', choices=['mortality', 'ARF', 'Shock'])
    parser.add_argument('--duration', default=12.0,type=float)
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('-alr', '--arch_learning_rate', default=0.00001, type=float, help='learning rate')
    parser.add_argument('-lrm', '--learning_rate_min', type=float, default=0.0001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--wdecay', default=0.001, type=float)
    parser.add_argument('--arch_wdecay', default=0.001, type=float)
    parser.add_argument('--steps', default=2, type=int)
    parser.add_argument('--steps2', default=3, type=int)
    parser.add_argument('--lamb', default=0.1, type=float)
    parser.add_argument('--clip', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--mode', default='train', choices=['train', 'pred', 'study'], help='run training or evaluation')
    parser.add_argument('--save_dir', default='./saved_models/', help='models output directory')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'pred':
        pred(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
    print(args)

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'models.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,train_auc,dev_auc,test_auc\n')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    datamodule = DataModule(device, task=args.target_disease, duration=args.duration)
    train_dataset, dev_dataset, test_dataset = datamodule.get_data()
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
    loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    if args.target_disease == 'mortality':
        v1, v2, v3, l = 6726, 1001, 96, 48
    elif args.target_disease == 'ARF':
        v1, v2, v3, l = 4452, 680, 96, 12
    elif args.target_disease == 'Shock':
        v1, v2, v3, l = 5056, 739, 97, 12
    else:
        raise ValueError('no such disease')
    model = Network(v1, v2, v3, args.d_model, args.steps, args.steps2, loss_func, args.lamb)
    model.to(device)
    text_encoder = Bert()
    text_encoder.to(device)
    text_encoder.eval()
    architect = Architect(model, args)
    optim = Adam(
        model.parameters(),
        args.learning_rate,
        weight_decay=args.wdecay)

    global_step, best_dev_epoch = 0, 0
    best_dev_auc, final_test_auc, total_loss = 0.0, 0.0, 0.0

    model.train()
    for epoch_id in range(args.n_epochs):
        lr = optim.param_groups[0]['lr']
        print('epoch: {:5} '.format(epoch_id))
        genotype = model.genotype()
        print(genotype)
        model.train()
        start_time = time.time()
        for i, data in enumerate(train_dataloader):
            x1, x2, s, ids, tokens, masks, label = data
            text = text_encoder(ids, tokens, masks)
            optim.zero_grad()
            for j, data1 in enumerate(dev_dataloader):
                dev_x1, dev_x2, dev_s, dev_ids, dev_tokens, dev_masks, dev_label = data1
                dev_text = text_encoder(dev_ids, dev_tokens, dev_masks)
                break
            architect.step((x1, x2, s, text), label, (dev_x1, dev_x2, dev_s, dev_text), dev_label, lr, optim, True)
            optim.zero_grad()
            out = model(x1, x2, s, text)
            loss = loss_func(out, label)
            loss.backward()
            total_loss += (loss.item() / label.size(0)) * args.batch_size
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()

            if (global_step + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                print('| step {:5} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step,
                                                                               total_loss,
                                                                               ms_per_batch))
                total_loss = 0.0
                start_time = time.time()
            global_step += 1

        model.eval()
        tr_roc_auc, tr_pr_auc, tr_avpre, tr_loss = eval_metric(train_dataloader, model, text_encoder, loss_func)
        d_roc_auc, d_pr_auc, d_avpre, d_loss = eval_metric(dev_dataloader, model, text_encoder, loss_func)
        t_roc_auc, t_pr_auc, t_avpre, t_loss = eval_metric(test_dataloader, model, text_encoder, loss_func)
        print('-' * 71)
        print('| step {:5} | train_auc {:7.4f} | dev_auc {:7.4f} | test_auc {:7.4f} '.format(global_step,
                                                                                             tr_roc_auc,
                                                                                             d_roc_auc,
                                                                                             t_roc_auc))
        print('| step {:5} | train_pr {:7.4f} | dev_pr {:7.4f} | test_pr {:7.4f} '.format(global_step,
                                                                                             tr_pr_auc,
                                                                                             d_pr_auc,
                                                                                             t_pr_auc))
        print('| step {:5} | train_avpre {:7.4f} | dev_avpre {:7.4f} | test_avpre {:7.4f} '.format(global_step,
                                                                                          tr_avpre,
                                                                                          d_avpre,
                                                                                          t_avpre))
        print('-' * 71)
        if d_pr_auc >= best_dev_auc:
            best_dev_auc = d_pr_auc
            final_test_auc = t_roc_auc
            best_dev_epoch = epoch_id
            torch.save([model, args], model_path)
            with open(log_path, 'a') as fout:
                fout.write('{},{},{},{}\n'.format(global_step, tr_pr_auc, d_pr_auc, t_pr_auc))
            print(f'models saved to {model_path}')
        if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
            break


    print()
    print('training ends in {} steps'.format(global_step))
    print('best dev auc: {:.4f} (at epoch {})'.format(best_dev_auc, best_dev_epoch))
    print('final test auc: {:.4f}'.format(final_test_auc))
    print()


def pred(args):
    print(args)
    model_path = os.path.join(args.save_dir, 'models.pt')
    model, old_args = torch.load(model_path)
    genotype = model.genotype()
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    datamodule = DataModule(device, task=old_args.target_disease, duration=old_args.duration)
    train_dataset, dev_dataset, test_dataset = datamodule.get_data()
    train_dataloader = DataLoader(train_dataset, old_args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, old_args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, old_args.batch_size, shuffle=False, collate_fn=collate_fn)
    model.to(device)
    model.eval()
    bert = Bert()
    bert.to(device)
    bert.eval()
    t_roc_auc, t_pr_auc, t_avpre, _ = eval_metric(test_dataloader, model, bert,
                                                  loss_func=nn.BCEWithLogitsLoss(reduction='mean'))
    log_path = os.path.join(args.save_dir, 'result.csv')
    with open(log_path, 'w') as fout:
        fout.write('test_auc,test_pr_auc,test_avpre\n')
        fout.write(
            '{},{},{}\n'.format(t_roc_auc, t_pr_auc, t_avpre))
        fout.write(str(genotype))


if __name__ == '__main__':
    main()