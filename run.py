import argparse
import os
import random
import sys
import time
from collections import defaultdict

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers.optimization import WarmupCosineSchedule


from model import CLHGT
from utils.data import load_data_group
from utils.pytorchtools import EarlyStopping

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.append('../../')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def struct_set(dataset='DBLP'):
    struct_list = []
    group_list = []
    if dataset == 'DBLP':
        name_list = ['A-P-A-P-A', 'A-P-V-P-A', 'A-P-T-P-A']
        edge_dict = {'A-P': 0, 'P-T': 1, 'P-V': 2,
                     'P-A': 3, 'T-P': 4, 'V-P': 5}
        for x in name_list:
            list_type = x.replace('-', ' ').split()
            edge_list = []
            for i in range(len(list_type)-1):
                edge_list.append(
                    edge_dict[str(list_type[i])+'-'+str(list_type[i+1])])
            struct_list.append(edge_list)
    elif dataset == 'ACM':
        # Special case P-P contains 0,1 two edge types.
        name_list = ['P-P-P-P', 'P-P-A-P', 'P-P-S-P']
        edge_dict = {'P-P': 0, 'P-A': 2, 'A-P': 3, 'P-S': 4, 'S-P': 5}
        group_list = [0, 1]
        for x in name_list:
            list_type = x.replace('-', ' ').split()
            edge_list = []
            for i in range(len(list_type)-1):
                edge_list.append(
                    edge_dict[str(list_type[i])+'-'+str(list_type[i+1])])
            struct_list.append(edge_list)
    elif dataset == 'IMDB':
        name_list = ['M-A-M-A-M', 'M-U-M-U-M', 'M-D-M-D-M']
        edge_dict = {'M-A': 0, 'M-U': 1, 'M-D': 2,
                     'A-M': 3, 'U-M': 4, 'D-M': 5}
        group_list = [0, 2, 4]
        for x in name_list:
            list_type = x.replace('-', ' ').split()
            edge_list = []
            for i in range(len(list_type)-1):
                edge_list.append(
                    edge_dict[str(list_type[i])+'-'+str(list_type[i+1])])
            struct_list.append(edge_list)
    return struct_list, group_list


def node2seq_fixlength(graph, nodes, k, maxlen):
    seqs = []
    for node in nodes:
        seq = []
        current_level = [node]
        for _ in range(k):
            next_level = []
            for root in current_level:
                neighbor_list = graph.successors(root).numpy().tolist()
                next_level.extend(neighbor_list)
                seq.extend(neighbor_list)
                if len(seq) >= maxlen:
                    break
            if len(seq) >= maxlen:
                break
            current_level = next_level
        seq = seq[:maxlen]
        if len(seq) < maxlen:
            seq.extend([node] * (maxlen - len(seq)))
        seqs.append(list(seq))
    return torch.tensor(seqs).long()


def sampled_list(dl, node_cnt, dataset='DBLP', num_sample=20):
    group_ind = node_cnt.copy()
    group_ind.insert(0, 0)

    struct_list, group_list = struct_set(dataset)

    n_sample = int(num_sample / len(struct_list))

    ifacm = False

    sampled_seqs = []
    if dataset == 'ACM':
        ifacm = True
    for x in struct_list:
        sampled_struct = []
        sp_adj = dl.get_meta_struct(x, ifacm, group_list)
        nonzero_indices = sp_adj.nonzero()
        neighbor_lists = defaultdict(list)
        for row_idx, col_idx in zip(nonzero_indices[0], nonzero_indices[1]):
            neighbor_lists[row_idx].append(col_idx)
        for i in range(node_cnt[0]):
            if len(neighbor_lists[i]) == 0:
                neighbor_lists[i] = [i]
            if n_sample > len(neighbor_lists[i]):
                sampled_n = np.random.choice(
                    neighbor_lists[i], size=n_sample, replace=True)
            else:
                sampled_n = np.random.choice(
                    neighbor_lists[i], size=n_sample, replace=False)
            sampled_struct.append(sampled_n)
        sampled_struct = torch.from_numpy(np.vstack(sampled_struct))
        sampled_seqs.append(sampled_struct)
    sampled_seqs = torch.cat(sampled_seqs, dim=1).long()

    return sampled_seqs, n_sample


def run_model_DBLP(args):
    set_seed(2023)
    t_start = time.time()
    features_list, adjM, labels, train_val_test_idx, dl = load_data_group(
        args.dataset)
    device = torch.device('cuda:' + str(args.device)
                          if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    node_cnt = [features.shape[0] for features in features_list]
    sum_node = 0
    for x in node_cnt:
        sum_node += x
    in_dims = [features.shape[1] for features in features_list]

    labels = torch.LongTensor(labels).to(device)


    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)

    t_end = time.time()

    print("Data loading finished! Time consumption: {:.4f}".format(
        t_end - t_start))

    if args.dataset == 'ACM':
        gnn_l = 2
        drop = 1 + ns * 2
        pick = 1 + ns * 2
        weight_decay = 1e-3
        lr = 1e-3
        batchsize = 512
        temper = 0.8
        tau = 1.5
        sample_s = 25
        sample_h = 75
    elif args.dataset == 'DBLP':
        gnn_l = 4
        drop = 1
        pick = 1
        weight_decay = 1e-3
        lr = 5e-4
        batchsize = 1024
        temper = 0.6
        tau = 1.5
        sample_s = 25
        sample_h = 75
    else:
        gnn_l = 3
        drop = 1 + ns * 2
        pick = 1 + ns * 2
        weight_decay = 1e-3
        lr = 1e-4
        batchsize = 512
        temper = 1.5
        tau = 0.9
        sample_s = 90
        sample_h = 30

    t_start = time.time()

    local_seq = node2seq_fixlength(g, nodes=np.arange(
        node_cnt[0]), k=4, maxlen=sample_s)
    hete_seq, ns = sampled_list(
        dl, node_cnt=node_cnt, dataset=args.dataset, num_sample=sample_h)

    node_seq = torch.cat(
        [torch.arange(node_cnt[0]).unsqueeze(1), hete_seq, local_seq], dim=1)

    t_end = time.time()

    print("Context sampling finished! Time consumption: {:.4f}".format(
        t_end - t_start))

    t_start = time.time()

    node_type = [i for i, z in zip(
        range(len(node_cnt)), node_cnt) for x in range(z)]
    
    g = g.to(device)

    micro_f1 = torch.zeros(args.repeat)
    macro_f1 = torch.zeros(args.repeat)

    num_classes = dl.labels_train['num_classes']
    node_type = torch.tensor(node_type).to(device)

    
    net = CLHGT(g, num_classes, in_dims, args.hidden_dim, args.num_layers, gnn_l,
                args.num_heads, args.dropout, temper=temper, tau=tau)

    net.to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=int(0.1 * args.epoch), t_total=args.epoch)

    # training loop
    net.train()

    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(args.epoch):

        if epoch > 0 and epoch % 10 == 0:

            hete_seq, _ = sampled_list(
                dl, node_cnt=node_cnt, dataset=args.dataset, num_sample=sample_h)

            node_seq = torch.cat(
                [torch.arange(node_cnt[0]).unsqueeze(1), hete_seq, local_seq], dim=1)

        t_start = time.time()

        unsup_loss = 0

        idx = np.random.permutation(node_cnt[0])

        if epoch < int(0.1 * args.epoch):
            for batch in range(0, node_cnt[0], batchsize):
                batch_seq = node_seq[idx][batch:batch +
                                          batchsize, :].to(device)

                u_loss = net.position_enc(features_list, batch_seq)

                unsup_loss += u_loss.item()

                loss = u_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:

            for batch in range(0, node_cnt[0], batchsize):
                batch_seq = node_seq[idx][batch:batch +
                                          batchsize, :]
                pairs = torch.randperm(batch_seq.shape[0])
                batch_seq = torch.cat((batch_seq[:, :drop], batch_seq[:, drop+ns:], batch_seq[pairs]
                                       [:, 0].unsqueeze(1), batch_seq[pairs][:, pick:pick+ns]), dim=1).to(device)

                u_loss = net.position_enc(features_list, batch_seq)

                unsup_loss += u_loss.item()

                loss = u_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        t_end = time.time()

        # print training info
        print('Epoch {:05d} | Unsup_Loss: {:.4f} | Time: {:.4f}'.format(
            epoch, unsup_loss, t_end-t_start))

        if unsup_loss < best - 1e-4:
            best = unsup_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(net.state_dict(
            ), 'checkpoint/dmgipre_{}_{}.pt'.format(args.dataset, args.device))
        else:
            cnt_wait += 1
        '''
        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        '''
        scheduler.step()

    print('Unsupervised learning finished. Best epoch: ', best_t)

    for k in range(0, 3):

        train_idx = train_val_test_idx['train_'+str((k+1)*10)]
        train_idx = np.sort(train_idx)
        val_idx = train_val_test_idx['val_'+str((k+1)*10)]
        val_idx = np.sort(val_idx)
        test_idx = train_val_test_idx['test_'+str((k+1)*10)]
        test_idx = np.sort(test_idx)

        for i in range(args.repeat):

            net.load_state_dict(torch.load(
                'checkpoint/dmgipre_{}_{}.pt'.format(args.dataset, args.device)))
            net.reset_dropout(args.dropout_ft)

            optimizer_ft = torch.optim.AdamW(
                net.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler_ft = WarmupCosineSchedule(optimizer_ft, warmup_steps=int(
                0.1 * args.epoch_nc), t_total=args.epoch_nc)

            print('Supervised training!')

            cnt_wait = 0
            best = 1e9
            best_t = 0

            for epoch in range(args.epoch_nc):

                if epoch % 10 == 0:
                    node_seq, _ = sampled_list(
                        dl, node_cnt=node_cnt, dataset=args.dataset, num_sample=args.sample_h)

                    node_seq = torch.cat([torch.arange(node_cnt[0]).unsqueeze(
                        1), node_seq, local_seq], dim=1).long()

                    train_seq = node_seq[train_idx].to(device)
                    val_seq = node_seq[val_idx].to(device)

                net.train()
                optimizer_ft.zero_grad()

                t_start = time.time()

                logits = net.forward_g(features_list, train_seq)
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp, labels[train_idx])

                train_loss.backward()
                optimizer_ft.step()

                t_end = time.time()

                print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(
                    epoch, train_loss.item(), t_end-t_start))

                net.eval()
                with torch.no_grad():
                    logits = net.forward_g(
                        features_list, val_seq)
                    logp = F.log_softmax(logits, 1)
                    val_loss = F.nll_loss(logp, labels[val_idx])
                    pred = logits.cpu().numpy().argmax(axis=1)
                    print(dl.evaluate_valid(
                        pred, labels[val_idx].cpu().numpy()))
                    print('Epoch {:05d} | Val_Loss {:.4f}'.format(
                        epoch, val_loss.item()))

                if val_loss < best:
                    best = val_loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(net.state_dict(
                    ), 'checkpoint/dmgift_{}_{}.pt'.format(args.dataset, args.device))
                else:
                    cnt_wait += 1

                if cnt_wait == args.patience:
                    print('Early stopping!')
                    break

                scheduler_ft.step()

            net.load_state_dict(torch.load(
                'checkpoint/dmgift_{}_{}.pt'.format(args.dataset, args.device)))
            net.eval()

            with torch.no_grad():
                for t in range(10):
                    node_seq, _ = sampled_list(
                        dl, node_cnt=node_cnt, dataset=args.dataset, num_sample=args.sample_h)

                    node_seq = torch.cat([torch.arange(node_cnt[0]).unsqueeze(
                        1), node_seq, local_seq], dim=1).long()

                    test_seq = node_seq[test_idx].to(device)

                    logits = net.forward_g(
                        features_list, test_seq)
                    if t == 0:
                        test_logits = logits
                    else:
                        test_logits += logits
                pred = test_logits.cpu().numpy().argmax(axis=1)
                result = dl.evaluate_valid(
                    pred, labels[test_idx].cpu().numpy())
                print(result)

                micro_f1[i] = result['micro-f1']
                macro_f1[i] = result['macro-f1']

        print("nlabel: ", str((k+1)*10))

        print('Micro-f1: %.4f, std: %.4f' %
              (micro_f1.mean().item(), micro_f1.std().item()))
        print('Macro-f1: %.4f, std: %.4f' %
              (macro_f1.mean().item(), macro_f1.std().item()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='CLHGT')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--hidden-dim', type=int, default=256,
                    help='Dimension of the node hidden state. Default is 32.')
    ap.add_argument('--rl-dim', type=int, default=4,
                    help='Dimension of the rl layer. Default is 4.')
    ap.add_argument('--num-heads', type=int, default=2,
                    help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=500,
                    help='Number of epochs.')
    ap.add_argument('--epoch-nc', type=int, default=500,
                    help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=50, help='Patience.')
    ap.add_argument('--repeat', type=int, default=5,
                    help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=4)
    ap.add_argument('--num-gnns', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-ft', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--dropout-ft', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=0)
    ap.add_argument('--weight-decay-ft', type=float, default=0)
    ap.add_argument('--len-seq', type=int, default=100)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--temper', type=float, default=1.0)
    ap.add_argument('--batchsize', type=int, default=512)
    ap.add_argument('--rep', type=float, default=0.1)
    ap.add_argument('--sample-s', type=int, default=25)
    ap.add_argument('--sample-h', type=int, default=75)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--drop', type=int, default=0)
    ap.add_argument('--pick', type=int, default=0)

    args = ap.parse_args()
    run_model_DBLP(args)
