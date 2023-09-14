import time
import dgl
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit

from model import HAN
from utils import setup_seed, read_graph, eval_metric
from args import parameter_parser

args = parameter_parser()
setup_seed(args.seed, torch.cuda.is_available())
print(args)

# read graph and split train/test indices
graph = read_graph(args.dataset)
all_labels = np.array(np.argmax(graph['labels'], axis=1))
num_etypes = np.max(all_labels) + 1

ss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size)
for train_idx, test_idx in ss.split(graph["edges"], graph["labels"]):
    train_idx.sort()
    test_idx.sort()
    train_labels = all_labels[train_idx]
    test_labels = all_labels[test_idx]
    train_edges = graph['edges'][train_idx]
    test_edges = graph['edges'][test_idx]

adjs = [np.zeros((graph["ncount"], graph["ncount"])) for _ in range(num_etypes)]
for i, edge in enumerate(train_edges):
    adjs[train_labels[i]][edge[0], edge[1]] = 1
adjs = [sparse.csc_matrix(x) for x in adjs]

## construct topology for each path
## for advogato and pgp
## 1-order
prop_path_1 = [[1],[2],[3],[4]]
adjs_1 = [adjs[path[0]-1] for path in prop_path_1]

## 2-order
prop_path_2 = [[1, 1], [1, 2], [1, 3], [1, 4],
             [2, 1], [2, 2], [2, 3], [2, 4],
             [3, 1], [3, 2], [3, 3], [3, 4],
             [4, 1], [4, 2], [4, 3], [4, 4]]
adjs_2 = [adjs[path[0]-1].dot(adjs[path[1]-1]) for path in prop_path_2]


## for ciao and epinions
## 1-order
# prop_path_1 = [[1],[2]]
# adjs_1 = [adjs[path[0]-1] for path in prop_path_1]

## 2-order
# prop_path_2 = [[1, 1], [1, 2],
#              [2, 1], [2, 2]]
# adjs_2 = [adjs[path[0]-1].dot(adjs[path[1]-1]) for path in prop_path_2]


# (1+2)order
prop_path = prop_path_1 + prop_path_2
adjs = adjs_1 + adjs_2

adjs = [(x != 0).astype(int) for x in adjs]


# construct dgl graph
adjs = [dgl.from_scipy(x) for x in adjs]
if args.no_self_loop:
    adjs = [dgl.remove_self_loop(g) for g in adjs]
else:
    adjs = [dgl.add_self_loop(g) for g in adjs]

# pouring to pytorch
train_labels = torch.LongTensor(train_labels).to(args.device)
test_labels = torch.LongTensor(test_labels).to(args.device)
train_edges = torch.LongTensor(train_edges).to(args.device)
test_edges = torch.LongTensor(test_edges).to(args.device)
adjs = [x.to(args.device) for x in adjs]


# in & out
adjs_out = adjs
adjs_in = [dgl.reverse(g) for g in adjs]

for i in range(num_etypes):
    for j in range(num_etypes-1, i, -1):
        # [a, b] corresponds to [b, a] when reversed
        adjs_in[i * num_etypes + j], adjs_in[j * num_etypes + i] = adjs_in[j * num_etypes + i], adjs_in[i * num_etypes + j]

mean_f1 = []
mean_mae = []
for t in range(10):
    # init model
    net = HAN(num_paths=len(adjs), num_etypes=num_etypes, num_nodes=graph["ncount"], out_size=num_etypes,
              path=prop_path, args=args)
    net.to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('model init finish')

    # training loop
    best_f1, best_mae, best_epoch = 0., 9e15, -1
    print('training...\n')
    for epoch in range(args.epochs):
        t = time.time()
        net.train()
        _, logits = net(adjs_in, adjs_out, train_edges)
        logp = F.log_softmax(logits, 1)
        train_loss = F.nll_loss(logp, train_labels)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_time = time.time() - t

        t = time.time()
        net.eval()
        with torch.no_grad():
            _, logits = net(adjs_in, adjs_out, test_edges)
        ac, f1_micro, f1_macro, f1_weighted, mae, rmse= eval_metric(
            logits.detach().cpu().numpy(), test_labels.cpu().numpy())
        test_time = time.time() - t

        print('epoch:{:03d} | train_loss:{:.6f} | train_time:{:.2f} | test_time:{:.2f} | '
              'F1_Micro:{:.4f} | F1_macro:{:.4f} | F1_Weighted:{:.4f} | MAE:{:.4f} | RMSE:{:.4f}'.format(
            epoch, train_loss.item(), train_time, test_time, f1_micro, f1_macro, f1_weighted, mae, rmse))


        if best_f1 < f1_micro:
            best_f1 = f1_micro
            best_mae = mae
            best_epoch = epoch
            torch.save(net.state_dict(), 'checkpoint/{}.pth'.format(args.dataset))

    print('best_epoch: {:3d} - {:.6f} - {:.4f}'.format(best_epoch, best_f1, best_mae))


    mean_f1.append(best_f1)
    mean_mae.append(best_mae)

mean_f1 = np.mean(mean_f1, axis=0)
mean_mae = np.mean(mean_mae, axis=0)
print(mean_f1)
print(mean_mae)
print('Done')



