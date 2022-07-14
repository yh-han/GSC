#<code># -- coding:UTF-8 --<code>
import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
# import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid

# from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
# from torch.utils.data import random_split

from model import Encoder, Encoder1, Model
from collections import defaultdict
from torch_geometric.utils import remove_self_loops, add_self_loops
import function as func
import logreg as logreg
import scipy.sparse as sp
import numpy as np

node_neighbor = {}
best = 1e9
best_t = 0
bestacc = 0

def train(model, x, edge_index, edge_index2, adj, node_neighbor_cen):
    model.train()
    optimizer.zero_grad()

    z1, z2 = model(x, edge_index, edge_index2)
    loss = model.loss(z1, z2, adj, node_neighbor_cen)
    loss.backward()
    optimizer.step()

    return loss.item(), loss.item(), loss.item()


def test(model, data, idx_train, idx_val, idx_test, num):
    model.eval()
    embeds = model.embed(data.x, data.edge_index).detach()
    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = data.y[idx_train]
    val_lbls = data.y[idx_val]
    test_lbls = data.y[idx_test]

    accs = []
    xent = nn.CrossEntropyLoss()

    for _ in range(num): 
        log = logreg.LogReg(num_hidden, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0 )
        log.cuda()

        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward(retain_graph=True)
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)

    maxnum = max(accs)
    minnum = min(accs)
    accs = torch.stack(accs)
    print(accs.mean().item(), accs.std().item(), '###', maxnum.item(), minnum.item())
    return accs.mean().item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--sub', type=int, default= 2)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--savepath', type=str, default='save/Cora.pkl')
    
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]

    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    num, k1 = config['num'], config['k1']

    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = Planetoid(path, args.dataset)
    nb_classes = dataset.num_classes
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    edge = data.edge_index.clone()
    adj = sp.coo_matrix((np.ones(edge.shape[1]), (edge.cpu()[0, :], edge.cpu()[1, :]) ), 
                                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32).toarray()
    adj = torch.from_numpy(adj).cuda()

    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask
    train_lbls = data.y[idx_train]
    test_lbls = data.y[idx_test]

    encoder = Encoder(dataset.num_features, num_hidden, activation, base_model=base_model).to(device)
    gene = Encoder1(num_hidden, num_hidden, activation).to(device)
    model = Model(encoder, gene, tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate , weight_decay=weight_decay)
    start = t()
    prev = start

    adj_lists = defaultdict(set)
    data.edge_index, _ = add_self_loops(data.edge_index)
    # 2*xxx -- N*xx 
    for i in range(data.edge_index.size(1)):
        adj_lists[data.edge_index[0][i].item()].add(data.edge_index[1][i].item())
    edge_index2, _ = remove_self_loops(data.edge_index)
    for epoch in range(1, num_epochs + 1):
        nodes_batch = torch.randint(0, data.num_nodes, (num, )) 
        node_neighbor_cen, node_neighbor, node_centor = func.sub_sam(nodes_batch, adj_lists, k1)
        loss, loss1, loss2 = train(model, data.x, data.edge_index, edge_index2, adj, node_neighbor_cen)

        if epoch%10==0:
            acc = test(model, data, idx_train, idx_val, idx_test, 5)
            if acc > bestacc:
                bestacc = acc
                best_t = epoch
                torch.save(model.state_dict(), args.savepath)
            print(f'acc={acc}, epoch={epoch}, bestacc={bestacc}, bestepoch={best_t}')

        now = t()
        print(f'(T)|E={epoch:03d},l={loss:.4f},t={now - prev:.2f},all={now - start:.2f},'
              f'{num},{k1},{args.dataset},{learning_rate}--{loss1:.4f},{loss2:.4f}')
        prev = now
    
    print(config)
    print("=== Final ===")
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(args.savepath))
    acc = test(model, data, idx_train, idx_val, idx_test, 50) 
    print(f'bestacc={bestacc}, bestepoch={best_t}')
    print(f'acc={acc}')
