import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
from torch_geometric.nn import GCNConv
import numpy as np
from torch_geometric.nn import GATConv

b_xent = nn.BCEWithLogitsLoss()

# GNN
class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv):
        super(Encoder, self).__init__()
        self.conv = [base_model(in_channels, 2 * out_channels)]
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.activation = activation

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x1 = self.activation(self.conv[0](x, edge_index))
        x2 = self.activation(self.conv[1](x1, edge_index))
        return x2

# Generation
class Genetation(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation):
        super(Genetation, self).__init__()
        self.conv = GATConv(in_channels, out_channels, add_self_loops=False)
        self.activation = activation

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.activation(self.conv(x, edge_index))
        return x

# Model
class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, gene, tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.ge = gene

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_index2: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        z_g = self.ge(z, edge_index2)
        return z, z_g

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        return z

    def loss(self, z1, z2, adj, sub_g1):
        loss = self.sub_loss_batch(z1, z2, adj, sub_g1)
        return loss

    def sub_loss_batch(self, z, z_g, adj, sub_g1):
        subz_s, sub_gene_s = self.subg_centor(z, z_g, sub_g1)

        num = torch.randint(0, len(sub_g1)-1, [len(sub_g1),])
        if num[0] == 0:
            num[0] = 1
        for i in range(1, len(num)):
            if num[i] == i:
                num[i] -= 1
        subg2_s_n = subz_s[num] # disrupt
        sub_gene_s_n = sub_gene_s[num]

        input1 = torch.cat((subz_s, subz_s, subz_s), dim=0)
        input2 = torch.cat((sub_gene_s, subg2_s_n, sub_gene_s_n), dim=0)
        
        # adj
        subg1_adj = self.sub_adj(adj, sub_g1)
        input_adj = torch.cat((subg1_adj, subg1_adj, subg1_adj), dim=0)
        
        lbl_1 = torch.ones(len(sub_g1)).cuda()
        lbl_2 = torch.zeros(len(sub_g1)*2).cuda()
        lbl = torch.cat((lbl_1, lbl_2), 0).cuda()

        # WD
        wd, T_wd = self.wd(input1, input2, self.tau)
        logits = torch.exp(-wd / 0.01)
        loss1 = b_xent(torch.squeeze(logits), lbl)

        # GWD
        gwd = self.gwd(input1.transpose(2,1), input2.transpose(2,1), T_wd, input_adj, self.tau)
        logits2 = torch.exp(-gwd / 0.1)
        loss2 = b_xent(torch.squeeze(logits2), lbl)

        a = 0.5
        loss = a* loss1 + (1-a)* loss2        
        return loss

    # adj
    def sub_adj(self, adj, sub_g1):
        subg1_adj = torch.zeros(len(sub_g1), len(sub_g1[0]), len(sub_g1[0]))
        for i in range(len(sub_g1)):
            subg1_adj[i] = adj[sub_g1[i]].t()[sub_g1[i]]
        return subg1_adj


    def subg_centor(self, z, z_g, sub_g1):
        sub = [element for lis in sub_g1 for element in lis]
        subz = z[sub] 
        subg = z_g[sub]

        sub_s = subz.reshape(len(sub_g1), len(sub_g1[0]), -1)
        subg_s = subg.reshape(len(sub_g1), len(sub_g1[0]), -1)
        return sub_s, subg_s

    # WD
    def wd(self, x, y, tau):
        cos_distance = self.cost_matrix_batch(torch.transpose(x, 2, 1), torch.transpose(y, 2, 1), tau)
        cos_distance = cos_distance.transpose(1,2)

        beta = 0.1
        min_score = cos_distance.min()
        max_score = cos_distance.max()
        threshold = min_score + beta * (max_score - min_score)
        cos_dist = torch.nn.functional.relu(cos_distance - threshold)
        
        wd, T_wd = self.OT_distance_batch(cos_dist, x.size(0), x.size(1), y.size(1), 40)
        return wd, T_wd

    def OT_distance_batch(self, C, bs, n, m, iteration=50):
        C = C.float().cuda()
        T = self.OT_batch(C, bs, n, m, iteration=iteration)
        temp = torch.bmm(torch.transpose(C,1,2), T)
        distance = self.batch_trace(temp, m, bs)
        return distance, T
    
    def OT_batch(self, C, bs, n, m, beta=0.5, iteration=50):
        sigma = torch.ones(bs, int(m), 1).cuda()/float(m)
        T = torch.ones(bs, n, m).cuda()
        A = torch.exp(-C/beta).float().cuda()
        for t in range(iteration):
            Q = A * T
            for k in range(1):
                delta = 1 / (n * torch.bmm(Q, sigma))
                a = torch.bmm(torch.transpose(Q,1,2), delta)
                sigma = 1 / (float(m) * a)
            T = delta * Q * sigma.transpose(2,1)
        return T

    def cost_matrix_batch(self, x, y, tau=0.5):
        bs = list(x.size())[0]
        D = x.size(1)
        assert(x.size(1)==y.size(1))
        x = x.contiguous().view(bs, D, -1)
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        
        cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)
        cos_dis = torch.exp(- cos_dis / tau)
        return cos_dis.transpose(2,1)

    def batch_trace(self, input_matrix, n, bs):
        a = torch.eye(n).cuda().unsqueeze(0).repeat(bs, 1, 1)
        b = a * input_matrix
        return torch.sum(torch.sum(b,-1),-1).unsqueeze(1)

    # GWD
    def gwd(self, X, Y, T_wd, input_adj, tau, lamda=1e-1, iteration=5, OT_iteration=20):
        m = X.size(2)
        n = Y.size(2)
        bs = X.size(0)
        p = (torch.ones(bs, m, 1)/m).cuda()
        q = (torch.ones(bs, n, 1)/n).cuda()
        return self.GW_distance(X, Y, p, q, T_wd, input_adj, tau, lamda=lamda, iteration=iteration, OT_iteration=OT_iteration)

    def GW_distance(self, X, Y, p, q, T_wd, input_adj, tau, lamda=0.5, iteration=5, OT_iteration=20):
        cos_dis = torch.exp(- input_adj / tau).cuda() 
        beta = 0.1
        min_score = cos_dis.min()
        max_score = cos_dis.max()
        threshold = min_score + beta * (max_score - min_score)
        res = cos_dis - threshold
        Cs = torch.nn.functional.relu(res.transpose(2,1))

        Ct = self.cos_batch(Y, Y, tau).float().cuda()
        bs = Cs.size(0)
        m = Ct.size(2)
        n = Cs.size(2)
        T, Cst = self.GW_batch(Cs, Ct, bs, n, m, p, q, beta=lamda, iteration=iteration, OT_iteration=OT_iteration)
        temp = torch.bmm(torch.transpose(Cst,1,2), T_wd)
        distance = self.batch_trace(temp, m, bs)
        return distance

    def GW_batch(self, Cs, Ct, bs, n, m, p, q, beta=0.5, iteration=5, OT_iteration=20):
        one_m = torch.ones(bs, m, 1).float().cuda()
        one_n = torch.ones(bs, n, 1).float().cuda()

        Cst = torch.bmm(torch.bmm(Cs**2, p), torch.transpose(one_m, 1, 2)) + \
            torch.bmm(one_n, torch.bmm(torch.transpose(q,1,2), torch.transpose(Ct**2, 1, 2)))
        gamma = torch.bmm(p, q.transpose(2,1))
        for i in range(iteration):
            C_gamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
            gamma = self.IPOT_torch_batch_uniform(C_gamma, bs, n, m, beta=beta, iteration=OT_iteration)
        Cgamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
        return gamma.detach(), Cgamma

    def cos_batch(self, x, y, tau):
        bs = x.size(0)
        D = x.size(1)
        assert(x.size(1)==y.size(1))
        x = x.contiguous().view(bs, D, -1)
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = torch.bmm(torch.transpose(x,1,2), y)
        cos_dis = torch.exp(- cos_dis / tau).transpose(1,2)
        
        beta = 0.1
        min_score = cos_dis.min()
        max_score = cos_dis.max()
        threshold = min_score + beta * (max_score - min_score)
        res = cos_dis - threshold
        return torch.nn.functional.relu(res.transpose(2,1))


