import math

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import precision_score
from torch.nn import init


import torch


def sample_without_d(n, s, d):
    samples = torch.randperm(n - 1)[:s]
    if d in samples:
        samples = samples[samples != d]
        new_sample = torch.randint(0, n, (1,))
        while new_sample == d or new_sample in samples:
            new_sample = torch.randint(0, n, (1,))
        samples = torch.cat((samples, new_sample))

    return samples


class GTLayer(nn.Module):
    def __init__(self, embeddings_dimension, nheads=2, dropout=0.5, temper=1.0, rl=False, rl_dim=4, alpha=0.5):
        '''
            embeddings_dimension: d = dp = dk = dq
            multi-heads: n
            
        '''

        super(GTLayer, self).__init__()

        self.nheads = nheads
        self.embeddings_dimension = embeddings_dimension
        self.dropout = dropout

        self.head_dim = self.embeddings_dimension // self.nheads

        self.rl_dim = rl_dim

        self.temper = temper
        self.alpha = alpha

        self.linear_k = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)
        self.linear_v = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)
        self.linear_q = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)

        if rl:
            self.r_k = nn.Linear(rl_dim, rl_dim, bias=False)
            self.r_q = nn.Linear(rl_dim, rl_dim, bias=False)

        self.linear_final = nn.Linear(
            self.head_dim * self.nheads, self.embeddings_dimension, bias=False)
        self.dropout_att = nn.Dropout(self.dropout)
        self.dropout_mlp = nn.Dropout(self.dropout)
        self.dropout_msa = nn.Dropout(self.dropout)
        self.dropout_ffn = nn.Dropout(self.dropout)

        self.activation = nn.LeakyReLU(0.2)

        self.FFN1 = nn.Linear(embeddings_dimension, embeddings_dimension)
        self.FFN2 = nn.Linear(embeddings_dimension, embeddings_dimension)
        self.LN1 = nn.LayerNorm(embeddings_dimension)
        self.LN2 = nn.LayerNorm(embeddings_dimension)

    def forward(self, h, dist=None, rh=None, mask=None, e=1e-12):
        q = self.linear_q(h)
        k = self.linear_k(h)
        v = self.linear_v(h)
        batch_size = k.size()[0]

        q_ = q.reshape(batch_size, -1, self.nheads,
                       self.head_dim).transpose(1, 2)
        k_ = k.reshape(batch_size, -1, self.nheads,
                       self.head_dim).transpose(1, 2)
        v_ = v.reshape(batch_size, -1, self.nheads,
                       self.head_dim).transpose(1, 2)

        batch_size, head, length, d_tensor = k_.size()
        k_t = k_.transpose(2, 3)
        score = (q_ @ k_t) / math.sqrt(d_tensor)
        if rh is not None:
            r_k = self.r_k(rh)
            r_q = self.r_q(rh)
            r_k_ = r_k.unsqueeze(1).transpose(1, 2)
            r_q_ = r_q.unsqueeze(1).transpose(1, 2)
            r_k_t = r_k_.transpose(2, 3)
            score = score + self.alpha * \
                (r_q_ @ r_k_t) / math.sqrt(self.rl_dim)

        if dist is not None:
            dist = dist.reshape(batch_size, 1, length, 1) * -self.alpha
            dist = dist.exp()
            score = torch.mul(score, dist)

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        score = F.softmax(score / self.temper, dim=-1)
        score = self.dropout_att(score)
        context = score @ v_

        h_sa = context.transpose(1, 2).reshape(
            batch_size, -1, self.head_dim * self.nheads)
        h_sa = self.activation(self.linear_final(h_sa))

        h_sa = self.dropout_msa(h_sa)
        h1 = self.LN1(h_sa + h)

        hf = self.activation(self.FFN1(h1))
        hf = self.dropout_mlp(hf)
        hf = self.FFN2(hf)
        hf = self.dropout_ffn(hf)

        h2 = self.LN2(h1+hf)
        return h2

class CLHGT(nn.Module):

    def __init__(self, g, num_class, input_dimensions, embeddings_dimension=64,  num_layers=2, num_GNNs=2, nheads=2, dropout=0, temper=1.0, alpha=1, tau=1.0):
        '''
            embeddings_dimension: d = dp = dk = dq
            multi-heads: n
            
        '''
        super(CLHGT, self).__init__()

        self.g = g

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_GNNs = num_GNNs
        self.num_class = num_class
        self.nheads = nheads
        self.temper = temper
        self.alpha = alpha
        self.tau = tau
        self.fc_list = nn.ModuleList([nn.Linear(
            in_dim, embeddings_dimension) for in_dim in input_dimensions])
        self.fc2_list = nn.ModuleList(
            [nn.Linear(embeddings_dimension, embeddings_dimension) for _ in input_dimensions])
        self.activation = nn.LeakyReLU(0.2)

        self.dropout = dropout

        self.GCNLayers = torch.nn.ModuleList()
        self.GTLayers = torch.nn.ModuleList()

        for layer in range(self.num_GNNs):
            self.GCNLayers.append(GraphConv(
                self.embeddings_dimension, self.embeddings_dimension, activation=None, weight=False, bias=False))
        for layer in range(self.num_layers):
            self.GTLayers.append(
                GTLayer(self.embeddings_dimension, self.nheads, self.dropout, temper=self.temper, alpha=self.alpha))

        self.predictor = nn.Linear(embeddings_dimension, num_class)

        self.proj = nn.Sequential(
            nn.Linear(embeddings_dimension, embeddings_dimension),
            nn.ELU(),
            nn.Linear(embeddings_dimension, embeddings_dimension)
        )

    def forward(self, features_list, seqs):
        h = []
        for fc, fc2, feature in zip(self.fc_list, self.fc2_list, features_list):
            h.append(fc2(self.activation(fc(feature))))

        h = torch.cat(h, 0)
        h = h[seqs]
        for layer in range(self.num_layers):
            h = self.GTLayers[layer](h)
        output = self.predictor(h[:, 0, :])
        output = output / (torch.norm(output, dim=1, keepdim=True)+1e-12)
        return output

    def pretrain(self, features_list, seqs):
        h = []
        for fc, fc2, feature in zip(self.fc_list, self.fc2_list, features_list):
            h.append(fc2(self.activation(fc(feature))))
        h1 = torch.cat(h, 0)
        h2 = h1.clone()
        h1 = h1[seqs]
        for layer in range(self.num_GNNs):
            h2 = self.GCNLayers[layer](self.g, h2)
        for layer in range(self.num_layers):
            h1 = self.GTLayers[layer](h1)

        output1 = h1[:, 0, :].reshape(-1, self.embeddings_dimension)
        output2 = h2[seqs[:, 0].reshape(-1,)]

        loss = self.contrast(output1, output2)

        return loss

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def contrast(self, z_1, z_2):
        z_proj_1 = self.proj(z_1)
        z_proj_2 = self.proj(z_2)
        matrix_1 = self.sim(z_proj_1, z_proj_2)
        matrix_2 = matrix_1.t()

        matrix_1 = matrix_1 / \
            (torch.sum(matrix_1, dim=1).view(-1, 1) + 1e-8)
        lori_1 = - \
            torch.log(torch.diag(matrix_1)).mean()

        matrix_2 = matrix_2 / \
            (torch.sum(matrix_2, dim=1).view(-1, 1) + 1e-8)
        lori_2 = - \
            torch.log(torch.diag(matrix_2)).mean()
        return 0.5 * (lori_1 + lori_2)

    def reset_dropout(self, p=0.5):
        for layer in range(self.num_layers):
            self.GTLayers[layer].dropout_att.p = p
            self.GTLayers[layer].dropout_msa.p = p
            self.GTLayers[layer].dropout_ffn.p = p
            self.GTLayers[layer].dropout_mlp.p = p


class CLHGT_CD(nn.Module):
    def __init__(self, g, num_class, input_dimensions, embeddings_dimension=64,  num_layers=2, num_GNNs=2, nheads=2, dropout=0, temper=1.0, alpha=1, tau=1.0):
        '''
            embeddings_dimension: d = dp = dk = dq
            multi-heads: n
            
        '''
        super(CLHGT_CD, self).__init__()

        self.g = g

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_GNNs = num_GNNs
        self.num_class = num_class
        self.nheads = nheads
        self.temper = temper
        self.alpha = alpha
        self.tau = tau
        self.fc_list = nn.ModuleList([nn.Linear(
            in_dim, embeddings_dimension) for in_dim in input_dimensions])
        self.fc2_list = nn.ModuleList(
            [nn.Linear(embeddings_dimension, embeddings_dimension) for _ in input_dimensions])
        self.activation = nn.LeakyReLU(0.2)

        self.dropout = dropout

        self.GCNLayers = torch.nn.ModuleList()
        self.GTLayers = torch.nn.ModuleList()

        for layer in range(self.num_GNNs):
            self.GCNLayers.append(GraphConv(
                self.embeddings_dimension, self.embeddings_dimension, activation=None, weight=False, bias=False))
        for layer in range(self.num_layers):
            self.GTLayers.append(
                GTLayer(self.embeddings_dimension, self.nheads, self.dropout, temper=self.temper, alpha=self.alpha))

        self.sig = nn.Sigmoid()
        self.predictor = nn.Linear(embeddings_dimension, num_class)

        self.proj = nn.Sequential(
            nn.Linear(embeddings_dimension, embeddings_dimension),
            nn.ELU(),
            nn.Linear(embeddings_dimension, embeddings_dimension)
        )

        self.score_matrix = nn.Parameter(torch.FloatTensor(
            embeddings_dimension, embeddings_dimension))
        nn.init.kaiming_uniform_(self.score_matrix)

    def forward(self, features_list, seqs):
        h = []
        for fc, fc2, feature in zip(self.fc_list, self.fc2_list, features_list):
            h.append(fc2(self.activation(fc(feature))))

        h = torch.cat(h, 0)
        h = h[seqs]
        for layer in range(self.num_layers):
            h = self.GTLayers[layer](h)
        output = self.predictor(h[:, 0, :])
        output = output / (torch.norm(output, dim=1, keepdim=True)+1e-12)
        return output

    def con_div(self, features_list, origin_seq, aug_seq, lenlist, topk):
        h = []
        for fc, fc2, feature in zip(self.fc_list, self.fc2_list, features_list):
            h.append(fc2(self.activation(fc(feature))))
        h1 = torch.cat(h, 0)
        h2 = h1[origin_seq].clone()
        h3 = h1[aug_seq].clone()
        for layer in range(self.num_GNNs):
            h1 = self.GCNLayers[layer](self.g, h1)
        for layer in range(self.num_layers):
            h2 = self.GTLayers[layer](h2)
            h3 = self.GTLayers[layer](h3)


        div = torch.norm(self.proj(h2[:, 0, :]) -
                         self.proj(h3[:, 0, :]), p=2, dim=1).mean()

        target = self.proj(h1[origin_seq[:, 0].reshape(-1,)])
        score = self.sig(
            torch.mm(torch.mm(self.proj(h2[:, 0, :]), self.score_matrix), target.T))

        first_score = torch.diag(score)

        list_ind = []

        for x in range(origin_seq.size(0)):
            row = sample_without_d(origin_seq.size(0), lenlist, x)
            list_ind.append(row)
        list_ind = torch.stack(list_ind).long().to(h3.device)

        score = score[torch.arange(score.size(0)).unsqueeze(1), list_ind]

        k = max(round(lenlist * topk), 1)

        values, _ = torch.topk(score, k=k, dim=1, largest=True)

        gap_score = values.reshape(-1, k)[:, -1]

        con = torch.sum(first_score >= gap_score).item() / score.size(0)

        return con, div

    def pretrain(self, features_list, seqs):
        h = []
        for fc, fc2, feature in zip(self.fc_list, self.fc2_list, features_list):
            h.append(fc2(self.activation(fc(feature))))
        h1 = torch.cat(h, 0)
        h2 = h1.clone()
        h1 = h1[seqs]
        for layer in range(self.num_GNNs):
            h2 = self.GCNLayers[layer](self.g, h2)
        for layer in range(self.num_layers):
            h1 = self.GTLayers[layer](h1)

        output1 = h1[:, 0, :].reshape(-1, self.embeddings_dimension)
        output2 = h2[seqs[:, 0].reshape(-1,)]

        loss = self.contrast(output1, output2)

        return loss
    
    def classify_cd(self, features_list, seqs):
        h = []
        for fc, fc2, feature in zip(self.fc_list, self.fc2_list, features_list):
            h.append(fc2(self.activation(fc(feature))))
        h1 = torch.cat(h, 0)
        h2 = h1.clone()
        h1 = h1[seqs]
        for layer in range(self.num_GNNs):
            h2 = self.GCNLayers[layer](self.g, h2)
        for layer in range(self.num_layers):
            h1 = self.GTLayers[layer](h1)

        lossFun = nn.BCELoss()

        output1 = self.proj(
            h1[:, 0, :].reshape(-1, self.embeddings_dimension)).detach()
        output2 = self.proj(h2[seqs[:, 0].reshape(-1,)]).detach()
        neg_idx = torch.randperm(seqs.shape[0])
        output3 = output2[neg_idx]

        target = torch.cat(
            [torch.ones(seqs.shape[0]), torch.zeros(seqs.shape[0])], dim=0).reshape(seqs.shape[0]*2, 1).to(output1.device)

        score_p = torch.mm(
            torch.mm(output1, self.score_matrix), output2.T).diag()
        score_n = torch.mm(
            torch.mm(output1, self.score_matrix), output3.T).diag()

        score = self.sig(torch.cat([score_p, score_n]).reshape(-1, 1))

        loss = lossFun(score, target)

        return loss


    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def contrast(self, z_1, z_2):
        z_proj_1 = self.proj(z_1)
        z_proj_2 = self.proj(z_2)
        matrix_1 = self.sim(z_proj_1, z_proj_2)
        matrix_2 = matrix_1.t()

        matrix_1 = matrix_1 / \
            (torch.sum(matrix_1, dim=1).view(-1, 1) + 1e-8)
        lori_1 = - \
            torch.log(torch.diag(matrix_1)).mean()

        matrix_2 = matrix_2 / \
            (torch.sum(matrix_2, dim=1).view(-1, 1) + 1e-8)
        lori_2 = - \
            torch.log(torch.diag(matrix_2)).mean()
        return 0.5 * (lori_1 + lori_2)

    def reset_dropout(self, p=0.5):
        for layer in range(self.num_layers):
            self.GTLayers[layer].dropout_att.p = p
            self.GTLayers[layer].dropout_msa.p = p
            self.GTLayers[layer].dropout_ffn.p = p
            self.GTLayers[layer].dropout_mlp.p = p
