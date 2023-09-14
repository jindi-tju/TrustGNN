import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch import GINConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w.mean(0), dim=0)
        score = beta.detach()
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1), score


class HANLayer(nn.Module):
    def __init__(self, num_paths, in_size, out_size, path, dropout):
        super(HANLayer, self).__init__()
        self.path = path
        self.num_paths = num_paths

        self.gat_layers = nn.ModuleList()
        for i in range(num_paths):
            func = nn.Linear(in_size, out_size)
            self.gat_layers.append(GINConv(func, 'sum'))

        self.semantic_attention = SemanticAttention(in_size=out_size, hidden_size=out_size // 2)

    def forward(self, gs, node_emb, edge_emb, tag):
        r_vec_1, r_vec_2 = torch.chunk(edge_emb, 2, dim=-1)
        r_vec = torch.cat([r_vec_1.unsqueeze(2), r_vec_2.unsqueeze(2)], dim=2)
        r_vec = F.normalize(r_vec, p=2, dim=2)
        h = node_emb.reshape(node_emb.shape[0], node_emb.shape[1] // 2, 2)

        semantic_embeddings = []
        for i, g in enumerate(gs):
            cur_path = self.path[i]
            temp1, temp2 = h[:, :, 0], h[:, :, 1]
            if tag == 'in':
                for etype in cur_path:
                    temp1 = temp1.clone() * r_vec[etype-1, :, 0] - temp2.clone() * r_vec[etype-1, :, 1]
                    temp2 = temp1.clone() * r_vec[etype-1, :, 1] + temp2.clone() * r_vec[etype-1, :, 0]
            elif tag == 'out':
                for etype in reversed(cur_path):
                    temp1 = temp1.clone() * r_vec[etype-1, :, 0] + temp2.clone() * r_vec[etype-1, :, 1]
                    temp2 = temp2.clone() * r_vec[etype-1, :, 0] - temp1.clone() * r_vec[etype-1, :, 1]
            else:
                raise Exception('tag error')
            h_new = torch.cat([temp1, temp2], dim=-1)
            semantic_embeddings.append(F.elu(self.gat_layers[i](g, h_new)))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)
        h, score = self.semantic_attention(semantic_embeddings)

        return h


class HAN(nn.Module):
    def __init__(self, num_paths, num_etypes, num_nodes, out_size, path, args):
        super(HAN, self).__init__()
        self.dropout = args.dropout
        self.num_layers = args.num_layers

        if args.node == 'random':
            self.node_feat = nn.Parameter(torch.empty(size=(num_nodes, args.node_dim)))
            nn.init.xavier_normal_(self.node_feat.data, gain=1.414)

            self.fc_node = nn.Linear(args.node_dim, args.hidden_dim, bias=True)
        elif args.node == 'node2vec':
            feature = []
            with open('dataset/{}/{}_node_vec.txt'.format(args.dataset, args.dataset)) as vec:
                for node in vec:
                    feature.append(node.split()[1:])
            self.node_feat = torch.FloatTensor(np.array(feature, np.float32)).to(args.device)

            self.fc_node = nn.Linear(self.node_feat.shape[1], args.hidden_dim, bias=True)
        else:
            raise Exception('args.node error')

        if args.edge == 'random':
            self.edge_feat_in = nn.Parameter(torch.empty(size=(num_etypes, 1, args.edge_dim)))
            self.edge_feat_out = nn.Parameter(torch.empty(size=(num_etypes, 1, args.edge_dim)))
            nn.init.xavier_normal_(self.edge_feat_in.data, gain=1.414)
            nn.init.xavier_normal_(self.edge_feat_out.data, gain=1.414)

            self.fc_edge_in = nn.Parameter(torch.empty(size=(num_etypes, args.edge_dim, args.hidden_dim)))
            self.fc_edge_out = nn.Parameter(torch.empty(size=(num_etypes, args.edge_dim, args.hidden_dim)))
            nn.init.xavier_normal_(self.fc_edge_in.data, gain=1.414)
            nn.init.xavier_normal_(self.fc_edge_out.data, gain=1.414)
        elif args.edge == 'one-hot':
            self.edge_feat_in = torch.eye(num_etypes, dtype=torch.float, device=args.device).unsqueeze(1)
            self.edge_feat_out = torch.eye(num_etypes, dtype=torch.float, device=args.device).unsqueeze(1)

            self.fc_edge_in = nn.Parameter(torch.empty(size=(num_etypes, num_etypes, args.hidden_dim)))
            self.fc_edge_out = nn.Parameter(torch.empty(size=(num_etypes, num_etypes, args.hidden_dim)))
            nn.init.xavier_normal_(self.fc_edge_in.data, gain=1.414)
            nn.init.xavier_normal_(self.fc_edge_out.data, gain=1.414)
        else:
            raise Exception('args.edge error')

        self.layers_in = nn.ModuleList()
        self.layers_in.append(HANLayer(num_paths, args.hidden_dim, args.hidden_dim, path, args.dropout))
        for k in range(1, self.num_layers):
            self.layers_in.append(HANLayer(num_paths, args.hidden_dim, args.hidden_dim, path, args.dropout))

        self.layers_out = nn.ModuleList()
        self.layers_out.append(HANLayer(num_paths, args.hidden_dim, args.hidden_dim, path, args.dropout))
        for k in range(1, self.num_layers):
            self.layers_out.append(HANLayer(num_paths, args.hidden_dim, args.hidden_dim, path, args.dropout))

        self.fc = nn.ModuleList([nn.Linear(args.hidden_dim * 2, args.hidden_dim) for _ in range(args.num_layers)])
        self.predict = nn.Linear(args.hidden_dim * 2, out_size, bias=False)
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, g_in, g_out, edge_indices):

        node_emb = F.dropout(F.elu(self.fc_node(self.node_feat)), self.dropout, training=self.training)
        edge_emb_in = F.elu(torch.bmm(self.edge_feat_in, self.fc_edge_in).squeeze())
        edge_emb_out = F.elu(torch.bmm(self.edge_feat_out, self.fc_edge_out).squeeze())

        for i in range(self.num_layers):
            node_emb_in = self.layers_in[i](g_in, node_emb, edge_emb_in, 'in')
            node_emb_out = self.layers_out[i](g_out, node_emb, edge_emb_out, 'out')
            node_emb = F.elu(self.fc[i](torch.cat([node_emb_in, node_emb_out], dim=-1)))

        node_pair_emb = F.embedding(edge_indices, node_emb)
        node_pair_emb = node_pair_emb.view(edge_indices.shape[0], -1)

        logits = self.predict(node_pair_emb)

        return node_emb, logits

