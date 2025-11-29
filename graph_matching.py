import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax, degree
from torch_geometric.nn import global_add_pool
# from pyg_lib import MessagePassing
from torch_geometric.nn import global_mean_pool

class GMNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GMNConv, self).__init__(aggr=aggr)
        self.emb_dim = emb_dim
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(3*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))

    def get_self_loop_attr(self, x, edge_attr):
        raise NotImplementedError('')

    def encode_edge_feat(self, edge_attr):
        raise NotImplementedError('')

    def forward(self, gid, x, edge_index, edge_attr, batch):

        emb_dim = self.emb_dim
        gx = x[batch == gid]

        deg = degree(edge_index[0], num_nodes=x.size(0))

        mask = deg > 1
        gx_filtered = gx[mask[batch == gid]]
        high_degree_x = x[mask]
        filtered_high_batch = batch[mask]
        # single -> multi for high degree nodes
        a = torch.matmul(high_degree_x, gx_filtered.T)  # (high_degree_node_num, g_node_num)
        a1 = softmax(a, filtered_high_batch)  # (high_degree_node_num, g_node_num
        mu1 = a1.unsqueeze(-1) * gx_filtered.unsqueeze(0)  # (high_degree_node_num, g_node_num, emb_dim)
        sum_mu1 = torch.sum(mu1, dim=1)  # (high_degree_node_num, emb_dim)
        # Concatenate the aggregated inter-graph message with inner aggregation output
        inner_inter_aggr_out1 = torch.cat([high_degree_x, sum_mu1], dim=-1)  #
        # Update node features using MLP
        out1 = self.mlp(inner_inter_aggr_out1)  # (high_degree_node_num, emb_dim)
        out_multi = out1
        # multi -> single for high degree nodes
        a2 = torch.softmax(a.transpose(1, 0), dim=0)  # (g_node_num, high_degree_node_num)
        mu2 = a2.unsqueeze(-1) * high_degree_x.unsqueeze(0)  # (g_node_num, high_degree_node_num, emb_dim)
        # Aggregate features globally
        sum_mu2 = global_add_pool(mu2.transpose(1, 0), filtered_high_batch)
        # Expand and concatenate features for MLP
        inner_aggr_out2 = high_degree_x[filtered_high_batch == gid].unsqueeze(0).expand(sum_mu2.shape[0], -1, -1)  # (batch_size, g_node_num, 2 * emb_dim)
        inner_inter_aggr_out2 = torch.cat([inner_aggr_out2, sum_mu2], dim=-1)  # (batch_size, g_node_num, 2 * emb_dim)
        # Update node features using MLP
        out2 = self.mlp(inner_inter_aggr_out2.view(-1, 2 * emb_dim)).view(sum_mu2.shape[0], -1, emb_dim)  # (batch_size, g_node_num, emb_dim)
        out_single = out2

        num_graphs = batch.max().item() + 1  # Determine number of graphs from batch
        # create graphs
        gid_graphs = torch.stack([gx.clone() for _ in range(num_graphs)])
        for i in range(num_graphs):
            gid_graphs[i][mask[batch == gid]] = out_single[i]
        # Normalize graph features
        out1 = torch.stack([F.normalize(g.sum(dim=0), dim=-1) for g in gid_graphs], dim=0)
        # Update high degree node features of x and normalize
        x[mask] = out_multi
        out2 = F.normalize(global_add_pool(x, batch), dim=-1)
        # calc squared elucidean distance
        # similarity_matrix = torch.sum((out1 - out2) ** 2, dim=-1)
        # Calculate similarity matrix
        similarity_matrix = torch.sum(out1 * out2, dim=-1)

        return similarity_matrix

    def message(self, x_j, edge_attr):
        return torch.cat([x_j, edge_attr], dim=1)
class BioGMNConv(GMNConv):
    def __init__(self, edge_dim, emb_dim, aggr='add'):
        super(BioGMNConv, self).__init__(emb_dim, aggr=aggr)
        self.edge_encoder = torch.nn.Linear(edge_dim, emb_dim)

    def get_self_loop_attr(self, x, edge_attr):
        self_loop_attr = torch.zeros(x.size(0), 9)
        self_loop_attr[:,7] = 1 # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        return self_loop_attr

    def encode_edge_feat(self, edge_attr):
        return self.edge_encoder(edge_attr)

class DarpaGMNConv(GMNConv):
    def __init__(self,edge_dim, emb_dim, aggr='add'):
        super(DarpaGMNConv, self).__init__(emb_dim, aggr=aggr)
        self.edge_dim = edge_dim
        self.edge_encoder = torch.nn.Linear(edge_dim, emb_dim)

    def get_self_loop_attr(self, x, edge_attr):
        self_loop_attr = torch.zeros(x.size(0), self.edge_dim)
        self_loop_attr[:,self.edge_dim-1] = 1 # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        return self_loop_attr

    def encode_edge_feat(self, edge_attr):
        # print(edge_attr.dtype)
        # print(self.edge_encoder.weight.dtype)

        return self.edge_encoder(edge_attr)

class ChemGMNConv(GMNConv):
    def __init__(self, emb_dim, aggr='add'):
        super().__init__(emb_dim, aggr=aggr)
        from chem_model import num_bond_type, num_bond_direction
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def get_self_loop_attr(self, x, edge_attr):
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        return self_loop_attr

    def encode_edge_feat(self, edge_attr):
        return self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
class AdaGMNConv(torch.nn.Module):
    def __init__(self, edge_dim, emb_dim, mode='bio'):
        super(AdaGMNConv, self).__init__()

        self.mode2conv = {
            'cadets': DarpaGMNConv,
            'trace': DarpaGMNConv,
            'theia': DarpaGMNConv,
            'optc': DarpaGMNConv,
        }

        self.gmnconv = self.mode2conv[mode](edge_dim, emb_dim)

    def forward(self, gid, x, edge_index, edge_attr, batch):
        return self.gmnconv(gid, x, edge_index, edge_attr, batch)
