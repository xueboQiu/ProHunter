import os
import networkx as nx
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

cnt = 0
class GraphAug:
    def __init__(self, aug_ratio, aug_methods='node_add'):
        self.name2method = {
            'edge_perturb': self.perturb_edges,
            'node_drop': self.node_drop,
            'subgraph': self.subgraph,
            'attr_mask': self.mask_nodes,
            'node_add': self.node_add,
        }

        self.aug_ratio = aug_ratio
        if isinstance(aug_methods, str):
            aug_methods = [aug_methods]
        self.funcs = [v for k, v in self.name2method.items() if k in aug_methods]
        self.num_funcs = len(self.funcs)
        print('aug_methods', aug_methods)

    def __call__(self, data):
        # global cnt
        # print("cnt", cnt)
        # cnt += 1
        rd = torch.randint(self.num_funcs, (2,))
        data.view1_x, data.view1_edge_index, data.view1_edge_attr = self.funcs[rd[0].item()](data)
        data.view2_x, data.view2_edge_index, data.view2_edge_attr = self.funcs[rd[1].item()](data)
        # print("coming GraphAug")
        return data

    def perturb_edges(self, data):
        _, edge_num = data.edge_index.size()
        permute_num = int(edge_num * self.aug_ratio)
        idx_delete = torch.randperm(edge_num)[permute_num:]

        new_edge_index = data.edge_index[:, idx_delete]
        new_edge_attr = data.edge_attr[idx_delete]
        return data.x, new_edge_index, new_edge_attr

    def node_drop(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num  * self.aug_ratio)

        idx_perm = torch.randperm(node_num).numpy()

        idx_drop = idx_perm[:drop_num]
        idx_nondrop = idx_perm[drop_num:]
        idx_nondrop.sort()
        idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

        edge_index = data.edge_index.numpy()
        edge_mask = np.array([n for n in range(edge_num) if not (edge_index[0, n] in idx_drop or edge_index[1, n] in idx_drop)])

        edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
        try:
            new_edge_index = torch.tensor(edge_index).transpose_(0, 1)
            new_x = data.x[idx_nondrop]
            new_edge_attr = data.edge_attr[edge_mask]
            return new_x, new_edge_index, new_edge_attr
        except:
            return data.x, data.edge_index, data.edge_attr

    def node_add(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        add_num = int(node_num * self.aug_ratio)

        sampled_indices = torch.randint(0, node_num, (add_num,))
        add_nodes = data.x[sampled_indices]
        new_x = torch.cat([data.x, add_nodes], dim=0)

        sampled_edges = data.edge_index[:,
                        torch.isin(data.edge_index[0], sampled_indices) | torch.isin(data.edge_index[1],
                                                                                     sampled_indices)]
        new_edge_index = torch.cat([data.edge_index, sampled_edges], dim=1)

        sampled_edge_attr = data.edge_attr[
            torch.isin(data.edge_index[0], sampled_indices) | torch.isin(data.edge_index[1], sampled_indices)]
        new_edge_attr = torch.cat([data.edge_attr, sampled_edge_attr], dim=0)

        return new_x, new_edge_index, new_edge_attr

    def subgraph(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        sub_num = int(node_num * self.aug_ratio)

        edge_index = data.edge_index.numpy()

        idx_sub = [torch.randint(node_num, (1,)).item()]
        idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = np.random.choice(list(idx_neigh))
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

        idx_drop = [n for n in range(node_num) if not n in idx_sub]
        idx_nondrop = idx_sub
        idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
        edge_mask = np.array([n for n in range(edge_num) if (edge_index[0, n] in idx_nondrop and edge_index[1, n] in idx_nondrop)])

        edge_index = data.edge_index.numpy()
        edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
        try:
            new_edge_index = torch.tensor(edge_index).transpose_(0, 1)
            new_x = data.x[idx_nondrop]
            new_edge_attr = data.edge_attr[edge_mask]
            return new_x, new_edge_index, new_edge_attr
        except:
            return data.x, data.edge_index, data.edge_attr

    def mask_nodes(self, data):
        node_num, feat_dim = data.x.size()
        mask_num = int(node_num * self.aug_ratio)

        token = data.x.mean(dim=0)
        idx_mask = torch.randperm(node_num)[:mask_num]
        data.x[idx_mask] = token.clone().detach()

        return data.x, data.edge_index, data.edge_attr
def graph_reduction(g):
    # 将相同名字且类型相同的节点合并
    # 解除冻结图
    sbg = g.copy()
    nname2id = {sbg.nodes[node]["type"] + sbg.nodes[node]["name"]: node for node in sbg.nodes}

    # print("nname2id:",nname2id)
    nodes = list(sbg.nodes)
    for node in nodes:
        node_name = sbg.nodes[node]["type"] + sbg.nodes[node]["name"]
        if node_name.endswith("/"):
            sbg.remove_node(node)
            continue
        if node == nname2id[node_name]:
            continue
        # 得到当前节点的所有边，并转移至节点nname2id[node_name]中
        for edge in list(sbg.in_edges(node, keys=True)) + list(sbg.out_edges(node, keys=True)):
            src, dst, key = edge
            if src == node:
                src, dst = dst, src
            sbg.add_edge(src, nname2id[node_name], key=key, **sbg.edges[edge])
        sbg.remove_node(node)

    # 冻结图
    sbg = nx.freeze(sbg)

    return sbg
def load_pois(dataset):
    in_path = f"../dataset/{dataset}/pois"
    files = os.listdir(in_path)

    pois_uuids = []
    for file in files:
        with open(f"{in_path}/{file}", 'r') as f:
            for line in f:
                pois_uuids.append(line.strip().split(":")[0])

    return pois_uuids
def trans2dgl(pyg_data):
    import dgl
    src_nodes = pyg_data.edge_index[0]
    dst_nodes = pyg_data.edge_index[1]

    num_nodes = pyg_data.x.size(0)
    dgl_graph = dgl.graph((src_nodes, dst_nodes),num_nodes=num_nodes)
    dgl_graph.ndata['x'] = pyg_data.x  # 添加节点特征
    dgl_graph.edata['edge_attr'] = torch.tensor([attr[0] for attr in pyg_data.edge_attr]).unsqueeze(1)

    return dgl_graph

def calculate_ged(graph1, graph2):
    from preprocess import GED
    ged = GED.graph_edit_distance(trans2dgl(graph1), trans2dgl(graph2),
                                  algorithm='bipartite')

    return ged
import numpy as np
from scipy.optimize import linear_sum_assignment

class GED:
    def __init__(self):
        pass

    def calculate_ged(self, graph1, graph2):
        # Create the node cost matrix
        size1 = graph1.number_of_nodes()
        size2 = graph2.number_of_nodes()
        node_cost_matrix = np.zeros((size1 + size2, size1 + size2)) + np.inf  # Fill with infinity

        # Fill the node cost matrix
        nodes1 = list(graph1.nodes(data=True))
        nodes2 = list(graph2.nodes(data=True))
        for i, node1 in enumerate(nodes1):
            for j, node2 in enumerate(nodes2):
                if node1[1]['type'] == node2[1]['type']:
                    node_cost_matrix[i, j] = 0  # No cost if the types match
                else:
                    node_cost_matrix[i, j] = 1  # Cost of 1 if the types do not match

        # Consider the cost of deleting and adding nodes
        for i in range(size1):
            node_cost_matrix[i, size2 + i] = 1  # Cost of deleting a node
        for j in range(size2):
            node_cost_matrix[size1 + j, j] = 1  # Cost of adding a node

        # Create the edge cost matrix
        edges1 = list(graph1.edges(data=True))
        edges2 = list(graph2.edges(data=True))
        num_edges1 = len(edges1)
        num_edges2 = len(edges2)
        edge_cost_matrix = np.zeros((num_edges1 + num_edges2, num_edges1 + num_edges2)) + np.inf

        # Fill the edge cost matrix
        for i, edge1 in enumerate(edges1):
            for j, edge2 in enumerate(edges2):
                if edge1[2]['type'] == edge2[2]['type']:
                    edge_cost_matrix[i, j] = 0  # No cost if the types match
                else:
                    edge_cost_matrix[i, j] = 1  # Cost of 1 if the types do not match

        # Consider the cost of deleting and adding edges
        for i in range(num_edges1):
            edge_cost_matrix[i, num_edges2 + i] = 1  # Cost of deleting an edge
        for j in range(num_edges2):
            edge_cost_matrix[num_edges1 + j, j] = 1  # Cost of adding an edge

        # Use the Hungarian algorithm to solve the minimum cost matching
        node_row_ind, node_col_ind = linear_sum_assignment(node_cost_matrix)
        edge_row_ind, edge_col_ind = linear_sum_assignment(edge_cost_matrix)

        # Calculate the total cost
        total_node_cost = node_cost_matrix[node_row_ind, node_col_ind].sum()
        total_edge_cost = edge_cost_matrix[edge_row_ind, edge_col_ind].sum()

        return total_node_cost + total_edge_cost
def parse_args():
    # Training settings
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--mode', type=str, default='cadets')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3, help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--model_file', type=str, default='models/', help='filename to output the pre-trained model')
    parser.add_argument('--dataset_path', type=str, default='./', help='root path to the dataset')
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument('--aug_ratio', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=0.1, help='softmax temperature (default: 0.1)')
    parser.add_argument('--input_epochs', type=int, default=0, help='number of epoch of input model')
    parser.add_argument('--sample_num', type=int, default=5, help='number of sample graph in one batch')
    parser.add_argument('--eval_sample_num', type=int, default=2, help='number of sample graph in one batch for evaluation')
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--split', type=str, default="random", help='Random or species split')
    parser.add_argument('--thre', type=float, default=0.3, help='graph matching threshold')
    parser.add_argument('--ablation_test', type=int, default=0, help='eval inter-graph message passing')
    parser.add_argument('--eval', type=bool, default=True, help='train or eval')
    return parser.parse_args()