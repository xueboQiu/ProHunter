import json
import os
import pickle
import random
import re

from tqdm import tqdm
import networkx as nx
import torch

from preprocess.dataset_preprocess import preprocess_dataset_cadets, preprocess_dataset_trace, preprocess_dataset_theia, \
    preprocess_dataset_fivedirections, preprocess_dataset_optc, networkx2pyg
from util import GED


def read_single_graph(dataset, path, test=False):

    with open("../dataset/{}/".format(dataset)+ 'node_type_map.json', 'r') as f:
        node_type_dict = json.load(f)
    with open("../dataset/{}/".format(dataset)+ 'edge_type_map.json', 'r') as f:
        edge_type_dict = json.load(f)
    with open("../dataset/{}/".format(dataset)+ 'names.json', 'r') as f:
        uuid2name = json.load(f)

    g = nx.MultiDiGraph()
    print('converting {} ...'.format(path))
    f = open(path, 'r')
    lines = []
    for l in f.readlines():
        split_line = l.split('\t')
        src, src_type, dst, dst_type, edge_type, ts = split_line

        if "optc" not in dataset:
            ts = int(ts)

        if 'READ' in edge_type or 'RECV' in edge_type or 'LOAD' in edge_type:
            lines.append([dst, src, dst_type, src_type, edge_type, ts])
        else:
            lines.append([src, dst, src_type, dst_type, edge_type, ts])
    lines.sort(key=lambda l: l[5])

    node_map = {}
    node_type_map = {}
    node_cnt = 0
    node_list = []

    for l in lines:
        if "optc" not in dataset:
            src, dst, src_type, dst_type, edge_type, ts = l
        else:
            src, src_type, dst, dst_type, edge_type,_ = l

        if src_type not in node_type_dict or dst_type not in node_type_dict or edge_type not in edge_type_dict:
            continue

        if src not in node_map:
            node_map[src] = node_cnt
            # g.add_node(node_cnt, node_type=one_hot_encode(src_type_id, node_type_cnt+1))
            g.add_node(node_cnt, type = src_type, ts = ts, name = uuid2name.get(src, "UNKNOWN"), uuid = src)
            node_list.append(src)
            node_type_map[src] = src_type
            node_cnt += 1
        if dst not in node_map:
            node_map[dst] = node_cnt
            g.add_node(node_cnt, type = dst_type, ts = ts, name = uuid2name.get(dst, "UNKNOWN"), uuid = dst)
            node_type_map[dst] = dst_type
            node_list.append(dst)
            node_cnt += 1
        if not g.has_edge(node_map[src], node_map[dst]):
            g.add_edge(node_map[src], node_map[dst], key=0, edge_type = [edge_type], ts = ts)
        elif edge_type not in g[node_map[src]][node_map[dst]][0]['edge_type']:
            g[node_map[src]][node_map[dst]][0]['edge_type'].append(edge_type)

    print('converting {} succeed!'.format(path))
    return node_map, g
def generate_traing_dataset(dataset):

    if os.path.exists(f'../dataset/{dataset}/train/raw/train_subgraph_matching.pt'):
        print("raw train samples already exist!")
        return
    if "cadets" in dataset:
        preprocess_dataset_cadets()
    elif "trace" in dataset:
        preprocess_dataset_trace()
    elif "theia" in dataset:
        preprocess_dataset_theia()
    elif "five" in dataset:
        preprocess_dataset_fivedirections()
    elif "optc" in dataset:
        preprocess_dataset_optc()

    subgraph_match_pairs = []

    file_paths = [f'../dataset/{dataset}/tuples/{file}' for file in os.listdir(f'../dataset/{dataset}/tuples/')]
    for file_path in file_paths:

        graph_path = file_path.replace('tuples', 'multi_digraphs').replace(".txt",".pkl")
        if not os.path.exists(graph_path):
            _, train_g = read_single_graph(dataset, file_path, False)
            # feature initialization
            train_g = fea_init(dataset, train_g)
            pickle.dump(train_g, open(graph_path, 'wb'))
            print("graph saved:"+ graph_path)
        else:
            train_g = pickle.load(open(graph_path, 'rb'))
            print("graph loaded:" + graph_path)

        # 得到train_g是个MultiDiGraph (Multi-Directed Graph)，从中选出几个子图来训练模型
        nodes = train_g.nodes()
        # 找到train_g中度大于2的所有节点
        nodes = [node for node in nodes if len(list(train_g.neighbors(node))) > 3 and 'PROCESS' in train_g.nodes[node]['type']]
        # 对每个度大于2的节点，得到其2阶段邻居子图
        for node in nodes:
            # get uuid
            center_uuid = train_g.nodes[node]['uuid']
            wholegraph_nodes = [node]  # 存储全部的邻居节点用于子图匹配
            subgraph_nodes = [node] # 存储子图节点，限制大小
            # 获取node一阶邻居
            neighbors = list(train_g.neighbors(node))
            # 过滤度小于3的邻居
            filtered_neighbors = [n for n in neighbors if len(list(train_g.neighbors(n))) > 3]
            if len(filtered_neighbors) < 3:
                if len(neighbors) < 5: continue
                selected_first_neigs = random.sample(neighbors, random.randint(min(len(neighbors),10), min(len(neighbors),15))) # 星形图
            else:
                # 随机选取3~5个节点
                selected_first_neigs = random.sample(filtered_neighbors,random.randint(3,min(len(filtered_neighbors),6)))

            for fn in selected_first_neigs:
                subgraph_nodes.append(fn)
                # 2阶邻居
                sec_neighs = list(train_g.neighbors(fn))
                if len(sec_neighs)>3:
                    selected_sec_neighs = random.sample(sec_neighs,random.randint(3,min(len(sec_neighs),5)))
                    subgraph_nodes.extend(selected_sec_neighs)
            # generate subgraph
            sb = train_g.subgraph(subgraph_nodes)
            if len(sb.edges()) > 10:
                pyg_sb = networkx2pyg(sb)
                # subgraphs.append(pyg_graph)
                wholegraph_nodes.extend(neighbors)
                for n in neighbors:
                    wholegraph_nodes.extend(list(train_g.neighbors(n)))
                wg = train_g.subgraph(wholegraph_nodes)

                pyg_wg = networkx2pyg(wg)
                subgraph_match_pairs.append((pyg_sb, pyg_wg,center_uuid,center_uuid))

    torch.save(subgraph_match_pairs, f'../dataset/{dataset}/train/raw/train_subgraph_matching.pt')
    print("raw train samples generate!")
def calc_samples_ged(dataset, pair_num=2000, ged_thre = 40):

    if os.path.exists((f'../dataset/{dataset}/train/raw/{pair_num}_ged_prohunter.pt')) and \
        os.path.exists((f'../dataset/{dataset}/train/raw/{pair_num}_ged_sg_matching.pt')):
        print("ged filtered samples already exist!")
        return

    sg_training_samples = torch.load(f'../dataset/{dataset}/train/raw/train_subgraph_matching.pt')

    sample_num = len(sg_training_samples)
    # 存储样本之间的GED
    prohunter_training_dataset_after_ged = []
    sg_match_training_dataset_after_ged_pos = []
    sg_match_training_dataset_after_ged_neg = []
    sampled_idx_set = set()

    def sample_indices(sample_num, sampled_idx_set):
        while True:
            i, j = random.sample(range(sample_num), 2)
            if (i, j) not in sampled_idx_set and (j, i) not in sampled_idx_set:
                sampled_idx_set.add((i, j))
                return i, j

    for k in tqdm(range(pair_num + 1), desc = "Calculating GED for training samples "+dataset):

        i, j = sample_indices(sample_num, sampled_idx_set)
        # multidigraph to pygdata
        from util import trans2dgl
        ged = GED.graph_edit_distance(trans2dgl(sg_training_samples[i][0]), trans2dgl(sg_training_samples[j][0]), algorithm='bipartite')

        while ged < ged_thre:
            i, j = sample_indices(sample_num, sampled_idx_set)
            ged = GED.graph_edit_distance(trans2dgl(sg_training_samples[i][0]), trans2dgl(sg_training_samples[j][0]),
                                          algorithm='bipartite')

        # for prohunter, directly push pyg data
        prohunter_training_dataset_after_ged.append((sg_training_samples[i][0], sg_training_samples[j][0]))
        # for subgraph matching, store pos and neg sample pairs
        sg_match_training_dataset_after_ged_pos.append(sg_training_samples[i])
        sg_match_training_dataset_after_ged_pos.append(sg_training_samples[j])
        sg_match_training_dataset_after_ged_neg.append((sg_training_samples[i][0],sg_training_samples[j][1],
                                                        sg_training_samples[i][2],sg_training_samples[j][3]))
                                                        # sg_training_samples[i][4],sg_training_samples[j][5]))
        sg_match_training_dataset_after_ged_neg.append((sg_training_samples[j][0],sg_training_samples[i][1],
                                                        sg_training_samples[j][2],sg_training_samples[i][3]))
                                                        # sg_training_samples[j][4],sg_training_samples[i][5]))

        if k % 1000 == 0:
            print(f'generate {k}_ged.pt')
            torch.save(prohunter_training_dataset_after_ged, f'../dataset/{dataset}/train/raw/{k}_ged_prohunter.pt')
            torch.save((sg_match_training_dataset_after_ged_pos,sg_match_training_dataset_after_ged_neg), f'../dataset/{dataset}/train/raw/{k}_ged_sg_matching.pt')
def combined_samples(dataset,pair_num = 2000):

    if os.path.exists((f'../dataset/{dataset}/train/processed/train_prohunter.pt'))\
            and os.path.exists(f'../dataset/{dataset}/train/processed/train_sg_matching.pt'):
        print("prohunter and sg matching training samples already combined!")
        return

    prohunter_ged_samples = torch.load(f'../dataset/{dataset}/train/raw/{pair_num}_ged_prohunter.pt')
    sg_matching_ged_samples = torch.load(f'../dataset/{dataset}/train/raw/{pair_num}_ged_sg_matching.pt')

    # combine prohunter training and testing samples
    prohunter_combined_samples = []
    for pair in prohunter_ged_samples:
        prohunter_combined_samples.append(pair[0])
        prohunter_combined_samples.append(pair[1])
    split_boundry = int(len(prohunter_combined_samples)/4*3)
    torch.save(prohunter_combined_samples[0:split_boundry], f'../dataset/{dataset}/train/processed/train_prohunter.pt')
    torch.save(prohunter_combined_samples[split_boundry:], f'../dataset/{dataset}/test/raw/test_prohunter.pt')
    print("prohunter training and test samples generated!")

    # process subgraph matching training and testing samples
    pos_pairs, neg_pairs = sg_matching_ged_samples
    pos_pairs = [pp for pp in pos_pairs if pp[0].num_nodes<=pp[1].num_nodes and pp[1].num_nodes<3000 and pp[1].num_nodes > 100]
    neg_pairs = [np for np in neg_pairs if np[0].num_nodes<=np[1].num_nodes and np[1].num_nodes<3000 and np[1].num_nodes > 100]

    split_boundry_pos_pairs_num = int(len(pos_pairs)/2)
    split_boundry_neg_pairs_num = int(len(neg_pairs)/2)
    train_pos_pairs = pos_pairs[0:split_boundry_pos_pairs_num]
    train_neg_pairs = neg_pairs[0:split_boundry_neg_pairs_num]
    # get the last 1000 negs
    test_neg_pairs = neg_pairs[split_boundry_neg_pairs_num:]
    if len(test_neg_pairs) > 1000:
        test_neg_pairs = test_neg_pairs[-1000:]

    sg_matching_combined_samples_train = (train_pos_pairs,train_neg_pairs)
    sg_matching_combined_samples_test = test_neg_pairs

    torch.save(sg_matching_combined_samples_train, f'../dataset/{dataset}/train/processed/train_sg_matching.pt')
    torch.save(sg_matching_combined_samples_test, f'../dataset/{dataset}/test/raw/test_sg_matching.pt')
    print("sg matching training and test samples generated!")

# 根据每个gt的节点uuid的重合度决定是否去重
def abstract_graph(gt):

    def abs_node_type(node_name, t):

        node_name = node_name.lower()
        if "p" in t:
            return next((process_type for process_type, process_names in abstract_process_map.items() if
                         any(name in node_name for name in process_names)), "UNKNOWN_PROCESS")
        elif "f" in t:
            return next((file_type for file_type, file_names in abstract_file_map.items() if
                         any(name in node_name for name in file_names)), "UNKNOWN_FILE")
        else:
            return "UNKNOWN"

    # iterate node
    for u in gt.nodes():
        node_type = gt.nodes[u]['type'].lower()
        if "process" in node_type:
            node_name = gt.nodes[u]['image_path'] if 'image_path' in gt.nodes[u] else gt.nodes[u]['name']
            abs_name = abs_node_type(node_name,"p")
        elif "file" in node_type:
            node_name = gt.nodes[u]['file_path'] if 'file_path' in gt.nodes[u] else gt.nodes[u]['name']
            abs_name = abs_node_type(node_name,'f')
        elif "module" in node_type:
            node_name = gt.nodes[u]['file_path'] if 'file_path' in gt.nodes[u] else gt.nodes[u]['name']
            abs_name = abs_node_type(node_name,'f')
        elif "flow" in node_type:
            node_name = gt.nodes[u]['remote_ip'] if 'remote_ip' in gt.nodes[u] else gt.nodes[u]['name']
            abs_name = "FLOW"
        elif "reg" in node_type:
            node_name = gt.nodes[u]['key'] if 'key' in gt.nodes[u] else gt.nodes[u]['name']
            abs_name = "REG"
        else:
            node_name = ""
            abs_name = "UNKNOWN"

        # print(f"original node type:{node_type}, node name:{node_name}, abstract node type:{abs_name}")
        gt.nodes[u]['node_type'] = abs_name

    return gt
def init_abstract_node_map():
    global abstract_process_map
    process_files = {
            "BROWSER_PROCESS": "BROWSER_PROCESS.txt",
            "OFFICE_PROCESS": "OFFICE_PROCESS.txt",
            "SERVER_PROCESS": "SERVER_PROCESS.txt",
            "DAEMON_PROCESS": "DAEMON_PROCESS.txt",
            "UTIL_PROCESS": "UTIL_PROCESS.txt",
            "SYSTEM_PROCESS": "SYSTEM_PROCESS.txt",
            "USER_PROCESS": "USER_PROCESS.txt",
        }

    for process_type, file_path in process_files.items():
        with open(f"../preprocess/abs_types/{file_path}", 'r') as file:
            process_names = file.read().splitlines()
            abstract_process_map[process_type] = process_names

    global abstract_file_map
    file_files = {
        "EXECUTABLE_FILE": "EXECUTABLE_FILE.txt",
        "ETC_FILE": "ETC_FILE.txt",
        "LIB_FILE": "LIB_FILE.txt",
        "LOG_FILE": "LOG_FILE.txt",
        "TMP_FILE": "TMP_FILE.txt",
        "SYS_FILE": "SYS_FILE.txt",
        "USR_FILE": "USR_FILE.txt",
        "DEV_FILE": "DEV_FILE.txt"
    }

    for file_type, file_path in file_files.items():
        with open(f"../preprocess/abs_types/{file_path}", 'r') as file:
            file_names = file.read().splitlines()
            abstract_file_map[file_type] = file_names
def fea_init(dataset, g):

    with open("../dataset/{}/".format(dataset) + 'edge_type_map.json', 'r') as f:
        edge_type_dict = json.load(f)

    # abstract node type dictionary
    abs_node_type_dict = {"BROWSER_PROCESS": 0, "OFFICE_PROCESS": 1, "SERVER_PROCESS": 2, "DAEMON_PROCESS": 3,
                          "UTIL_PROCESS": 4, "SYSTEM_PROCESS": 5, "USER_PROCESS": 6,
                          "EXECUTABLE_FILE": 7, "ETC_FILE": 8, "LIB_FILE": 9, "LOG_FILE": 10, "TMP_FILE": 11,
                          "SYS_FILE": 12, "USR_FILE": 13, "DEV_FILE": 14, "UNKNOWN_PROCESS": 15,
                          "UNKNOWN_FILE": 16, "FLOW": 17, "REG": 18, "UNKNOWN":19}

    edge_type_cnt = len(edge_type_dict)
    # convert edge features to multi-hot encoding
    for (u, v, k) in g.edges(keys=True):
        edge_types = g[u][v][k]['edge_type']
        multiu_hot_encodings = [0] * (edge_type_cnt + 1)
        for et in edge_types:
            if et in edge_type_dict:
                multiu_hot_encodings[edge_type_dict[et]] = 1
        g[u][v][k]['edge_type'] = torch.tensor(multiu_hot_encodings)

    g = abstract_graph(g)
    # assign abstract node types
    for u in g.nodes():
        node_type = g.nodes[u]['node_type']
        g.nodes[u]['node_type'] = torch.tensor(abs_node_type_dict.get(node_type, abs_node_type_dict["UNKNOWN"]))

    return g
def generate_test_dataset(dataset):

    from subgraph.subgraph_extraction import read_json_graph

    def load_and_initialize_graphs(file_paths, dataset):
        graphs = []
        for file_path in file_paths:
            graph = read_json_graph(file_path)
            graph = fea_init(dataset, graph)
            graphs.append((graph, os.path.basename(file_path)))
        return graphs

    # 生成 groundtruth 和 sampled_graphs 文件的路径
    gt_filepaths = [os.path.join(f'../dataset/{dataset}/groundtruth/', file) for file in os.listdir(f'../dataset/{dataset}/groundtruth/')]
    sg_filepaths = [os.path.join(f'../dataset/{dataset}/sampled_graphs/', file) for file in os.listdir(f'../dataset/{dataset}/sampled_graphs/')]
    # 加载和初始化图数据
    gts = load_and_initialize_graphs(gt_filepaths, dataset)
    sgs = load_and_initialize_graphs(sg_filepaths, dataset)
    # 移除节点过多的gt graphs
    gts = [gt for gt in gts if len(gt[0].nodes) < 50]
    sgs = [sg for sg in sgs if len(sg[0].nodes) < 60]
    # 移除相似的gt graphs
    dedu_gts, dedup_idxes = deduplicate(gts)
    query_samples = [networkx2pyg(gt[0]) for gt in dedu_gts]
    # used to vision
    nx_query_samples = [gt[0] for gt in dedu_gts]

    sampled_samples = []
    nx_sampled_samples = []
    for sg in sgs:
        query_idx = -1
        for i, gt in enumerate(dedu_gts):
            if is_duplicate(gt, sg, 0.7):
                query_idx = i
                # print(f"==============gt file:{gt[1]}, gt num:{len(gt[0].nodes)}. sg file:{sg[1]}, sg num:{len(sg[0].nodes)}, query_idx:{query_idx}")
                break
        if query_idx == -1:
            print("missing query_idx")
            continue
        sampled_samples.append((networkx2pyg(sg[0]),query_idx))
        # used to vision positive sample pairs
        nx_sampled_samples.append((sg[0],query_idx))
    # 加载negative test samples
    prohunter_test = torch.load(f'../dataset/{dataset}/test/raw/test_prohunter.pt')
    benign_graphs = [bg for bg in prohunter_test if bg.num_nodes < 60]
    for bg in benign_graphs:
        sampled_samples.append((bg,-1))
    # store porhunter test dataset
    torch.save((query_samples,sampled_samples), f'../dataset/{dataset}/test/processed/test_prohunter.pt')
    print("test prohunter samples generate!")

    return (nx_query_samples,nx_sampled_samples)

def subgraph_abstract_vision(test_dataset):
    def show(vis_graph):
        # abstract node type dictionary
        abs_node_type_dict = {"BROWSER_PROCESS": 0, "OFFICE_PROCESS": 1, "SERVER_PROCESS": 2, "DAEMON_PROCESS": 3,
                              "UTIL_PROCESS": 4, "SYSTEM_PROCESS": 5, "USER_PROCESS": 6,
                              "EXECUTABLE_FILE": 7, "ETC_FILE": 8, "LIB_FILE": 9, "LOG_FILE": 10, "TMP_FILE": 11,
                              "SYS_FILE": 12, "USR_FILE": 13, "DEV_FILE": 14, "UNKNOWN_PROCESS": 15,
                              "UNKNOWN_FILE": 16, "FLOW": 17, "REG": 18, "UNKNOWN": 19}

        abs_node_type_dict_reverse = {v: k for k, v in abs_node_type_dict.items()}
        from matplotlib import pyplot as plt
        # pyg transform to nxg transform
        nodes_labels = {}
        for node_id, node_attr in list(vis_graph.nodes(data=True)):
            nodes_labels[node_id] = abs_node_type_dict_reverse[node_attr["node_type"].item()]

        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(vis_graph, k=1.5)
        # 定义每种类型的节点样式和颜色
        shapes = {
            'PROCESS': 's',  # square
            'FILE': 'o',  # circle
            'FLOW': 'd',  # diamond
            'REG': 'p',  # pentagon
            'SHELL': 'h'  # hexagon
        }
        colors = {
            'PROCESS': '#C3C5C7',
            'FILE': '#D19494',
            'FLOW': '#8FC3E4',
            'REG': '#D19494',
            'SHELL': '#D15424',
        }
        # 绘制每种类型的节点
        for node_type, shape in shapes.items():
            nodes = [n for n in vis_graph.nodes if vis_graph.nodes[n]['type'] == node_type]
            poi_nodes = [n for n in nodes if 'poi' in vis_graph.nodes[n]]
            normal_nodes = [n for n in nodes if 'poi' not in vis_graph.nodes[n]]
            # 绘制没有 poi 属性的节点
            if normal_nodes:
                nx.draw_networkx_nodes(
                    vis_graph, pos,
                    nodelist=normal_nodes,
                    node_color=colors[node_type],
                    node_shape=shape,
                    node_size=1200,
                    alpha=0.9
                )
            # 绘制具有 poi 属性的节点
            if poi_nodes:
                nx.draw_networkx_nodes(
                    vis_graph, pos,
                    nodelist=poi_nodes,
                    node_color='red',  # 红色
                    node_shape=shape,
                    node_size=1200,
                    alpha=0.9
                )
        # 绘制边
        nx.draw_networkx_edges(vis_graph, pos, arrowstyle='-|>', arrowsize=8, edge_color="grey")
        nx.draw_networkx_edge_labels(vis_graph, pos,
                                     edge_labels={(e1, e2): edge_attr['type'] for e1, e2, edge_attr in
                                                  list(vis_graph.edges(data=True))},
                                     font_color='red',
                                     font_size=5,
                                     )
        nx.draw_networkx_labels(vis_graph, pos, labels=nodes_labels, font_size=10, font_color='black',
                                font_weight='bold')

        plt.axis('off')
        plt.show()
    query_samples, sampled_samples = test_dataset

    for i, sampled_sample in enumerate(sampled_samples):
        tmp_sampled_sample = sampled_sample[0]
        show(tmp_sampled_sample)
        query_sample = query_samples[sampled_sample[1]]
        show(query_sample)
        print(f"sampled_sample:{i}, query_sample:{sampled_sample[1]}")

if __name__ == '__main__':

    # init abstract node map
    init_abstract_node_map()
    for dataset in ['darpa_cadets']:
        print("processing dataset:",dataset)
        # 生成不同数据集的训练集
        generate_traing_dataset(dataset)
        # 测试GED，筛选训练样本
        calc_samples_ged(dataset)
        # 合并GED过滤后的样本
        combined_samples(dataset)
        # 生成测试集
        test_dataset = generate_test_dataset(dataset)