import json
import re

import networkx as nx
from matplotlib import pyplot as plt
from networkx.readwrite import json_graph
import os


def trans2nx(lines,uuid2name,uuid2type):

    g = nx.MultiDiGraph()

    node_map = {}
    node_type_map = {}
    node_cnt = 0
    node_list = []

    unique_nodes = set()
    for i,l in enumerate(lines):
        if len(l) == 0: continue

        sps = l.strip().split(" ")
        if len(sps) == 5:
            # 处理格式为: src edge_type dst
            src, edge_type, dst = sps[0], sps[2], sps[3]
        elif len(sps) == 3:
            src, edge_type, dst = sps[0], sps[1], sps[2]
        elif len(sps) == 6:
            src, edge_type, dst = sps[0], sps[1], sps[3]
        else:
            print("Invalid line format:", l.strip())
            exit(1)

        if src not in uuid2name or dst not in uuid2name:
            print("Invalid")
            continue

        src_type = uuid2type[src]
        dst_type = uuid2type[dst]
        src_name = uuid2name[src]
        dst_name = uuid2name[dst]
        # print(i)
        # print(src_name, edge_type, dst_name)

        if src not in node_map:
            node_map[src] = node_cnt
            g.add_node(node_cnt, node_type=src_type, type = src_type, uuid = src, name = src_name)
            node_list.append(src)
            node_type_map[src] = src_type
            node_cnt += 1
        if dst not in node_map:
            node_map[dst] = node_cnt
            g.add_node(node_cnt, node_type=dst_type, type = dst_type,uuid = dst, name = dst_name)
            node_type_map[dst] = dst_type
            node_list.append(dst)
            node_cnt += 1
        if not g.has_edge(node_map[src], node_map[dst]):
            # 添加一条新的边，您可以提供一个唯一的键
            g.add_edge(node_map[src], node_map[dst], key=0, edge_type=[edge_type])
        # 检查是否存在特定类型的边
        elif edge_type not in g[node_map[src]][node_map[dst]][0]['edge_type']:
            # 如果没有找到特定类型的边，则添加一条新的边
            g[node_map[src]][node_map[dst]][0]['edge_type'].append(edge_type)

    # for un in unique_nodes:
        # print(un,uuid2name[un])

    return g
def attach_attributes(g,poi_uuid=""):

    # node_map_reverse = {node_map[node]: node for node in node_map}

    for node in g.nodes():
        node_uuid = g.nodes[node]['uuid'].upper()
        node_type = g.nodes[node]['node_type'].upper()
        node_name = g.nodes[node]['name'].upper()

        if node_uuid == poi_uuid:
            g.nodes[node]['poi'] = True

        if "FILE" in node_type or "MODULE" in node_type:
            g.nodes[node]['type'] = 'FILE'
            g.nodes[node]['file_path'] = node_name

        elif "PROCESS" in node_type:
            g.nodes[node]['type'] = 'PROCESS'
            g.nodes[node]['image_path'] = node_name

        elif "FLOW" in node_type:
            g.nodes[node]['type'] = 'FLOW'
            g.nodes[node]['remote_ip'] = node_name
        elif "REG" in node_type:
            g.nodes[node]['type'] = 'REG'
            g.nodes[node]['key'] = node_name
        elif "SHELL" in node_type:
            g.nodes[node]['type'] = 'SHELL'
            g.nodes[node]['command'] = node_name
        else:
            print("Unknown node type: ", node_type)

    for e1, e2, edge_attr in g.edges(data=True):
        # 使用multi-hot向量来选择对应的字符串
        selected_strings = ",".join([val for val in edge_attr['edge_type']])
        selected_strings = selected_strings.replace("EVENT_", "")
        edge_attr['type'] = selected_strings

    # 将相同名字且类型相同的节点合并
    # 解除冻结图
    sbg = g.copy()
    nname2id = {sbg.nodes[node]["type"] + sbg.nodes[node]["name"]: node for node in sbg.nodes}

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
        # 如果存在poi属性，则添加到新节点中
        if 'poi' in sbg.nodes[node]:
            sbg.nodes[nname2id[node_name]]['poi'] = True
        sbg.remove_node(node)

    # remove attribute name
    for node in sbg.nodes:
        sbg.nodes[node].pop("name")
        sbg.nodes[node].pop("node_type")

    # 冻结图
    sbg = nx.freeze(sbg)

    return sbg

def get_subgraphs_attributes(dataset,query_graph=False):
    attributes = {}
    if dataset in "darpa_cadets":
        if query_graph:
            attributes['process'] = 'image_path'
            attributes['file'] = 'file_path'
            attributes['flow'] = 'remote_ip'
        else:
            attributes['process'] = 'NA'
            attributes['pipe'] = 'NA'
            attributes['file'] = 'object_paths'
            attributes['flow'] = 'remote_ip'
    elif dataset in "darpa_theia":
        if query_graph:
            attributes['process'] = 'image_path'
            attributes['file'] = 'file_path'
            attributes['flow'] = 'remote_ip'
            attributes['memory'] = "image_path"
        else:
            attributes['process'] = 'command_lines'
            attributes['file'] = 'NA'
            attributes['memory'] = 'NA'
            attributes['flow'] = 'remote_ip'
    elif dataset in "darpa_trace":
        if query_graph:
            attributes['process'] = 'image_path'
            attributes['file'] = 'file_path'
            attributes['flow'] = 'remote_ip'
            attributes['memory'] = "image_path"
        else:
            attributes['process'] = 'command_lines'
            attributes['file'] = 'object_paths'
            attributes['flow'] = 'remote_ip'
            attributes['memory'] = 'NA'
    elif dataset in "darpa_optc":
        if query_graph:
            attributes['process'] = 'image_path'
            attributes['file'] = 'file_path'
            attributes['flow'] = 'remote_ip'
            attributes['reg'] = 'key'
            attributes['shell'] = 'command'
        else:
            attributes['process'] = 'image_path'
            attributes['file'] = 'file_path'
            attributes['flow'] = 'remote_ip'
            attributes['reg'] = 'key'
            attributes['shell'] = 'command'
    elif 'e5' in dataset:
        if query_graph:
            attributes['process'] = 'image_path'
            attributes['file'] = 'file_path'
            attributes['flow'] = 'remote_ip'
        else:
            attributes['process'] = 'image_path'
            attributes['file'] = 'file_path'
            attributes['flow'] = 'remote_ip'
    else:
        print("Undefined dataset")
    return attributes

def draw_graph(vis_graph,dataset,host=""):
    nodes_labels = {}
    l_ts = []
    attributes = get_subgraphs_attributes(dataset,query_graph=True)

    for node_id, node_attr in list(vis_graph.nodes(data=True)):
        if node_attr['type'].lower() in attributes:
            label_type = attributes[node_attr['type'].lower()]
        if label_type and label_type in node_attr:
            # 按每15个字符换行
            nodes_labels[node_id] = "\n".join([node_attr[label_type][i:i+15] for i in range(0, len(node_attr[label_type]), 15)])
            # if "SHELL" not in node_attr['type']:  # show full name for shell nodes
            #     nodes_labels[node_id] = node_attr[label_type].split(" ")[0].split("\\")[-1].split("/")[-1]
            # else:
            #     nodes_labels[node_id] = node_attr[label_type]
            # if "optc" in dataset:
            #     nodes_labels[node_id] = nodes_labels[node_id].split(" -")[0]
        else:
            nodes_labels[node_id] = node_attr['type'].lower()

        # 判断是否存在ioc标签
        if 'poi' in node_attr:
            nodes_labels[node_id] += " (poi)"
            l_ts.append(host + ":" + node_attr['uuid'] + " : " + nodes_labels[node_id])

    plt.figure(figsize=(15,15))
    pos = nx.spring_layout(vis_graph,k=1.5)
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
        'FILE':  '#D19494',
        'FLOW': '#8FC3E4',
        'REG': '#D19494',
        'SHELL': '#D15424',
        'poi': 'red'  # 定义具备poi属性的节点颜色为红色
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
        edge_labels={ (e1,e2):edge_attr['type'] for e1,e2,edge_attr in list(vis_graph.edges(data=True))},
        font_color='red',
        font_size=5,
    )
    nx.draw_networkx_labels(vis_graph, pos, labels=nodes_labels, font_size=10, font_color='black', font_weight='bold')

    l_ts = "\n".join([f"{name}" for name in l_ts])
    # 添加l_ts到图中垂直显示
    plt.text(0.5, -0.05, f"{l_ts}", fontsize=11,
             ha='center', va='center', transform=plt.gca().transAxes)

    plt.axis('off')
    plt.show()

    # 保存图片
    # plt.savefig(f"groundtruth/{dataset}/vis_prov/{dataset}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

if __name__ == '__main__':

    datasets = ["darpa_cadets"]

    for dataset in datasets:

        with open("../dataset/{}/".format(dataset)+ 'node_type_map.json', 'r') as f:
            node_type_dict = json.load(f)
        with open("../dataset/{}/".format(dataset)+ 'edge_type_map.json', 'r') as f:
            edge_type_dict = json.load(f)
        with open("../dataset/{}/".format(dataset)+ 'names.json', 'r') as f:
            uuid2name = json.load(f)
        with open("../dataset/{}/".format(dataset)+ 'types.json', 'r') as f:
            uuid2type = json.load(f)

        in_dir = "../dataset/{}/sgs_demo".format(dataset)
        files = os.listdir(in_dir)

        for f in files:

            poi = re.findall( r'([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})', f)[0]
            print("current file: {}:{}".format(f,uuid2name[poi]))

            # read each file lines
            with open(in_dir + '/' + f, 'r') as fin:
                lines = fin.readlines()

            nx_muldi_graph = trans2nx(lines,uuid2name,uuid2type)

            nx_muldi_graph = attach_attributes(nx_muldi_graph,poi)

            json_data = json_graph.node_link_data(nx_muldi_graph)

            draw_graph(nx_muldi_graph,dataset)