from collections import deque
from datetime import datetime

import json
import os
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pytz
import torch
from matplotlib import pyplot as plt
from networkx.readwrite import json_graph

from preprocess.parse_trace import read_single_graph

explosion_nodes = ["libc.so.7","null","ld-elf.so.1","libmap.conf","ld-elf.so.hints","tty"]
low_value_nodes = ["log","127.0.0.1","cat","libthr.so.3","..","libutil.so.9","libpcap.so.8","hpet0","libm.so.5"]
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
    else:
        print("Undefined dataset")
    return attributes
def draw_query_graph(vis_graph,dataset):
    nodes_labels = {}
    l_ts = []

    attributes = get_subgraphs_attributes(dataset,query_graph=True)
    for node_id, node_attr in list(vis_graph.nodes(data=True)):
        if node_attr['type'].lower() in attributes:
            label_type = attributes[node_attr['type'].lower()]
        if label_type and label_type in node_attr:
            nodes_labels[node_id] = node_attr[label_type].split("\\")[-1].split("/")[-1]
            # nodes_labels[node_id] = node_attr[label_type]
        else:
            nodes_labels[node_id] = node_attr['type'].lower()
        # Check if the node has the "ioc" tag
        if 'ioc' in node_attr:
            nodes_labels[node_id] += " (IOC)"
            # Store node timestamps:
            l_ts.append((node_attr['ts'],nodes_labels[node_id]))

    # print("node_labels:",nodes_labels)
    # Sort l_ts to select the earliest and latest timestamps
    l_ts.sort(key=lambda x: x[0])
    # Convert timestamps
    l_ts = [(datetime.fromtimestamp(ts // 1000000000, tz=pytz.timezone("America/Nipigon")).strftime('%Y-%m-%d %H:%M:%S'),name) for ts, name in l_ts]

    plt.figure(figsize=(12,12))
    pos = nx.spring_layout(vis_graph,k=1.5)
    # Define node styles and colors for each type
    shapes = {
        'PROCESS': 's',  # square
        'FILE': 'o',  # circle (used for ellipses)
        'FLOW': 'd'  # diamond
    }
    colors = {
        'PROCESS': '#C3C5C7',
        'FILE':  '#D19494',
        'FLOW': '#8FC3E4',
        'IOC': 'red'  # Define nodes with ioc attribute as red
    }
    # Draw nodes for each type
    for node_type, shape in shapes.items():
        nodes = [n for n in vis_graph.nodes if vis_graph.nodes[n]['type'] == node_type]
        ioc_nodes = [n for n in nodes if 'ioc' in vis_graph.nodes[n]]
        normal_nodes = [n for n in nodes if 'ioc' not in vis_graph.nodes[n]]
        # Draw nodes without the ioc attribute
        if normal_nodes:
            nx.draw_networkx_nodes(
                vis_graph, pos,
                nodelist=normal_nodes,
                node_color=colors[node_type],
                node_shape=shape,
                node_size=1200,
                alpha=0.9
            )
        # Draw nodes with the ioc attribute
        if ioc_nodes:
            nx.draw_networkx_nodes(
                vis_graph, pos,
                nodelist=ioc_nodes,
                node_color='red',  # red
                node_shape=shape,
                node_size=1200,
                alpha=0.9
            )
    nx.draw_networkx_labels(vis_graph, pos, labels=nodes_labels, font_size=10, font_color='black', font_weight='bold')
    # Draw edges
    nx.draw_networkx_edges(vis_graph, pos, arrowstyle='-|>', arrowsize=8, edge_color="grey")
    nx.draw_networkx_edge_labels(vis_graph, pos,
        edge_labels={ (e1,e2):edge_attr['type'] for e1,e2,edge_attr in list(vis_graph.edges(data=True))},
        font_color='red',
        font_size=5,
    )
    l_ts = "\n".join([f"{ts}: {name}" for ts,name in l_ts])
    # Add l_ts vertically to the graph
    plt.text(0.5, -0.05, f"{l_ts}", fontsize=11,
             ha='center', va='center', transform=plt.gca().transAxes)
    # Draw node labels
    plt.axis('off')
    plt.show()

    # Save the image
    # plt.savefig(f"groundtruth/{dataset}/vis_prov/{dataset}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

def read_json_graph(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)

 # Filter sampled nodes
def filter_nodes(nodes,uuid2name,node_map_reverse):
    # Merge explosion_nodes and low_value_nodes into a set to speed up lookup
    combined_nodes = set(explosion_nodes) | set(low_value_nodes)

    rt_nodes = []
    for node in nodes:
        node_name = uuid2name[node_map_reverse[node]]
        # Use the any() function to check if any exclusion terms appear in the node name
        if not any(explosion in node_name for explosion in combined_nodes):
            rt_nodes.append(node)
    return rt_nodes
def trans2sbg(g, subgraph_nodes, node_map_reverse, uuid2type, uuid2name, edge_type_dict):
    for node in subgraph_nodes:
        node_uuid = node_map_reverse[node]
        node_type = uuid2type[node_uuid]
        node_name = uuid2name[node_uuid]

        if "FILE" in node_type:
            g.nodes[node]['type'] = 'FILE'
            g.nodes[node]['file_path'] = node_name
        elif "PROCESS" in node_type:
            g.nodes[node]['type'] = 'PROCESS'
            g.nodes[node]['image_path'] = node_name
        elif "Flow" in node_type:
            g.nodes[node]['type'] = 'FLOW'
            g.nodes[node]['remote_ip'] = node_name
    sbg = g.subgraph(subgraph_nodes)

    # Convert edge attributes to strings
    edge_types = list(edge_type_dict.keys())
    for e1, e2, edge_attr in sbg.edges(data=True):
        # Use multi-hot vectors to select corresponding strings
        selected_strings = ",".join([edge_types[val] for val in edge_attr['edge_type']])
        selected_strings = selected_strings.replace("EVENT_", "")
        edge_attr['type'] = selected_strings

    # Merge nodes with the same name and type
    # Unfreeze the graph
    sbg = sbg.copy()
    nname2id = {uuid2type[node_map_reverse[node]] + uuid2name[node_map_reverse[node]]: node for node in sbg.nodes}
    nodes = list(sbg.nodes)  # Create a copy of the node list
    for node in nodes:
        node_name = uuid2type[node_map_reverse[node]] + uuid2name[node_map_reverse[node]]
        if node_name.endswith("/"):
            sbg.remove_node(node)
            continue
        if node == nname2id[node_name]:
            continue
        # Transfer all edges of the current node to nname2id[node_name]
        for edge in list(sbg.in_edges(node, keys=True)) + list(sbg.out_edges(node, keys=True)):
            src, dst, key = edge
            if src == node:
                src, dst = dst, src
            sbg.add_edge(src, nname2id[node_name], key=key, **sbg.edges[edge])
        sbg.remove_node(node)
    # Freeze the graph
    sbg = nx.freeze(sbg)
    return sbg

# Extract subgraphs based on IOC, deprecated, before PPG
def extract_subgraphs(node_map, train_g, iocs2uuids, node_type_dict, uuid2name, uuid2type, edge_type_dict, iocs, khop=2):
    ioc2sbgs = []
    # Get the UUIDs of IOCs
    current_iocs_uuids = [(ioc, uuid) for ioc, uuids in iocs2uuids.items() for uuid in uuids if uuid in node_map]
    # Add tags to all IOC nodes in the graph
    for ioc, uuid in current_iocs_uuids:
        node_cnt = node_map[uuid]
        train_g.nodes[node_cnt]['ioc'] = 1
    # Mapping from node index to UUID
    node_map_reverse = {v: k for k, v in node_map.items()}
    for ioc, uuid in current_iocs_uuids:
        node_cnt = node_map[uuid]
        # Nodes corresponding to the subgraph
        # Use a queue to implement BFS
        queue = deque([(node_cnt, 0)])  # (current node, current depth)
        visited = set([node_cnt])  # Set of visited nodes
        matched_iocs = set([ioc])
        while queue:
            current_node, depth = queue.popleft()
            # Direct neighbors of the current node
            current_neigs = set(train_g.predecessors(current_node)).union(set(train_g.successors(current_node)))
            # Filter nodes at the current level
            # current_neigs = filter_nodes(current_neigs, uuid2name, node_map_reverse)
            # Avoid adding more than 70 neighbors
            if len(current_neigs) > 70:
                # Retain only process nodes
                # current_node_name = uuid2name[node_map_reverse[current_node]]
                # print(f"Number of nodes exceeds 70: {current_node_name}, {len(current_neigs)}")
                # current_neigs = set([node for node in current_neigs if "PROCESS" in uuid2type[node_map_reverse[node]]])
                continue
            # Add suitable neighbors to the set of subgraph nodes
            for neighbor in current_neigs:
                if neighbor not in visited:
                    visited.add(neighbor)
                    # If an IOC is sampled, recursively sample
                    nei_name = uuid2name[node_map_reverse[neighbor]]
                    if nei_name in iocs:
                        matched_iocs.add(nei_name)
                        queue.append((neighbor, 0))
                    else:
                        if depth < khop - 1:  # Only add neighbor nodes to the queue if k-hop depth is not reached
                            queue.append((neighbor, depth + 1))

        subgraph_nodes = visited
        if len(subgraph_nodes) < 10:
            print(f"less than 5 nodes: {ioc}")
            continue

        for old_ioc, old_sb in ioc2sbgs.copy():
            duplicated_nodes = set(old_sb) & set(subgraph_nodes)
            if len(duplicated_nodes) >= min(len(subgraph_nodes), len(old_sb)):
                print(f"==============================================High overlap subgraph: "
                      f"{old_ioc}:{ioc}, {len(old_sb)}:{len(subgraph_nodes)}, duplicate nodes: {len(duplicated_nodes)}")
                # Replace with subgraph containing more IOC nodes
                if len(matched_iocs) >= len(old_ioc):
                    ioc2sbgs.remove((old_ioc, old_sb))
        ioc2sbgs.append((matched_iocs, subgraph_nodes))
    # Convert to subgraph
    for ioc, nodes in ioc2sbgs.copy():
        # print(f"{ioc}: {len(nodes)}")
        ioc2sbgs.append((ioc, trans2sbg(train_g, nodes, node_map_reverse, uuid2type, uuid2name, edge_type_dict)))
        ioc2sbgs.remove((ioc, nodes))
    return ioc2sbgs

def containIoCs(name, iocs):
    for ioc in iocs:
        if ioc in name:
            return True
    return False

# , deprecated, before PPG
def subgraph_construct(dataset):
    with open("../dataset/{}/".format(dataset) + 'node_type_map.json', 'r') as f:
        node_type_dict = json.load(f)
    with open("../dataset/{}/".format(dataset) + 'edge_type_map.json', 'r') as f:
        edge_type_dict = json.load(f)
    with open("../dataset/{}/".format(dataset) + 'names.json', 'r') as f:
        uuid2name = json.load(f)
        # name2uuid = {v: k for k, v in uuid2name.items()}
    with open("../dataset/{}/".format(dataset) + 'types.json', 'r') as f:
        uuid2type = json.load(f)
    with open("../dataset/{}/".format(dataset) + 'query_iocs', 'r') as f:
    # with open("./groundtruth/{}/".format(dataset) + 'query_graphs_IOCs.json', 'r') as f:
        lines = f.readlines()
        graph2iocs = {}
        iocs = set()
        for line in lines:
            graph, gt_iocs = line.split(",")[0], line.split(",")[1:]
            graph2iocs[graph] = []
            for ioc in gt_iocs:
                iocs.add(ioc)
                graph2iocs[graph].append(ioc)

    iocs2uuids = {}
    for uuid, name in uuid2name.items():
        if containIoCs(name, iocs):
            if name not in iocs2uuids:
                iocs2uuids[name] = []
            iocs2uuids[name].append(uuid)

    sbg_idx = 0
    for file in os.listdir('../dataset/{}/tuples/'.format(dataset)):
        path = '../dataset/{}/tuples/'.format(dataset) + file
        node_map, train_g, _, _ = read_single_graph(dataset, path, False)
        ioc2sbgs = extract_subgraphs(node_map, train_g, iocs2uuids, node_type_dict, uuid2name, uuid2type, edge_type_dict, iocs)

        # Visualize and output subgraph data
        for iocs, sbg in ioc2sbgs:
            draw_query_graph(sbg, dataset)
            # Convert MultiDiGraph object to JSON data
            json_data = json_graph.node_link_data(sbg)
            # Match query graph index
            query_graph_idx_set = {graph_idx for ioc in iocs for graph_idx, gt_iocs in graph2iocs.items() for gt_ioc in gt_iocs if gt_ioc in ioc}
            query_graph_idx = "_".join(query_graph_idx_set)
            if query_graph_idx == "":
                print(iocs)
                exit()
            # Save graph data to JSON file
            file_path = f'../dataset/{dataset}/subgraphs/{sbg_idx}_graph_{query_graph_idx}.json'
            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            sbg_idx += 1

if __name__ == "__main__":

    # datasets = ["darpa_cadets","darpa_theia","darpa_trace","darpa_optc"]
    datasets = ["darpa_optc"]

    for dataset in datasets:
        # in_dir = f"MEGR_groundtruth/{dataset}/query_graphs" # visualize MEGR-APT query graphs
        in_dir = f"../dataset/{dataset}/groundtruth/" # visualize xxx query graphs
        files = os.listdir(in_dir)

        for f in files:
            print(f)

            # show sampled subgraph
            tmp_in_dir = in_dir.replace("groundtruth", "sampled_graphs")
            tmp_f = f.replace("gt", "sg")

            if not os.path.exists(f"{tmp_in_dir}/{tmp_f}"):
                print(f"{tmp_in_dir}/{tmp_f} not exists")
                continue
            # show gt
            gt = read_json_graph(f"{in_dir}/{f}")
            draw_query_graph(gt,dataset)

            sg = read_json_graph(f"{tmp_in_dir}/{tmp_f}")
            # draw_query_graph(sg,dataset)

        # subgraph_construct("darpa_cadets")