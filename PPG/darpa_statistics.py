import json
import os
import pickle
import sys
from datetime import datetime
from dateutil import parser
import networkx as nx
import pytz
from tqdm import tqdm


def graph_analysis(sce2events):
    print(
        "================================================ Graph Analysis =================================================")
    for sce_id in sce2events:

        statistics = {}
        if sce_id not in statistics:
            statistics[sce_id] = {
                "num_edges": 0,  # Number of edges
                "nodes": set(),  # Set of unique nodes
                "start_ts": 0,  # Start timestamp
                "end_ts": 0  # End timestamp
            }
        f_lines = sce2events[sce_id]

        # Increment the number of edges
        statistics[sce_id]["num_edges"] += len(f_lines)
        for l in f_lines:
            # Collect unique nodes
            statistics[sce_id]["nodes"].add(l[0])
            statistics[sce_id]["nodes"].add(l[1])

        # Sort the events by timestamp (6th column)
        f_lines.sort(key=lambda l: l[5])
        statistics[sce_id]["start_ts"] = f_lines[0][5]  # Earliest timestamp
        statistics[sce_id]["end_ts"] = f_lines[-1][5]  # Latest timestamp

        for sce_id in statistics:
            print(f"Scenario {sce_id}:")
            print(f"Number of edges: {statistics[sce_id]['num_edges']}")
            print(f"Number of nodes: {len(statistics[sce_id]['nodes'])}")

            # Convert start and end timestamps to human-readable format
            start_ts = datetime.fromtimestamp(
                statistics[sce_id]['start_ts'] // 1000000000,
                tz=pytz.timezone("America/Nipigon")
            )
            print(f"Start timestamp: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}")

            end_ts = datetime.fromtimestamp(
                statistics[sce_id]['end_ts'] // 1000000000,
                tz=pytz.timezone("America/Nipigon")
            )
            print(f"End timestamp: {end_ts.strftime('%Y-%m-%d %H:%M:%S')}")

            # Calculate the time difference
            time_difference = end_ts - start_ts
            total_seconds = time_difference.total_seconds()

            # Convert the time difference into hours, minutes, and seconds
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            # Print the time difference in HH:MM:SS format
            print(f"Time difference: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Mark as seen
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size
def npg_construct(lines, node_type_dict, edge_type_dict,node_names_dict):
    g = nx.MultiDiGraph()

    node_map = {}
    node_type_map = {}
    node_cnt = 0
    node_list = []

    for l in lines:
        src, dst, src_type, dst_type, edge_type, ts = l
        src_type_id = node_type_dict[src_type]
        dst_type_id = node_type_dict[dst_type]
        edge_type_id = edge_type_dict[edge_type]
        if src not in node_map:
            node_map[src] = node_cnt
            src_name = node_names_dict[src]
            # g.add_node(node_cnt, node_type=one_hot_encode(src_type_id, node_type_cnt+1))
            g.add_node(node_cnt, node_type=src_type_id,name=src_name)
            node_list.append(src)
            node_type_map[src] = src_type
            node_cnt += 1
        if dst not in node_map:
            node_map[dst] = node_cnt
            dst_name = node_names_dict[dst]
            g.add_node(node_cnt, node_type=dst_type_id,name=dst_name)
            node_type_map[dst] = dst_type
            node_list.append(dst)
            node_cnt += 1
        if not g.has_edge(node_map[src], node_map[dst]):
            g.add_edge(node_map[src], node_map[dst], key=0, edge_type=edge_type_id, ts=ts)
        else:
            g.add_edge(node_map[src], node_map[dst], key=len(g[node_map[src]][node_map[dst]]), edge_type=edge_type_id, ts=ts)
    return g
def ppg_construct(lines, node_type_dict, edge_type_dict):

    g = nx.MultiDiGraph()

    node_map = {}
    node_type_map = {}
    node_cnt = 0
    node_list = []

    for l in lines:
        src, dst, src_type, dst_type, edge_type, ts = l
        src_type_id = node_type_dict[src_type]
        dst_type_id = node_type_dict[dst_type]
        edge_type_id = edge_type_dict[edge_type]
        if src not in node_map:
            node_map[src] = node_cnt
            g.add_node(node_cnt, node_type=src_type_id)
            node_list.append(src)
            node_type_map[src] = src_type
            node_cnt += 1
        if dst not in node_map:
            node_map[dst] = node_cnt
            g.add_node(node_cnt, node_type=dst_type_id)
            node_type_map[dst] = dst_type
            node_list.append(dst)
            node_cnt += 1
        if not g.has_edge(node_map[src], node_map[dst]):
            g.add_edge(node_map[src], node_map[dst], key=0, edge_type=[edge_type_id], ts=1)
        elif edge_type_id not in g[node_map[src]][node_map[dst]][0]['edge_type']:
            g[node_map[src]][node_map[dst]][0]['edge_type'].append(edge_type_id)
    return g
def compact_analysis(dataset,day2events):
    print("================================================ Compact Analysis =================================================")

    with open(f"../dataset/darpa_{dataset}/node_type_map.json", 'r') as f:
        node_type_dict = json.load(f)
    with open(f"../dataset/darpa_{dataset}/edge_type_map.json", 'r') as f:
        edge_type_dict = json.load(f)
    with open(f"../dataset/darpa_{dataset}/names.json", 'r') as f:
        node_names_dict = json.load(f)

    for day_num in day2events:

        lines = day2events[day_num]

        npg = npg_construct(lines,node_type_dict, edge_type_dict,node_names_dict)
        ppg = ppg_construct(lines,node_type_dict, edge_type_dict)

        print(f"Day {day_num}:")
        npg_graph_size_gb = get_size(npg)/1024/1024/1024
        ppg_graph_size_gb = get_size(ppg)/1024/1024/1024

        max_dist = 0
        for e in ppg.edges(data=True):

            src, dst, edge_data = e

            dist = abs(src-dst)
            if dist > max_dist:
                max_dist = dist
        print(f"Max distance between nodes: {max_dist}")
        print("npg edge num: ",len(npg.edges))
        print("ppg edge num: ",len(ppg.edges))

        compact_ratio = (1-(ppg_graph_size_gb/npg_graph_size_gb)) * 100

        print("Original Graph Size(GB): {:.2f}".format(npg_graph_size_gb))
        print("Compact Graph Size(GB): {:.2f}".format(ppg_graph_size_gb))
        print(f"Compact Ratio: {compact_ratio:.2f}%")
def combine_events(dataset):

    in_dir = f"../dataset/{dataset}/tuples"
    out_file = f"../dataset/{dataset}/sce_combines.pkl"
    sce2events = {}
    sce2dedu_num = {}

    if os.path.exists(out_file):
        with open(out_file, "rb") as f:
            sce2events = pickle.load(f)
        for sce_id in sce2events:
            print(f"Scenario {sce_id}: {len(sce2events[sce_id])} events")
            # 判断day2events是否按照时间戳排序,并排序
            events = sce2events[sce_id]
            events.sort(key=lambda l: l[5])

            with open(f"../dataset/{dataset}/sce{sce_id}.txt", "w", encoding="utf-8") as f:
                for l in sce2events[sce_id]:
                    f.write(f"{l[0]}\t{l[1]}\t{l[2]}\t{l[3]}\t{l[4]}\t{l[5]}\n")
        return sce2events

    # Define hostid2idx based on the dataset outside the loop
    dataset_to_hostid = {
        "darpa_cadets": {"o": "040611", "1": "041115", "2": "041214"},
        "darpa_theia": {"1": "0", "3": "1", "5": "2", "6": "041314"},
        "darpa_trace": {"o": "041010", "1": "041213"},
        "darpa_five": {"o": 1, "2":2, "3": 3},
    }

    # Process each file in the directory
    for file in os.listdir(in_dir):
        print(f"Processing {file}...")

        path = os.path.join(in_dir, file)
        sce_id = file.split(".json")[0].split("-")[-1][0]
        sce_id = dataset_to_hostid[dataset].get(sce_id, None)

        if sce_id is None:
            continue  # Skip files that do not have a valid host_id

        print(f"Scenario id: {sce_id}")
        if sce_id not in sce2events:
            sce2events[sce_id] = []
        if sce_id not in sce2dedu_num:
            sce2dedu_num[sce_id] = 0

        with open(path, "r") as f:
            lines = [line.split('\t') for line in f]
            # print("original lines num: ",len(lines))
            # deduplicate logs
            d_lines = deduplicate_adjacent_same_events(lines)
            sce2dedu_num[sce_id] += len(lines) - len(d_lines)
            # print("single dedu num: ",len(lines) - len(d_lines))
            dd_lines = deduplicate_adjacent_same_events_in2(d_lines)
            sce2dedu_num[sce_id] += len(d_lines) - len(dd_lines)
            # print("dual dedu num: ",len(d_lines) - len(dd_lines))

            processed_lines = [[src, dst, src_type, dst_type, edge_type, int(ts)] for
                               src, src_type, dst, dst_type, edge_type, ts in dd_lines]
            sce2events[sce_id].extend(processed_lines)

        # print(sce2dedu_num)
    # print deduplicated lines num
    for sce_id in sce2dedu_num:
        print(f"Scenario {sce_id}: original events num: {len(sce2events[sce_id]) + sce2dedu_num[sce_id]}, {sce2dedu_num[sce_id]} semantic reduction events")

    with open(out_file, "wb") as f:
        pickle.dump(sce2events, f)

    return sce2events
def deduplicate_adjacent_same_events_in2(lines):
    result = []
    previous_entry1 = ""
    previous_entry2 = ""

    # Iterate over the lines in pairs
    i = 0
    while i < len(lines):
        if i + 1 >= len(lines):
            # Handle the case where the number of lines is odd
            result.append(lines[i])
            break

        # Remove timestamp from both lines
        trimmed_line1 = lines[i][:-1]
        trimmed_line2 = lines[i + 1][:-1]
        # Combine the two lines into a single string for comparison
        combined_line = trimmed_line1 + trimmed_line2
        # If the current combined entry is different from the previous combined entry, add it to result
        if combined_line != previous_entry1 + previous_entry2:
            result.append(lines[i])
            result.append(lines[i + 1])
            previous_entry1 = trimmed_line1  # Update previous entry
            previous_entry2 = trimmed_line2  # Update previous entry
        # else:
        #     print("Current line: ",lines[i] + lines[i + 1])
        #     print("Previous line: ",previous_entry1 + previous_entry2)
        i += 2

    return result
def deduplicate_adjacent_same_events(lines):
    result = []
    previous_entry = ""

    # Iterate over the lines
    for line in lines:
        # Remove timestamp
        trimmed_line = line[:-1]

        # If the current entry is different from the previous entry, add it to result
        if trimmed_line != previous_entry:
            result.append(line)
            previous_entry = trimmed_line  # Update previous entry
        # else:
        #     print("Current line: ",line)
        #     print("Previous line: ",previous_entry)
    return result

def graph_size_analysis():

    g = nx.MultiDiGraph()

    print("================================================ Graph Size Analysis =================================================")
    print("Empty Graph Size(B): {:.2f}".format(get_size(g)))

    g.add_node(0, node_type=0)
    g.add_node(1, node_type=1)

    g.add_edge(0, 1, key=0, edge_type=[0], ts=1)

    print("Graph Size with 2 node and 1 edge(B): {:.2f}".format(get_size(g)))

    print("one integer size: ",sys.getsizeof(1))
    print("one float size: ",sys.getsizeof(1.0))

    print(1522706861813350340/1e9)


def optc_analysis():
    statistics = {"num_edges": 0, "nodes": set(), "start_ts": 0, "end_ts": 0}

    in_file = os.path.join("../dataset/darpa_optc/tuples/benign_system0201.json.txt")
    with open(in_file, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            src, dst, src_type, dst_type, edge_type, ts = line.split("\t")
            statistics["num_edges"] += 1
            statistics["nodes"].add(src)
            statistics["nodes"].add(dst)

            # Use dateutil to parse timestamps, accurate to millisecond Unix timestamps
            ts = parser.parse(ts.strip()).timestamp() * 1000
            # ts = datetime.fromisoformat(ts).astimezone(timezone.utc).timestamp() * 1000
            # compare string timestamps
            if ts < statistics["start_ts"] or statistics["start_ts"] == 0:
                statistics["start_ts"] = ts
            if ts > statistics["end_ts"]:
                statistics["end_ts"] = ts

        print(f"Number of edges: {statistics['num_edges']}")
        print(f"Number of nodes: {len(statistics['nodes'])}")
        ori_start_time = datetime.fromtimestamp(statistics["start_ts"] // 1000)
        print(f"Start timestamp: {ori_start_time}")
        ori_end_time = datetime.fromtimestamp(statistics["end_ts"] // 1000)
        print(f"End timestamp: {ori_end_time}")
        # Calculate time difference
        time_difference = statistics["end_ts"] - statistics["start_ts"]
        # Convert time difference to seconds
        total_seconds = time_difference / 1000
        # Calculate hours, minutes, and seconds
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        # Print time difference
        print(f"Time difference: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    # Additional statistics analysis
    print("\nAdditional Statistics:")
    print(f"Total unique nodes: {len(statistics['nodes'])}")
    print(f"Time span in milliseconds: {statistics['end_ts'] - statistics['start_ts']} ms")
    print(f"Start time (ISO format): {ori_start_time.isoformat()}")
    print(f"End time (ISO format): {ori_end_time.isoformat()}")

    # Save results to a file
    out_file = os.path.join("../dataset/darpa_optc/statistics_summary.txt")
    with open(out_file, 'w') as f:
        f.write(f"Number of edges: {statistics['num_edges']}\n")
        f.write(f"Number of nodes: {len(statistics['nodes'])}\n")
        f.write(f"Start timestamp: {ori_start_time}\n")
        f.write(f"End timestamp: {ori_end_time}\n")
        f.write(f"Time difference: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n")
        f.write(f"Total unique nodes: {len(statistics['nodes'])}\n")
        f.write(f"Time span in milliseconds: {statistics['end_ts'] - statistics['start_ts']} ms\n")
        f.write(f"Start time (ISO format): {ori_start_time.isoformat()}\n")
        f.write(f"End time (ISO format): {ori_end_time.isoformat()}\n")

    print(f"Statistics summary saved to {out_file}")

if __name__ == "__main__":
    # datasets = ["darpa_cadets", "darpa_theia", "darpa_trace", "darpa_optc"]
    datasets = ["darpa_cadets"]

    for dataset in datasets:
        print(f"============================ Processing dataset: {dataset}")

        if "optc" not in dataset:
            sce2events = combine_events(dataset)
            graph_analysis(sce2events)
            # compact_analysis(dataset,host2events)
            # graph_size_analysis()
        else:
            optc_analysis()
