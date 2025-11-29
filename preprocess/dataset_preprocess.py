import json
import os
import re
import numpy as np
from tqdm import tqdm
import networkx as nx
import torch

listened_event_types = ["write","rename","read","create","link","modify","connect","recv","send","exec","fork","exit","clone"]

pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1_path = re.compile(r'predicateObjectPath\":\{\"string\":\"(.*?)\"')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2_path = re.compile(r'predicateObject2Path\":\{\"string\":\"(.*?)\"')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')
pattern_src_path = re.compile(r'exec\":\"(.*?)\"')

abstract_process_map = {}
abstract_file_map = {}

def parse_shell_event(payload,command):

    re_value = re.compile(r"value=(.*)")
    # find all values in payload
    values = re_value.findall(payload)
    if len(values) == 0 or "cannot" in payload:
        return command
    else:
        for v in values:
            if len(v) > 100: # truncate long values
                v = v[ :100]
            command += "," + v

    return command

def preprocess_dataset_optc():

    pattern_src_id = re.compile(r'actorID\":"(.*?)\"')
    pattern_src_process_name = re.compile(r'"image_path":"(.*?)"')
    pattern_par_src_process_name = re.compile(r'parent_image_path":"(.*?)"')
    pattern_edge_type = re.compile(r"\"action\":\"(.*?)\"")
    pattern_dst_id = re.compile(r'objectID\":\"(.*?)\"')
    pattern_dst_shell_id = re.compile(r'id\":\"(.*?)\"')
    pattern_dst_type = re.compile(r'object\":\"(.*?)\"')
    pattern_dst_file_name = re.compile(r'file_path\":\"(.*?)\"')
    pattern_dst_module_name = re.compile(r'module_path\":\"(.*?)\"')
    pattern_dst_flow_in_ip_name = re.compile(r'src_ip\":\"(.*?)\"')
    pattern_dst_flow_in_port_name = re.compile(r'src_port\":\"(.*?)\"')
    pattern_dst_flow_out_ip_name = re.compile(r'dest_ip\":\"(.*?)\"')
    pattern_dst_flow_out_port_name = re.compile(r'dest_port\":\"(.*?)\"')
    pattern_dst_reg_name = re.compile(r'key\":\"(.*?)\"')
    pattern_dst_process_name = re.compile(r'command_line":"(.*?)"')
    pattern_dst_shell_name = re.compile(r'payload":"(.*?)"')
    pattern_time = re.compile(r'timestamp\":\"(.*?)\"')
    pattern_dst_flow_direction = re.compile(r'direction\":\"(.*?)\"')

    # listened shell commands,
    listened_shell_commands = {"get-","invoke-", "job","out-file","test-"}
    pattern_dst_shell_command = re.compile(r'CommandInvocation\((.*?)\)')

    dir = '../dataset/darpa_optc/'
    raw_dir = dir + 'raws/'
    tuples_dir = dir + 'tuples/'

    # store unique shell for each process
    id_nodetype_map = {}
    id_nodename_map = {}
    edge_type_dict = {}
    node_type_dict = {"PROCESS":0}
    for file in os.listdir(raw_dir):

        in_file = raw_dir + file
        out_file = tuples_dir + file + '.txt'
        if os.path.exists(out_file):
            print('file {} already exists!'.format(file))
            # continue

        # =================================================================debug
        # if "0501" not in file:
        #     continue

        with open(in_file, 'r', encoding='utf-8') as f:
            f_lines = f.readlines()
        print('processing {} ...'.format(file))

        edges = []
        for line in tqdm(f_lines):

            # replace characters
            line = line.replace(r"\\","\\").replace("\\","/").replace("/\"","").replace("/??/","")

            dstType = pattern_dst_type.findall(line)[0]
            if any(skip_text.upper() in dstType for skip_text in ['user_session', 'thread',"task","service","host"]): continue

            if dstType not in node_type_dict:
                node_type_dict[dstType] = len(node_type_dict)

            srcId = pattern_src_id.findall(line)[0]
            if srcId is None: continue

            edgeType = pattern_edge_type.findall(line)[0]
            if edgeType not in edge_type_dict:
                edge_type_dict[edgeType] = len(edge_type_dict)

            # filter process open events
            if "OPEN" in edgeType and "PROCESS" in dstType:
                continue

            try:
                if "PROCESS" in dstType:
                    srcName = pattern_par_src_process_name.findall(line)
                else:
                    srcName = pattern_src_process_name.findall(line)

                if srcName is None or len(srcName) == 0:    continue

                srcName = srcName[0]

                if srcName == "System": continue
                if srcId not in id_nodename_map:
                    id_nodename_map[srcId] = srcName
            except:
                print(line)
                exit()

            if srcId not in id_nodetype_map:
                id_nodetype_map[srcId] = 'PROCESS'

            srcType = id_nodetype_map[srcId]

            # process shell type event,filter commonly used commands
            if "SHELL" in dstType:
                shell_id = pattern_dst_shell_id.findall(line)[0]

                if shell_id not in id_nodename_map:
                    payload = pattern_dst_shell_name.findall(line)[0]
                    shell_command = pattern_dst_shell_command.findall(payload)[0]
                    shell_command = shell_command.lower()
                    # skip shell commands
                    if not any(listen_command in shell_command for listen_command in listened_shell_commands)\
                            or "random" in shell_command or "acl" in shell_command or "variable" in shell_command\
                            or "service" in shell_command or "path" in shell_command: continue

                    print("shell:",shell_command)
                    dstName = parse_shell_event(payload,shell_command)
                    if dstName == shell_command:   # failed command
                        continue
                    print("dstName:",dstName)
                    id_nodename_map[shell_id] = dstName
                if shell_id not in id_nodetype_map:
                    id_nodetype_map[shell_id] = 'SHELL'
                dstName = id_nodename_map[shell_id]

                dstId = shell_id

            else:
                dstId = pattern_dst_id.findall(line)[0]
                if dstId is None: continue

                if dstId not in id_nodetype_map:
                    id_nodetype_map[dstId] = dstType

                try:
                    if dstId not in id_nodename_map:
                        if "FILE" in dstType:
                            dstName = pattern_dst_file_name.findall(line)
                            if dstName is None or len(dstName) == 0:
                                continue
                            dstName = dstName[0]
                        elif "FLOW" in dstType:
                            direction = pattern_dst_flow_direction.findall(line)[0]

                            if "in" in direction:
                                dstName = pattern_dst_flow_in_ip_name.findall(line)[0]
                            elif "out" in direction:
                                dstName = pattern_dst_flow_out_ip_name.findall(line)[0]
                            else:
                                print("error direction")
                                print(line)
                                exit()
                            # filter local ip
                            if "127.0.0.1" in dstName or "0.0.0.0" in dstName:
                                continue
                        elif "PROCESS" in dstType:
                            dstName = pattern_dst_process_name.findall(line)

                            if dstName is None or len(dstName) == 0:
                                continue
                            dstName = dstName[0]
                        elif "MODULE" in dstType:
                            dstName = pattern_dst_module_name.findall(line)[0]
                        elif "REG" in dstType:
                            dstName = pattern_dst_reg_name.findall(line)[0]

                        id_nodename_map[dstId] = dstName
                except:
                    print(dstType)
                    print(line)
                    exit()

                dstName = id_nodename_map[dstId]

            timestamp = pattern_time.findall(line)[0]

            def listen_edge_type(edgeType):
                listened_event_types = ["write", "add", "read", "create", "start", "open", "message","load","edit","remove","rename","modify","command","delete"]
                for et in listened_event_types:
                    if et.lower() in edgeType.lower():
                        return True
                return False

            if not listen_edge_type(edgeType):
                continue

            this_edge2 = str(srcId) + '\t' + str(dstId) + '\t' + str(srcType) + '\t' + str(
                dstType) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
            edges.append(this_edge2)

        with open(out_file, 'w', encoding='utf-8') as fw:
            fw.write("".join(edges))
    print(len(id_nodetype_map))
    print(len(id_nodename_map))

    if len(id_nodename_map) != 0:
        with open(dir + 'names.json', 'w', encoding='utf-8') as fw:
            json.dump(id_nodename_map, fw)
        print("len of nodename:", len(id_nodename_map))
    if len(id_nodetype_map) != 0:
        with open(dir + 'types.json', 'w', encoding='utf-8') as fw:
            json.dump(id_nodetype_map, fw)
        print("len of nodetype:", len(id_nodetype_map))
    if len(edge_type_dict) != 0:
        with open(dir + 'edge_type_map.json', 'w', encoding='utf-8') as fw:
            json.dump(edge_type_dict, fw)
        print("len of edgetype:", len(edge_type_dict))
    if len(node_type_dict) != 0:
        with open(dir + 'node_type_map.json', 'w', encoding='utf-8') as fw:
            json.dump(node_type_dict, fw)
        print("len of nodetype:", len(node_type_dict))
def preprocess_dataset_cadets():
    pattern_file_name = re.compile(r'map\":\{\"path\":\"(.*?)\"')
    pattern_process_name = re.compile(r'map\":\{\"name\":\"(.*?)\"')
    pattern_netflow_object_name = re.compile(r'remoteAddress\":\"(.*?)\"')

    id_nodetype_map = {}
    id_nodename_map = {}
    edge_type_dict = {}
    node_type_dict = {}

    dir = '../dataset/darpa_cadets/'
    raw_dir = dir + 'raws/'
    tuples_dir = dir + 'tuples/'
    if os.path.exists(dir + 'init_names.json') and os.path.exists(dir + 'init_types.json'):
        print("init_names.json and init_types.json already exist! loading now...")
        id_nodename_map = json.load(open(dir + 'init_names.json', 'r'))
        id_nodetype_map = json.load(open(dir + 'init_types.json', 'r'))
    else:
        for file in os.listdir(raw_dir):
            print('reading {} ...'.format(file))

            f = open(raw_dir + file, 'r', encoding='utf-8')
            flines = f.readlines()
            for line in tqdm(flines):

                if any(skip_text in line for skip_text in [
                    'com.bbn.tc.schema.avro.cdm18.Event', 'com.bbn.tc.schema.avro.cdm18.Host',
                    'com.bbn.tc.schema.avro.cdm18.TimeMarker', 'com.bbn.tc.schema.avro.cdm18.StartMarker',
                    'com.bbn.tc.schema.avro.cdm18.UnitDependency', 'com.bbn.tc.schema.avro.cdm18.EndMarker']):
                    continue

                if len(pattern_uuid.findall(line)) == 0: print(line)
                uuid = pattern_uuid.findall(line)[0]

                subject_type = pattern_type.findall(line)
                if len(subject_type) < 1:
                    if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                        subject_type = 'MemoryObject'
                    if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                        subject_type = 'NetFlowObject'
                    if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                        subject_type = 'UnnamedPipeObject'
                else:
                    subject_type = subject_type[0]

                if subject_type not in node_type_dict:
                    node_type_dict[subject_type] = len(node_type_dict)

                if uuid == '00000000-0000-0000-0000-000000000000' or subject_type in ['SUBJECT_UNIT']:
                    id_nodename_map[uuid] = None
                    continue
                id_nodetype_map[uuid] = subject_type
                id_nodename_map[uuid] = None
                if 'FILE' in subject_type and len(pattern_file_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_file_name.findall(line)[0]
                elif subject_type == 'SUBJECT_PROCESS' and len(pattern_process_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_process_name.findall(line)[0]
                elif subject_type == 'NetFlowObject' and len(pattern_netflow_object_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_netflow_object_name.findall(line)[0]
        # 写入文件
        json.dump(id_nodename_map, open(dir + 'init_names.json', 'w', encoding='utf-8'))
        json.dump(id_nodetype_map, open(dir + 'init_types.json', 'w', encoding='utf-8'))

    for file in os.listdir(raw_dir):

        in_file = raw_dir + file
        out_file = tuples_dir + file + '.txt'
        if os.path.exists(out_file):
            print('file {} already exists!'.format(file))
            continue

        with open(in_file, 'r', encoding='utf-8') as f:
            f_lines = f.readlines()
        print('processing {} ...'.format(file))

        edges = []
        for line in tqdm(f_lines):
            if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                edgeType = pattern_type.findall(line)[0]
                timestamp = pattern_time.findall(line)[0]
                srcId = pattern_src.findall(line)
                srcIdname = pattern_src_path.findall(line)

                if len(srcId) == 0 or len(srcIdname) == 0: continue
                srcId = srcId[0]
                srcIdname = srcIdname[0]

                if (srcIdname == 'null' or srcIdname == 'unknown'):
                    continue
                if srcId not in id_nodetype_map:
                    id_nodetype_map[srcId] = 'SUBJECT_PROCESS'
                if srcId not in id_nodename_map or id_nodename_map[srcId] is None:
                    id_nodename_map[srcId] = srcIdname

                srcType = id_nodetype_map[srcId]

                def listen_edge_type(edgeType):
                    for et in listened_event_types:
                        if et.lower() in edgeType.lower():
                            return True
                    return False

                if not listen_edge_type(edgeType):
                    continue

                if edgeType not in edge_type_dict:
                    edge_type_dict[edgeType] = len(edge_type_dict)

                dstId1 = pattern_dst1.findall(line)
                dstId1name = pattern_dst1_path.findall(line)
                # id_nodename_map[dstId1[0]] != None 为判断Netflow对象
                if len(dstId1) > 0 and dstId1[0] != 'null':

                    dstId1 = dstId1[0]
                    # 若当前对象没有声明，则根据事件类型判断其类型
                    if "FORK" in edgeType:
                        id_nodetype_map[dstId1] = 'SUBJECT_PROCESS'
                        id_nodename_map[dstId1] = srcIdname
                    elif "EXECUTE" in edgeType:
                        assert srcId in id_nodename_map
                        id_nodetype_map[dstId1] = 'FILE_OBJECT_FILE'

                        if len(dstId1name) > 0 and dstId1name[0] != 'null':
                            id_nodename_map[dstId1] = dstId1name[0]
                            id_nodename_map[srcId] = dstId1name[0]
                    elif "MODIFY_PROCESS" in edgeType:
                        id_nodetype_map[dstId1] = 'SUBJECT_PROCESS'
                    elif "READ" in edgeType or "WRITE" in edgeType or "LINK" in edgeType or "RENAME" in edgeType or "FILE" in edgeType:
                        if dstId1 not in id_nodetype_map:
                            id_nodetype_map[dstId1] = 'FILE_OBJECT_FILE'

                        if dstId1 not in id_nodename_map or id_nodename_map[dstId1] is None:
                            if len(dstId1name) > 0 and dstId1name[0] != 'null':
                                id_nodename_map[dstId1] = dstId1name[0]

                    dstType1 = id_nodetype_map[dstId1]

                    if dstId1 in id_nodename_map and id_nodename_map[dstId1] is not None and id_nodename_map[
                        dstId1] != '<unknown>' and id_nodename_map[dstId1] != 'null':
                        this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(
                            dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        edges.append(this_edge1)

                dstId2 = pattern_dst2.findall(line)
                dstId2name = pattern_dst2_path.findall(line)
                # id_nodename_map[dstId2[0]] != None 为判断Netflow对象
                if len(dstId2) > 0 and dstId2[0] != 'null':
                    dstId2 = dstId2[0]
                    if dstId2 not in id_nodetype_map:
                        # 若当前对象没有声明，则根据事件类型判断其类型
                        id_nodetype_map[dstId2] = 'FILE_OBJECT_FILE'

                    if dstId2 not in id_nodename_map or id_nodename_map[dstId2] is None:
                        if len(dstId2name) > 0 and dstId2name[0] != 'null':
                            id_nodename_map[dstId2] = dstId2name[0]

                    dstType2 = id_nodetype_map[dstId2]

                    if dstId2 in id_nodename_map and id_nodename_map[dstId2] is not None and id_nodename_map[
                        dstId2] != '<unknown>' and id_nodename_map[dstId2] != 'null':
                        this_edge2 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId2) + '\t' + str(
                            dstType2) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        edges.append(this_edge2)

        with open(out_file, 'w', encoding='utf-8') as fw:
            fw.write("".join(edges))

    # 过滤掉没有声明的节点
    filtered_id_set = {k for k, v in id_nodename_map.items() if v is None or "<unknown>" == v or "null" == v}
    print(len(filtered_id_set))
    # 使用集合进行高效的成员测试，合并两个字典的更新到一个循环中
    new_id_nodetype_map = {}
    new_id_nodename_map = {}
    for k in id_nodetype_map:
        if k not in filtered_id_set:
            new_id_nodetype_map[k] = id_nodetype_map[k]
            new_id_nodename_map[k] = id_nodename_map.get(k, None)
    # 更新原字典
    id_nodetype_map = new_id_nodetype_map
    id_nodename_map = new_id_nodename_map

    print(len(id_nodetype_map))
    print(len(id_nodename_map))

    if len(id_nodename_map) != 0:
        fw = open(dir + 'names.json', 'w', encoding='utf-8')
        json.dump(id_nodename_map, fw)
    if len(id_nodetype_map) != 0:
        fw = open(dir + 'types.json', 'w', encoding='utf-8')
        json.dump(id_nodetype_map, fw)
    if len(edge_type_dict) != 0:
        fw = open(dir + 'edge_type_map.json', 'w', encoding='utf-8')
        json.dump(edge_type_dict, fw)
    if len(node_type_dict) != 0:
        fw = open(dir + 'node_type_map.json', 'w', encoding='utf-8')
        json.dump(node_type_dict, fw)


def networkx2pyg(g):
    # 创建节点特征矩阵
    x = torch.tensor([a[1] for a in g.nodes(data="node_type")], dtype=torch.float32).view(-1, 1)
    edge_attr = torch.stack([a[2] for a in g.edges(data="edge_type")], dim=0).to(torch.float32)

    # 获取图G邻接矩阵的稀疏表示
    adj = nx.to_scipy_sparse_matrix(g, weight=None, format='coo')

    # 获取非零元素行索引
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    # 获取非零元素列索引
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)

    # 将行和列进行拼接，shape变为[2, num_edges], 包含两个列表，第一个是row, 第二个是col
    edge_index = torch.stack([row, col], dim=0).to(torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, center_node_idx=0)

def preprocess_dataset_trace():
    pattern_file_name = re.compile(r'"path":"(.*?)"')
    pattern_process_name = re.compile(r'map":{"name":"(.*?)"')
    pattern_netflow_object_name = re.compile(r'remoteAddress":"(.*?)"')

    id_nodetype_map = {}
    id_nodename_map = {}
    edge_type_dict = {}
    node_type_dict = {}

    dir = '../dataset/darpa_trace/'
    raw_dir = dir + 'raws/'
    tuples_dir = dir + 'tuples/'
    if os.path.exists(dir + 'init_names.json') and os.path.exists(dir + 'init_types.json'):
        print("init_names.json and init_types.json already exist! loading now...")
        id_nodename_map = json.load(open(dir + 'init_names.json', 'r'))
        id_nodetype_map = json.load(open(dir + 'init_types.json', 'r'))
    else:
        for file in os.listdir(raw_dir):
            if "tar.gz" in file or ".rar" in file: continue

            # ptint the id_nodename_map size
            print("size of id_nodename_map:" + str(len(id_nodename_map)))

            print('reading {} ...'.format(file))

            f = open(raw_dir + file, 'r', encoding='utf-8')
            flines = f.readlines()
            for line in tqdm(flines):

                if any(skip_text in line for skip_text in [
                    'com.bbn.tc.schema.avro.cdm18.Event', 'com.bbn.tc.schema.avro.cdm18.Host',
                    'com.bbn.tc.schema.avro.cdm18.TimeMarker', 'com.bbn.tc.schema.avro.cdm18.StartMarker',
                    'com.bbn.tc.schema.avro.cdm18.UnitDependency', 'com.bbn.tc.schema.avro.cdm18.EndMarker']):
                    continue

                if len(pattern_uuid.findall(line)) == 0: print(line)
                uuid = pattern_uuid.findall(line)[0]

                subject_type = pattern_type.findall(line)
                if len(subject_type) < 1:
                    if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                        subject_type = 'MemoryObject'
                    if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                        subject_type = 'NetFlowObject'
                    if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                        subject_type = 'UnnamedPipeObject'
                else:
                    subject_type = subject_type[0]

                if subject_type not in node_type_dict:
                    node_type_dict[subject_type] = len(node_type_dict)

                if uuid == '00000000-0000-0000-0000-000000000000' or subject_type in ['SUBJECT_UNIT']:
                    id_nodename_map[uuid] = None
                    continue
                id_nodetype_map[uuid] = subject_type
                id_nodename_map[uuid] = None
                if 'FILE' in subject_type and len(pattern_file_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_file_name.findall(line)[0]
                elif subject_type == 'SUBJECT_PROCESS' and len(pattern_process_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_process_name.findall(line)[0]
                elif subject_type == 'NetFlowObject' and len(pattern_netflow_object_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_netflow_object_name.findall(line)[0]
        # 写入文件
        json.dump(id_nodename_map, open(dir + 'init_names.json', 'w', encoding='utf-8'))
        json.dump(id_nodetype_map, open(dir + 'init_types.json', 'w', encoding='utf-8'))
        json.dump(node_type_dict, open(dir + 'node_type_map.json', 'w', encoding='utf-8'))

    for file in os.listdir(raw_dir):

        in_file = raw_dir + file
        out_file = tuples_dir + file + '.txt'
        # if os.path.exists(out_file):
        #     print('file {} already exists!'.format(file))
        # exit(0)
        # continue
        if "json" not in file: continue

        with open(in_file, 'r', encoding='utf-8') as f:
            f_lines = f.readlines()
        print('processing {} ...'.format(file))

        edges = []
        for line in tqdm(f_lines):
            if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                try:
                    edgeType = pattern_type.findall(line)[0]
                    timestamp = pattern_time.findall(line)[0]
                except Exception as e:
                    print('error processing')
                    print(line)
                    continue

                # 只监听需要的事件类型
                def listen_edge_type(edgeType):
                    for et in listened_event_types:
                        if et.lower() in edgeType.lower():
                            return True
                    return False

                # 跳过非监听事件
                if not listen_edge_type(edgeType):
                    continue

                srcId = pattern_src.findall(line)

                if len(srcId) == 0: continue
                srcId = srcId[0]
                if not srcId in id_nodetype_map:
                    continue
                # 若主体对象没有初始化，则跳过事件
                if srcId not in id_nodename_map:
                    continue

                srcType = id_nodetype_map[srcId]
                srcIdname = id_nodename_map[srcId]

                if edgeType not in edge_type_dict:
                    edge_type_dict[edgeType] = len(edge_type_dict)

                dstId1 = pattern_dst1.findall(line)
                dstId1name = pattern_dst1_path.findall(line)
                if len(dstId1) > 0 and dstId1[0] != 'null':

                    dstId1 = dstId1[0]

                    if dstId1 in id_nodetype_map:
                        dstType1 = id_nodetype_map[dstId1]

                        if dstId1 in id_nodename_map and id_nodename_map[dstId1] is not None and id_nodename_map[
                            dstId1] != '<unknown>' and id_nodename_map[dstId1] != 'null':
                            this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(
                                dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                            edges.append(this_edge1)

                dstId2 = pattern_dst2.findall(line)
                # id_nodename_map[dstId2[0]] != None 为判断Netflow对象
                if len(dstId2) > 0 and dstId2[0] != 'null':
                    dstId2 = dstId2[0]
                    if not dstId2 in id_nodetype_map.keys():
                        continue

                    dstType2 = id_nodetype_map[dstId2]
                    if dstId2 in id_nodename_map and id_nodename_map[dstId2] is not None and id_nodename_map[
                        dstId2] != '<unknown>' and id_nodename_map[dstId2] != 'null':
                        this_edge2 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId2) + '\t' + str(
                            dstType2) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        edges.append(this_edge2)

        with open(out_file, 'w', encoding='utf-8') as fw:
            fw.write("".join(edges))

    # 过滤掉没有声明的节点
    filtered_id_set = {k for k, v in id_nodename_map.items() if v is None or "<unknown>" == v or "null" == v}
    print(len(filtered_id_set))
    # 使用集合进行高效的成员测试，合并两个字典的更新到一个循环中
    new_id_nodetype_map = {}
    new_id_nodename_map = {}
    for k in id_nodetype_map:
        if k not in filtered_id_set:
            new_id_nodetype_map[k] = id_nodetype_map[k]
            new_id_nodename_map[k] = id_nodename_map.get(k, None)
    # 更新原字典
    id_nodetype_map = new_id_nodetype_map
    id_nodename_map = new_id_nodename_map

    print(len(id_nodetype_map))
    print(len(id_nodename_map))

    if len(id_nodename_map) != 0:
        fw = open(dir + 'names.json', 'w', encoding='utf-8')
        json.dump(id_nodename_map, fw)
    if len(id_nodetype_map) != 0:
        fw = open(dir + 'types.json', 'w', encoding='utf-8')
        json.dump(id_nodetype_map, fw)
    if len(edge_type_dict) != 0:
        fw = open(dir + 'edge_type_map.json', 'w', encoding='utf-8')
        json.dump(edge_type_dict, fw)


def preprocess_dataset_theia():
    pattern_file_name = re.compile(r'\"filename\":\"(.*?)\"')
    pattern_process_name = re.compile(r'\"path\":\"(.*?)\"')
    pattern_netflow_object_name = re.compile(r'remoteAddress\":\"(.*?)\"')

    id_nodetype_map = {}
    id_nodename_map = {}
    edge_type_dict = {}
    node_type_dict = {}

    dir = '../dataset/darpa_theia/'
    raw_dir = dir + 'raws/'
    tuples_dir = dir + 'tuples/'
    if os.path.exists(dir + 'init_names.json') and os.path.exists(dir + 'init_types.json'):
        print("init_names.json and init_types.json already exist! loading now...")
        id_nodename_map = json.load(open(dir + 'init_names.json', 'r'))
        id_nodetype_map = json.load(open(dir + 'init_types.json', 'r'))
    else:
        for file in os.listdir(raw_dir):
            print('reading {} ...'.format(file))

            f = open(raw_dir + file, 'r', encoding='utf-8')
            flines = f.readlines()
            for line in tqdm(flines):

                if any(skip_text in line for skip_text in [
                    'com.bbn.tc.schema.avro.cdm18.Event', 'com.bbn.tc.schema.avro.cdm18.Host',
                    'com.bbn.tc.schema.avro.cdm18.TimeMarker', 'com.bbn.tc.schema.avro.cdm18.StartMarker',
                    'com.bbn.tc.schema.avro.cdm18.UnitDependency', 'com.bbn.tc.schema.avro.cdm18.EndMarker']):
                    continue

                if len(pattern_uuid.findall(line)) == 0: print(line)
                uuid = pattern_uuid.findall(line)[0]

                subject_type = pattern_type.findall(line)
                if len(subject_type) < 1:
                    if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                        subject_type = 'MemoryObject'
                    if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                        subject_type = 'NetFlowObject'
                    if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                        subject_type = 'UnnamedPipeObject'
                else:
                    subject_type = subject_type[0]

                if subject_type not in node_type_dict:
                    node_type_dict[subject_type] = len(node_type_dict)

                if uuid == '00000000-0000-0000-0000-000000000000' or subject_type in ['SUBJECT_UNIT']:
                    id_nodename_map[uuid] = None
                    continue
                id_nodetype_map[uuid] = subject_type
                id_nodename_map[uuid] = None
                if 'FILE' in subject_type and len(pattern_file_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_file_name.findall(line)[0]
                elif subject_type == 'SUBJECT_PROCESS' and len(pattern_process_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_process_name.findall(line)[0]
                elif subject_type == 'NetFlowObject' and len(pattern_netflow_object_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_netflow_object_name.findall(line)[0]
        # 写入文件
        json.dump(id_nodename_map, open(dir + 'init_names.json', 'w', encoding='utf-8'))
        json.dump(id_nodetype_map, open(dir + 'init_types.json', 'w', encoding='utf-8'))
        json.dump(node_type_dict, open(dir + 'node_type_map.json', 'w', encoding='utf-8'))

    for file in os.listdir(raw_dir):

        in_file = raw_dir + file
        out_file = tuples_dir + file + '.txt'
        # if os.path.exists(out_file):
        #     print('file {} already exists!'.format(file))
        # exit(0)
        # continue

        with open(in_file, 'r', encoding='utf-8') as f:
            f_lines = f.readlines()
        print('processing {} ...'.format(file))

        edges = []
        for line in tqdm(f_lines):
            if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                edgeType = pattern_type.findall(line)[0]
                timestamp = pattern_time.findall(line)[0]
                srcId = pattern_src.findall(line)

                if len(srcId) == 0: continue
                srcId = srcId[0]

                # 若主体对象不存在，则跳过该事件
                if srcId not in id_nodetype_map:
                    id_nodetype_map[srcId] = 'SUBJECT_PROCESS'
                # 若主体对象没有初始化，则跳过事件
                if srcId not in id_nodename_map:
                    continue

                srcType = id_nodetype_map[srcId]
                srcIdname = id_nodename_map[srcId]

                # 只监听需要的事件类型
                def listen_edge_type(edgeType):
                    for et in listened_event_types:
                        if et.lower() in edgeType.lower():
                            return True
                    return False

                # 跳过非监听事件
                if not listen_edge_type(edgeType):
                    continue

                if edgeType not in edge_type_dict:
                    edge_type_dict[edgeType] = len(edge_type_dict)

                dstId1 = pattern_dst1.findall(line)
                dstId1name = pattern_dst1_path.findall(line)
                # id_nodename_map[dstId1[0]] != None 为判断Netflow对象
                if len(dstId1) > 0 and dstId1[0] != 'null':

                    dstId1 = dstId1[0]
                    # 若当前对象没有声明，则根据事件类型判断其类型
                    if "CLONE" in edgeType:
                        assert srcId in id_nodename_map
                        assert dstId1 in id_nodename_map

                        id_nodetype_map[dstId1] = 'SUBJECT_PROCESS'
                        if dstId1 not in id_nodename_map:
                            id_nodename_map[dstId1] = srcIdname

                    elif "EXECUTE" in edgeType:
                        assert srcId in id_nodename_map
                        assert dstId1 in id_nodename_map
                        # pattern_cmd_path = re.compile(r'map":{"cmdLine":"(.*?)"')
                        # cmd_path = pattern_cmd_path.findall(line)

                        # id_nodetype_map[dstId1] = 'SUBJECT_PROCESS'
                        # if len(cmd_path) > 0 and cmd_path[0] != 'null':
                        #     pname = cmd_path[0].split(' ')[0]
                        #     id_nodename_map[dstId1] = pname
                        #     print('cmd_path:',pname)
                    elif "MODIFY_PROCESS" in edgeType:
                        id_nodetype_map[dstId1] = 'SUBJECT_PROCESS'
                    elif "READ" in edgeType or "WRITE" in edgeType or "LINK" in edgeType or "RENAME" in edgeType or "FILE" in edgeType:
                        if dstId1 not in id_nodetype_map:
                            id_nodetype_map[dstId1] = 'FILE_OBJECT_BLOCK'

                        if dstId1 not in id_nodename_map or id_nodename_map[dstId1] is None:
                            if len(dstId1name) > 0 and dstId1name[0] != 'null':
                                id_nodename_map[dstId1] = dstId1name[0]

                    dstType1 = id_nodetype_map[dstId1]

                    if dstId1 in id_nodename_map and id_nodename_map[dstId1] is not None and id_nodename_map[
                        dstId1] != '<unknown>' and id_nodename_map[dstId1] != 'null':
                        this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(
                            dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        edges.append(this_edge1)

                dstId2 = pattern_dst2.findall(line)
                dstId2name = pattern_dst2_path.findall(line)
                # id_nodename_map[dstId2[0]] != None 为判断Netflow对象
                if len(dstId2) > 0 and dstId2[0] != 'null':
                    dstId2 = dstId2[0]
                    if dstId2 not in id_nodetype_map:
                        # 若当前对象没有声明，则根据事件类型判断其类型
                        id_nodetype_map[dstId2] = 'FILE_OBJECT_BLOCK'

                    if dstId2 not in id_nodename_map or id_nodename_map[dstId2] is None:
                        if len(dstId2name) > 0 and dstId2name[0] != 'null':
                            id_nodename_map[dstId2] = dstId2name[0]

                    dstType2 = id_nodetype_map[dstId2]

                    if dstId2 in id_nodename_map and id_nodename_map[dstId2] is not None and id_nodename_map[
                        dstId2] != '<unknown>' and id_nodename_map[dstId2] != 'null':
                        this_edge2 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId2) + '\t' + str(
                            dstType2) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        edges.append(this_edge2)

        with open(out_file, 'w', encoding='utf-8') as fw:
            fw.write("".join(edges))

    # 过滤掉没有声明的节点
    filtered_id_set = {k for k, v in id_nodename_map.items() if v is None or "<unknown>" == v or "null" == v}
    print(len(filtered_id_set))
    # 使用集合进行高效的成员测试，合并两个字典的更新到一个循环中
    new_id_nodetype_map = {}
    new_id_nodename_map = {}
    for k in id_nodetype_map:
        if k not in filtered_id_set:
            new_id_nodetype_map[k] = id_nodetype_map[k]
            new_id_nodename_map[k] = id_nodename_map.get(k, None)
    # 更新原字典
    id_nodetype_map = new_id_nodetype_map
    id_nodename_map = new_id_nodename_map

    print(len(id_nodetype_map))
    print(len(id_nodename_map))

    if len(id_nodename_map) != 0:
        fw = open(dir + 'names.json', 'w', encoding='utf-8')
        json.dump(id_nodename_map, fw)
    if len(id_nodetype_map) != 0:
        fw = open(dir + 'types.json', 'w', encoding='utf-8')
        json.dump(id_nodetype_map, fw)
    if len(edge_type_dict) != 0:
        fw = open(dir + 'edge_type_map.json', 'w', encoding='utf-8')
        json.dump(edge_type_dict, fw)

def preprocess_dataset_fivedirections():
    id_nodetype_map = {}
    id_nodename_map = {}
    edge_type_dict = {}
    node_type_dict = {}

    pattern_file_name = re.compile(r'map\":\{\"path\":\"(.*?)\"')
    pattern_process_name = re.compile(r'\"string\":\"(.*?)\"')
    pattern_netflow_object_name = re.compile(r'remoteAddress\":\"(.*?)\"')
    pattern_type = re.compile(r'\"type\":\"(.*?)\",')
    pattern_src = re.compile(r'subject\":{\"UUID\":\"(.*?)\"}')
    pattern_dst1 = re.compile(r'predicateObject\":{\"UUID\":\"(.*?)\"}')
    pattern_dst2 = re.compile(r'predicateObject2\":{\"UUID\":\"(.*?)\"}')

    dir = '../dataset/darpa_fivedirections/'
    raw_dir = dir + 'raws/'
    tuples_dir = dir + 'tuples/'
    if os.path.exists(dir + 'init_names.json') and os.path.exists(dir + 'init_types.json'):
        print("init_names.json and init_types.json already exist! loading now...")
        id_nodename_map = json.load(open(dir + 'init_names.json', 'r'))
        id_nodetype_map = json.load(open(dir + 'init_types.json', 'r'))
    else:
        for file in os.listdir(raw_dir):
            print('reading {} ...'.format(file))

            try:
                f = open(raw_dir + file, 'r', encoding='utf-8')
                flines = f.readlines()
            except Exception as e:
                f = open(raw_dir + file, 'r', encoding='gbk')
                flines = f.readlines()

            for line in tqdm(flines):

                if any(skip_text in line for skip_text in [
                    'Event\":{\"', 'Host\":{\"', 'TimeMarker\":{\"', 'StartMarker\":{\"', 'UnitDependency\":{\"',
                    'EndMarker\":{\"']):
                    continue

                if len(pattern_uuid.findall(line)) == 0:
                    print("error line")
                    print(line)
                    continue
                uuid = pattern_uuid.findall(line)[0]

                subject_type = pattern_type.findall(line)
                if len(subject_type) < 1:
                    if 'MemoryObject' in line:
                        subject_type = 'MemoryObject'
                    if 'NetFlowObject' in line:
                        subject_type = 'NetFlowObject'
                    if 'UnnamedPipeObject' in line:
                        subject_type = 'UnnamedPipeObject'
                else:
                    subject_type = subject_type[0]

                if len(subject_type) == 0:
                    continue

                if subject_type not in node_type_dict:
                    node_type_dict[subject_type] = len(node_type_dict)

                if uuid == '00000000-0000-0000-0000-000000000000' or subject_type in ['SUBJECT_UNIT']:
                    id_nodename_map[uuid] = None
                    continue
                id_nodetype_map[uuid] = subject_type
                id_nodename_map[uuid] = None
                if 'FILE' in subject_type and len(pattern_file_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_file_name.findall(line)[0]
                elif subject_type == 'SUBJECT_PROCESS' and len(pattern_process_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_process_name.findall(line)[0]
                elif subject_type == 'NetFlowObject' and len(pattern_netflow_object_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_netflow_object_name.findall(line)[0]
        # 写入文件
        json.dump(id_nodename_map, open(dir + 'init_names.json', 'w', encoding='utf-8'))
        json.dump(id_nodetype_map, open(dir + 'init_types.json', 'w', encoding='utf-8'))
        json.dump(node_type_dict, open(dir + 'node_type_map.json', 'w', encoding='utf-8'))

    for file in os.listdir(raw_dir):

        if "-3" in file or "-2" in file:
            pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
            pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
            pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')

        in_file = raw_dir + file
        out_file = tuples_dir + file + '.txt'
        if os.path.exists(out_file):
            print('file {} already exists!'.format(file))
            # exit(0)
            # continue

        with open(in_file, 'r', encoding='utf-8') as f:
            f_lines = f.readlines()
        print('processing {} ...'.format(file))

        edges = []
        for line in tqdm(f_lines):
            if r'Event":{"' in line:
                edgeType = pattern_type.findall(line)[0]
                timestamp = pattern_time.findall(line)[0]
                srcId = pattern_src.findall(line)

                if len(srcId) == 0: continue
                srcId = srcId[0]

                # 若主体对象不存在，则跳过该事件
                if srcId not in id_nodename_map or id_nodename_map[srcId] is None: continue

                if srcId not in id_nodetype_map:
                    id_nodetype_map[srcId] = 'SUBJECT_PROCESS'

                srcType = id_nodetype_map[srcId]
                srcIdname = id_nodename_map[srcId]

                # 只监听需要的事件类型
                def listen_edge_type(edgeType):
                    for et in listened_event_types:
                        if et.lower() in edgeType.lower():
                            return True
                    return False

                if not listen_edge_type(edgeType):
                    continue

                if edgeType not in edge_type_dict:
                    edge_type_dict[edgeType] = len(edge_type_dict)

                dstId1 = pattern_dst1.findall(line)
                dstId1name = pattern_dst1_path.findall(line)
                # id_nodename_map[dstId1[0]] != None 为判断Netflow对象
                if len(dstId1) > 0 and dstId1[0] != 'null':

                    dstId1 = dstId1[0]
                    # 若当前对象没有声明，则根据事件类型判断其类型
                    if "FORK" in edgeType:
                        id_nodetype_map[dstId1] = 'SUBJECT_PROCESS'
                        id_nodename_map[dstId1] = srcIdname
                    elif "EXECUTE" in edgeType:
                        assert srcId in id_nodename_map
                        id_nodetype_map[dstId1] = 'FILE_OBJECT_FILE'

                        if len(dstId1name) > 0 and dstId1name[0] != 'null':
                            id_nodename_map[dstId1] = dstId1name[0]
                            id_nodename_map[srcId] = dstId1name[0]
                    elif "MODIFY_PROCESS" in edgeType:
                        id_nodetype_map[dstId1] = 'SUBJECT_PROCESS'
                    elif "READ" in edgeType or "WRITE" in edgeType or "LINK" in edgeType or "RENAME" in edgeType or "FILE" in edgeType:
                        if dstId1 not in id_nodetype_map:
                            id_nodetype_map[dstId1] = 'FILE_OBJECT_FILE'

                        if dstId1 not in id_nodename_map or id_nodename_map[dstId1] is None:
                            if len(dstId1name) > 0 and dstId1name[0] != 'null':
                                id_nodename_map[dstId1] = dstId1name[0]

                    dstType1 = id_nodetype_map[dstId1]

                    if dstId1 in id_nodename_map and id_nodename_map[dstId1] is not None and id_nodename_map[
                        dstId1] != '<unknown>' and id_nodename_map[dstId1] != 'null':
                        this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(
                            dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        edges.append(this_edge1)

                dstId2 = pattern_dst2.findall(line)
                dstId2name = pattern_dst2_path.findall(line)
                # id_nodename_map[dstId2[0]] != None 为判断Netflow对象
                if len(dstId2) > 0 and dstId2[0] != 'null':
                    dstId2 = dstId2[0]
                    if dstId2 not in id_nodetype_map:
                        # 若当前对象没有声明，则根据事件类型判断其类型
                        id_nodetype_map[dstId2] = 'FILE_OBJECT_FILE'

                    if dstId2 not in id_nodename_map or id_nodename_map[dstId2] is None:
                        if len(dstId2name) > 0 and dstId2name[0] != 'null':
                            id_nodename_map[dstId2] = dstId2name[0]

                    dstType2 = id_nodetype_map[dstId2]

                    if dstId2 in id_nodename_map and id_nodename_map[dstId2] is not None and id_nodename_map[
                        dstId2] != '<unknown>' and id_nodename_map[dstId2] != 'null':
                        this_edge2 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId2) + '\t' + str(
                            dstType2) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        edges.append(this_edge2)

        with open(out_file, 'w', encoding='utf-8') as fw:
            fw.write("".join(edges))

    # 过滤掉没有声明的节点
    filtered_id_set = {k for k, v in id_nodename_map.items() if v is None or "<unknown>" == v or "null" == v}
    print(len(filtered_id_set))
    # 使用集合进行高效的成员测试，合并两个字典的更新到一个循环中
    new_id_nodetype_map = {}
    new_id_nodename_map = {}
    for k in id_nodetype_map:
        if k not in filtered_id_set:
            new_id_nodetype_map[k] = id_nodetype_map[k]
            new_id_nodename_map[k] = id_nodename_map.get(k, None)
    # 更新原字典
    id_nodetype_map = new_id_nodetype_map
    id_nodename_map = new_id_nodename_map

    print(len(id_nodetype_map))
    print(len(id_nodename_map))

    if len(id_nodename_map) != 0:
        fw = open(dir + 'names.json', 'w', encoding='utf-8')
        json.dump(id_nodename_map, fw)
    if len(id_nodetype_map) != 0:
        fw = open(dir + 'types.json', 'w', encoding='utf-8')
        json.dump(id_nodetype_map, fw)
    if len(edge_type_dict) != 0:
        fw = open(dir + 'edge_type_map.json', 'w', encoding='utf-8')
        json.dump(edge_type_dict, fw)
    if len(node_type_dict) != 0:
        fw = open(dir + 'node_type_map.json', 'w', encoding='utf-8')
        json.dump(node_type_dict, fw)


def load_init_maps(path):
    id_nodetype_map, id_nodename_map = {}, {}

    file_id2name = json.load(open(f'{path}/id_maps/file_id2name.json', 'r', encoding='utf-8'))
    netflow_id2name = json.load(open(f'{path}/id_maps/netflow_id2name.json', 'r', encoding='utf-8'))
    sbj_id2name = json.load(open(f'{path}/id_maps/sbj_id2name.json', 'r', encoding='utf-8'))

    # merge file_id2name, netflow_id2name, sbj_id2name into id_nodename_map
    id_nodename_map.update(file_id2name)
    id_nodename_map.update(netflow_id2name)
    id_nodename_map.update(sbj_id2name)

    id_nodetype_map.update({k: 'FILE_OBJECT_FILE' for k in file_id2name.keys()})
    id_nodetype_map.update({k: 'NetFlowObject' for k in netflow_id2name.keys()})
    id_nodetype_map.update({k: 'SUBJECT_PROCESS' for k in sbj_id2name.keys()})

    return id_nodetype_map, id_nodename_map
def preprocess_e5_cadets():

    edge2obj_type = {
        'EVENT_READ': 'FILE_OBJECT_FILE',
        'EVENT_WRITE': 'FILE_OBJECT_FILE',
        'EVENT_MODIFY_FILE_ATTRIBUTES': 'FILE_OBJECT_FILE',
        'EVENT_EXECUTE': 'FILE_OBJECT_FILE',
        'EVENT_CLONE': 'SUBJECT_PROCESS',
        'EVENT_RECVFROM': 'NetFlowObject',
        'EVENT_RECVMSG': 'NetFlowObject',
        'EVENT_SENDMSG': 'NetFlowObject',
        'EVENT_SENDTO': 'NetFlowObject',
    }
    include_edge_type = edge2obj_type.keys()
    pattern_file_name = re.compile(r'com.bbn.tc.schema.avro.cdm20.FileObject":{"uuid":"(.*?)')
    pattern_process = re.compile(r'com.bbn.tc.schema.avro.cdm20.Subject":{"uuid":"(.*?)')
    pattern_netflow = re.compile(
        r'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":{"string":"(.*?)"},"localPort":{"int":(.*?)},"remoteAddress":{"string":"(.*?)"},"remotePort":{"int":(.*?)}')

    fail_count = 0
    datalist = []
    edge_type = set()
    total_event_count = 0
    reverse = ["EVENT_READ", "EVENT_RECVFROM", "EVENT_RECVMSG"]

    dir = './dataset/darpa_cadets_e5/'
    raw_dir = dir + 'raws/'
    output_dir = dir + 'tuples/'

    id_nodetype_map, id_nodename_map = load_init_maps(dir)

    # order file lists
    json_files = os.listdir(raw_dir)
    json_files.sort()
    for file in tqdm(json_files):
        print("processing file: ", file)
        with open(raw_dir + file, "r") as f:
            lines = f.readlines()
        for line in tqdm(lines):

            if "schema.avro.cdm20.Event" in line:
                relation_type = re.findall('"type":"(.*?)"', line)[0]
                if relation_type not in include_edge_type:
                    continue
                try:
                    pattern = '"subject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"},(.*?)"exec":"(.*?)",'
                    match_ans = re.findall(pattern, line)
                    subject_uuid = match_ans[0][0]
                    if subject_uuid not in id_nodetype_map:
                        # id_nodename_map[subject_uuid] = match_ans[0][-1]
                        # id_nodetype_map[subject_uuid] = 'SUBJECT_PROCESS'
                        continue
                except:
                    fail_count += 1

                try:
                    predicateObject_uuid = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"},', line)[0]
                    object_path = re.findall('"predicateObjectPath":{"string":"(.*?)"}', line)
                    if predicateObject_uuid not in id_nodetype_map:
                        # id_nodename_map[predicateObject_uuid] = object_path[0] if len(object_path) != 0 else 'null'
                        # id_nodetype_map[predicateObject_uuid] = edge2obj_type.get(relation_type)
                        continue

                except:
                    fail_count += 1

                total_event_count += 1
                if len(subject_uuid) > 0 and len(predicateObject_uuid) > 0:
                    # if subject_uuid[0] in id_nodetype_map and (predicateObject_uuid[0] in id_nodetype_map or predicateObject_uuid[0] in id_nodetype_map):
                    time_rec = re.findall('"timestampNanos":(.*?),', line)[0]
                    time_rec = int(time_rec)
                    edge_type.add(relation_type)
                    if relation_type in reverse:
                        datalist.append([predicateObject_uuid, id_nodetype_map[predicateObject_uuid], relation_type,
                                         subject_uuid, id_nodetype_map[subject_uuid], time_rec])
                    else:
                        datalist.append([subject_uuid, id_nodetype_map[subject_uuid], relation_type,
                                         predicateObject_uuid,id_nodetype_map[predicateObject_uuid], time_rec])
                    if len(datalist) % 500000 == 0:
                        print(f"parsing {len(datalist)} edges ...")

            # elif "schema.avro.cdm20.Subject" in line:
            #     uuid = pattern_process.findall(line)
            #     if len(uuid) == 0:  continue
            #     uuid = uuid[0]
            #     if uuid in id_nodetype_map: continue
            #     id_nodetype_map[uuid] = 'SUBJECT_PROCESS'
            # elif "schema.avro.cdm20.FileObject" in line:
            #     uuid = pattern_file_name.findall(line)
            #     if len(uuid) == 0:  continue
            #     uuid = uuid[0]
            #     if uuid in id_nodetype_map:
            #         continue
            #     id_nodetype_map[uuid] = 'FILE_OBJECT_FILE'
            # elif "schema.avro.cdm20.NetFlowObject" in line:
            #     res = pattern_netflow.search(line)
            #     if not res: continue
            #     res = res[0]
            #     uuid = res[0]
            #     srcaddr = res[2]
            #     srcport = res[3]
            #     dstaddr = res[4]
            #     dstport = res[5]
            #     name = f"{srcaddr}:{srcport},{dstaddr}:{dstport}"
            #     if uuid in id_nodetype_map:
            #         if id_nodename_map[uuid] is None:
            #             id_nodename_map[uuid] = name
            #         continue
            #     id_nodename_map[uuid] = name
            #     id_nodetype_map[uuid] = 'NetFlowObject'
            # else:
            #     pass

        # output all edges
        if len(datalist) > 0:
            print(f"writing {len(datalist)} edges ...")
            with open(output_dir + file, 'w', encoding='utf-8') as f:
                for edge in datalist:
                    f.write('\t'.join([edge[0], edge[1], edge[2], edge[3], edge[4], str(edge[5])]) + '\n')
            datalist = []
    # output initialization files
    json.dump(id_nodetype_map, open(dir + 'node_types.json', 'w', encoding='utf-8'))
    json.dump(id_nodename_map, open(dir + 'node_names.json', 'w', encoding='utf-8'))

def preprocess_e5_theia():

    edge2obj_type = {
        'EVENT_READ': 'FILE_OBJECT_FILE',
        'EVENT_WRITE': 'FILE_OBJECT_FILE',
        'EVENT_MODIFY_FILE_ATTRIBUTES': 'FILE_OBJECT_FILE',
        'EVENT_EXECUTE': 'FILE_OBJECT_FILE',
        'EVENT_CLONE': 'SUBJECT_PROCESS',
        'EVENT_RECVFROM': 'NetFlowObject',
        'EVENT_RECVMSG': 'NetFlowObject',
        'EVENT_SENDMSG': 'NetFlowObject',
        'EVENT_SENDTO': 'NetFlowObject',
    }
    include_edge_type = edge2obj_type.keys()
    pattern_file_name = re.compile(r'avro.cdm20.FileObject":{"uuid":"(.*?)",(.*?)"filename":"(.*?)"')
    pattern_process = re.compile(r'avro.cdm20.Subject":{"uuid":"(.*?)",(.*?)"path":"(.*?)"')
    pattern_netflow = re.compile(
        r'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":{"string":"(.*?)"},"localPort":{"int":(.*?)},"remoteAddress":{"string":"(.*?)"},"remotePort":{"int":(.*?)}')

    id_nodetype_map = {}
    id_nodename_map = {}
    fail_count = 0
    datalist = []
    edge_type = set()
    total_event_count = 0
    reverse = ["EVENT_READ", "EVENT_RECVFROM", "EVENT_RECVMSG"]

    dir = './dataset/darpa_theia_e5/'
    raw_dir = dir + 'raws/'
    output_dir = dir + 'tuples/'

    id_nodetype_map, id_nodename_map = load_init_maps(dir)

    # order file lists
    json_files = os.listdir(raw_dir)
    json_files.sort()
    for file in tqdm(json_files):
        print("processing file: ", file)
        # if "bin.30" not in file: continue
        with open(raw_dir + file, "r") as f:
            lines = f.readlines()
        for line in tqdm(lines):

            if "schema.avro.cdm20.Event" in line:
                relation_type = re.findall('"type":"(.*?)"', line)[0]
                if relation_type not in include_edge_type:
                    continue
                try:
                    pattern = '"subject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"'
                    match_ans = re.findall(pattern, line)
                    subject_uuid = match_ans[0]
                    if subject_uuid not in id_nodetype_map:
                        # id_nodetype_map[subject_uuid] = 'SUBJECT_PROCESS'
                        continue

                except:
                    fail_count += 1

                try:
                    predicateObject_uuid = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"},', line)[0]
                    object_path = re.findall('"predicateObjectPath":{"string":"(.*?)"}', line)
                    if predicateObject_uuid not in id_nodetype_map:
                        # id_nodename_map[predicateObject_uuid] = object_path[0] if len(object_path) != 0 else 'null'
                        # id_nodetype_map[predicateObject_uuid] = edge2obj_type.get(relation_type)
                        continue

                except:
                    fail_count += 1

                total_event_count += 1
                if len(subject_uuid) > 0 and len(predicateObject_uuid) > 0:
                    # if subject_uuid[0] in id_nodetype_map and (predicateObject_uuid[0] in id_nodetype_map or predicateObject_uuid[0] in id_nodetype_map):
                    time_rec = re.findall('"timestampNanos":(.*?),', line)[0]
                    time_rec = int(time_rec)
                    edge_type.add(relation_type)
                    if relation_type in reverse:
                        datalist.append([predicateObject_uuid, id_nodetype_map[predicateObject_uuid], relation_type,
                                         subject_uuid, id_nodetype_map[subject_uuid], time_rec])
                    else:
                        datalist.append([subject_uuid, id_nodetype_map[subject_uuid], relation_type,
                                         predicateObject_uuid, id_nodetype_map[predicateObject_uuid], time_rec])
                    if len(datalist) % 500000 == 0:
                        print(f"parsing {len(datalist)} edges ...")

            # elif "schema.avro.cdm20.Subject" in line:
            #     subject_uuid = pattern_process.findall(line)
            #     if len(subject_uuid) == 0:
            #         continue
            #     uuid = subject_uuid[0][0]
            #     if uuid not in id_nodetype_map:
            #         id_nodetype_map[uuid] = 'SUBJECT_PROCESS'
            #     if uuid not in id_nodename_map:
            #         id_nodename_map[uuid] = subject_uuid[0][-1]
            # elif "schema.avro.cdm20.FileObject" in line:
            #     object_uuid = pattern_file_name.findall(line)
            #     if len(object_uuid) == 0:
            #         continue
            #     uuid = object_uuid[0][0]
            #     if uuid not in id_nodetype_map:
            #         id_nodetype_map[uuid] = 'FILE_OBJECT_FILE'
            #     if uuid not in id_nodename_map:
            #         id_nodename_map[uuid] = object_uuid[0][-1]
            # elif "schema.avro.cdm20.NetFlowObject" in line:
            #     res = pattern_netflow.findall(line)[0]
            #     if not res: continue
            #     uuid = res[0]
            #     srcaddr = res[2]
            #     srcport = res[3]
            #     dstaddr = res[4]
            #     dstport = res[5]
            #     name = f"{srcaddr}:{srcport},{dstaddr}:{dstport}"
            #     if id_nodename_map[uuid] is None:
            #         id_nodename_map[uuid] = name
            #         id_nodetype_map[uuid] = 'NetFlowObject'
            else:
                pass

        # output all edges
        if len(datalist) > 0:
            print(f"writing {len(datalist)} edges ...")
            with open(output_dir + file, 'w', encoding='utf-8') as f:
                for edge in datalist:
                    f.write('\t'.join([edge[0], edge[1], edge[2], edge[3], edge[4], str(edge[5])]) + '\n')
            datalist = []
    # output initialization files
    json.dump(id_nodetype_map, open(dir + 'node_types.json', 'w', encoding='utf-8'))
    json.dump(id_nodename_map, open(dir + 'node_names.json', 'w', encoding='utf-8'))

def preprocess_e5_clearscope(dataset):

    edge2obj_type = {
        'EVENT_READ': 'FILE_OBJECT_FILE',
        'EVENT_WRITE': 'FILE_OBJECT_FILE',
        'EVENT_EXECUTE': 'FILE_OBJECT_FILE',
        'EVENT_MODIFY_FILE_ATTRIBUTES': 'FILE_OBJECT_FILE',
        'EVENT_CLONE': 'SUBJECT_PROCESS',
        'EVENT_RECVFROM': 'NetFlowObject',
        'EVENT_RECVMSG': 'NetFlowObject',
        'EVENT_SENDMSG': 'NetFlowObject',
        'EVENT_SENDTO': 'NetFlowObject',
    }
    include_edge_type = edge2obj_type.keys()
    pattern_file_name = re.compile(r'cdm20.FileObject":{"uuid":"(.*?)",(.*?){"map":{"path":"(.*?)"')
    pattern_process = re.compile(r'avro.cdm20.Subject":{"uuid":"(.*?)",(.*?)"cmdLine":{"string":"(.*?)"}')
    pattern_netflow = re.compile(
        r'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":{"string":"(.*?)"},"localPort":{"int":(.*?)},"remoteAddress":{"string":"(.*?)"},"remotePort":{"int":(.*?)}')

    id_nodetype_map = {}
    id_nodename_map = {}
    fail_count = 0
    datalist = []
    edge_type = set()
    total_event_count = 0
    reverse = ["EVENT_READ", "EVENT_RECVFROM", "EVENT_RECVMSG"]

    dir = './dataset/darpa_clearscope1_e5/'
    raw_dir = dir + 'raws/'
    output_dir = dir + 'tuples/'

    id_nodetype_map, id_nodename_map = load_init_maps(dir)

    # order file lists
    json_files = os.listdir(raw_dir)
    json_files.sort()
    for file in tqdm(json_files):
        print("processing file: ", file)
        # if "bin.30" not in file: continue
        with open(raw_dir + file, "r") as f:
            lines = f.readlines()
        for line in tqdm(lines):

            if "schema.avro.cdm20.Event" in line:
                relation_type = re.findall('"type":"(.*?)"', line)[0]
                if relation_type not in include_edge_type:
                    continue
                try:
                    pattern = '"subject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"'
                    match_ans = re.findall(pattern, line)
                    subject_uuid = match_ans[0]
                    if subject_uuid not in id_nodetype_map:
                        # id_nodetype_map[subject_uuid] = 'SUBJECT_PROCESS'
                        continue

                except:
                    fail_count += 1

                try:
                    predicateObject_uuid = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"},', line)[0]
                    object_path = re.findall('"predicateObjectPath":{"string":"(.*?)"}', line)
                    if predicateObject_uuid not in id_nodetype_map:
                        id_nodename_map[predicateObject_uuid] = object_path[0] if len(object_path) != 0 else 'null'
                        id_nodetype_map[predicateObject_uuid] = edge2obj_type.get(relation_type)
                        # continue

                except:
                    fail_count += 1

                total_event_count += 1
                # if len(subject_uuid) > 0 and len(predicateObject_uuid) > 0 and predicateObject_uuid in id_nodetype_map and subject_uuid in id_nodetype_map:
                if len(subject_uuid) > 0 and len(predicateObject_uuid) > 0:
                    # if subject_uuid[0] in id_nodetype_map and (predicateObject_uuid[0] in id_nodetype_map or predicateObject_uuid[0] in id_nodetype_map):
                    time_rec = re.findall('"timestampNanos":(.*?),', line)[0]
                    time_rec = int(time_rec)
                    edge_type.add(relation_type)
                    if relation_type in reverse:
                        datalist.append([predicateObject_uuid, id_nodetype_map[predicateObject_uuid], relation_type,
                                         subject_uuid, id_nodetype_map[subject_uuid], time_rec])
                    else:
                        datalist.append([subject_uuid, id_nodetype_map[subject_uuid], relation_type,
                                         predicateObject_uuid, id_nodetype_map[predicateObject_uuid], time_rec])

            elif "schema.avro.cdm20.Subject" in line:
                subject_uuid = pattern_process.findall(line)
                if len(subject_uuid) == 0:
                    continue
                uuid = subject_uuid[0][0]
                if uuid not in id_nodetype_map:
                    id_nodetype_map[uuid] = 'SUBJECT_PROCESS'
                if uuid not in id_nodename_map:
                    id_nodename_map[uuid] = subject_uuid[0][-1]
            elif "schema.avro.cdm20.FileObject" in line:
                object_uuid = pattern_file_name.findall(line)
                if len(object_uuid) == 0:
                    continue
                uuid = object_uuid[0][0]
                if uuid not in id_nodetype_map:
                    id_nodetype_map[uuid] = 'FILE_OBJECT_FILE'
                if uuid not in id_nodename_map:
                    id_nodename_map[uuid] = object_uuid[0][-1]
            elif "schema.avro.cdm20.NetFlowObject" in line:
                try:
                    res = pattern_netflow.findall(line)[0]
                    if not res: continue
                    uuid = res[0]
                    srcaddr = res[2]
                    srcport = res[3]
                    dstaddr = res[4]
                    dstport = res[5]
                    name = f"{srcaddr}:{srcport},{dstaddr}:{dstport}"
                    if id_nodename_map[uuid] is None:
                        id_nodename_map[uuid] = name
                        id_nodetype_map[uuid] = 'NetFlowObject'
                except:
                    # print(line)
                    pass
            else:
                pass

        # output all edges
        if len(datalist) > 0:
            print(f"writing {len(datalist)} edges ...")
            with open(output_dir + file, 'w', encoding='utf-8') as f:
                for edge in datalist:
                    f.write('\t'.join([edge[0], edge[1], edge[2], edge[3], edge[4], str(edge[5])]) + '\n')
            datalist = []
    # output initialization files
    json.dump(id_nodetype_map, open(dir + 'node_types.json', 'w', encoding='utf-8'))
    json.dump(id_nodename_map, open(dir + 'node_names.json', 'w', encoding='utf-8'))

if __name__ == '__main__':

    preprocess_e5_cadets()
    preprocess_e5_theia()
    preprocess_e5_clearscope("clearscope2_515")
    preprocess_e5_clearscope("clearscope2_517")
    preprocess_e5_clearscope("clearscope1")