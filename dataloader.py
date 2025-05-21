import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch_geometric.data import Data
import numpy as np
import util
from torch_geometric.data import Batch
import torch.utils.data
from tqdm import tqdm

from batch import BatchFinetune, BatchAE, BatchMultiView
def collate_batch(data_list):
    return BatchMultiView.from_data_list(data_list)
class DataLoaderContrastive(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderContrastive, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_batch,
            **kwargs)
        self.batch_cache = None    # Add a variable to store batch cache

    # def __iter__(self):
    #     if self.batch_cache is None:
    #         # If the cache is empty, it means this is the first iteration and the cache needs to be built
    #         self.batch_cache = list(super(DataLoaderContrastive, self).__iter__())
    #     # Return an iterator to iterate over the pre-cached batches
    #     return iter(self.batch_cache)
    #
    # def refresh_cache(self):
    #     # A method to manually refresh the cache in case the dataset is updated and needs to be reloaded
    #     self.batch_cache = list(super(DataLoaderContrastive, self).__iter__())

class DataLoaderMEGRAPT():
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, preloader_path = None, shuffle=True, is_train=True,node_dim=18,**kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_train = is_train
        self.kwargs = kwargs
        self.MEGRAPT_GEDS = None
        self.graphs = None
        self.coordinates = None
        self.preloader_path = preloader_path
        self.node_dim = node_dim

        if self.preloader_path is None:
            print("Preloading filepath is not provided. Exiting...")
            exit()
        # test dataset size
        if os.path.exists(self.preloader_path):
            datas = torch.load(self.preloader_path)
            self.MEGRAPT_GEDS, self.graphs, self.coordinates = datas["MEGRAPT_GEDS"], datas["graphs"], datas["coordinates"]

            self.graphs = util.compat2MEGRAPT(self.graphs,is_train=True,node_dim = self.node_dim)
            # remove unaligned attrs
            for data in self.graphs:
                remove_attrs = ["center_node_idx","species_id","view1_edge_attr","view1_edge_index","view1_x","view2_edge_attr","view2_edge_index","view2_x"]
                for attr in remove_attrs:
                    if hasattr(data,attr):
                        delattr(data,attr)

        else:
            print("Preloading file not found. constructing...")
            self.init_graphs_geds()
            # self.graphs = [self.convert_data_to_long(data) for data in self.graphs]

            data = dict()
            data["MEGRAPT_GEDS"] = self.MEGRAPT_GEDS
            data["graphs"] = self.graphs
            data["coordinates"] = self.coordinates
            torch.save(data, self.preloader_path)

    def init_graphs_geds(self):

        scale_size = 100
        self.dataset = self.dataset[:scale_size]
        self.coordinates = [(i, j) for i in range(scale_size) for j in range(scale_size)]

        self.graphs = []
        # Extract graphs from view1
        for i, data in enumerate(self.dataset):
            # Extract attributes of view1
            view1_x = data.view1_x
            view1_edge_index = data.view1_edge_index
            view1_edge_attr = data.view1_edge_attr
            # Create Data object for view1
            view1_data = Data(x=view1_x, edge_index=view1_edge_index, edge_attr=view1_edge_attr)
            # Extract attributes of view2
            # view2_x = data.view2_x
            # view2_edge_index = data.view2_edge_index
            # view2_edge_attr = data.view2_edge_attr
            # Create Data object for view2
            # view2_data = Data(x=view2_x, edge_index=view2_edge_index, edge_attr=view2_edge_attr)

            self.graphs.append(data)
            self.graphs.append(view1_data)
            # self.graphs.append(view2_data)

        self.MEGRAPT_GEDS = torch.empty((len(self.graphs), len(self.graphs)), dtype=torch.float32)
        for i in tqdm(range(0, len(self.graphs))):
            for j in range(i, len(self.graphs)):
                ga, gb = self.graphs[i], self.graphs[j]
                norm_ged_value = util.calculate_ged(ga, gb) / (0.5 * (ga.num_nodes + gb.num_nodes))
                self.MEGRAPT_GEDS[i][j] = self.MEGRAPT_GEDS[j][i] = torch.tensor(np.exp(-norm_ged_value),
                                                                                 dtype=torch.float32)

    def get_batch(self):

        geds = []
        graph_pairs = []
        random_coordinates = random.sample(self.coordinates, self.batch_size)

        if self.is_train:
            for i, j in random_coordinates:
                geds.append(self.MEGRAPT_GEDS[i][j])
                graph_pairs.append((self.graphs[i], self.graphs[j]))
        else:
            for i, j in random_coordinates:
                graph_pairs.append((self.graphs[i], self.graphs[j]))

        np_geds = np.array(geds)

        new_data = dict()
        new_data["g1"] = Batch.from_data_list([pair[0] for pair in graph_pairs])
        new_data["g2"] = Batch.from_data_list( [pair[1] for pair in graph_pairs])
        new_data["target"] = torch.from_numpy(np_geds).to(torch.float32)

        return new_data

class DataLoaderFinetune(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderFinetune, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchFinetune.from_data_list(data_list),
            **kwargs)
class DataLoaderAE(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderAE, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchAE.from_data_list(data_list),
            **kwargs)
