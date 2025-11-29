import torch
from torch_geometric.data import InMemoryDataset, Batch
import util

class DarpaDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 data_type,
                 empty=False,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: the data directory that contains a raw and processed dir
        :param data_type: either supervised or unsupervised
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        :param transform:
        :param pre_transform:
        :param pre_filter:
        """
        self.root = root
        self.data_type = data_type

        super(DarpaDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if not empty:
            data_list = torch.load(self.processed_paths[0])
            # Add species_id attribute to each node of each Data object
            for i, data in enumerate(data_list):
                num_nodes = data.num_nodes
                # Create a species_id tensor with the same value
                species_id = torch.full((num_nodes,), i, dtype=torch.long)
                # Add species_id to the Data object
                data.species_id = species_id.view(-1, 1)

                if pre_transform:
                    data = pre_transform(data)

            self.data, self.slices = self.collate(data_list)

    @property
    def raw_file_names(self):
        #raise NotImplementedError('Data is assumed to be processed')
        file_name_list = ['0_ged_prohunter.pt']
        return file_name_list

    @property
    def processed_file_names(self):
        return 'train_prohunter.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        raise NotImplementedError('Data is assumed to be processed')