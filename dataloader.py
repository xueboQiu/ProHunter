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
