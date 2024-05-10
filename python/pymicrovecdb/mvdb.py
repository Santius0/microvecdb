import microvecdb as mvdb_c
import numpy as np
from enum import Enum, unique

@unique
class DataType(Enum):
    INT8 = 0
    INT16 = 1
    INT32 = 2
    INT64 = 3
    UINT8 = 4
    UINT16 = 5
    UINT32 = 6
    UINT64 = 7
    FLOAT = 8
    DOUBLE = 9

@unique
class IndexType(Enum):
    DISKANN = 0
    FAISS_FLAT = 1
    FLAT = 2
    SPANN = 3
    ANNOY = 4

@unique
class DistanceMetric(Enum):
    L2_DISTANCE = 0
    COSINE = 2

def _create_faiss_flat_named_args(**kwargs):
    return mvdb_c.FaissFlatIndexNamedArgs_init()

def _create_annoy_named_args(**kwargs):
    n_trees = 10
    n_threads = -1
    if 'n_trees' in kwargs:
        n_trees = kwargs['n_trees']
    if 'n_threads' in kwargs:
        n_threads = kwargs['n_threads']
    return mvdb_c.AnnoyIndexNamedArgs_init(n_trees, n_threads)

def _create_spann_named_args(**kwargs):
    return mvdb_c.SPANNIndexNamedArgs_init()

def create_named_args(index_type: IndexType, **kwargs):
    if index_type == IndexType.FAISS_FLAT:
        return _create_faiss_flat_named_args(**kwargs)
    elif index_type == IndexType.ANNOY:
        return _create_annoy_named_args(**kwargs)
    elif index_type == IndexType.SPANN:
        return _create_spann_named_args(**kwargs)
    else:
        raise RuntimeError(f"Unknown IndexType {index_type}")
class MVDB:
    def __init__(self, data_type: DataType = DataType.FLOAT):
        self.built = False
        self.index_type = None
        self.named_args = None
        self.data_type = data_type
        self.mvdb_obj = mvdb_c.MVDB_init(data_type.value)

    def create(self, index_type: IndexType, dims: np.int64, path: str, initial_data_path: str = "", initial_data: np.array = np.array([]), initial_data_size: np.int64 = 0, **kwargs):
        self.index_type = index_type
        self.named_args = create_named_args(index_type, **kwargs)
        mvdb_c.MVDB_create(self.data_type.value, self.mvdb_obj, index_type.value, dims, path, initial_data_path, initial_data, initial_data_size, self.named_args)
        self.built = True

    def open(self, path: str):
        mvdb_c.MVDB_open(self.data_type.value, self.mvdb_obj, path)

    def topk(self, num_queries: np.uint64, query: np.array, k: np.uint64 = 5,
             metric: DistanceMetric = DistanceMetric.L2_DISTANCE, c: np.float32 = 100.0, **kwargs):
        return mvdb_c.MVDB_topk(self.data_type.value, self.mvdb_obj, num_queries, query, k, metric.value, c)

    def num_items(self):
        return mvdb_c.MVDB_num_items(self.data_type.value, self.mvdb_obj)