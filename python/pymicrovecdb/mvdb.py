import microvecdb as mvdb_c
import numpy as np
from enum import Enum, unique

@unique
class DATA_TYPE(Enum):
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
class DISTANCE_METRIC(Enum):
    L2_DISTANCE = 0
    MAHALANOBIS_DISTANCE = 1

class MVDB:
    def __init__(self, data_type: DATA_TYPE = DATA_TYPE.FLOAT):
        self.built = False
        self.index_type = None
        self.data_type = data_type
        self.mvdb_obj = mvdb_c.MVDB_init(data_type.value)

    def create(self, index_type: IndexType, dims: np.int64, path: str, initial_data_path: str = "", initial_data: np.array = np.array([]), initial_data_size: np.int64 = 0, *args, **kwargs):
        mvdb_c.MVDB_create(self.data_type.value, self.mvdb_obj, index_type.value, dims, path, initial_data_path, initial_data, initial_data_size, None)
        self.index_type = index_type
        self.built = True

    def open(self, path: str):
        mvdb_c.MVDB_open(self.data_type.value, self.mvdb_obj, path)

    def topk(self, num_queries: np.uint64, query: np.array, k: np.uint64 = 5,
             metric: DISTANCE_METRIC = DISTANCE_METRIC.L2_DISTANCE, c: np.float32 = 100.0):
        return mvdb_c.MVDB_topk(self.data_type.value, self.mvdb_obj, num_queries, query, k, metric.value, c)