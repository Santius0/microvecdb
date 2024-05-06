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

class MVDB:
    def __init__(self, data_type: DATA_TYPE):
        self.data_type = data_type
        self.mvdb_ = mvdb_c.MVDB_init(data_type)

    def create(self, index_type: IndexType, dims: np.int64, path: str, initial_data_path: str = "", initial_data: np.ndarray = None, initial_data_size: np.int64 = 0, *args, **kwargs):
        mvdb_c.MVDB_create(self.data_type, self.mvdb_, index_type, dims, path, initial_data_path, initial_data, initial_data_size, None)

    def open(self, path: str):
        mvdb_c.MVDB_open(self.data_type, self.mvdb_, path)


    def topk(self, k: np.int64, c: np.float = 100.0):
        return mvdb_c.MVDB_topk(self.data_type, self.mvdb_, k, c)