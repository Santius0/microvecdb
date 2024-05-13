import microvecdb as mvdb_c
import numpy as np
from enum import Enum, unique
import warnings

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
        assert (len(initial_data) == 0 and initial_data_path == "") != True, "either a initial_data or an initial_data_path must be specified"
        assert (len(initial_data) > 0 and initial_data_path != "") != True, "only exactly one of initial_data or an initial_data_path must be specified"
        # assert (initial_data_size == -1 or initial_data_size > 0), "invalid value initial_data_size"
        # if len(initial_data) > 0:
        #     if initial_data_size != -1:
        #         warnings.warn("initial_data_size is automatically calculated using initial_data and dims. "
        #                       "initial_data_size value entered is unnecessary and will be ignored")
        #     initial_data_size = int(len(initial_data)/dims)
        self.index_type = index_type
        self.named_args = create_named_args(index_type, **kwargs)
        mvdb_c.MVDB_create(self.data_type.value, self.mvdb_obj, index_type.value, dims, path, initial_data_path, initial_data, initial_data_size, self.named_args)
        self.built = True

    def open(self, path: str):
        mvdb_c.MVDB_open(self.data_type.value, self.mvdb_obj, path)

    @property
    def dims(self):
        assert mvdb_c.MVDB_get_built, "can't get dims db not built"
        return mvdb_c.MVDB_get_dims(self.data_type.value, self.mvdb_obj)

    def topk(self, query: np.array = np.array([]), query_file = "", k: np.uint64 = 5, num_queries: np.uint64 = 0,
             metric: DistanceMetric = DistanceMetric.L2_DISTANCE, c: np.float32 = 10000.0, **kwargs):
        assert mvdb_c.MVDB_get_built, "cannot search index. db not built"
        # assert (len(query) == 0 and query_file == "") != True, "either a query or a query_file must be specified"
        # assert (len(query) > 0 and query_file != "") != True, "only exactly one of either a query or a query_file must be specified"
        # assert (num_queries == 0), "invalid value 0 for num_queries"
        if len(query) > 0:
            if num_queries > -1:
                warnings.warn("num_queries is automatically calculated using query and quer.dims. "
                              "num_queries value entered is unnecessary and will be ignored")
            num_queries = int(len(query)/self.dims)
        print(f"num_queries={num_queries}")
        return mvdb_c.MVDB_topk(self.data_type.value, self.mvdb_obj, num_queries, query, k, metric.value, c)

    def num_items(self):
        assert mvdb_c.MVDB_get_built, "can't get num_items db not built"
        return mvdb_c.MVDB_get_num_items(self.data_type.value, self.mvdb_obj)
