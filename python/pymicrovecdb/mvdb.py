import microvecdb as mvdb_c
import numpy as np
from enum import Enum, unique
import warnings
import struct

@unique
class DataType(Enum):
    INT8 = 0
    FLOAT32 = 1
    def __str__(self):
        return f"{self.name.lower()}"

def str_to_data_type(type_str: str):
    if type_str == str(DataType.INT8):
        return DataType.INT8
    elif type_str == str(DataType.FLOAT32):
        return DataType.FLOAT32
    else:
        raise ValueError(f"Unsupported index type '{type_str}'")

def np_dtype(data_type: DataType):
    if data_type == DataType.INT8:
        return np.int8
    elif data_type == DataType.FLOAT32:
        return np.float32
    else:
        raise ValueError("Unsupported data type")

@unique
class IndexType(Enum):
    FAISS_FLAT = 0
    ANNOY = 1
    SPANN = 2
    def __str__(self):
        return f"{self.name.lower()}"

def str_to_index_type(type_str: str):
    if type_str == str(IndexType.FAISS_FLAT):
        return IndexType.FAISS_FLAT
    elif type_str == str(IndexType.ANNOY):
        return IndexType.ANNOY
    elif type_str == str(IndexType.SPANN):
        return IndexType.SPANN
    else:
        raise ValueError("Unsupported index type")

@unique
class DistanceMetric(Enum):
    L2_DISTANCE = 0
    COSINE = 2

def _create_faiss_flat_named_args(**kwargs):
    return mvdb_c.FaissFlatIndexNamedArgs_init()

def _create_annoy_named_args(**kwargs):
    n_trees = 10
    n_threads = -1
    search_k = -1
    if 'n_trees' in kwargs:
        n_trees = kwargs['n_trees']
    if 'n_threads' in kwargs:
        n_threads = kwargs['n_threads']
    if 'search_k' in kwargs:
        search_k = kwargs['search_k']
    return mvdb_c.AnnoyIndexNamedArgs_init(n_trees, n_threads, search_k)

def _create_spann_named_args(**kwargs):
    build_config_path = ""
    quantizer_path = ""
    meta_mapping = False
    normalized = False
    thread_num = 0
    if 'build_config_path' in kwargs:
        build_config_path = kwargs['build_config_path']
    if 'quantizer_path' in kwargs:
        quantizer_path = kwargs['quantizer_path']
    if 'meta_mapping' in kwargs:
        meta_mapping = kwargs['meta_mapping']
    if 'normalized' in kwargs:
        normalized = kwargs['normalized']
    if 'thread_num' in kwargs:
        thread_num = kwargs['thread_num']
    return mvdb_c.SPANNIndexNamedArgs_init(build_config_path, quantizer_path, meta_mapping, normalized, thread_num)

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
    def __init__(self, dtype: DataType = DataType.FLOAT32):
        self.index_type = None
        self.named_args = None
        self.dtype = dtype
        self.mvdb_obj = mvdb_c.MVDB_init(dtype.value)

    @property
    def built(self):
        return mvdb_c.MVDB_get_built(self.dtype.value, self.mvdb_obj)

    @property
    def dims(self):
        assert mvdb_c.MVDB_get_built, "can't get dims db not built"
        return mvdb_c.MVDB_get_dims(self.dtype.value, self.mvdb_obj)

    def create(self, index_type: IndexType, dims: int, path: str,
               initial_data: np.array = None, initial_data_path: str = None, **kwargs):
        if dims <= 0:
            raise ValueError("dims must be a positive integer")
        if initial_data is not None and initial_data_path is not None:
            raise ValueError("Specify only one of initial_data or initial_data_path")
        if initial_data is None and initial_data_path is None:
            raise ValueError("Either initial_data or initial_data_path must be specified")

        if initial_data is not None:
            if initial_data.size % dims != 0:
                raise ValueError("The total size of initial_data must be a multiple of dims")
            initial_data_size = initial_data.shape[0]
            if initial_data.shape[0] > 1:
                initial_data = initial_data.flatten(order='C')
        else:
            initial_data_size = 0

        if initial_data_path:
            if not initial_data_path.endswith('.fvec') and not initial_data_path.endswith('.ivec') \
                    and not initial_data_path.endswith('.fvecs') and not initial_data_path.endswith('.ivecs'):
                raise ValueError("Unsupported file format, only .xvecs (fvecs or ivecs) files are allowed")

        self.index_type = index_type
        self.named_args = create_named_args(index_type, **kwargs)

        mvdb_c.MVDB_create(
            self.dtype.value,
            self.mvdb_obj,
            index_type.value,
            dims,
            path,
            initial_data_path if initial_data_path else "",
            initial_data if initial_data is not None else np.array([], dtype=np.float32),
            initial_data_size,
            self.named_args
        )

    def open(self, path: str):
        mvdb_c.MVDB_open(self.dtype.value, self.mvdb_obj, path)

    def topk(self, query: np.array = None, query_file: str = None, k: np.uint64 = 5,
             metric: DistanceMetric = DistanceMetric.L2_DISTANCE, c: np.float32 = 10000.0, **kwargs):
        assert mvdb_c.MVDB_get_built, "cannot search index. db not built"
        if (query is not None and len(query) > 0) and (query_file != "" and query_file is not None):
            raise ValueError("Specify only one of query or query_file")
        if (query is None or len(query) == 0) and (query_file == "" or query_file is None):
            raise ValueError("Either query or query_file must be specified")

        if query is not None:
            if query.size % self.dims != 0:
                raise ValueError("The total size of initial_data must be a multiple of dims")
            if query.ndim > 1:
                num_queries = query.shape[0]
                query = query.flatten(order='C')
            else:
                num_queries = int(query.shape[0]/self.dims)
        else:
            num_queries = 0

        result_path = None

        res = mvdb_c.MVDB_topk(
            self.dtype.value,
            self.mvdb_obj,
            query if query is not None else np.array([], dtype=np.float32),
            num_queries,
            query_file if query_file else "",
            result_path if result_path else "",
            k,
            metric.value,
            c
        )
        if result_path == "" or result_path is None:
            return res[0].reshape(num_queries, k), res[1].reshape(num_queries, k), res[2]
        return None


    @property
    def num_items(self):
        assert mvdb_c.MVDB_get_built, "can't get num_items. db not built"
        return mvdb_c.MVDB_get_num_items(self.dtype.value, self.mvdb_obj)
