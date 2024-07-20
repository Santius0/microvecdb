import microvecdb as mvdb_c
import numpy as np
from enum import Enum, unique
import warnings
import struct
import cv2
import pickle

@unique
class DataType(Enum):
    INT8 = 0
    FLOAT32 = 1
    def __str__(self):
        return f"{self.name.lower()}"

def str_to_data_type(type_str: str):
    type_str = type_str.lower()
    if type_str == str(DataType.INT8).lower():
        return DataType.INT8
    elif type_str == str(DataType.FLOAT32).lower():
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
    type_str = type_str.lower()
    if type_str == str(IndexType.FAISS_FLAT).lower():
        return IndexType.FAISS_FLAT
    elif type_str == str(IndexType.ANNOY).lower():
        return IndexType.ANNOY
    elif type_str == str(IndexType.SPANN).lower():
        return IndexType.SPANN
    else:
        raise ValueError("Unsupported index type: " + type_str)

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
    truth_path = ""
    meta_mapping = False
    normalized = False
    thread_num = 3
    batch_size = 10000
    BKTKmeansK = -1
    Samples = -1
    TPTNumber = -1
    RefineIterations = -1
    NeighborhoodSize = -1
    CEF = -1
    MaxCheckForRefineGraph = -1
    NumberOfInitialDynamicPivots = -1
    GraphNeighborhoodScale = -1
    NumberOfOtherDynamicPivots = -1
    if 'build_config_path' in kwargs:
        build_config_path = kwargs['build_config_path']
    if 'quantizer_path' in kwargs:
        quantizer_path = kwargs['quantizer_path']
    if 'truth_path' in kwargs:
        truth_path = kwargs['truth_path']
    if 'meta_mapping' in kwargs:
        meta_mapping = kwargs['meta_mapping']
    if 'normalized' in kwargs:
        normalized = kwargs['normalized']
    if 'thread_num' in kwargs:
        thread_num = kwargs['thread_num']
    if 'batch_size' in kwargs:
        batch_size = int(kwargs['batch_size'])
    if 'BKTKmeansK' in kwargs:
        BKTKmeansK = int(kwargs['BKTKmeansK'])
    if 'Samples' in kwargs:
        Samples = int(kwargs['Samples'])
    if 'TPTNumber' in kwargs:
        TPTNumber = int(kwargs['TPTNumber'])
    if 'RefineIterations' in kwargs:
        RefineIterations = int(kwargs['RefineIterations'])
    if 'NeighborhoodSize' in kwargs:
        NeighborhoodSize = int(kwargs['NeighborhoodSize'])
    if 'CEF' in kwargs:
        CEF = int(kwargs['CEF'])
    if 'MaxCheckForRefineGraph' in kwargs:
        MaxCheckForRefineGraph = int(kwargs['MaxCheckForRefineGraph'])
    if 'NumberOfInitialDynamicPivots' in kwargs:
        NumberOfInitialDynamicPivots = int(kwargs['NumberOfInitialDynamicPivots'])
    if 'GraphNeighborhoodScale' in kwargs:
        GraphNeighborhoodScale = int(kwargs['GraphNeighborhoodScale'])
    if 'NumberOfOtherDynamicPivots' in kwargs:
        NumberOfOtherDynamicPivots = int(kwargs['NumberOfOtherDynamicPivots'])
    return mvdb_c.SPANNIndexNamedArgs_init(build_config_path, truth_path, quantizer_path, meta_mapping, normalized, thread_num, batch_size,
                                           BKTKmeansK, Samples, TPTNumber, RefineIterations, NeighborhoodSize, CEF,
                                           MaxCheckForRefineGraph, NumberOfInitialDynamicPivots, GraphNeighborhoodScale,
                                           NumberOfOtherDynamicPivots)

def create_named_args(index_type: IndexType, **kwargs):
    if index_type == IndexType.FAISS_FLAT:
        return _create_faiss_flat_named_args(**kwargs)
    elif index_type == IndexType.ANNOY:
        return _create_annoy_named_args(**kwargs)
    elif index_type == IndexType.SPANN:
        return _create_spann_named_args(**kwargs)
    else:
        raise RuntimeError(f"Unknown IndexType {index_type}")

def read_binary_file(filepath):
    with open(filepath, 'rb') as f:
        return f.read()

class MVDB:
    def __init__(self, dtype: DataType = DataType.FLOAT32):
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
               initial_data: np.array = None, initial_objs: np.array = None, **kwargs):
        if dims <= 0:
            raise ValueError("dims must be a positive integer")

        if initial_data is not None:
            if initial_data.size % dims != 0:
                raise ValueError("The total size of initial_data must be a multiple of dims")
            initial_data_size = initial_data.shape[0]
            if initial_data.shape[0] > 1:
                initial_data = initial_data.flatten(order='C')
        else:
            initial_data_size = 0

        named_args = create_named_args(index_type, **kwargs)

        mvdb_c.MVDB_create(
            self.dtype.value,
            index_type.value,
            self.mvdb_obj,
            dims,
            path,
            initial_data if initial_data is not None else np.array([], dtype=np.float32),
            initial_objs,
            initial_data_size,
            named_args
        )

    def add(self, data: np.array = None, objs: np.array = None):
        if data is not None:
            if data.size % self.dims != 0:
                raise ValueError("The total size of initial_data must be a multiple of dims")
            n = data.shape[0]
            if data.shape[0] > 1:
                data = data.flatten(order='C')
        else:
            n = 0

        mvdb_c.MVDB_add(
            self.dtype.value,
            self.mvdb_obj,
            data if data is not None else np.array([], dtype=np.float32),
            objs,
            n,
        )

    def open(self, path: str):
        mvdb_c.MVDB_open(self.dtype.value, self.mvdb_obj, path)

    def knn(self, query: np.array, k: np.int64 = 5, **kwargs):
        assert mvdb_c.MVDB_get_built, "cannot search index. db not built"

        if query.size % self.dims != 0:
            raise ValueError("The total size of initial_data must be a multiple of dims")
        if query.ndim > 1:
            num_queries = query.shape[0]
            query = query.flatten(order='C')
        else:
            num_queries = int(query.shape[0]/self.dims)

        result_path = None

        named_args = create_named_args(self.index_type, **kwargs)

        res = mvdb_c.MVDB_knn(
            self.index_type.value,
            self.dtype.value,
            self.mvdb_obj,
            query if query is not None else np.array([], dtype=np.float32),
            num_queries,
            k,
            named_args
        )
        if result_path == "" or result_path is None:
            return res[0].reshape(num_queries, k), res[1].reshape(num_queries, k), res[2]
        return None

    def get(self, ids: np.array):
        return mvdb_c.MVDB_get(self.dtype.value, self.mvdb_obj, ids.astype(np.uint64))
    @property
    def index_type(self):
        index_type_byte_val = mvdb_c.MVDB_get_index_type(self.dtype.value, self.mvdb_obj)
        try:
            return IndexType(index_type_byte_val)
        except ValueError:
            raise ValueError(f"Unknown byte value: {index_type_byte_val}")

    @property
    def num_items(self):
        assert mvdb_c.MVDB_get_built, "can't get num_items. db not built"
        return mvdb_c.MVDB_get_num_items(self.dtype.value, self.mvdb_obj)
