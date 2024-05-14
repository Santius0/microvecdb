import microvecdb as mvdb_c
import numpy as np
from enum import Enum, unique
import warnings
import struct

def read_vector_file(filename):
    """
    Read vectors from a file with .fvecs or .ivecs extension.

    Parameters:
    - filename (str): Path to the file to read.

    Returns:
    - numpy.ndarray: Array of vectors stored in the file.
    """
    # determine the type based on file extension
    file_type = 'f' if filename.endswith('.fvecs') else 'i'
    dtype = np.float32 if file_type == 'f' else np.int32

    vectors = []
    with open(filename, 'rb') as file:
        while True:
            # Read the dimensionality of the vector (first 4 bytes)
            dim_bytes = file.read(4)
            if not dim_bytes:
                break
            dimension = struct.unpack('i', dim_bytes)[0]

            # Read the vector data
            count = dimension
            if file_type == 'f':
                vector_bytes = file.read(4 * count)  # Each float takes 4 bytes
                vector = struct.unpack(f'{count}f', vector_bytes)
            else:
                vector_bytes = file.read(4 * count)  # Each int takes 4 bytes
                vector = struct.unpack(f'{count}i', vector_bytes)

            vectors.append(vector)

    return np.array(vectors, dtype=dtype)

@unique
class DataType(Enum):
    INT8 = 0
    INT16 = 1
    UINT8 = 2
    FLOAT = 3

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
    build_config_path = ""
    quantizer_path = ""
    meta_mapping = False
    normalized = False
    thread_num = 0
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
    def __init__(self, data_type: DataType = DataType.FLOAT):
        self.index_type = None
        self.named_args = None
        self.data_type = data_type
        self.mvdb_obj = mvdb_c.MVDB_init(data_type.value)

    @property
    def built(self):
        return mvdb_c.MVDB_get_built(self.data_type.value, self.mvdb_obj)

    @property
    def dims(self):
        assert mvdb_c.MVDB_get_built, "can't get dims db not built"
        return mvdb_c.MVDB_get_dims(self.data_type.value, self.mvdb_obj)

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
            self.data_type.value,
            self.mvdb_obj,
            index_type.value,
            dims,
            path,
            initial_data_path if initial_data_path else "",
            initial_data if initial_data is not None else np.array([]),
            initial_data_size,
            self.named_args
        )

    def open(self, path: str):
        mvdb_c.MVDB_open(self.data_type.value, self.mvdb_obj, path)

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
