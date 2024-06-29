import os
import gc
import shutil
import numpy as np
from pymicrovecdb import mvdb, utils as mv_utils

def delete_directory(path, verbose=False):
    if os.path.exists(path):
        shutil.rmtree(path)
        if verbose:
            print(f"Directory '{path}' has been deleted.")
    elif verbose:
        print(f"Directory '{path}' does not exist.")


def short_code(num):
    suffixes = ['K', 'M', 'B', 'T']
    base = 1000
    num_str = str(num)
    for suffix in suffixes:
        if num < base:
            return num_str
        num /= base
        num_str = f"{num:.1f}".rstrip('0').rstrip('.') + suffix
    return num_str

def dataset_make(base_path, output_path, n, dims=None, dtype="float32", quant_mul=1):
    """
    Creates a new dataset from the original dataset.

    Args:
        base_path (str): Path to the original dataset.
        n (int): Number of vectors.
        dims (int): Number of dimensions.
        output_path (str): Path to save the new dataset.
    """
    if not os.path.exists(output_path):
        print(f"Building {output_path}")
        vecs = mv_utils.read_vector_file(base_path, n)
        vecs = vecs[:, :dims]

        if dtype == "int8":
            vecs = np.clip(vecs, -128, 127)
            vecs *= quant_mul
            vecs = vecs.astype(np.int8).astype(np.float32)

        mv_utils.write_vector_file(vecs, output_path)
        print(f'{output_path} successfully built')
    else:
        print(f'{output_path} already exists')

BASE_DATA_DIR = '/home/santius/ann_data'
BASE_INDEX_DIR = '/home/santius/ann_indices'

DATASET_CONFIGS = {
    'deep': {
        'base_path': f'{BASE_DATA_DIR}/deep/deep10M.fvecs',
        'query_path': f'{BASE_DATA_DIR}/deep/deep1B_queries.fvecs',
        'sizes': [10000, 100000, 500000, 1000000],
        'dimensions': [96],
        'dtype': ['float32', 'int8'],
        'index_types': ['annoy', 'spann'],
        'annoy_index_params': {'n_trees': 10, 'n_threads': 12},
        'spann_index_params': {
            'build_config_path': "buildconfig.ini",
            'BKTKmeansK': 8,
            'Samples': 4000,
            'TPTNumber': 112,
            'RefineIterations': 2,
            'NeighborhoodSize': 144,
            'CEF': 1800,
            'MaxCheckForRefineGraph': 7168,
            'NumberOfInitialDynamicPivots': 30,
            'GraphNeighborhoodScale': 2,
            'NumberOfOtherDynamicPivots': 2,
            'batch_size': 2000
        }
    },
    'sift': {
        'base_path': f'{BASE_DATA_DIR}/sift/sift_base.fvecs',
        'query_path': f'{BASE_DATA_DIR}/sift/sift_query.fvecs',
        'sizes': [10000, 100000, 500000, 1000000],
        'dimensions': [128],
        'dtype': ['float32', 'int8'],
        'index_types': ['annoy', 'spann'],
        'annoy_index_params': {'n_trees': 10, 'n_threads': 12},
        'spann_index_params': {
            'build_config_path': "buildconfig.ini",
            'BKTKmeansK': 8,
            'Samples': 4000,
            'TPTNumber': 112,
            'RefineIterations': 2,
            'NeighborhoodSize': 144,
            'CEF': 1800,
            'MaxCheckForRefineGraph': 7168,
            'NumberOfInitialDynamicPivots': 30,
            'GraphNeighborhoodScale': 2,
            'NumberOfOtherDynamicPivots': 2,
            'batch_size': 2000
        }
    },
    'gist': {
        'base_path': f'{BASE_DATA_DIR}/gist/gist_base.fvecs',
        'query_path': f'{BASE_DATA_DIR}/gist/gist_query.fvecs',
        'sizes': [10000, 100000, 500000, 1000000],
        'dimensions': [256, 512, 960],
        'dtype': ['float32', 'int8'],
        'index_types': ['annoy', 'spann'],
        'annoy_index_params': {'n_trees': 10, 'n_threads': 12},
        'spann_index_params': {
            'build_config_path': "buildconfig.ini",
            'BKTKmeansK': 8,
            'Samples': 4000,
            'TPTNumber': 112,
            'RefineIterations': 2,
            'NeighborhoodSize': 144,
            'CEF': 1800,
            'MaxCheckForRefineGraph': 7168,
            'NumberOfInitialDynamicPivots': 30,
            'GraphNeighborhoodScale': 2,
            'NumberOfOtherDynamicPivots': 2,
            'batch_size': 2000
        }
    },
}

def build_indices():
    os.makedirs(BASE_INDEX_DIR, exist_ok=True)
    for key, config in DATASET_CONFIGS.items():
        for size in config['sizes']:
            for dims in config['dimensions']:
                for dtype in config['dtype']:
                    for index_type in config['index_types']:
                        dataset_name = f'{key}{short_code(size)}_{dims}D_{dtype}'
                        data_path = os.path.join(BASE_DATA_DIR, dataset_name)

                        index_name = f'{dataset_name}.{index_type}'
                        index_path = os.path.join(BASE_INDEX_DIR, index_name)

                        print(f"Building {index_path}")

                        dims = mv_utils.get_fvecs_dim_size(f'{data_path}/{dataset_name}_base.fvecs')
                        data = mv_utils.read_vector_file(f'{data_path}/{dataset_name}_base.fvecs')
                        if dtype == "int8":
                            data = data.astype(np.int8)

                        params = config['annoy_index_params'] if index_type == 'annoy' else config['spann_index_params']

                        db = mvdb.MVDB(dtype=mvdb.str_to_data_type(dtype))
                        db.create(
                            index_type=mvdb.str_to_index_type(index_type),
                            dims=dims,
                            path=index_path,
                            initial_data=data,
                            **params
                        )
                        print(f"Successfully built {index_path}")

def build_datasets():
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    os.makedirs(BASE_INDEX_DIR, exist_ok=True)
    for key, config in DATASET_CONFIGS.items():
        for size in config['sizes']:
            for dims in config['dimensions']:
                for dtype in config['dtype']:
                    name = f'{key}{short_code(size)}_{dims}D_{dtype}'
                    output_dir = f'{BASE_DATA_DIR}/{name}'
                    output_path = f'{output_dir}/{name}_base.fvecs'
                    os.makedirs(output_dir, exist_ok=True)
                    dataset_make(config['base_path'], output_path, size, dims, dtype=dtype, quant_mul=10 if key == "deep" or key == "gist" else 1)
                    gc.collect()

                    faiss_index_path = f'{BASE_INDEX_DIR}/faissflat_{name}'
                    data = mv_utils.read_vector_file(output_path, size)
                    db = mvdb.MVDB(mvdb.DataType.FLOAT32) # faiss only support float32 so that's what we use when computing groundtruths with faissflat
                    db.create(
                        index_type=mvdb.IndexType.FAISS_FLAT,
                        dims=dims,
                        path=faiss_index_path,
                        initial_data=data
                    )
                    print(f'{faiss_index_path} successfully built')
                    query = mv_utils.read_vector_file(config['query_path'])
                    if dtype == "int8":
                        query = np.clip(query, -128, 127)
                        query *= 10 if key == "deep" or key == "gist" else 1
                        query = query.astype(np.int8).astype(np.float32)
                    ids, dists, _ = db.topk(query=query, k=100)
                    mv_utils.write_vector_file(ids, f'{output_path}_groundtruth.ivecs')
                    mv_utils.write_vector_file(dists, f'{output_path}_groundtruth_dists.fvecs')
                    print(f'{output_path}_groundtruth and distances successfully built')
                    delete_directory(faiss_index_path)
                    gc.collect()


if __name__ == '__main__':
    # build_datasets()
    build_indices()
    # deep = mv_utils.read_vector_file(DATASET_CONFIGS['deep']['base_path'], 1000000)
    # sift = mv_utils.read_vector_file(DATASET_CONFIGS['sift']['base_path'])
    # gist = mv_utils.read_vector_file(DATASET_CONFIGS['gist']['base_path'])
    #
    # # print(deep[0])
    # # print(sift[0])
    # # print(gist[0])
    #
    # deep *= 10
    # deep = deep.astype(np.int8)
    # print(f'deep_max = {np.min(deep)}')
    # print(f'sift_max = {np.min(sift)}')
    # print(f'gist_max = {np.min(gist)}')
