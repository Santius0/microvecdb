import os
import gc
import shutil
from pymicrovecdb import mvdb, utils as mv_utils


BASE_DATA_DIR = '~/ann_data'

SIFT10K = f'{BASE_DATA_DIR}/sift10K/sift10K_base.fvecs'
SIFT100K = f'{BASE_DATA_DIR}/sift100K/sift100K_base.fvecs'
SIFT500K = f'{BASE_DATA_DIR}/sift500K/sift500K_base.fvecs'
SIFT1M = f'{BASE_DATA_DIR}/sift1M/sift/sift_base.fvecs'

GIST10K = f'{BASE_DATA_DIR}/gist10K/gist10K_base.fvecs'
GIST100K = f'{BASE_DATA_DIR}/gist100K/gist100K_base.fvecs'
GIST500K = f'{BASE_DATA_DIR}/gist500K/gist500K_base.fvecs'
GIST1M = f'{BASE_DATA_DIR}/gist1M/gist/gist_base.fvecs'

DEEP10K = f'{BASE_DATA_DIR}/deep10K/deep10K_base.fvecs'
DEEP100K = f'{BASE_DATA_DIR}/deep100K/deep100K_base.fvecs'
DEEP500K = f'{BASE_DATA_DIR}/deep500K/deep500K_base.fvecs'
DEEP1M = f'{BASE_DATA_DIR}/deep1M/deep1M_base.fvecs'
DEEP5M = f'{BASE_DATA_DIR}/deep5M/deep5M_base.fvecs'
DEEP10M = f'{BASE_DATA_DIR}/deep10M/deep10M_base.fvecs'


dataset_configs = {
    'sift': {'base_path': f'{BASE_DATA_DIR}/sift1M/sift/sift_base.fvecs', 'query_path': f'{BASE_DATA_DIR}/sift1M/sift/sift_query.fvecs' , 'sizes': [10000, 100000, 500000]},
    'gist': {'base_path': f'{BASE_DATA_DIR}/gist1M/gist/gist_base.fvecs', 'query_path': f'{BASE_DATA_DIR}/gist1M/gist/gist_query.fvecs', 'sizes': [10000, 100000, 500000]},
    'deep': {'base_path': f'{BASE_DATA_DIR}/deep10M/deep10M_base.fvecs', 'query_path': f'{BASE_DATA_DIR}/deep10M/deep1B_queries.fvecs', 'sizes': [10000, 100000, 500000, 1000000, 5000000, 10000000]},
}

index_configs = [
    # FAISS_FLAT
    # {'dataset_name': 'sift10K', 'initial_data_path': SIFT10K, 'index_type': mvdb.IndexType.FAISS_FLAT, 'data_type': mvdb.DataType.FLOAT32, 'params': {}, 'extra_labels': ""},
    # {'dataset_name': 'sift100K', 'initial_data_path': SIFT100K, 'index_type': mvdb.IndexType.FAISS_FLAT, 'data_type': mvdb.DataType.FLOAT32, 'params': {}, 'extra_labels': ""},
    # {'dataset_name': 'sift500K', 'initial_data_path': SIFT500K, 'index_type': mvdb.IndexType.FAISS_FLAT, 'data_type': mvdb.DataType.FLOAT32, 'params': {}, 'extra_labels': ""},
    # {'dataset_name': 'sift1M', 'initial_data_path': SIFT1M, 'index_type': mvdb.IndexType.FAISS_FLAT, 'data_type': mvdb.DataType.FLOAT32, 'params': {}, 'extra_labels': ""},

    # ANNOY (n_trees = 10)
    {'dataset_name': 'sift10K', 'initial_data_path': SIFT10K, 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 10, 'n_threads': 12}, 'extra_labels': 'n_trees=10'},
    {'dataset_name': 'sift100K', 'initial_data_path': SIFT100K, 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 10, 'n_threads': 12}, 'extra_labels': 'n_trees=10'},
    {'dataset_name': 'sift500K', 'initial_data_path': SIFT500K, 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 10, 'n_threads': 12}, 'extra_labels': 'n_trees=10'},
    {'dataset_name': 'sift1M', 'initial_data_path': SIFT1M, 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 10, 'n_threads': 12}, 'extra_labels': 'n_trees=10'},
    {'dataset_name': 'gist10K', 'initial_data_path': GIST10K, 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 10, 'n_threads': 12}, 'extra_labels': 'n_trees=10'},
    {'dataset_name': 'gist100K', 'initial_data_path': GIST100K, 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 10, 'n_threads': 12}, 'extra_labels': 'n_trees=10'},
    {'dataset_name': 'gist500K', 'initial_data_path': GIST500K, 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 10, 'n_threads': 12}, 'extra_labels': 'n_trees=10'},
    {'dataset_name': 'gist1M', 'initial_data_path': GIST1M, 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 10, 'n_threads': 12}, 'extra_labels': 'n_trees=10'},
    {'dataset_name': 'deep10K', 'initial_data_path': DEEP10K, 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 10, 'n_threads': 12}, 'extra_labels': 'n_trees=10'},
    {'dataset_name': 'deep100K', 'initial_data_path': DEEP100K, 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 10, 'n_threads': 12}, 'extra_labels': 'n_trees=10'},
    {'dataset_name': 'deep500K', 'initial_data_path': DEEP500K, 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 10, 'n_threads': 12}, 'extra_labels': 'n_trees=10'},
    {'dataset_name': 'deep1M', 'initial_data_path': DEEP1M, 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 10, 'n_threads': 12}, 'extra_labels': 'n_trees=10'},


    # SPANN
    # {'dataset_name': 'sift10K', 'initial_data_path': SIFT10K, 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'sift100K', 'initial_data_path': SIFT100K, 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'sift500K', 'initial_data_path': SIFT500K, 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'sift1M', 'initial_data_path': SIFT1M, 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'gist10K', 'initial_data_path': GIST10K, 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'gist100K', 'initial_data_path': GIST100K, 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'gist500K', 'initial_data_path': GIST500K, 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'gist1M', 'initial_data_path': GIST1M, 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'deep10K', 'initial_data_path': DEEP10K, 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'deep100K', 'initial_data_path': DEEP100K, 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'deep500K', 'initial_data_path': DEEP500K, 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'deep1M', 'initial_data_path': DEEP1M, 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
]

def delete_directory(path, verbose=False):
    if os.path.exists(path):
        shutil.rmtree(path)
        if verbose:
            print(f"Directory '{path}' has been deleted.")
    elif verbose:
        print(f"Directory '{path}' does not exist.")

def short_code(num):
    if num < 1000:
        return str(num)
    elif num < 1_000_000:
        short_num = num / 1_000
        suffix = "K"
    elif num < 1_000_000_000:
        short_num = num / 1_000_000
        suffix = "M"
    elif num < 1_000_000_000_000:
        short_num = num / 1_000_000_000
        suffix = "B"
    else:
        short_num = num / 1_000_000_000_000
        suffix = "T"

    if short_num == int(short_num):
        short_num = int(short_num)
    else:
        short_num = f"{short_num:.1f}".rstrip('0').rstrip('.')

    return f"{short_num}{suffix}"

def dataset_make(base_path, n, output_name):
    if not os.path.exists(f'{output_name}_base.fvecs'):
        print(f"Building {base_path} variant")
        vecs = mv_utils.read_vector_file(base_path, n)
        mv_utils.write_vector_file(vecs, f'{output_name}_base.fvecs')
        print(f'{output_name}_base.fvecs successfully built')
    else:
        print(f'{output_name}_base.fvecs already exists')

def build_datasets():
    os.makedirs("indices", exist_ok=True)
    for conf_key in dataset_configs:
        dataset_config = dataset_configs[conf_key]
        for size in dataset_config['sizes']:
            name = f'{conf_key}{short_code(size)}'
            path = f'{BASE_DATA_DIR}/{name}'
            os.makedirs(path, exist_ok=True)
            dataset_make(dataset_config['base_path'], size, f'{path}/{name}')
            gc.collect()
            db = mvdb.MVDB()
            db.create(
                index_type=mvdb.IndexType.FAISS_FLAT,
                dims=mv_utils.get_fvecs_dim_size(f'{path}/{name}_base.fvecs'),
                path=f'indices/faissflat_{name}',
                initial_data_path=f'{path}/{name}_base.fvecs'
            )
            print(f'./indices/{name} successfully built')
            ids, dists, _ = db.topk(query=mv_utils.read_vector_file(dataset_config['query_path']), k=100)
            mv_utils.write_vector_file(ids, f'{path}/{name}_groundtruth.ivecs')
            print(f'{path}/{name}_groundtruth.ivecs successfully built')
            mv_utils.write_vector_file(dists, f'{path}/{name}_groundtruth_dists.fvecs')
            print(f'{path}/{name}_groundtruth_dists.fvecs successfully built')
            gc.collect()

def build_indices():
    os.makedirs("indices", exist_ok=True)
    for config in index_configs:
        gc.collect()
        index_name = f"{config['index_type']}_{config['dataset_name']}_{config['data_type']}"
        if config['extra_labels'] != "":
            index_name += f"_{config['extra_labels']}"
        print(f"Building ./indices/{index_name}")
        db = mvdb.MVDB(dtype=config['data_type'])
        db.create(
            index_type=config['index_type'],
            dims=mv_utils.get_fvecs_dim_size(config['initial_data_path']),
            path=f"indices/{index_name}",
            initial_data_path=config['initial_data_path'],
            **config['params']
        )
        print(f"Successfully built ./indices/{index_name}")

def main():
    # build_datasets()
    build_indices()


if __name__ == '__main__':
    main()
