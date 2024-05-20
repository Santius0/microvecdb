import os
import gc
import shutil
from pymicrovecdb import mvdb, utils as mv_utils

dataset_configs = {
    'sift': {'base_path': '../../ann_data/sift1M/sift/sift_base.fvecs', 'query_path': '/../../ann_data/sift1M/sift/sift_query.fvecs' , 'sizes': [10000, 100000, 500000]},
    'gist': {'base_path': '../../ann_data/gist1M/gist/gist_base.fvecs', 'query_path': '/../../ann_data/gist1M/gist/gist_query.fvecs', 'sizes': [10000, 100000, 500000]},
    'deep': {'base_path': '../../ann_data/deep10M/deep10M.fvecs', 'query_path': '../../ann_data/deep10M/deep1B_queries.fvecs', 'sizes': [10000, 100000, 500000, 1000000, 5000000]},
}

index_configs = [
    # FAISS_FLAT
    # {'dataset_name': 'sift10K', 'initial_data_path': '../../ann_data/sift10K/sift10K_base.fvecs', 'index_type': mvdb.IndexType.FAISS_FLAT, 'data_type': mvdb.DataType.FLOAT32, 'params': {}, 'extra_labels': ""},
    # {'dataset_name': 'sift100K', 'initial_data_path': '../../ann_data/sift100K/sift100K_base.fvecs', 'index_type': mvdb.IndexType.FAISS_FLAT, 'data_type': mvdb.DataType.FLOAT32, 'params': {}, 'extra_labels': ""},
    # {'dataset_name': 'sift500K', 'initial_data_path': '../../ann_data/sift500K/sift500K_base.fvecs', 'index_type': mvdb.IndexType.FAISS_FLAT, 'data_type': mvdb.DataType.FLOAT32, 'params': {}, 'extra_labels': ""},
    # {'dataset_name': 'sift1M', 'initial_data_path': '../../ann_data/sift1M/sift/sift_base.fvecs', 'index_type': mvdb.IndexType.FAISS_FLAT, 'data_type': mvdb.DataType.FLOAT32, 'params': {}, 'extra_labels': ""},

    # ANNOY
    # {'dataset_name': 'sift10K', 'initial_data_path': '../../ann_data/sift10K/sift10K_base.fvecs', 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 100, 'n_threads': 10}, 'extra_labels': 'n_tree=200'},
    # {'dataset_name': 'sift10K', 'initial_data_path': '../../ann_data/sift10K/sift10K_base.fvecs', 'index_type': mvdb.IndexType.ANNOY, 'data_type': mvdb.DataType.FLOAT32, 'params': {'n_trees': 100, 'n_threads': 10}, 'extra_labels': 'n_tree=400'},

    # SPANN
    # {'dataset_name': 'sift10K', 'initial_data_path': '../../ann_data/sift10K/sift10K_base.fvecs', 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'sift100K', 'initial_data_path': '../../ann_data/sift100K/sift100K_base.fvecs', 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'sift500K', 'initial_data_path': '../../ann_data/sift500K/sift500K_base.fvecs', 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
    # {'dataset_name': 'sift1M', 'initial_data_path': '../../ann_data/sift1M/sift/sift_base.fvecs', 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},

    # {'dataset_name': 'gist10K', 'initial_data_path': '../../ann_data/gist10K/gist10K_base.fvecs', 'index_type': mvdb.IndexType.SPANN, 'data_type': mvdb.DataType.FLOAT32, 'params': {'build_config_path': "/home/santius/microvecdb/SPTAG/buildconfig.ini"}, 'extra_labels': ''},
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
    print(f"Building {base_path} variant")
    vecs = mv_utils.read_vector_file(base_path, n)
    mv_utils.write_vector_file(vecs, f'{output_name}_base.fvecs')
    print(f'{output_name}_base.fvecs successfully built')

def build_datasets():
    for conf_key in dataset_configs:
        dataset_config = dataset_configs[conf_key]
        for size in dataset_config['sizes']:
            name = f'{conf_key}{short_code(size)}'
            path = f'/home/santius/ann_data/{name}'
            os.makedirs(path, exist_ok=True)
            dataset_make(dataset_config['base_path'], size, f'{path}/{name}')
            gc.collect()
            db = mvdb.MVDB()
            db.create(
                index_type=mvdb.IndexType.FAISS_FLAT,
                dims=mv_utils.get_fvecs_dim_size(f'{path}/{name}_base.fvecs'),
                path=f'./indices/faissflat_{name}',
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
    os.makedirs("./indices", exist_ok=True)
    for config in index_configs:
        gc.collect()
        index_name = f"{str(config['index_type'])}_{config['dataset_name']}_{config['data_type']}"
        if config['extra_labels'] != "":
            index_name = f"_{config['extra_labels']}"
        print(f"Building ./indices/{index_name}")
        db = mvdb.MVDB(data_type=config['data_type'])
        db.create(
            index_type=config['index_type'],
            dims=mv_utils.get_fvecs_dim_size(config['initial_data_path']),
            path=f"./indices/{index_name}",
            initial_data_path=config['initial_data_path'],
            **config['params']
        )
        print(f"Successfully built ./indices/{index_name}")

def main():
    # build_datasets()
    build_indices()


if __name__ == '__main__':
    main()