import gc
import csv
import os
import time

import numpy as np
from memory_profiler import memory_usage

from pymicrovecdb import mvdb, utils as mv_utils
import nano_utils


query_sizes = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 3840, 5120, 6400, 7680, 8960, 10000]

k_values = [1, 10, 50, 100]

BASE_DATA_DIR = '/home/santius/ann_data'
BASE_INDEX_DIR = '/home/santius/ann_indices'

datasets = {
    'sift10K': {'query':  f'{BASE_DATA_DIR}/sift1M/sift/sift_query.fvecs', 'ground':  f'{BASE_DATA_DIR}/sift10K/sift10K_groundtruth.ivecs'},
    'sift100K': {'query':  f'{BASE_DATA_DIR}/sift1M/sift/sift_query.fvecs', 'ground':  f'{BASE_DATA_DIR}/sift100K/sift100K_groundtruth.ivecs'},
    'sift500K': {'query':  f'{BASE_DATA_DIR}/sift1M/sift/sift_query.fvecs', 'ground':  f'{BASE_DATA_DIR}/sift500K/sift500K_groundtruth.ivecs'},
    'sift1M': {'query':  f'{BASE_DATA_DIR}/sift1M/sift/sift_query.fvecs', 'ground':  f'{BASE_DATA_DIR}/sift1M/sift/sift_groundtruth.ivecs'},
}

indices = [
    'annoy_sift100K_float32_n_trees=10',
    'annoy_deep1M_float32_n_trees=10',
    'annoy_gist10K_float32_n_trees=10',
    'annoy_sift1M_float32_n_trees=10',
    'annoy_gist100K_float32_n_trees=10',
    'annoy_deep100K_float32_n_trees=10',
    'annoy_sift500K_float32_n_trees=10',
    'annoy_deep10K_float32_n_trees=10',
    'annoy_deep500K_float32_n_trees=10',
    'annoy_gist1M_float32_n_trees=10',
    'annoy_gist500K_float32_n_trees=10',
    'annoy_sift10K_float32_n_trees=10'
]


    # 'deep10K': {'query_path': 'deep10K', 'dims': 96, 'size': 10000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep100K': {'query_path': 'deep100K', 'dims': 96, 'size': 100000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep500K': {'query_path': 'deep500K', 'dims': 96, 'size': 500000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep1M': {'query_path': 'deep1M', 'dims': 96, 'size': 1000000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep5M': {'query_path': 'deep5M', 'dims': 96, 'size': 5000000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep10M': {'query_path': 'deep10M', 'dims': 96, 'size': 10000000, 'dtype': mvdb.DataType.FLOAT},

    # 'spann_sift10K_float32': {'index': './indices/spann_sift10K_float32', 'extra_args': {}, 'query': '../../ann_data/sift1M/sift/sift_query.fvecs', 'ground': '../../ann_data/sift10K/sift10K_groundtruth.ivecs'},
    # 'sift100K': {'query_path': 'sift100K', 'dims': 128, 'size': 100000, 'dtype': mvdb.DataType.FLOAT},
    # 'sift500K': {'query_path': 'sift500K', 'dims': 128, 'size': 500000, 'dtype': mvdb.DataType.FLOAT},
    # 'sift1M': {'query_path': '../data/sift1M/sift/sift_query.fvecs', 'ground_path': '../data/sift1M/sift/sift_groundtruth.ivecs', 'dims': 128, 'size': 1000000, 'dtype': mvdb.DataType.FLOAT},
    # 'sift5M': {'query_path': 'sift5M', 'dims': 128, 'size': 5000000, 'dtype': mvdb.DataType.FLOAT},
    # 'sift10M': {'query_path': 'sift10M', 'dims': 128, 'size': 10000000, 'dtype': mvdb.DataType.FLOAT},
    #
    # 'gist10K': {'query_path': 'gist10K', 'dims': 960, 'size': 10000, 'dtype': mvdb.DataType.FLOAT},
    # 'gist100K': {'query_path': 'gist100K', 'dims': 960, 'size': 100000, 'dtype': mvdb.DataType.FLOAT},
    # 'gist500K': {'query_path': 'gist500K', 'dims': 960, 'size': 500000, 'dtype': mvdb.DataType.FLOAT},
    # 'gist1M': {'query_path': 'gist1M', 'dims': 960, 'size': 1000000, 'dtype': mvdb.DataType.FLOAT},
# }

def topk_wrapper(db_, q, k, params=None):
    if params is None:
        params = {}
    results = db_.topk(query=q, k=k, **params)
    print('DONE')
    return results

def recall1(qr, gt, k = 100):
    recalls = np.zeros(len(qr))
    for i in range(len(qr)):
        actual = 0
        for id in qr[i][:k]:
            if id in gt[i][:k]:
                actual += 1
        # recalls[i] = actual/float(k)
        recalls[i] = actual
    # return np.mean(recalls), np.std(recalls), recalls
    # return np.mean(recalls) / float(k), np.std(recalls) / float(k), recalls
    return np.mean(recalls) / float(k)

def recall2(qr, gt, k = 100):
    return (qr[:, :k] == gt[:, :1]).sum() / float(qr.shape[0])

def q_copy(arr: np.array, flat: bool = False):
    external = []
    if flat:
        for a in range(len(arr)):
            for b in range(len(arr[a])):
                external.append(arr[a][b])
    else:
        for a in range(len(arr)):
            internal = []
            for b in range(len(arr[a])):
                internal.append(arr[a][b])
            external.append(internal)
    return external

cpu_info = mv_utils.get_cpu_info()
ram_info = mv_utils.get_ram_info()
storage_info = mv_utils.get_storage_info()
battery_info = mv_utils.get_battery_info()
uptime = mv_utils.get_system_uptime()
processes_df = mv_utils.get_concurrent_processes()
load_average = mv_utils.get_system_load()
cpu_env = {**cpu_info, **ram_info, **storage_info, **battery_info}

def main():
    result_dir = './results'
    tegrastats_dir = './tegrastats'
    result_file = f'{result_dir}/results.csv'
    os.makedirs(result_dir, exist_ok=True)
    is_nano = nano_utils.is_jetson_nano()
    for index in indices:
        gc.collect()
        index_key_list = index.split('_')
        index_type, dataset_name, dtype = mvdb.str_to_index_type(index_key_list[0]), index_key_list[1], mvdb.str_to_data_type(index_key_list[2])
        dataset = datasets[dataset_name]
        db = mvdb.MVDB(dtype=dtype)
        db.open(f"{BASE_INDEX_DIR}/{index}")
        queries = mv_utils.read_vector_file(dataset['query'])
        ground = mv_utils.read_vector_file(dataset['ground'])
        for q_size in query_sizes:
            gc.collect()
            q = queries[:q_size]
            gt = ground[:q_size]
            for k in k_values:
                internal_config = f"{index}_{q_size}_{k}"
                print(f"processing: {internal_config}...")
                if is_nano:
                    nano_utils.start_tegrastats(f"{tegrastats_dir}/{internal_config}")
                start_time = time.time()
                peak_dram, results = memory_usage((topk_wrapper, (db, q, k, {'n_threads': 3})), retval=True, max_usage=True)
                query_time = time.time() - start_time
                if is_nano:
                    nano_utils.stop_tegrastats()
                row = cpu_env
                row['dataset'] = dataset
                row['dims'] = db.dims
                row['index_size'] = db.num_items
                row['k'] = k
                row['distance_metric'] = 'L2'
                row['peak_dram_(MB)'] = peak_dram
                row['peak_WSS_(MB)'] = results[2]
                row['index'] = index
                row['index_type'] = str(index_type)
                row['latency_(s)'] = query_time
                row['recall1'] = recall1(qr=results[0], gt=gt, k=k)
                row['recall2'] = recall2(qr=results[0], gt=gt, k=k)
                print(f"{internal_config} completed in {query_time} seconds")
                file_exists = os.path.isfile(result_file)
                with open(result_file, mode='a', newline='') as file:
                    fieldnames = row.keys()
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)
    print("**DONE**")

if __name__ == '__main__':
    main()
