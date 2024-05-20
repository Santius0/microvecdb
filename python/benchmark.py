import gc

import csv
import os

import numpy as np
import time
from memory_profiler import profile, memory_usage

from pymicrovecdb import mvdb, utils as mv_utils
import nano_utils

indices = {
    'spann_sift10K_float32': {'type': mvdb.IndexType.SPANN, 'path': './indices/spann_sift10K_float32', 'dims': 128, 'misc.': ['n_threads=12'], 'dtype': mvdb.DataType.FLOAT32},
}

query_sizes = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 3840, 5120, 6400, 7680, 8960, 10000]

k_values = [1, 10, 50, 100]

configs = {
    # 'deep10K': {'query_path': 'deep10K', 'dims': 96, 'size': 10000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep100K': {'query_path': 'deep100K', 'dims': 96, 'size': 100000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep500K': {'query_path': 'deep500K', 'dims': 96, 'size': 500000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep1M': {'query_path': 'deep1M', 'dims': 96, 'size': 1000000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep5M': {'query_path': 'deep5M', 'dims': 96, 'size': 5000000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep10M': {'query_path': 'deep10M', 'dims': 96, 'size': 10000000, 'dtype': mvdb.DataType.FLOAT},

    'spann_sift10K_float32': {'index': './indices/spann_sift10K_float32', 'extra_args': {}, 'query': '../../ann_data/sift1M/sift/sift_query.fvecs', 'ground': '../../ann_data/sift10K/sift10K_groundtruth.ivecs'},
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
}

def topk_wrapper(db_, q, k):
    results = db_.topk(query=q, k=k)
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
    os.makedirs('./benchmark_results', exist_ok=True)
    is_nano = nano_utils.is_jetson_nano()
    for dataset_key in datasets:
        print(f'Processing: {dataset_key}...')
        dataset = datasets[dataset_key]
        queries = mv_utils.read_vector_file(dataset['query_path'])
        ground = mv_utils.read_vector_file(dataset['ground_path'])
        for index_key in indices:
            gc.collect()
            index = indices[index_key]
            db = mvdb.MVDB(index['dtype'])
            db.open(index['path'])
            for q_size in query_sizes:
                q = queries[:q_size]
                gt = ground[:q_size]
                for k in k_values:
                    gc.collect()
                    print(f'running: {dataset_key} => {index_key} => q_size = {q_size} => k = {k}')
                    if is_nano:
                        nano_utils.start_tegrastats(f'./tegrastats/{dataset_key}_{index_key}_{q_size}_{k}')
                    start_time = time.time()
                    peak_dram, results = memory_usage((topk_wrapper, (db, q, k)), retval=True, max_usage=True)
                    query_time = time.time() - start_time
                    if is_nano:
                        nano_utils.stop_tegrastats()
                    row = cpu_env
                    row['dataset'] = dataset_key
                    row['dims'] = dataset['dims']
                    row['index_size'] = dataset['size']
                    row['k'] = k
                    row['distance_metric'] = 'L2'
                    row['peak_dram_(MB)'] = peak_dram
                    row['peak_WSS_(MB)'] = results[2]
                    row['index'] = index_key
                    row['index_type'] = str(index['type'])
                    row['latency_(s)'] = query_time
                    row['recall1'] = recall1(results[0], gt, k)
                    row['recall2'] = recall2(results[0], gt, k)
                    file_exists = os.path.isfile(f"./benchmark_results/res.csv")
                    with open(f"./benchmark_results/res.csv", mode='a', newline='') as file:
                        fieldnames = row.keys()
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(row)
                    print(f'complete')
    print("**DONE**")

if __name__ == '__main__':
    main()