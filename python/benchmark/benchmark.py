import gc
import csv
import os
import time
import re

import numpy as np
from memory_profiler import memory_usage

from pymicrovecdb import mvdb, utils as mv_utils
import nano_utils


query_sizes = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 3840, 5120, 6400, 7680, 8960, 10000]

k_values = [1, 10, 50, 100]

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
        'annoy_index_params': {'n_trees': 10, 'n_threads': 4},
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
            'batch_size': 2000,
            'thread_num': 4,
            'search_k': 6500
        }
    },
    'sift': {
        'base_path': f'{BASE_DATA_DIR}/sift/sift_base.fvecs',
        'query_path': f'{BASE_DATA_DIR}/sift/sift_query.fvecs',
        'sizes': [10000, 100000, 500000, 1000000],
        'dimensions': [128],
        'dtype': ['float32', 'int8'],
        'index_types': ['annoy', 'spann'],
        'annoy_index_params': {'n_trees': 10, 'n_threads': 4},
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
            'batch_size': 2000,
            'thread_num': 4,
            'search_k': 6500
        }
    },
    # 'gist': {
    #     'base_path': f'{BASE_DATA_DIR}/gist/gist_base.fvecs',
    #     'query_path': f'{BASE_DATA_DIR}/gist/gist_query.fvecs',
    #     'sizes': [10000, 100000, 500000, 1000000],
    #     'dimensions': [256, 512, 960],
    #     'dtype': ['float32', 'int8'],
    #     'index_types': ['annoy', 'spann'],
    #     'annoy_index_params': {'n_trees': 10, 'n_threads': 4},
    #     'spann_index_params': {
    #         'build_config_path': "buildconfig.ini",
    #         'BKTKmeansK': 8,
    #         'Samples': 4000,
    #         'TPTNumber': 112,
    #         'RefineIterations': 2,
    #         'NeighborhoodSize': 144,
    #         'CEF': 1800,
    #         'MaxCheckForRefineGraph': 7168,
    #         'NumberOfInitialDynamicPivots': 30,
    #         'GraphNeighborhoodScale': 2,
    #         'NumberOfOtherDynamicPivots': 2,
    #         'batch_size': 2000,
    #         'thread_num': 4,
    #         'search_k': 6500
    #     }
    # },
}

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

def get_cpu_env():
    cpu_info = mv_utils.get_cpu_info()
    ram_info = mv_utils.get_ram_info()
    storage_info = mv_utils.get_storage_info()
    battery_info = mv_utils.get_battery_info()
    uptime = mv_utils.get_system_uptime()
    processes_df = mv_utils.get_concurrent_processes()
    load_average = mv_utils.get_system_load()
    cpu_env = {**cpu_info, **ram_info, **storage_info, **battery_info}
    cpu_env['system_uptime'] = uptime
    cpu_env['concurrent_processes'] = processes_df
    cpu_env['load_average'] = load_average
    return cpu_env


def parse_tegrastats(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    lines = data.strip().split('\n')
    parsed_data = []

    for line in lines:
        ram = re.search(r'RAM (\d+)/(\d+)MB', line).groups()
        swap = re.search(r'SWAP (\d+)/(\d+)MB', line).groups()
        cpu = re.findall(r'(\d+)%@(\d+)', line)
        emc_freq = re.search(r'EMC_FREQ (\d+)%', line).group(1)
        gr3d_freq = re.search(r'GR3D_FREQ (\d+)%', line).group(1)
        temps = re.findall(r'(\w+)@([\d.]+)C', line)

        parsed_line = {
            'RAM_Used_MB': int(ram[0]),
            'RAM_Total_MB': int(ram[1]),
            'SWAP_Used_MB': int(swap[0]),
            'SWAP_Total_MB': int(swap[1]),
            'CPU0_Usage_Percent': int(cpu[0][0]),
            'CPU0_Freq_MHz': int(cpu[0][1]),
            'CPU1_Usage_Percent': int(cpu[1][0]),
            'CPU1_Freq_MHz': int(cpu[1][1]),
            'CPU2_Usage_Percent': int(cpu[2][0]),
            'CPU2_Freq_MHz': int(cpu[2][1]),
            'CPU3_Usage_Percent': int(cpu[3][0]),
            'CPU3_Freq_MHz': int(cpu[3][1]),
            'EMC_FREQ_Percent': int(emc_freq),
            'GR3D_FREQ_Percent': int(gr3d_freq),
        }

        for temp in temps:
            parsed_line[temp[0]] = float(temp[1])

        parsed_data.append(parsed_line)

    averages = {}
    count = len(parsed_data)
    for key in parsed_data[0]:
        averages[key] = sum(d[key] for d in parsed_data) / count

    return averages


def benchmark():
    get_cpu_env() # run once at the beginning so script immediately asks for sudo password
    result_dir = './results'
    tegrastats_dir = './tegrastats'
    result_file = f'{result_dir}/results.csv'
    os.makedirs(tegrastats_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    is_nano = nano_utils.is_jetson_nano()

    for key, config in DATASET_CONFIGS.items():
        for size in config['sizes']:
            for dims in config['dimensions']:
                for dtype in config['dtype']:
                    for index_type in config['index_types']:
                        dataset_name = f'{key}{short_code(size)}_{dims}D_{dtype}'
                        data_path = os.path.join(BASE_DATA_DIR, dataset_name)
                        index_name = f'{dataset_name}.{index_type}'
                        index_path = os.path.join(BASE_INDEX_DIR, index_name)

                        db = mvdb.MVDB(dtype=mvdb.str_to_data_type(dtype))
                        db.open(index_path)

                        queries = mv_utils.read_vector_file(config['query_path']).astype(np.int8) if dtype == "int8" else mv_utils.read_vector_file(config['query_path'])
                        ground = mv_utils.read_vector_file(f'{data_path}/{dataset_name}_base.fvecs_groundtruth.ivecs')

                        for q_size in query_sizes:
                            q = queries[:q_size]
                            gt = ground[:q_size]

                            for k in k_values:
                                internal_config = f"{index_name}_{q_size}_{k}"
                                print(f'Processing {index_name}:\n\tq_size={q_size}, k={k}')

                                if is_nano:
                                    nano_utils.start_tegrastats(f"{tegrastats_dir}/{internal_config}")

                                start_time = time.time()

                                # optimal search_k for SPANN = 6500
                                peak_dram, results = memory_usage((topk_wrapper, (db, q, k, config['annoy_index_params'] if index_type == "annoy" else config['spann_index_params'])), retval=True, max_usage=True)

                                query_time = time.time() - start_time

                                tegrastats = {}
                                if is_nano:
                                    nano_utils.stop_tegrastats()
                                    tegrastats = parse_tegrastats(f"{tegrastats_dir}/{internal_config}")

                                row = {
                                    **get_cpu_env(),
                                    **tegrastats,
                                    'dataset': f'{key}{short_code(size)}',
                                    'dims': db.dims, 'index_size': db.num_items, 'k': k, 'distance_metric': 'L2',
                                    'peak_dram_(MB)': peak_dram,
                                    'peak_WSS_(MB)': results[2],
                                    'index': index_name,
                                    'index_type': str(index_type),
                                    'dtype': str(dtype),
                                    'latency_(s)': query_time,
                                    'recall1': recall1(qr=results[0], gt=gt, k=k),
                                    'recall2': recall2(qr=results[0], gt=gt, k=k)
                                }

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
    parsed = parse_tegrastats('./results_sunday/tegrastats/deep10K_96D_float32.spann_10000_100')
    print(parsed)
    # benchmark()