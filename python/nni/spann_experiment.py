import os
import gc
import shutil
import time
import nni
import numpy as np
from pymicrovecdb import mvdb, utils as mv_utils
from memory_profiler import memory_usage
import json
import re

datasets = [
    {'name': 'sift1M', 'base': '../../../ann_data/sift1M/sift/sift_base.fvecs', 'query': '../../../ann_data/sift1M/sift/sift_query.fvecs', 'ground': '../../../ann_data/sift1M/sift/sift_groundtruth.ivecs'},
    {'name': 'gist1M', 'base': '../../../ann_data/gist1M/gist/gist_base.fvecs', 'query': '../../../ann_data/gist1M/gist/gist_query.fvecs', 'ground': '../../../ann_data/gist1M/gist/gist_groundtruth.ivecs'},
    {'name': 'deep1M', 'base': '../../../ann_data/deep1M/deep1M_base.fvecs', 'query': '../../../ann_data/deep10M/deep1B_queries.fvecs', 'ground': '../../../ann_data/deep1M/deep1M_groundtruth.ivecs'},
]

k_values = [1, 10, 50, 100]


def dict_values_to_safe_filepath_string(d):
    # Extract values from the dictionary and convert them to strings
    values = [str(value) for value in d.values()]

    # Replace non-safe characters in each value with underscores
    safe_values = [re.sub(r'[^A-Za-z0-9]', '_', value) for value in values]

    # Join the values with underscores
    safe_str = '_'.join(safe_values)
    return safe_str

def recall1(qr, gt, k=100):
    recalls_ = np.zeros(len(qr))
    for i in range(len(qr)):
        actual = 0
        for id_ in qr[i, :k]:
            if id_ in gt[i, :k]:
                actual += 1
        recalls_[i] = actual
    r1 = np.mean(recalls_)/float(k)
    print(f"Recall1 (r1): {r1:.4f}")
    return r1

def recall2(qr, gt, k=100):
    r2 = (qr[:, :k] == gt[:, :1]).sum() / float(qr.shape[0])
    print(f"Recall2 (r2): {r2:.4f}")
    return r2

def delete_directory(path, verbose=False):
    if os.path.exists(path):
        shutil.rmtree(path)
        if verbose:
            print(f"Directory '{path}' has been deleted.")
    elif verbose:
        print(f"Directory '{path}' does not exist.")

def wrapper(db, dataset, k, kwargs=None):
    if kwargs is None:
        kwargs = {}
    return db.topk(query=(mv_utils.read_vector_file(dataset['query'])), k=k, **kwargs)

def normalize(value, min_value, max_value):
    return max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))

def objective(latency, recall, dram, weight_latency=0.2, weight_recall=0.4, weight_dram=0.4, min_latency=0, max_latency=30, min_dram=0, max_dram=2048):
    latency_norm = normalize(latency, min_latency, max_latency)
    dram_norm = normalize(dram, min_dram, max_dram)
    objective_value = (weight_recall * recall) - (weight_latency * latency_norm) - (weight_dram * dram_norm)
    print(f"Latency Normalized: {latency_norm:.4f}")
    print(f"DRAM Normalized: {dram_norm:.4f}")
    print(f"Objective Value: {objective_value:.4f}")
    return objective_value

def evaluate_spann(params):
    index_dir = "./indices"
    os.makedirs(index_dir, exist_ok=True)
    gc.collect()
    latencies = np.zeros(len(datasets) * len(k_values))
    drams = np.zeros(len(datasets) * len(k_values))
    recalls = np.zeros(len(datasets) * len(k_values))
    objectives = np.zeros(len(datasets) * len(k_values))
    for i_d, dataset in enumerate(datasets):
        gc.collect()
        index_path = f"{index_dir}/spann_{dataset['name']}_{dict_values_to_safe_filepath_string(params)}"
        db = mvdb.MVDB()
        if not os.path.exists(index_path):
            print(f"Building index for {index_path}...")
            db.create(
                index_type=mvdb.IndexType.SPANN,
                dims=mv_utils.get_fvecs_dim_size(dataset['query']),
                path=index_path,
                initial_data=mv_utils.read_vector_file(dataset['base'])[:1000],
                # initial_data_path=dataset['base'],
                thread_num=10,
                build_config_path='../../SPTAG/buildconfig.ini',
                **params
            )
            print(f"Build successful for {dataset['name']} at {index_path}")
        else:
            print(f"Index at {index_path} already exists. No build needed.")
            db.open(index_path)

        ground = mv_utils.read_vector_file(dataset['ground'])

        for i_k, k in enumerate(k_values):
            print("\n")

            pos = i_d * len(k_values) + i_k
            print(f"pos [{pos}]")
            print(f"k: {k}")
            gc.collect()

            start_time = time.time()
            peak_dram, results = memory_usage((wrapper, (db, dataset, k, {'thread_num': 10, **params})), retval=True, max_usage=True)
            ids = results[0]
            latency = time.time() - start_time
            recall = recall2(qr=ids, gt=ground, k=k)
            latencies[pos] = -latency
            drams[pos] = -peak_dram
            recalls[pos] = recall
            objective_val = objective(latency, recall, peak_dram)
            objectives[pos] = objective_val

            current = {'-latency': -latency, '-dram': -peak_dram, 'recall': recall, 'default': objective_val}
            print(f"Current Performance: {current}")

            mask = objectives != 0
            intermediate = {'-latency': np.mean(latencies[mask]), '-dram': np.mean(drams[mask]), 'recall': np.mean(recalls[mask]), 'default': np.mean(objectives[mask])}
            print(f"Intermediate Performance: {intermediate}")
            nni.report_intermediate_result(intermediate)
            print("\n")

    return {'-latency': np.mean(latencies), '-dram': np.mean(drams), 'recall': np.mean(recalls), 'default': np.mean(objectives)}



if __name__ == "__main__":
    params = nni.get_next_parameter()
    print(f"Hyperparameters:\n{params}")
    performance = evaluate_spann(params)
    print(f"Final Performance: {performance}")
    nni.report_final_result(performance)


# notes: best experiment results obtained. best params are as follows:
#   1. n_trees = 10, search_k = 6500
#   2. n_trees = 10, search_k = 5500
#   3. n_trees = 10, search_k = 7000
#   4. n_trees = 10, search_k = 4500
#   5. n_trees = 10, search_k = 7500
#   6. n_trees = 10, search_k = 6000
#   7. n_trees = 10, search_k = 3500
#   8. n_trees = 10, search_k = 4000
#   9. n_trees = 10, search_k = 5000
#   10. n_trees = 10, search_k = 8000
