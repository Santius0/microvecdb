import os
import gc
import shutil
import time
import nni
import numpy as np
from pymicrovecdb import mvdb, utils as mv_utils
from memory_profiler import memory_usage

datasets = [
    # {'name': 'deep10M', 'base': '../../../ann_data/deep10M/deep10M_base.fvecs', 'query': '../../../ann_data/deep10M/deep1B_queries.fvecs', 'ground': '../../../ann_data/deep10M/deep10M_groundtruth.ivecs'},
    {'name': 'sift1M', 'base': '../../../ann_data/sift1M/sift/sift_base.fvecs', 'query': '../../../ann_data/sift1M/sift/sift_query.fvecs', 'ground': '../../../ann_data/sift1M/sift/sift_groundtruth.ivecs'},
    {'name': 'gist1M', 'base': '../../../ann_data/gist1M/gist/gist_base.fvecs', 'query': '../../../ann_data/gist1M/gist/gist_query.fvecs', 'ground': '../../../ann_data/gist1M/gist/gist_groundtruth.ivecs'},
    {'name': 'deep1M', 'base': '../../../ann_data/deep1M/deep1M_base.fvecs', 'query': '../../../ann_data/deep10M/deep1B_queries.fvecs', 'ground': '../../../ann_data/deep1M/deep1M_groundtruth.ivecs'},
]

k_values = [1, 10, 50, 100]

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

def objective(latency, recall, dram, weight_latency=0.2, weight_recall=0.4, weight_dram=0.4, min_latency=0, max_latency=30, min_dram=0, max_dram=1536):
    latency_norm = normalize(latency, min_latency, max_latency)
    dram_norm = normalize(dram, min_dram, max_dram)
    objective_value = (weight_recall * recall) - (weight_latency * latency_norm) - (weight_dram * dram_norm)
    print(f"Latency Normalized: {latency_norm:.4f}")
    print(f"DRAM Normalized: {dram_norm:.4f}")
    print(f"Objective Value: {objective_value:.4f}")
    return objective_value

def evaluate_annoy(n_trees_, search_k_):
    gc.collect()
    latencies = np.zeros(len(datasets) * len(k_values))
    drams = np.zeros(len(datasets) * len(k_values))
    recall1s = np.zeros(len(datasets) * len(k_values))
    recall2s = np.zeros(len(datasets) * len(k_values))
    recalls = np.zeros(len(datasets) * len(k_values))
    objectives = np.zeros(len(datasets) * len(k_values))
    for i_d, dataset in enumerate(datasets):
        gc.collect()
        index_path = f"./annoy_{dataset['name']}_{n_trees_}"
        # delete_directory(index_path, verbose=True)  # Commented out code
        db = mvdb.MVDB()
        if not os.path.exists(index_path):
            print(f"Building index for {dataset['name']}...")
            db.create(
                index_type=mvdb.IndexType.ANNOY,
                dims=mv_utils.get_fvecs_dim_size(dataset['query']),
                path=index_path,
                initial_data_path=dataset['base'],
                n_trees=n_trees_,
                n_threads=1 if dataset['name'] == 'deep10M' or (dataset['name'] == 'gist1M' and n_trees_ == 1000) else -1
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
            peak_dram, results = memory_usage((wrapper, (db, dataset, k, {'search_k': search_k_, 'n_threads': 12})), retval=True, max_usage=True)
            ids = results[0]
            latency = time.time() - start_time
            recall1_ = recall1(qr=ids, gt=ground, k=k)
            recall2_ = recall2(qr=ids, gt=ground, k=k)
            recall = (recall1_ + recall2_)/2
            latencies[pos] = -latency
            drams[pos] = -peak_dram
            recall1s[pos] = recall1_
            recall2s[pos] = recall2_
            recalls[pos] = recall
            objective_val = objective(latency, recall, peak_dram)
            objectives[pos] = objective_val

            intermediate = {'-latency': -latency, '-dram': -peak_dram, 'reacall1': recall1_, 'reacall2': recall2_, 'recall': recall, 'default': objective_val}
            print(f"Intermediate Performance: {intermediate}")
            nni.report_intermediate_result(intermediate)

            print("\n")
        # delete_directory(index_path, verbose=True)
    return {'-latency': np.mean(latencies), '-dram': np.mean(drams), 'reacall1': np.mean(recall1s), 'reacall2': np.mean(recall2s),'recall': np.mean(recalls), 'default': np.mean(objectives)}



if __name__ == "__main__":
    params = nni.get_next_parameter()
    n_trees = int(params["n_trees"])
    search_k = int(params["search_k"])
    print(f"Running with parameters:\n\tn_trees = {n_trees}\n\tsearch_k = {search_k}")
    performance = evaluate_annoy(n_trees_=n_trees, search_k_=search_k)
    print(f"Hyperparameters: n_trees = {n_trees}, search_k = {search_k}")
    print(f"Final Performance: {performance}")
    nni.report_final_result(performance)