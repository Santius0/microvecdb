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
    # print(f"query result({len(qr)} x {len(qr[0])}): {qr[:, :k]}")
    # print(f"ground truth({len(gt)} x {len(gt[0, :k])}): {gt[:, :k]}\n")
    recalls_ = np.zeros(len(qr))
    for i in range(len(qr)):
        actual = 0
        for id_ in qr[i, :k]:
            # print(id_)
            # print(gt[i, :k])
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

def objective(latency, recall, dram, weight_latency=0.3, weight_recall=0.6, weight_dram=0.4, min_latency=0, max_latency=600, min_dram=0, max_dram=2048):
    latency_norm = normalize(latency, min_latency, max_latency)
    dram_norm = normalize(dram, min_dram, max_dram)
    # objective_value = (weight_latency * latency_norm) + (weight_dram * dram_norm) - (weight_recall * recall)
    objective_value = (weight_dram * dram_norm) - (weight_recall * recall)
    print(f"Latency Normalized: {latency_norm:.4f}")
    print(f"DRAM Normalized: {dram_norm:.4f}")
    print(f"Objective Value: {objective_value:.4f}")
    return objective_value

def evaluate_annoy(n_trees, search_k):
    gc.collect()
    latencies = np.zeros(len(datasets) * len(k_values))
    drams = np.zeros(len(datasets) * len(k_values))
    recalls = np.zeros(len(datasets) * len(k_values))
    objectives = np.zeros(len(datasets) * len(k_values))
    for i_d, dataset in enumerate(datasets):
        gc.collect()
        index_path = f"./annoy_{dataset['name']}_{n_trees}"
        # delete_directory(index_path, verbose=True)  # Commented out code
        db = mvdb.MVDB()
        if not os.path.exists(index_path):
            print(f"Building index for {dataset['name']}...")
            db.create(
                index_type=mvdb.IndexType.ANNOY,
                dims=mv_utils.get_fvecs_dim_size(dataset['query']),
                path=index_path,
                initial_data_path=dataset['base'],
                n_trees=n_trees,
                n_threads=1 if dataset['name'] == 'deep10M' or (dataset['name'] == 'gist1M' and n_trees == 1000) else -1
            )
            print(f"Build successful for {dataset['name']} at {index_path}")
        else:
            print(f"Index at {index_path} already exists. No build needed.")
            db.open(index_path)

        for i_k, k in enumerate(k_values):
            print("\n")

            pos = i_d * len(k_values) + i_k
            print(f"pos [{pos}]")
            print(f"k: {k}")
            gc.collect()

            start_time = time.time()
            # ids, dists, mem_usage = db.topk(query=mv_utils.read_vector_file(dataset['query']), k=k)  # Commented out code
            peak_dram, results = memory_usage((wrapper, (db, dataset, k, {'search_k': search_k})), retval=True, max_usage=True)
            ids = results[0]
            # dists = results[1]  # Commented out code
            # mem_usage = results[2]  # Commented out code
            latency = time.time() - start_time
            # recall = (recall1(qr=ids, gt=mv_utils.read_vector_file(dataset['ground']), k=k) + recall2(qr=ids, gt=mv_utils.read_vector_file(dataset['ground']), k=k)) / 2  # Commented out code
            recall = recall1(qr=ids, gt=mv_utils.read_vector_file(dataset['ground']), k=k)
            latencies[pos] = latency
            drams[pos] = peak_dram
            recalls[pos] = recall
            objective_val = objective(latency, recall, peak_dram)
            objectives[pos] = objective_val

            intermediate = {'latency': latency, 'dram': peak_dram, 'recall': recall, 'default': objective_val}
            print(f"Intermediate Performance: {intermediate}")
            nni.report_intermediate_result(intermediate)

            print("\n")
        # delete_directory(index_path, verbose=True)  # Commented out code
    return {'latency': np.mean(latencies), 'dram': np.mean(drams), 'recall': np.mean(recalls), 'default': np.mean(objectives)}



if __name__ == "__main__":
    params = nni.get_next_parameter()
    n_trees = int(params["n_trees"])
    search_k = int(params["search_k"])
    print(f"Running with parameters:\n\tn_trees = {n_trees}\n\tsearch_k = {search_k}")
    performance = evaluate_annoy(n_trees=n_trees, search_k=search_k)
    print(f"Hyperparameters: n_trees = {n_trees}, search_k = {search_k}")
    print(f"Final Performance: {performance}")
    nni.report_final_result(performance)