import os
import gc
import shutil
import time
import nni
import numpy as np
from pymicrovecdb import mvdb, utils as mv_utils
from memory_profiler import memory_usage

datasets = [
    {'name': 'sift1M', 'base': '../../../ann_data/sift1M/sift/sift_base.fvecs', 'query': '../../../ann_data/sift1M/sift/sift_query.fvecs', 'ground': '../../../ann_data/sift1M/sift/sift_groundtruth.ivecs'},
    {'name': 'gist1M', 'base': '../../../ann_data/gist1M/gist/gist_base.fvecs', 'query': '../../../ann_data/gist1M/gist/gist_query.fvecs', 'ground': '../../../ann_data/gist1M/gist/gist_groundtruth.ivecs'},
    {'name': 'deep1M', 'base': '../../../ann_data/deep1M/deep1M_base.fvecs', 'query': '../../../ann_data/deep10M/deep1B_queries.fvecs', 'ground': '../../../ann_data/deep1M/deep1M_groundtruth.ivecs'},
    {'name': 'deep10M', 'base': '../../../ann_data/deep10M/deep10M_base.fvecs', 'query': '../../../ann_data/deep10M/deep1B_queries.fvecs', 'ground': '../../../ann_data/deep10M/deep10M_groundtruth.ivecs'},
]

k_values = [1, 10, 50, 100]

def recall1(qr, gt, k = 100):
    recalls = np.zeros(len(qr))
    for i in range(len(qr)):
        actual = 0
        for id in qr[i][:k]:
            if id in gt[i][:k]:
                actual += 1
        recalls[i] = actual
    return np.mean(recalls) / float(k)

def recall2(qr, gt, k = 100):
    return (qr[:, :k] == gt[:, :1]).sum() / float(qr.shape[0])

def delete_directory(path, verbose=False):
    if os.path.exists(path):
        shutil.rmtree(path)
        if verbose:
            print(f"Directory '{path}' has been deleted.")
    elif verbose:
        print(f"Directory '{path}' does not exist.")

def wrapper(db, dataset, k):
    return db.topk(query=mv_utils.read_vector_file(dataset['query']), k=k)

def normalize(value, min_value, max_value):
    return max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))
def objective(latency, recall, dram, weight_latency=0.3, weight_recall=0.3, weight_dram=0.4, min_latency=0, max_latency=600, min_dram=0, max_dram=2048):
    latency_norm = normalize(latency, min_latency, max_latency)
    dram_norm = normalize(dram, min_dram, max_dram)
    objective_value = (weight_latency * latency_norm) + (weight_dram * dram_norm) - (weight_recall * recall) # recall is already normalised between 0 and 1
    print(f"latency_norm: {latency_norm}")
    print(f"normalised_peak_dram: {dram_norm}")
    print(f"objective_value: {objective_value}")
    return objective_value

def evaluate_annoy(n_trees):
    gc.collect()
    perf = np.zeros(len(datasets) * len(k_values))
    for i_d, dataset in enumerate(datasets):
        gc.collect()
        index_path = f"./annoy_{dataset['name']}_{n_trees}"
        delete_directory(index_path, verbose=True)
        db = mvdb.MVDB()
        print(f"building for {dataset['name']} ...")
        db.create(
            index_type=mvdb.IndexType.ANNOY,
            dims=mv_utils.get_fvecs_dim_size(dataset['query']),
            path=index_path,
            initial_data_path=dataset['base'],
            n_trees = n_trees
        )
        print(f"build for {dataset['name']} @ f{index_path} successful")
        for i_k, k in enumerate(k_values):
            gc.collect()
            start_time = time.time()
            # ids, dists, mem_usage = db.topk(query=mv_utils.read_vector_file(dataset['query']), k=k)
            peak_dram, results = memory_usage((wrapper, (db, dataset, k)), retval=True, max_usage=True)
            ids = results[0]
            # dists = results[1]
            # mem_usage = results[2]
            latency = time.time() - start_time
            recall = (recall1(dataset['query'], dataset['ground'], k = k_values[i_k])  + recall2(ids, gt=mv_utils.read_vector_file(dataset['ground']), k=k))/2
            print(f"k: {k}")
            print(f"latency: {latency}")
            print(f"peak_dram: {peak_dram}")
            print(f"recall: {recall}")
            perf[i_d + i_k] = objective(latency, recall, peak_dram)
            print(f"perf[i_d + i_k]: {perf[i_d + i_k]}")
        delete_directory(index_path, verbose=True)
    return np.mean(perf)


if __name__ == "__main__":
    params = nni.get_next_parameter()
    n_trees = int(params["n_trees"])
    print("Running n_trees = {}".format(n_trees))
    performance = float(evaluate_annoy(n_trees))
    print("Performance: {}".format(performance))
    nni.report_final_result(performance)
