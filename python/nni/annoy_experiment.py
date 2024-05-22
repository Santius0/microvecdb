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
    {'name': 'deep10M', 'base': '../../../ann_data/deep10M/deep10M.fvecs', 'query': '../../../ann_data/deep10M/deep1B_queries.fvecs', 'ground': '../../../ann_data/deep10M/deep1B_groundtruth.ivecs'},
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
def evaluate_annoy(n_trees):
    gc.collect()
    index_path = f"./annoy_nni_{os.getpid()}"
    delete_directory(index_path, verbose=True)
    max_dram = 2048 # 2 GB
    max_latency = 600 # 10 mins
    weight_latency = 0.3
    weight_recall = 0.4
    weight_dram = 0.3
    perf = np.zeros(len(datasets) * len(k_values))
    for i_d, dataset in enumerate(datasets):
        db = mvdb.MVDB()
        print(f"building for {dataset['name']} ...")
        db.create(
            index_type=mvdb.IndexType.ANNOY,
            dims=mv_utils.get_fvecs_dim_size(dataset['query']),
            path=index_path,
            initial_data=mv_utils.read_vector_file(dataset['base']),
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
            normalised_latency = latency / max_latency
            normalised_peak_dram = peak_dram / max_dram
            recall = (recall1(dataset['query'], dataset['ground'], k = k_values[i_k])  + recall2(ids, gt=mv_utils.read_vector_file(dataset['ground']), k=k))/2
            perf[i_d + i_k] = weight_recall * recall - weight_dram * normalised_peak_dram - weight_latency * normalised_latency
            print(f"k: {k}")
            print(f"latency: {latency}")
            print(f"normalised_latency: {normalised_latency}")
            print(f"peak_dram: {peak_dram}")
            print(f"normalised_men_usage: {normalised_peak_dram}")
            print(f"recall: {recall}")
            print(f"perf[i_d + i_k]: {perf[i_d + i_k]}")
            break
        delete_directory(index_path, verbose=True)
        break
    return np.mean(perf)


if __name__ == "__main__":
    params = nni.get_next_parameter()
    n_trees = int(params["n_trees"])
    print("Running n_trees = {}".format(n_trees))
    performance = float(evaluate_annoy(n_trees))
    print("Performance: {}".format(performance))
    nni.report_final_result(performance)
