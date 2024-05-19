import numpy as np
import time
from memory_profiler import profile, memory_usage

from pymicrovecdb import mvdb, utils

indices = {
    'spann_128_sift1m_file_index': {'type': mvdb.IndexType.SPANN, 'path': './indexes/spann_128_sift1m_file_index', 'dims': 128, 'misc.': ['n_threads=12'], 'dtype': mvdb.DataType.FLOAT},
}

query_sizes = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 3840, 5120, 6400, 7680, 8960, 10000]

k_values = [1, 10, 50, 100]

datasets = {
    # 'deep10K': {'query_path': 'deep10K', 'dims': 96, 'size': 10000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep100K': {'query_path': 'deep100K', 'dims': 96, 'size': 100000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep500K': {'query_path': 'deep500K', 'dims': 96, 'size': 500000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep1M': {'query_path': 'deep1M', 'dims': 96, 'size': 1000000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep5M': {'query_path': 'deep5M', 'dims': 96, 'size': 5000000, 'dtype': mvdb.DataType.FLOAT},
    # 'deep10M': {'query_path': 'deep10M', 'dims': 96, 'size': 10000000, 'dtype': mvdb.DataType.FLOAT},

    # 'sift10K': {'query_path': 'sift10K', 'dims': 128, 'size': 10000, 'dtype': mvdb.DataType.FLOAT},
    # 'sift100K': {'query_path': 'sift100K', 'dims': 128, 'size': 100000, 'dtype': mvdb.DataType.FLOAT},
    # 'sift500K': {'query_path': 'sift500K', 'dims': 128, 'size': 500000, 'dtype': mvdb.DataType.FLOAT},
    'sift1M': {'query_path': '../data/sift1M/sift/sift_query.fvecs', 'ground_path': '../data/sift1M/sift/sift_groundtruth.ivecs', 'dims': 128, 'size': 1000000, 'dtype': mvdb.DataType.FLOAT},
    # 'sift5M': {'query_path': 'sift5M', 'dims': 128, 'size': 5000000, 'dtype': mvdb.DataType.FLOAT},
    # 'sift10M': {'query_path': 'sift10M', 'dims': 128, 'size': 10000000, 'dtype': mvdb.DataType.FLOAT},
    #
    # 'gist10K': {'query_path': 'gist10K', 'dims': 960, 'size': 10000, 'dtype': mvdb.DataType.FLOAT},
    # 'gist100K': {'query_path': 'gist100K', 'dims': 960, 'size': 100000, 'dtype': mvdb.DataType.FLOAT},
    # 'gist500K': {'query_path': 'gist500K', 'dims': 960, 'size': 500000, 'dtype': mvdb.DataType.FLOAT},
    # 'gist1M': {'query_path': 'gist1M', 'dims': 960, 'size': 1000000, 'dtype': mvdb.DataType.FLOAT},
}

def topk_wrapper(db_, q, k):
    results = mvdb.tk(db_.mvdb_obj, q.flatten(), len(q), k)
    print('DONE')
    return results

def recall(qr, gt, k = 100):
    recalls1 = np.zeros(len(qr))
    recalls2 = np.zeros(len(qr))
    for i in range(len(qr)):
        actual = 0
        for id in qr[i][:k]:
            if id in gt[i][:k]:
                actual += 1
        recalls1[i] = actual
        recalls2[i] = actual/float(k)
    print("AAAAAAAAAAAAA")
    print(np.mean(recalls1) / float(k), np.std(recalls1) / float(k), recalls1)
    print("BBBBBBBBBBBBB")
    print(np.mean(recalls2), np.std(recalls2), recalls2)
    return None
def run_query(db_, query, ground, dataset, index, k = 100):
    start_time = time.time()
    # peak_dram, res = memory_usage((topk_wrapper, (db_, query, k)), retval=True, max_usage=True)
    # topk_wrapper(db_, query, k)
    query_time = time.time() - start_time
    # recall_ = recall(res[1], ground)

    # row_dict = cpu_env
    row_dict = {}
    # row_dict['dataset'] = dataset['name']
    # # row_dict['dims'] = dataset['dims']
    # # row_dict['index_size'] = dataset['size']
    # # row_dict['k'] = k
    # # row_dict['distance_metric'] = 'L2'
    # # # row_dict['energy_usage'] = ''
    # row_dict['peak_dram_(MB)'] = peak_dram
    # # row_dict['index'] = index['type'].value
    # row_dict['latency_(s)'] = query_time
    # # row_dict['recall'] = recall
    return row_dict

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

def main():
    db = mvdb.MVDB(mvdb.DataType.FLOAT)
    db.open('./indexes/spann_128_sift1m_file_index')
    queries = utils.read_vector_file('../data/sift1M/sift/sift_query.fvecs')
    q = queries[:500]
    print(db.mvdb_obj, q.flatten())
    # print(db.dims)
    # mvdb.tk(db.mvdb_obj, q.flatten())
    for dataset_key in datasets:
        print(f'Processing: {dataset_key}...')
        dataset = datasets[dataset_key]
        queries = utils.read_vector_file(dataset['query_path'])
        ground = utils.read_vector_file(dataset['ground_path'])
        for index_key in indices:
            index = indices[index_key]
            db = mvdb.MVDB(mvdb.DataType.FLOAT)
            db.open(index['path'])
            for q_size in query_sizes:
                q = queries[:q_size]
                gt = ground[:q_size]
                for k in k_values:
                    print(f'running: {dataset_key} => {index_key} => q_size = {q_size} => k = {k}')
                    # mvdb.tk(db.mvdb_obj, q.flatten(), q_size, k)
                    results = db.topk(query=q, k=k)
                    # peak_dram, res = memory_usage((topk_wrapper, (db, q, k)), retval=True, max_usage=True)
                    # print(peak_dram)
                    # topk_wrapper(db, q, k)
                    # res = run_query(db_=db, query=q, ground=gt, dataset=dataset, index=index, k=k)
                    print(f'complete')
    print("**DONE**")

if __name__ == '__main__':
    main()