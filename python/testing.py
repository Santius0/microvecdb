import os
import shutil
import numpy as np
from memory_profiler import profile

from pymicrovecdb import mvdb, utils


SIFT1M_BASE = "../data/sift1M/sift/sift_base.fvecs"
SIFT1M_QUERY = "../data/sift1M/sift/sift_query.fvecs"
SIFT1M_TRUTH = "../data/sift1M/sift/sift_groundtruth.ivecs"

def test_create_function(db_):
    initial_data = np.random.rand(100, 64).astype(np.float32)  # 100 random vectors of dimension 64
    vecs = utils.read_vector_file(SIFT1M_BASE)

    # FAISS_FLAT
    # db_.create(mvdb.IndexType.FAISS_FLAT, 64, "./indexes/faissflat_64_rand_direct_index", initial_data=initial_data)         # direct numpy array input
    # db_.create(mvdb.IndexType.FAISS_FLAT, 128, "./indexes/faissflat_128_sift1m_file_index", initial_data_path=SIFT1M_BASE)   # .xvec file input
    # db_.create(mvdb.IndexType.FAISS_FLAT, 128, "./indexes/faissflat_128_sift1m_direct_index", initial_data=vecs)             # loading from xvecs and then doing direct numpy array input

    # ANNOY
    # db_.create(mvdb.IndexType.ANNOY, 64, "./indexes/annoy_64_rand_direct_index", initial_data=initial_data, n_trees=10, n_threads = -1)
    # db_.create(mvdb.IndexType.ANNOY, 128, "./indexes/annoy_128_sift1m_file_index", initial_data_path=SIFT1M_BASE, n_trees=40, n_threads = -1)
    # db_.create(mvdb.IndexType.ANNOY, 128, "./indexes/annoy_128_sift1m_direct_index", initial_data=vecs, n_trees=25, n_threads = 6)

    # SPANN
    # db_.create(mvdb.IndexType.SPANN, 64, "./indexes/spann_64_rand_direct_index", initial_data=initial_data, build_config_path="/home/santius/microvecdb/SPTAG/buildconfig.ini")
    db_.create(mvdb.IndexType.SPANN, 128, "./indexes/spann_128_sift1m_file_index", initial_data_path=SIFT1M_BASE, build_config_path="/home/santius/microvecdb/SPTAG/buildconfig.ini")
    # db_.create(mvdb.IndexType.SPANN, 128, "./indexes/spann_128_sift1m_direct_index", initial_data=vecs, build_config_path="/home/santius/microvecdb/SPTAG/buildconfig.ini")

def test_topk_function(db_):
    rand_queries = np.random.rand(10, 64).astype(np.float32)  # 10 random vectors of dimension 64
    queries = utils.read_vector_file(SIFT1M_QUERY)

    # FAISS_FLAT
    # db_.open("./indexes/faissflat_64_rand_direct_index")
    # db_.topk(query=rand_queries, k = 10)
    # db_.topk(query=queries, k = 10)
    # db_.topk(query_file=SIFT1M_QUERY, k = 10)

    # SPANN
    db_.open("./indexes/spann_128_sift1m_file_index")
    # print(db_.dims)
    # res = db_.topk(query=rand_queries, k = 10)
    # print(res)
    res = db_.topk(query=queries, k = 100)
    print(res)
    # res = db_.topk(query_file=SIFT1M_QUERY, k = 100)
    # print(res)
    print(res[0].shape)
    print(res[1].shape)
    # print(res)

queries = utils.read_vector_file("../../ann_data/deep10M/deep1B_queries.fvecs")

# @profile
def main():
    if os.path.exists("./test_annoy_idx"):
        shutil.rmtree("./test_annoy_idx")

    annoy_db = mvdb.MVDB()
    annoy_db.create(
      index_type=mvdb.IndexType.ANNOY,
      dims=128,
      path="./test_annoy_idx",
      initial_data=utils.read_vector_file("../../ann_data/sift10K/sift10K_base.fvecs")
    )
    print(annoy_db.num_items)
    print(annoy_db.dims)
    # print(queries)
    # print(len(queries))
    # print(len(queries[33]))
    # db1 = mvdb.MVDB()
    # db1.open("./indices/spann_sift10K_float32")
    # print(f'python num items {db1.num_items}')

    # db2 = mvdb.MVDB()
    # db2.open("./indices/spann_sift1M_float32")
    # print(f'python num items {db2.num_items}')
    # res = db_.topk(query=queries, k = 100)
    # print(res[0])
    # print(utils.read_vector_file("../../ann_data/deep1M/deep1M_groundtruth.ivecs"))
# test_create_function(db)
    # test_topk_function(db)

if __name__ == '__main__':
    main()
