import numpy as np
from memory_profiler import profile

# import pymicrovecdb as mvdb_c
from pymicrovecdb import mvdb

import os
import shutil
import struct

SIFT1M_BASE = "../data/sift1m/sift_base.fvecs"
SIFT1M_QUERY = "../data/sift1m/sift_query.fvecs"
SIFT1M_TRUTH = "../data/sift1m/sift_groundtruth.ivecs"

def test_create_function(db_):
    initial_data = np.random.rand(100, 64).astype(np.float32)  # 100 random vectors of dimension 64
    vecs = mvdb.read_vector_file(SIFT1M_BASE)

    # FAISS_FLAT
    db_.create(mvdb.IndexType.FAISS_FLAT, 64, "./indexes/faissflat_64_rand_direct_index", initial_data=initial_data)         # direct numpy array input
    db_.create(mvdb.IndexType.FAISS_FLAT, 128, "./indexes/faissflat_128_sift1m_file_index", initial_data_path=SIFT1M_BASE)   # .xvec file input
    db_.create(mvdb.IndexType.FAISS_FLAT, 128, "./indexes/faissflat_128_sift1m_direct_index", initial_data=vecs)             # loading from xvecs and then doing direct numpy array input

    # ANNOY
    db_.create(mvdb.IndexType.ANNOY, 64, "./indexes/annoy_64_rand_direct_index", initial_data=initial_data, n_trees=10, n_threads = -1)
    db_.create(mvdb.IndexType.ANNOY, 128, "./indexes/annoy_128_sift1m_file_index", initial_data_path=SIFT1M_BASE, n_trees=40, n_threads = -1)
    db_.create(mvdb.IndexType.ANNOY, 128, "./indexes/annoy_128_sift1m_direct_index", initial_data=vecs, n_trees=25, n_threads = 6)

    # SPANN
    db_.create(mvdb.IndexType.SPANN, 64, "./indexes/spann_64_rand_direct_index", initial_data=initial_data)
    db_.create(mvdb.IndexType.SPANN, 128, "./indexes/spann_128_sift1m_file_index", initial_data_path=SIFT1M_BASE)
    db_.create(mvdb.IndexType.SPANN, 128, "./indexes/spann_128_sift1m_direct_index", initial_data=vecs)


def main():
    db = mvdb.MVDB()
    test_create_function(db)

if __name__ == '__main__':
    main()
