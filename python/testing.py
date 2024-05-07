import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from memory_profiler import profile

# import microvecdb as mvdb_c
# import pymicrovecdb as mvdb
from pymicrovecdb import MVDB, DATA_TYPE, IndexType

import os
import shutil

def delete_directory(path):
    # Check if the directory exists
    if os.path.exists(path):
        # Remove the directory and all its contents
        shutil.rmtree(path)
        print(f"Directory '{path}' has been deleted.")
    else:
        print(f"Directory '{path}' does not exist.")
# class TestMVDB(unittest.TestCase):
#     def setUp(self):
#         self.data_type = DATA_TYPE.INT32
#         self.index_type = IndexType.SPANN
#         self.dims = np.int64(128)
#         self.path = "./python_db_test"
#         self.initial_data_path = ""
#         self.initial_data = np.array([1, 2, 3], dtype=np.int32)
#         self.initial_data_size = np.int64(3)
#
#         # Mocking the C-extension module functions
#         self.mvdb_init_mock = patch('microvecdb.MVDB_init', return_value=MagicMock()).start()
#         self.mvdb_create_mock = patch('microvecdb.MVDB_create').start()
#         self.mvdb_open_mock = patch('microvecdb.MVDB_open').start()
#         self.mvdb_topk_mock = patch('microvecdb.MVDB_topk', return_value=[1, 2, 3]).start()
#
#         self.mvdb = MVDB(self.data_type)
#
#     def tearDown(self):
#         patch.stopall()
#
#     def test_init(self):
#         self.mvdb_init_mock.assert_called_once_with(self.data_type)
#         self.assertEqual(self.mvdb.data_type, self.data_type)
#
#     def test_create(self):
#         self.mvdb.create(self.index_type, self.dims, self.path, self.initial_data_path, self.initial_data, self.initial_data_size)
#         self.mvdb_create_mock.assert_called_once_with(self.data_type, self.mvdb.mvdb_, self.index_type, self.dims, self.path, self.initial_data_path, self.initial_data, self.initial_data_size, None)
#
#     def test_open(self):
#         self.mvdb.open(self.path)
#         self.mvdb_open_mock.assert_called_once_with(self.data_type, self.mvdb.mvdb_, self.path)
#
#     def test_topk(self):
#         k = np.int64(10)
#         c = np.float(200.0)
#         result = self.mvdb.topk(k, c)
#         self.mvdb_topk_mock.assert_called_once_with(self.data_type, self.mvdb.mvdb_, k, c)
#         self.assertEqual(result, [1, 2, 3])

@profile
def main():
    dims = 3
    test_data = np.array([
        1.1, 2.2, 3.3,
        1.2, 2.3, 3.4
    ], dtype=np.float32)
    db_ = MVDB(DATA_TYPE.FLOAT)
    delete_directory("./test_faiss_flat_db")
    # db_.create(IndexType.FAISS_FLAT, 2, "./test_faiss_flat_db", "", test_data, int(len(test_data)/dims))
    db_.create(IndexType.FAISS_FLAT, 128, "./test_faiss_flat_db", "../SPTAG/datasets/sift/sift_base.fvecs")
    db_.open("./test_faiss_flat_db")
    res = db_.topk(1, np.array([1.0, 1.1, 1.2], dtype=np.float32))
    print(db_)
    print(res)
    # unittest.main()

if __name__ == '__main__':
    main()
