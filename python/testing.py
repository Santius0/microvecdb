import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from memory_profiler import profile

# import microvecdb as mvdb_c
# import pymicrovecdb as mvdb
from pymicrovecdb import MVDB, DATA_TYPE, IndexType

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
    db_ = MVDB(DATA_TYPE.FLOAT)
    # db_.create(IndexType.FAISS_FLAT, 96, "./test_faiss_flat_db")
    db_.open("./test_faiss_flat_db")
    db_.topk(5)
    print(db_)
    # unittest.main()

if __name__ == '__main__':
    main()
