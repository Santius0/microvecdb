import unittest
import numpy as np
import pymicrovecdb as mvdb
import os
import shutil

class TestDB(unittest.TestCase):
    def setUp(self):
        self.db_name = "test_db"
        self.db_path = "./test_db"
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        self.dims = 3
        self.db = mvdb.DB(self.db_name, self.db_path, self.dims)

    def test_add_vector(self):
        print("test_add_vector start")
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = self.db.add_vector(vector)
        self.assertTrue(result)
        print("test_add_vector complete")

    def test_search_with_vector(self):
        print("test_search_with_vector start")
        vector_to_add = np.array([4.0, 5.0, 6.0,
                                  1.0, 2.0, 3.0,
                                  7.0, 8.0, 9.0], dtype=np.float32)
        self.db.add_vector(vector_to_add)

        # Use a similar vector for the search to ensure a match
        query_vector = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        k = 5
        ids, distances = self.db.search_with_vector(query_vector, k)

        expected_ids = np.array([1, 0, 2], dtype=np.int64)
        expected_ids = np.append(expected_ids, [np.int64(-1) for i in range(k - len(vector_to_add)//self.dims)])
        expected_distances = np.array([1.73205, 3.4641, 8.66025], np.float32)
        expected_distances = np.append(expected_distances, [np.float32(-1.0) for i in range(k - len(vector_to_add)//self.dims)])
        self.assertEqual(len(ids), k)
        self.assertEqual(len(distances), k)
        self.assertTrue(np.array_equal(ids, expected_ids))
        self.assertTrue(np.allclose(distances, expected_distances, atol=1e-5))
        print("test_search_with_vector complete")

    def tearDown(self):
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

if __name__ == '__main__':
    unittest.main()
