import microvecdb as mvdb
import numpy as np


class DB:
    def __init__(self, db_name: str, db_path: str, dims: int):
        self.db_name = db_name
        self.db_path = db_path
        self.dims = dims
        self.db = mvdb.DB_create(db_name, db_path, dims)

    def add_vector(self, v: np.ndarray):
        return mvdb.DB_add_vector(self.db, v, int(len(v)/self.dims))

    def add_data(self, d):
        raise NotImplementedError("add data function not implemented")

    def add_data_with_vector(self, d, v: np.ndarray):
        raise NotImplementedError("add data with vector not implemented")

    def search_with_vector(self, q: np.ndarray, k: int = 5):
        return mvdb.DB_search_with_vector(self.db, int(len(q)/self.dims), q, k)

    def search(self, d):
        raise NotImplementedError("generic search with data function not implemented")

    def start(self, db_path: str):
        raise NotImplementedError("server start not implemented")

    def connect(self, db_path: str):
        raise NotImplementedError("server connect not implemented")
