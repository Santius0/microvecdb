import microvecdb as mvdb
import numpy as np

class SearchResult:
    def __init__(self, search_result_obj):
        self.__search_result_obj = search_result_obj

    @property
    def ids(self):
        return self.__search_result_obj.ids_

    @property
    def distances(self):
        return self.__search_result_obj.distances_

    @property
    def data(self):
        return self.__search_result_obj.data_

    @property
    def size(self):
        return self.__search_result_obj.size_

    def __str__(self):
        return f"ids: {self.ids}\ndistances: {self.distances}\ndata: {self.data}"

class DB:
    def __init__(self, db_path, db_name, dims):
        self.db_path = db_path
        self.db_name = db_name
        self.dims = dims
        self.db = mvdb.DB_create(db_path, db_name, dims)

    def add_vector(self, v: np.ndarray):
        return mvdb.DB_add_vector(self.db, v, int(len(v)/self.dims))

    def search_with_vector(self, v:np.ndarray, k = 5, ret_data = False):
        res = mvdb.DB_search_with_vector(self.db, int(len(v)/self.dims), v, k, ret_data)
        return SearchResult(res)



def hello_world(msg):
    mvdb.hello_world(msg)