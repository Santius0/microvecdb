
class Collection:
    def __init__(self, name):
        self.name = name
        self.kv_store_ = None
        self.vector_index = None
        self.vectorizer = None

    def add_vector(self, vector):
        pass

    def remove_vector(self, idx):
        pass

    def add_data(self, data):
        pass

    def remove_data(self, key):
        pass

    def add(self):
        pass

    def remove(self):
        pass

class CollectionList:
    def __init__(self):
        self.collections_

class MicroVecDB:
    def __init__(self):
        self.collections = CollectionList()

