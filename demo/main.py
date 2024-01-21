# pip install memory_profiler
# python -m memory_profiler script.py

# import microvecdb as mvdb
from pymicrovecdb import *
import numpy as np
import struct
import gc
import time


def get_vector_count(filename):
    count = 0
    with open(filename, 'rb') as f:
        while True:
            bytes = f.read(4)  # Read the dimension (4 bytes)
            if not bytes:
                break  # End of file
            dim = struct.unpack('i', bytes)[0]
            f.seek(dim * 4, 1)  # Skip the vector data
            count += 1
    return count
def get_fvec_dims(filename):
    with open(filename, 'rb') as f:
        # Read the dimension of the first vector (4 bytes => integer)
        bytes = f.read(4)
        if not bytes:
            return None  # File is empty or not properly formatted
        dim = struct.unpack('i', bytes)[0]
        return dim

def read_fvecs(f, max_vectors=np.inf):
    i = 0
    while i < max_vectors:
        # Read the dimension of the vector (4 bytes => integer)
        bytes = f.read(4)
        if not bytes:
            break
        dim = struct.unpack('i', bytes)[0]
        # Read the vector (dim floats)
        vec = struct.unpack(f'{dim}f', f.read(dim * 4))
        yield vec
        i += 1

def index_fvecs():
    filename = '../benchmarks/data/deep10M.fvecs'
    # vector_count = get_vector_count(filename)
    dims = get_fvec_dims(filename)
    # print(f"The file contains {vector_count} vectors each of {dims} dimensions")
    db = DB("./deep10M_test_db", "deep10M_test_db", dims)


    batch_size = 500000
    total_vectors = 0

    with open(filename, 'rb') as f:
        for batch in iter(lambda: list(read_fvecs(f, max_vectors=batch_size)), []):
            vecs = np.array(batch, dtype=np.float32).flatten()
            db.add_vector(vecs)
            total_vectors += len(batch)
            print(f"Processed {total_vectors} vectors")
            del batch, vecs
            gc.collect()

def query():
    filename = '../benchmarks/data/deep1B_queries.fvecs'
    vector_count = get_vector_count(filename)
    dims = get_fvec_dims(filename)
    print(f"The file contains {vector_count} vectors each of {dims} dimensions")
    db = DB("./deep10M_test_db", "deep10M_test_db", dims)
    batch_size = 1
    total_vectors = 0
    with open(filename, 'rb') as f:
        for batch in iter(lambda: list(read_fvecs(f, max_vectors=batch_size)), []):
            vecs = np.array(batch, dtype=np.float32).flatten()
            # print(vecs)
            res = db.search_with_vector(vecs, 5, False)
            # print(res)
            total_vectors += len(batch)
            if(total_vectors >= 1):
                break
            print(f"Processed {total_vectors} vectors")
            del batch, vecs
            gc.collect()
@profile
def main():
    # start_time = time.time()
    query()
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Execution time: {elapsed_time} seconds")


if __name__ == '__main__':
    main()
