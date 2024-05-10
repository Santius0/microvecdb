import numpy as np
from memory_profiler import profile

# import microvecdb as mvdb_c
# import pymicrovecdb as mvdb
from pymicrovecdb import MVDB, DataType, IndexType

import os
import shutil
import struct

def delete_directory(path):
    # Check if the directory exists
    if os.path.exists(path):
        # Remove the directory and all its contents
        shutil.rmtree(path)
        print(f"Directory '{path}' has been deleted.")
    else:
        print(f"Directory '{path}' does not exist.")


def read_vector_file(filename, dtype = np.float32):
    """
    Read vectors from a file with .fvecs or .ivecs extension.

    Parameters:
    - filename (str): Path to the file to read.

    Returns:
    - numpy.ndarray: Array of vectors stored in the file.
    """
    # Determine the type based on file extension
    file_type = 'f' if filename.endswith('.fvecs') else 'i'
    dtype = np.float32 if file_type == 'f' else np.int32

    vectors = []
    with open(filename, 'rb') as file:
        while True:
            # Read the dimensionality of the vector (first 4 bytes)
            dim_bytes = file.read(4)
            if not dim_bytes:
                break
            dimension = struct.unpack('i', dim_bytes)[0]

            # Read the vector data
            count = dimension
            if file_type == 'f':
                vector_bytes = file.read(4 * count)  # Each float takes 4 bytes
                vector = struct.unpack(f'{count}f', vector_bytes)
            else:
                vector_bytes = file.read(4 * count)  # Each int takes 4 bytes
                vector = struct.unpack(f'{count}i', vector_bytes)

            vectors.append(vector)

    return np.array(vectors, dtype=dtype)



# @profile
def main():
    dims = 3
    test_data = np.array([
        1.1, 2.2, 3.3,
        1.2, 2.3, 4.4,
        8.2, 3.3, 6.4,
        4.2, 5.3, 3.4
    ], dtype=np.float32)
    db_ = MVDB(DataType.FLOAT)
    # delete_directory("./test_faiss_flat_db")
    # db_.create(IndexType.ANNOY, dims, "./test_faiss_flat_db", "", test_data, int(len(test_data)/dims))
    # db_.create(IndexType.ANNOY, 128, "./test_faiss_flat_db", "../data/sift1m/sift_base.fvecs")
    # print(f"num_items: {db_.num_items()}")
    db_.open("./test_faiss_flat_db")
    res = db_.topk(2, np.array([
                                1.0, 1.1, 1.2,
                                8.2, 3.3, 6.3], dtype=np.float32))
    # queries = read_vector_file("../data/sift1m/sift_query.fvecs")
    # query_vector = queries[10]
    # print(query_vector)
    # res = db_.topk(1, query_vector, dtype=np.float32)
    # print(db_)
    print(res)
    # unittest.main()

if __name__ == '__main__':
    main()
