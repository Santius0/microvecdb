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
