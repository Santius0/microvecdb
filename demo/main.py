# pip install memory_profiler
# python -m memory_profiler script.py

import microvecdb as mvdb
import numpy as np
# from microvecdb import hello_world

# microvecdb.hello_world("jimmy!")
# @profile
def main():
    index = mvdb.FaissFlatIndex_create("./test_index", 5)
    mvdb.FaissFlatIndex_open(index)
    # query = np.array([1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5, 1.2, 2.3, 3.4, 4.5], dtype=np.float32)
    # microvecdb.FaissFlatIndex_add(index, query, 3)
    # top_k = mvdb.FaissFlatIndex_search(index, 5, query, 10)
    # print(top_k)

if __name__ == '__main__':
    main()
