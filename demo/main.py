# pip install memory_profiler
# python -m memory_profiler script.py

import microvecdb
import numpy as np
# from microvecdb import hello_world

# microvecdb.hello_world("jimmy!")
# @profile
def main():
    index = microvecdb.FaissFlatIndex_create("./test_index", 5)
    microvecdb.FaissFlatIndex_open(index)
    query = np.array([1.2, 2.3, 3.4, 4.5], dtype=np.float32)
    microvecdb.FaissFlatIndex_add(index, query, 1)
    top_k = microvecdb.FaissFlatIndex_search(index, 5, query, 5)
    print(top_k)

if __name__ == '__main__':
    main()
