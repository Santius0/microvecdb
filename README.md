# MicroVecDB

## Requirements
1. C++17
2. [RocksDB](https://github.com/facebook/rocksdb/)
3. [MKL (used to build FAISS)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.1eup6s)
4. [FAISS](https://github.com/facebookresearch/faiss/)

### Installing
#### RocksDB (prebuilt binaries):
1. Install RocksDB using: `sudo apt-get install librocksdb-dev` and `sudo apt-get install rocksdb-tools` to gain access 
to tools included in the rocksdb build such as db_bench.
2. Verify RocksDB installation and version: `apt-cache policy librocksdb-dev`

#### FAISS (built from source):
1. [Download FAISS](https://github.com/facebookresearch/faiss/) 
2. Build make using `sudo cmake -B build -DBUILD_TESTING=OFF -DFAISS_ENABLE_GPU=OFF -DFAISS_OPT_LEVEL=avx2 -DFAISS_ENABLE_C_API=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release`
3. Make using `sudo make -C build -j faiss_c`

### Compilation
`g++ main.cpp -o main -lrocksdb -std=c++11 -lpthread -ldla` or use cmake

[//]: # (Might need `sudo apt-get install libopenblas-dev`. Not sure if just installing MKL gived everything needed.)
[//]: # (sudo ln -s /opt/cmake-3.*your_version*/bin/* /usr/local/bin)

