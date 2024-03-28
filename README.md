# MicroVecDB - CPP

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
2. Build make using `sudo cmake -B build -DBUILD_TESTING=OFF -DFAISS_ENABLE_GPU=OFF -DFAISS_OPT_LEVEL=generic -DFAISS_ENABLE_C_API=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Debug -DBLA_VENDOR=OpenBLAS`
3. Make using `sudo make -C build -j faiss_c`

### Compilation
`g++ main.cpp -o main -lrocksdb -std=c++11 -lpthread -ldla` or use cmake

#### Installing ZMQ
`sudo apt-get install libzmq3-dev`

#### Install Memory Profiler
`pip install memory_profiler`

Profiler Settings:
`valgrind --tool=massif --massif-out-file=./massif.out ./microvecdb_main`
`--leak-check=full --leak-resolution=med --track-origins=yes --vgdb=no`
`python -m memory_profiler main.py`

to tansfer files to and connect to the jetson nano, connect via it's micro-usb port then:

For example if your user name is nvidia for Jetson module;<br/>
    `scp <file_name> nvidia@192.168.55.1:/home/nvidia`

and also you connect to your Jetson module<br/>
    `ssh nvidia@192.168.55.1`

`sudo apt-get install libeigen3-dev`

Google benchmark: `sudo apt-get install libbenchmark-dev` 

Might need `sudo apt-get install libopenblas-dev libblas-dev libatlas-base-dev`. Not sure if just installing MKL gived everything needed.
[//]: # (sudo ln -s /opt/cmake-3.*your_version*/bin/* /usr/local/bin)
[//]: # (cmake-3.26.5-linux-x86_64/bin/cmake --build /home/santius/microvecdb/build --target all -- -j 10)
