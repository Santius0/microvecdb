# ConfluxDB

### Installing 
Install RocksDB

1. Install RocksDB using: `sudo apt-get install librocksdb-dev` and `sudo apt-get install rocksdb-tools` to gain access 
to tools included in the rocksdb build such as db_bench.
2. Verify RocksDB installation and version: `apt-cache policy librocksdb-dev`

### Compilation
`g++ main.cpp -o main -lrocksdb -std=c++11 -lpthread -ldla`

or

`make` with the following line in cmake: `target_link_libraries(rocksdb)`

<br/><br/><br/><br/><br/>

for faiss we have to run sudo apt-get install libopenblas-dev,

sudo apt install intel-mkl then source /opt/intel/mkl/bin/mklvars.sh intel64


sudo ln -s /opt/cmake-3.*your_version*/bin/* /usr/local/bin


# building  faiss
sudo cmake -B build -DBUILD_TESTING=OFF -DFAISS_ENABLE_GPU=OFF -DFAISS_OPT_LEVEL=avx2 -DFAISS_ENABLE_C_API=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
sudo make -C build -j faiss_c