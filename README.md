# ConfluxDB

### Installing 
Install RocksDB

1. Install RocksDB using: `sudo apt-get install librocksdb-dev` 
2. Verify RocksDB installation and version: `apt-cache policy librocksdb-dev`

### Compilation
`g++ main.cpp -o main -lrocksdb -std=c++11 -lpthread -ldla`

or

`make` with cmake file
