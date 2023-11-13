#include "RocksDBDatabase.h"

RocksDBDatabase::RocksDBDatabase(const std::string &db_path){
    this->options.create_if_missing = true;
    rocksdb::DB::Open(this->options, db_path, &db);
}

RocksDBDatabase::~RocksDBDatabase(){
    if(this->db != nullptr) {
        delete this->db;
        this->db = nullptr;
    }
}

RocksDBDatabase* RocksDBDatabase::getInstance(const std::string &db_path){
    if (instance == nullptr)
        instance = new RocksDBDatabase(db_path);
    return instance;
}

void RocksDBDatabase::destroyInstance() {
    if(instance != nullptr) {
        delete instance;
        instance = nullptr;
    }
}

rocksdb::Status RocksDBDatabase::getStatus(){
    return this->status;
}

rocksdb::Options RocksDBDatabase::getOptions(){
    return this->options;
}