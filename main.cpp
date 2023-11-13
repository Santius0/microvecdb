#include <iostream>
#include <rocksdb/db.h>
#include "RocksDBDatabase.h"

int main() {
    RocksDBDatabase *rdb = RocksDBDatabase::getInstance("./testdb");
    RocksDBDatabase::destroyInstance();
    rdb->getStatus();
//    rocksdb::DB* db;
//    rocksdb::Options options;
//    options.create_if_missing = true;
//    rocksdb::Status status =
//            rocksdb::DB::Open(options, "./testdb", &db);
//    assert(status.ok());
//    if (!status.ok()) std::cerr << status.ToString() << std::endl;
//
//    std::string value = "key 1's value";
//    std::string fetched;
//    rocksdb::Status s = db->Put(rocksdb::WriteOptions(), "key1", value);
//    if (s.ok()) s = db->Get(rocksdb::ReadOptions(), "key1", &fetched);
//    std::cout << "key1" << " => " << fetched << std::endl;
//    if (s.ok()) s = db->Delete(rocksdb::WriteOptions(), "key1");
//
//    delete db;
    return 0;
}
