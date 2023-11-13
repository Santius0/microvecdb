#ifndef CONFLUXDB_ROCKSDBDATABASE_H
#define CONFLUXDB_ROCKSDBDATABASE_H

#include <rocksdb/db.h>

// Class to manage rocksdb database object using singleton design pattern
class RocksDBDatabase {
private:
    explicit RocksDBDatabase(const std::string &db_path);
    ~RocksDBDatabase();

    static RocksDBDatabase *instance;

    rocksdb::DB *db{};
    rocksdb::Options options;
    rocksdb::Status status;
public:
    RocksDBDatabase(const RocksDBDatabase&) = delete;
    RocksDBDatabase& operator=(const RocksDBDatabase&) = delete;

    static RocksDBDatabase *getInstance(const std::string &db_path);
    static void destroyInstance();

    rocksdb::Status getStatus();
    rocksdb::Options getOptions();
};


#endif //CONFLUXDB_ROCKSDBDATABASE_H
