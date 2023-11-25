#ifndef CONFLUXDB_CONFLUXSTORAGE_H
#define CONFLUXDB_CONFLUXSTORAGE_H

#include <rocksdb/db.h>

// Class to manage rocksdb database object using singleton design pattern
class ConfluxStorage {
private:
    explicit ConfluxStorage(const std::string &db_path);
    ~ConfluxStorage();

    static ConfluxStorage *instance;

    rocksdb::DB *db{};
    rocksdb::Options options;
    rocksdb::Status status;

public:
    ConfluxStorage(const ConfluxStorage&) = delete;
    ConfluxStorage& operator=(const ConfluxStorage&) = delete;

    static ConfluxStorage *getInstance(const std::string &db_path);
    static void destroyInstance();

    rocksdb::Status getStatus();
    rocksdb::Options getOptions();

    bool put(const std::string& key, const std::string& put_value);
    bool get(const std::string& key, std::string &fetched_value);

    unsigned long get_latest_sequence_number();
};

std::string compressData(const std::string& data);
std::string decompressData(const std::string& compressedData);

#endif //CONFLUXDB_CONFLUXSTORAGE_H
