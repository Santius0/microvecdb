#include "kv_store.h"
#include <stdexcept>

namespace mvdb {

    KvStore::KvStore(const std::string& path, const bool createNew, const bool overwrite) {
        options.create_if_missing = createNew;

        if (overwrite) {
            // Delete the existing database if it exists and overwrite is true;
            if (const rocksdb::Status status = rocksdb::DestroyDB(path, options); !status.ok() && !status.IsNotFound()) {
                throw std::runtime_error("Failed to destroy existing database: " + status.ToString());
            }
        }

        // Open the database
        rocksdb::DB* raw_db_pointer = nullptr;
        if (const rocksdb::Status status = rocksdb::DB::Open(options, path, &raw_db_pointer); !status.ok()) {
            throw std::runtime_error("Failed to open/create database: " + status.ToString());
        }
        db.reset(raw_db_pointer);
    }

    bool KvStore::put(const std::string& key, const std::string& value) const {
        const rocksdb::Status status = db->Put(rocksdb::WriteOptions(), key, value);
        return status.ok();
    }

    bool KvStore::putMany(const std::vector<std::pair<std::string, std::string>>& pairs) const {
        rocksdb::WriteBatch batch;
        for (const auto& pair : pairs) {
            batch.Put(pair.first, pair.second);
        }
        const rocksdb::Status status = db->Write(rocksdb::WriteOptions(), &batch);
        return status.ok();
    }

    std::string KvStore::get(const std::string& key) const {
        std::string value;
        if (const rocksdb::Status status = db->Get(rocksdb::ReadOptions(), key, &value); status.ok())
            return value;
        return "";
    }

    bool KvStore::remove(const std::string& key) const {
        const rocksdb::Status status = db->Delete(rocksdb::WriteOptions(), key);
        return status.ok();
    }

}
