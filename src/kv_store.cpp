#include "kv_store.hpp"
#include "utils.hpp"
#include <stdexcept>
#include <string>

namespace mvdb {

    KvStoreMetadata::KvStoreMetadata(const rocksdb::Options& options) {
        create_if_missing = options.create_if_missing;
    }

    void KvStoreMetadata::serialize(std::ostream& out) const {
        serializeNumeric(out, create_if_missing);
    }

    void KvStoreMetadata::deserialize(std::istream& in) {
        create_if_missing = deserializeNumeric(in);
    }


    KvStore::KvStore(const std::string& path, const bool createNew) {
        options.create_if_missing = createNew;
        rocksdb::DB* raw_db_pointer = nullptr;
        if (const rocksdb::Status status = rocksdb::DB::Open(options, path, &raw_db_pointer); !status.ok())
            throw std::runtime_error("Failed to open/create database: " + status.ToString());
        db.reset(raw_db_pointer);
    }

    bool KvStore::put(const std::string& key, const std::string& value) const {
        const rocksdb::Status status = db->Put(rocksdb::WriteOptions(), key, value);
        return status.ok();
    }

    bool KvStore::putAutoKey(const std::string& value) const {
        return KvStore::put(std::to_string(db->GetLatestSequenceNumber()), value);
    }

    bool KvStore::putMany(const std::vector<std::pair<std::string, std::string>>& pairs) const {
        rocksdb::WriteBatch batch;
        for (const auto& pair : pairs) {
            batch.Put(pair.first, pair.second);
        }
        const rocksdb::Status status = db->Write(rocksdb::WriteOptions(), &batch);
        return status.ok();
    }

    bool KvStore::putManyAutoKey(const std::vector<std::string>&values) const {
        std::vector<std::pair<std::string, std::string>> pairs;
        for(int i = 0; i < values.size(); i++) {
            std::string key_str = std::to_string(this->db->GetLatestSequenceNumber() + i);
            pairs.emplace_back(key_str, values[i]);
        }
        return KvStore::putMany(pairs);
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
