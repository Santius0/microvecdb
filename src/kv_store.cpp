#include "kv_store.hpp"
#include "utils.hpp"
#include <stdexcept>
#include <string>
#include <utility>

namespace mvdb {

    KvStoreMetadata::KvStoreMetadata(std::string  dataDirectoryPath, const rocksdb::Options& options):
    dataDirectoryPath(std::move(dataDirectoryPath)) {
        options_ = options;
    }

    void KvStoreMetadata::serialize(std::ostream& out) const {
        serialize_string(out, dataDirectoryPath);
        serialize_numeric<bool>(out, options_.create_if_missing);
    }

    void KvStoreMetadata::deserialize(std::istream& in) {
        dataDirectoryPath = deserialize_string(in);
        options_.create_if_missing = deserialize_numeric<bool>(in);
    }

    KvStore::KvStore(const KvStoreMetadata& metadata) {
        options = metadata.options_;
        rocksdb::DB* raw_db_pointer = nullptr;
        if (const rocksdb::Status status = rocksdb::DB::Open(options, metadata.dataDirectoryPath, &raw_db_pointer); !status.ok())
            throw std::runtime_error("Failed to open/create database: " + status.ToString());
        db.reset(raw_db_pointer);
    }

    bool KvStore::put(const std::string& key, const std::string& value) const {
        const rocksdb::Status status = db->Put(rocksdb::WriteOptions(), key, value);
        return status.ok();
    }

    int64_t KvStore::putAutoKey(const std::string& value) const {
        const auto key = static_cast<int64_t>(db->GetLatestSequenceNumber());
        if(KvStore::put(std::to_string(key), value))
            return key;
        return -1;
    }

//    bool KvStore::putMany(const std::vector<std::pair<std::string, std::string>>& pairs) const {
//        rocksdb::WriteBatch batch;
//        for (const auto& pair : pairs) {
//            batch.Put(pair.first, pair.second);
//        }
//        const rocksdb::Status status = db->Write(rocksdb::WriteOptions(), &batch);
//        return status.ok();
//    }

//    bool KvStore::putManyAutoKey(const std::vector<std::string>&values) const {
//        std::vector<std::pair<std::string, std::string>> pairs;
//        for(int i = 0; i < values.size(); i++) {
//            std::string key_str = std::to_string(this->db->GetLatestSequenceNumber() + i);
//            pairs.emplace_back(key_str, values[i]);
//        }
//        return KvStore::putMany(pairs);
//    }

    std::string KvStore::get(const std::string& key) const {
        std::string value;
        const rocksdb::Status status = db->Get(rocksdb::ReadOptions(), key, &value);
        if (status.ok())
            return value;
        return status.ToString();
    }

    bool KvStore::remove(const std::string& key) const {
        const rocksdb::Status status = db->Delete(rocksdb::WriteOptions(), key);
        return status.ok();
    }

}
