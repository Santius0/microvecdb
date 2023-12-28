#include "kv_store.hpp"
#include "utils.hpp"
#include <stdexcept>
#include <string>
#include <utility>

namespace mvdb {
    std::ostream& operator<<(std::ostream& os, const KvStore& obj) {
        return os   << "data_dir_path_: " << obj.data_dir_path_ << std::endl
                    << "create_if_missing: " << obj.options_.create_if_missing << std::endl
                    ;
    }

    std::ostream& operator<<(std::ostream& os, const KvStore* obj) {
        return os << "*(" << *obj << ")";
    }

    void KvStore::serialize(std::ostream& out) const {
        serialize_string(out, data_dir_path_);
        serialize_numeric<bool>(out, options_.create_if_missing);
    }

    void KvStore::deserialize(std::istream& in) {
        data_dir_path_ = deserialize_string(in);
        options_.create_if_missing = deserialize_numeric<bool>(in);
    }

    KvStore::KvStore(std::string  data_dir_path, const bool& creating_if_missing, const bool& open): data_dir_path_(std::move(data_dir_path)){
        options_.create_if_missing = creating_if_missing;
        if(open) this->open();
    }

    void KvStore::open(const std::string &path) {
        if(!path.empty()) data_dir_path_ = path;
        rocksdb::DB* raw_db_pointer = nullptr;
        if (const rocksdb::Status status = rocksdb::DB::Open(options_, data_dir_path_, &raw_db_pointer); !status.ok())
            throw std::runtime_error("Failed to open/create database: " + status.ToString());
        db_.reset(raw_db_pointer);
        is_open_ = true;
    }

    void KvStore::close() {
        db_.reset();
        is_open_ = false;
    }

    bool KvStore::is_open() const {
        return is_open_;
    }

    bool KvStore::put(const std::string& key, const std::string& value) const {
        const rocksdb::Status status = db_->Put(rocksdb::WriteOptions(), key, value);
        return status.ok();
    }

    int64_t KvStore::putAutoKey(const std::string& value) const {
        const auto key = static_cast<int64_t>(db_->GetLatestSequenceNumber());
        if(KvStore::put(std::to_string(key), value))
            return key;
        return -1;
    }

    bool KvStore::putMany(const std::vector<std::pair<std::string, std::string>>& pairs) const {
        rocksdb::WriteBatch batch;
        for (const auto& pair : pairs) {
            batch.Put(pair.first, pair.second);
        }
        const rocksdb::Status status = db_->Write(rocksdb::WriteOptions(), &batch);
        return status.ok();
    }

    bool KvStore::putManyAutoKey(const std::vector<std::string>&values) const {
        std::vector<std::pair<std::string, std::string>> pairs;
        for(int i = 0; i < values.size(); i++) {
            std::string key_str = std::to_string(this->db_->GetLatestSequenceNumber() + i);
            pairs.emplace_back(key_str, values[i]);
        }
        return KvStore::putMany(pairs);
    }

    std::string KvStore::get(const std::string& key) const {
        std::string value;
        const rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), key, &value);
        if (status.ok())
            return value;
        return status.ToString();
    }

    bool KvStore::remove(const std::string& key) const {
        const rocksdb::Status status = db_->Delete(rocksdb::WriteOptions(), key);
        return status.ok();
    }

}
