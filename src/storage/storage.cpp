#include "storage.h"

 namespace mvdb {
     std::ostream& operator<<(std::ostream& os, const Storage& obj) {
         return os   << "data_dir_path_: " << obj.data_dir_path_ << std::endl
                     << "create_if_missing: " << obj.options_.create_if_missing << std::endl
                 ;
     }

     std::ostream& operator<<(std::ostream& os, const Storage* obj) {
         return os << "*(" << *obj << ")";
     }

     void Storage::serialize_(std::ostream& out) const {
         serialize_string(out, data_dir_path_);
         serialize_numeric<bool>(out, options_.create_if_missing);
     }

     void Storage::deserialize_(std::istream& in) {
         data_dir_path_ = deserialize_string(in);
         options_.create_if_missing = deserialize_numeric<bool>(in);
     }

     Storage::Storage(std::string  data_dir_path, const bool& creating_if_missing, const bool& open): data_dir_path_(std::move(data_dir_path)) {
         options_.create_if_missing = creating_if_missing;
         if(open) this->open();
     }

     void Storage::open(const std::string &path) {
         if(!path.empty()) data_dir_path_ = path;
         rocksdb::DB* raw_db_pointer = nullptr;
         if (const rocksdb::Status status = rocksdb::DB::Open(options_, data_dir_path_, &raw_db_pointer); !status.ok())
             throw std::runtime_error("Failed to open/create database: " + status.ToString());
         db_.reset(raw_db_pointer);
         is_open_ = true;
     }

     void Storage::close() {
         db_.reset();
         is_open_ = false;
     }

     bool Storage::is_open() const {
         return is_open_;
     }

     bool Storage::put(const size_t& n, const uint64_t* keys, char* values, size_t* value_sizes) const {
         if(n == 1){
             rocksdb::Slice value(reinterpret_cast<char*>(values[0]), value_sizes[0]);
             const rocksdb::Status status = db_->Put(rocksdb::WriteOptions(), std::to_string(keys[0]), value);
             return status.ok();
         }
         rocksdb::WriteBatch batch;
         size_t bytes_processed = 0;
         for (size_t i = 0; i < n; i++){
             rocksdb::Slice value(reinterpret_cast<char*>(values[bytes_processed]), value_sizes[i]);
             batch.Put(std::to_string(keys[i]), value);
             bytes_processed += value_sizes[i];
         }
         const rocksdb::Status status = db_->Write(rocksdb::WriteOptions(), &batch);
         return status.ok();
     }

     uint64_t* Storage::putAutoKey(const size_t& n, char* values, size_t* value_sizes) const {
         auto* keys = new uint64_t[n];
         for (size_t i = 0; i < n; i++)
             keys[i] = db_->GetLatestSequenceNumber() + i;
         if(Storage::put(n, keys, values, value_sizes)) return keys;
         return nullptr;
     }

     std::string Storage::get(const std::string& key) const {
         std::string value;
         const rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), key, &value);
         if (status.ok())
             return value;
         return status.ToString();
     }

     bool Storage::remove(const uint64_t& key) const {
         const rocksdb::Status status = db_->Delete(rocksdb::WriteOptions(), std::to_string(key));
         return status.ok();
     }
}