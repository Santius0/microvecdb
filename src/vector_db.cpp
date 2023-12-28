#include "vector_db.hpp"
#include "utils.hpp"
#include "constants.hpp"
#include <filesystem>
#include <stdexcept>
#include <fstream>

namespace mvdb {

     std::ostream& operator<<(std::ostream& os, const VectorDB& obj) {
        return os   << "path_: " << obj.path_ << std::endl
                    << "dbname_: " << obj.dbname_ << std::endl
                    << "dims_: " << obj.dims_ << std::endl
                    << "index_type_: " << obj.index_type_ << std::endl
                    << "vector_index_:\n" << obj.vector_index_.get() << std::endl
                    << "kv_store_:\n" << obj.kv_store_.get();
     }

     std::ostream& operator<<(std::ostream& os, const VectorDB* obj) {
         return os << "VectorDB*(" << *obj << ")";
     }

     void VectorDB::serialize(std::ostream &out) const {
        serialize_string(out, path_);
        serialize_string(out, dbname_);
        serialize_numeric<uint64_t>(out, dims_);
        serialize_numeric<int8_t>(out, static_cast<int8_t>(index_type_));
        vector_index_->serialize(out);
        kv_store_->serialize(out);
    }

    void VectorDB::deserialize(std::istream &in) {
        path_ = deserialize_string(in);
        dbname_ = deserialize_string(in);
        dims_ = deserialize_numeric<uint64_t>(in);
        index_type_ = static_cast<VectorIndexType>(deserialize_numeric<int8_t>(in));
        vector_index_->deserialize(in);
        kv_store_->deserialize(in);
    }

    VectorDB::VectorDB(const std::string& path, const std::string& dbname, const uint64_t& dims, const VectorIndexType& index_type,
                       VectorizerModelType vec_model): path_(trim(path)), dbname_(dbname), dims_(dims), index_type_(index_type),
                       vec_model_(vec_model){

        const std::string index_path = path + std::filesystem::path::preferred_separator + dbname + INDEX_EXT;
        const std::string data_path = path + std::filesystem::path::preferred_separator + dbname + KV_STORE_EXT;
        metadata_path_ = path + std::filesystem::path::preferred_separator + dbname_ + META_FILE_EXTENSION;

        if(!std::filesystem::exists(path))  // if directory doesn't exist create it
            std::filesystem::create_directory(path);
        else if(!std::filesystem::is_directory(path)) // else if file exists but is not a directory throw error
            throw::std::runtime_error("invalid database path \"" + path + "\"");

        if(std::filesystem::exists(metadata_path_)) load();
        else {
            vector_index_ = std::make_unique<VectorIndex>(index_path, dims_,  index_type_);
            kv_store_ = std::make_unique<KvStore>(data_path,true, false);
        }
    }

    void VectorDB::save(const std::string& save_path) {
        std::ofstream file(save_path.empty() ? metadata_path_ : save_path);
        if (!file) throw std::runtime_error("Error opening file for writing: \"" + path_ + "\"\n");
        serialize(file);
        file.close();
    }

    void VectorDB::load(const std::string& load_path) {
        std::ifstream file(load_path.empty() ? metadata_path_ : load_path);
        if (!file) throw std::runtime_error("Error opening file for writing: \"" + path_ + "\"\n");
        deserialize(file);
        file.close();
    }

    VectorIndex* VectorDB::index() {
        return vector_index_.get();
    }

    KvStore* VectorDB::storage() {
         return kv_store_.get();
    }

    bool VectorDB::add_data_vector(const std::string& data, float* vec) const {
         if(!kv_store_->is_open()) kv_store_->open();
         if(!vector_index_->is_open()) vector_index_->open();
         const std::vector<uint64_t> keys = vector_index_->add(1, vec);
         return kv_store_->put(std::to_string(keys[0]), data);
    }

    bool VectorDB::add_data(const std::string& data) const {
        Vectorizer vectorizer = Vectorizer("../models/cc.en.300.bin", dims_);
        fasttext::Vector vec = vectorizer.get_word_vector(data);
        return add_data_vector(data, vec.data());
    }

    SearchResult VectorDB::search_with_vector(const std::vector<float>& query, const long& k, const bool& ret_data) const {
         if(!kv_store_->is_open()) kv_store_->open();
         if(!vector_index_->is_open()) vector_index_->open();
         auto *ids = new int64_t[k];
         auto *distances = new float[k];
         auto *data = new std::string[k];
         vector_index_->search(query, ids, distances, k);
         if(ret_data) for(int i = 0 ; i < k; i++) data[i] = kv_store_->get(std::to_string(ids[i]));
         return {ids, distances, data, k};
    }

    SearchResult VectorDB::search(const std::string& data, const long& k, const bool& ret_data) const {
         Vectorizer vectorizer = Vectorizer("../models/cc.en.300.bin", dims_);
         fasttext::Vector query = vectorizer.get_word_vector(data);
         return search_with_vector(query.get_data_(), k, ret_data);
    }
}