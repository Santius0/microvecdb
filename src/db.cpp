#include "db.h"
#include "utils.h"
#include "constants.h"
#include "preprocess.h"
#include "faiss_flat_index.h"
#include "fasttext.h"
#include <filesystem>
#include <stdexcept>
#include <fstream>

namespace mvdb {

     std::ostream& operator<<(std::ostream& os, const DB& obj) {
        return os   << "path_: " << obj.path_ << std::endl
                    << "dbname_: " << obj.dbname_ << std::endl
                    << "dims_: " << obj.dims_ << std::endl
                    << "index_type_: " << obj.index_type_ << std::endl
                    << "vector_index_:\n" << obj.index_.get() << std::endl
                    << "kv_store_:\n" << obj.storage_.get();
     }

     std::ostream& operator<<(std::ostream& os, const DB* obj) {
         return os << "DB*(" << *obj << ")";
     }

     void DB::serialize(std::ostream &out) const {
        serialize_string(out, path_);
        serialize_string(out, dbname_);
        serialize_numeric<uint64_t>(out, dims_);
        serialize_numeric<int8_t>(out, static_cast<int8_t>(index_type_));
        index_->serialize(out);
        storage_->serialize(out);
    }

    void DB::deserialize(std::istream &in) {
        path_ = deserialize_string(in);
        dbname_ = deserialize_string(in);
        dims_ = deserialize_numeric<uint64_t>(in);
        index_type_ = static_cast<IndexType>(deserialize_numeric<int8_t>(in));
        index_->deserialize(in);
        storage_->deserialize(in);
    }

    DB::DB(const std::string& path, const std::string& dbname, const uint64_t& dims, const IndexType& index_type,
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
            make_index_(index_path);
//            kv_store_ = std::make_unique<KvStore>(data_path,true, false);
        }
    }

    void DB::make_index_(const std::string& index_path){
         switch (index_type_) {
             case IndexType::FLAT:
                 return;
             default:
                 index_ = std::make_unique<FaissFlatIndex>(index_path, dims_);
         }
     }

    void DB::save(const std::string& save_path) {
        std::ofstream file(save_path.empty() ? metadata_path_ : save_path);
        if (!file) throw std::runtime_error("Error opening file for writing: \"" + path_ + "\"\n");
        serialize(file);
        file.close();
    }

    void DB::load(const std::string& load_path) {
        std::ifstream file(load_path.empty() ? metadata_path_ : load_path);
        if (!file) throw std::runtime_error("Error opening file for writing: \"" + path_ + "\"\n");
        deserialize(file);
        file.close();
    }

    Index* DB::index() {
        return index_.get();
    }

    Storage* DB::storage() {
         return storage_.get();
    }

    uint64_t* DB::add_vector(const size_t& nv, void* v, const std::string& v_d_type) const {
         if(!index_->is_open()) index_->open();
         auto* keys = new uint64_t[nv];
         preprocess_vector(nv * dims_, dims_, v, v_d_type, false, false);
         bool success = index_->add(nv, static_cast<float*>(v), keys);
         if(success) return keys;
         return nullptr;
    }

//    bool DB::add_data(const size_t& nv, char* data, size_t* data_sizes, std::string* data_formats) const {
//        Vectorizer vectorizer = Vectorizer("../models/cc.en.300.bin", dims_);
//        auto * vecs = new float[nv * index_->dims()];
//        size_t processed_bytes = 0;
//        for(size_t i = 0; i < nv; i++) {
//            fasttext::Vector vec = vectorizer.get_word_vector(std::string(data + processed_bytes, data_sizes[i]));
//            processed_bytes += data_sizes[i];
//            for(size_t j = 0; j < index_->dims(); j++)
//                vecs[(i * index_->dims()) + j] = vec.data()[j];
//        }
////        return add_vector_data(nv, data, vecs);
//        return true;
//    }

    bool DB::add_vector_data(const size_t& nv, void* data, void* v) const {
//         // TODO: implement data add
//         bool success = false;
////         if(!kv_store_->is_open()) kv_store_->open();
//         if(!index_->is_open()) index_->open();
//         auto* keys = new uint64_t[nv];
//         success = index_->add(nv, static_cast<float*>(v), keys);
//         return kv_store_->put(std::to_string(keys[0]), data);
        return true;
    }

    SearchResult* DB::search_with_vector(const size_t& nq, void* query, const long& k, const bool& ret_data) const {
         if(!storage_->is_open()) storage_->open();
         if(!index_->is_open()) index_->open();
         auto *ids = new int64_t[k];
         auto *distances = new float[k];
         auto *data = new std::string[k];
         index_->search(nq, static_cast<float*>(query), ids, distances, k);
         if(ret_data) for(int i = 0 ; i < k; i++) data[i] = storage_->get(std::to_string(ids[i]));
         return new SearchResult(ids, distances, data, k);
    }

//    SearchResult DB::search(const std::string& data, const long& k, const bool& ret_data) const {
//         Vectorizer vectorizer = Vectorizer("../models/cc.en.300.bin", dims_);
//         fasttext::Vector query = vectorizer.get_word_vector(data);
//         return search_with_vector(query.get_data_(), k, ret_data);
//    }

    void read_from_files(const size_t& n, const std::string* file_paths, char* data, size_t* sizes) {
        size_t curr_size = 0;
        for (size_t i = 0; i < n; i++) {
            std::ifstream file(file_paths[i], std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                std::cerr << "Failed to open " << file_paths[i] << std::endl;
                continue;
            }
            size_t size = file.tellg();
            char* buffer = new char[curr_size + size];
            std::copy(data, data + curr_size, buffer);
            delete[] data;
            data = buffer;
            file.seekg(0, std::ios::beg);
            file.read(data + curr_size, size);
            file.close();
            curr_size += size;
            sizes[i] = size;
        }
    }
}