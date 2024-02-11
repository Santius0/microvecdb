#include "db.h"
#include "utils.h"
#include "constants.h"
#include "preprocess.h"
//#include "faiss_flat_index.h"
#include "flat_index.h"
#include "fasttext.h"
#include "exception.h"
#include <filesystem>
#include <stdexcept>
#include <fstream>

namespace mvdb {

    std::ostream& operator<<(std::ostream& os, const DB& obj) {
        return os   << "path_: " << obj.dbpath_ << std::endl
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
        serialize_string(out, dbpath_);
        serialize_string(out, dbname_);
        serialize_numeric<idx_t>(out, dims_);
        serialize_numeric<int8_t>(out, static_cast<int8_t>(index_type_));
//        index_->serialize(out);   // index and storage have their own save methods. we are not storing everything in one file serialising them here doesn't make sense
//        storage_->serialize(out);
    }

    void DB::deserialize(std::istream &in) {
        dbpath_ = deserialize_string(in);
        dbname_ = deserialize_string(in);
        dims_ = deserialize_numeric<idx_t>(in);
        index_type_ = static_cast<IndexType>(deserialize_numeric<int8_t>(in));
//        index_->deserialize(in);  // index and storage are automatically loaded if necessary at object instantiation
//        storage_->deserialize(in);
    }

    DB::DB(const std::string& dbname, const std::string& dbpath, const idx_t& dims, const IndexType& index_type,
                       VectorizerModelType vec_model): dbname_(dbname), dbpath_(trim(dbpath)), dims_(dims),
                       index_type_(index_type), vec_model_(vec_model) {
        const std::string index_path = dbpath + std::filesystem::path::preferred_separator + dbname + INDEX_EXT;
        const std::string data_path = dbpath + std::filesystem::path::preferred_separator + dbname + KV_STORE_EXT;
        metadata_path_ = dbpath + std::filesystem::path::preferred_separator + dbname_ + META_FILE_EXTENSION;

        if(!std::filesystem::exists(dbpath))  // if directory doesn't exist create it
            std::filesystem::create_directory(dbpath);
        else if(!std::filesystem::is_directory(dbpath)) // else if file exists but is not a directory throw error
            throw::std::runtime_error("invalid database path \"" + dbpath + "\"");
        make_index_(index_path);
        make_storage_(data_path);
        if(std::filesystem::exists(metadata_path_)) load();
    }

    DB::~DB(){
         delete[] add_ids_;
         delete[] search_ids_;
         delete[] search_distances_;
    }

    void DB::make_index_(const std::string& index_path){
         switch (index_type_) {
             case IndexType::FLAT:
                 index_ = std::make_unique<FlatIndex>(index_path, dims_);
                 break;
             case IndexType::FAISS_FLAT:
                 throw std::runtime_error("'" + std::to_string(index_type_) + "' is no longer a supported IndexType");
//                 index_ = std::make_unique<FaissFlatIndex>(index_path, dims_);

                 break;
             default:
                 throw std::runtime_error("'" + std::to_string(index_type_) + "' is not a valid IndexType");
         }
    }

    void DB::make_storage_(const std::string& data_path) {
        storage_ = std::make_unique<Storage>(data_path, true, false);
    }

    void DB::save(const std::string& save_path) {
        index_->save();
        // we don't actually save storage metadata, and the actual data part is kept up by rocksdb
        // TODO: maybe revisit that and actually store storage metadata
        std::ofstream file(save_path.empty() ? metadata_path_ : save_path);
        if (!file) throw std::runtime_error("Error opening file for writing: \"" + dbpath_ + "\"\n");
        serialize(file);
        file.close();
    }

    void DB::load(const std::string& load_path) {
        std::ifstream file(load_path.empty() ? metadata_path_ : load_path);
        if (!file) throw std::runtime_error("Error opening file for writing: \"" + dbpath_ + "\"\n");
        deserialize(file);
        file.close();
    }

    Index* DB::index() {
        return index_.get();
    }

    Storage* DB::storage() {
         return storage_.get();
    }

    idx_t* DB::add_vector(const idx_t& nv, value_t* v) {
         // TODO: perform write ahead log for vector data here
         if(!index_->is_open()) index_->open();
         delete[] add_ids_; // free old ids if they haven't been yet
         auto* ids = new idx_t[nv];
//         preprocess_vector(nv * dims_, dims_, v, v_d_type, false, false);
         bool success = index_->add(nv, v, ids);
         add_ids_ = ids;
         if(success) return ids;
         return nullptr;
    }

    std::future<idx_t*> DB::add_vector_async(const idx_t& nv, value_t* v) {
        // std::launch::async ensures the function is executed in a new thread immediately
        return std::async(std::launch::async, &DB::add_vector, this, nv, v);
    }

    // pre-processes data, generates embedding then passes them both to add_data_with_vector
    bool DB::add_data(const idx_t& nv, char* data, idx_t* data_sizes, const DataFormat* data_formats) {
        throw not_implemented("bool DB::add_data(const idx_t& nv, char* data, idx_t* data_sizes, const DataFormat* data_formats) const");
//        auto * vecs = new float[nv * index_->dims()];
//        size_t processed_bytes = 0;
//        for(size_t i = 0; i < nv; i++) {
//            if(data_formats[i] == RAW_TEXT) {
//                Vectorizer vectorizer = Vectorizer("../models/cc.en.300.bin", dims_);
//                fasttext::Vector vec = vectorizer.get_word_vector(std::string(data + processed_bytes, data_sizes[i]));
//                processed_bytes += data_sizes[i];
//                for(size_t j = 0; j < index_->dims(); j++)
//                    vecs[(i * index_->dims()) + j] = vec.data()[j];
//            } else {
//                // TODO: process and embed other data types
//                std::runtime_error("DataFormat '" + data_formats[i] + "''");
//            }
//        }
//        return add_data_with_vector(nv, data, data_sizes, data_formats, vecs, v_d_type);
//    }
//
//    bool DB::add_data_with_vector(const size_t& nv, char* data, size_t* data_sizes, const DataFormat* data_formats, value_t* v, const DataType& v_d_type) const {
//        uint64_t* keys = add_vector(nv, v, v_d_type);
//        if(!keys) return false;
//        auto* keys_str = new std::string[nv];
//        delete[] keys;
//        // TODO: perform write ahead log for char* data here
//        if(!storage_->is_open()) storage_->open();
//        return storage_->put(nv, keys_str, data, data_sizes);
    }

    void DB::search_with_vector(const size_t& nq, value_t* query, const long& k, idx_t* ids, value_t* distances) {
         if(!storage_->is_open()) storage_->open();
         if(!index_->is_open()) index_->open();
         index_->search(nq, static_cast<float*>(query), reinterpret_cast<idx_t*>(ids), distances, k);
         search_ids_ = ids;
         search_distances_ = distances;
    }

    std::future<void> DB::search_with_vector_async(const size_t &nq, mvdb::value_t *query, const long &k,
                                                   mvdb::idx_t *ids, mvdb::value_t *distances) {
        return std::async(std::launch::async, &DB::search_with_vector, this, nq, query, k, ids, distances);
    }

//    SearchResult* DB::search(const size_t& nq, const char* data, const size_t* data_sizes, const DataFormat* data_formats, const long& k, const bool& ret_data) const {
//        auto * vecs = new float[nq * index_->dims()];
//        size_t processed_bytes = 0;
//        for(size_t i = 0; i < nq; i++) {
//            if(data_formats[i] == RAW_TEXT) {
//                Vectorizer vectorizer = Vectorizer("../models/cc.en.300.bin", dims_);
//                fasttext::Vector vec = vectorizer.get_word_vector(std::string(data + processed_bytes, data_sizes[i]));
//                processed_bytes += data_sizes[i];
//                for(size_t j = 0; j < index_->dims(); j++)
//                    vecs[(i * index_->dims()) + j] = vec.data()[j];
//            } else {
//                // TODO: process and embed other data types
//            }
//        }
//        return search_with_vector(nq, vecs, k, ret_data);
//    }

//    void read_from_files(const size_t& n, const std::string* file_paths, char* data, size_t* sizes) {
//        size_t curr_size = 0;
//        for (size_t i = 0; i < n; i++) {
//            std::ifstream file(file_paths[i], std::ios::binary | std::ios::ate);
//            if (!file.is_open()) {
//                std::cerr << "Failed to open " << file_paths[i] << std::endl;
//                continue;
//            }
//            size_t size = file.tellg();
//            char* buffer = new char[curr_size + size];
//            std::copy(data, data + curr_size, buffer);
//            delete[] data;
//            data = buffer;
//            file.seekg(0, std::ios::beg);
//            file.read(data + curr_size, size);
//            file.close();
//            curr_size += size;
//            sizes[i] = size;
//        }
//    }

    value_t* DB::get(idx_t& n, idx_t* keys) const {
        if(!index_->is_open()) index_->open();
        return index_->get(n, keys);
    }
}