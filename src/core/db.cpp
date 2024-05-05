#include "db.h"
#include "storage.h"
#include "utils.h"
#include "filesystem.h"
#include "faiss_flat_index.h"

#include <fstream>
#include <iostream>

namespace mvdb {

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const DB_<T>& obj) {
        return os   << "_path: " << obj._path << std::endl
                    << "_path: " << obj._index_path << std::endl
                    << "_path: " << obj._storage_path << std::endl
                    << "_dims: " << obj._dims << std::endl
                    << "vector_index_:\n" << obj._index.get() << std::endl
                    << "_storage:\n" << obj._storage.get();
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const DB_<T>* obj) {
        return os << "DB*(" << *obj << ")";
    }

    template <typename T>
    void DB_<T>::serialize_(std::ostream &out) const {
        serialize_string(out, _path);
        serialize_string(out, _index_path);
        serialize_string(out, _storage_path);
        serialize_numeric<idx_t>(out, _dims);
//        out.write(reinterpret_cast<const char*>(_records->data()), ntotal_ * dims_ * sizeof(value_t));
        _index->serialize(out);
        _storage->serialize_(out);
    }

    template <typename T>
    void DB_<T>::deserialize_(std::istream &in) {
        _path = deserialize_string(in);
        _index_path = deserialize_string(in);
        _storage_path = deserialize_string(in);
        _dims = deserialize_numeric<idx_t>(in);

        _storage = std::make_unique<Storage>(_storage_path, true, false);
        _index = std::make_unique<index::FaissFlatIndex<T>>(_dims, _index_path);

        _index->deserialize(in);
        _storage->deserialize_(in);
    }

    template <typename T>
    DB_<T>::~DB_(){
         delete[] _add_ids;
         delete[] _search_ids;
         delete[] _search_distances;
    }

    template <typename T>
    typename DB_<T>::Status* DB_<T>::status() const {
        return status_.get();
    }

    template <typename T>
    bool DB_<T>::open(const std::string &path) {
        std::ifstream file(path);
        if (!file) {
            std::cerr << "Error opening file for writing: \"" + path + "\"\n";
            return false;
        }
        _path = path;
        deserialize_(file);
        file.close();
        return true;
    }

    template <typename T>
    bool DB_<T>::create(const std::string& path, const idx_t& dims) {
        _path = path;
        _storage_path = path + fs::preferred_separator + KV_STORE_EXT;
        _index_path = path + fs::preferred_separator + INDEX_EXT;

        if(!fs::exists(path) && !fs::exists(_storage_path) && !fs::exists(_index_path)) {
            if (!fs::create_directory(path))
                return false;
        } else throw::std::runtime_error("database path \"" + path + "\" already exists");

        _storage = std::make_unique<Storage>(_storage_path, true, false);
        _index = std::make_unique<index::FaissFlatIndex<T>>();
        _index.build(dims, _index_path);
        return true;
    }

    template <typename T>
    void DB_<T>::_save(const std::string& save_path) {
        _index->save(_index_path);
        // we don't actually save storage metadata, and the actual data part is kept up by rocksdb
        std::string path = save_path.empty() ? _path + fs::preferred_separator + META_FILE_EXTENSION  : save_path;
        std::ofstream file(path);
        if (!file) throw std::runtime_error("Error opening file for writing: \"" + path + "\"\n");
        serialize_(file);
        file.close();
    }

    template <typename T>
    Storage* DB_<T>::storage() {
         return _storage.get();
    }

    template <typename T>
    index::Index<T>* DB_<T>::index() {
        return _index.get();
    }

//    template <typename T>
//    idx_t* DB_<T>::add_vector(const idx_t& nv, T* v) {
//        delete _add_ids;
////        insert_(this, nv, _dims, v, nullptr, operators::InsertOperatorDataType::VECTOR, nullptr, nullptr);
//        return _add_ids;
////        if(!_index) return nullptr;
////         delete[] _add_ids; // free old ids if they haven't been yet
////         auto* ids = new idx_t[nv];
////         bool success = _index->add(nv, v, ids);
////         _add_ids = ids;
////         if(success){
////             _save();
////             return ids;
////         }
////         return nullptr;
//    }
//
//    template <typename T>
//    void DB_<T>::search_with_vector(const idx_t& nq, T* query, const long& k, idx_t* ids, T* distances) {
//         if(!_storage || !_index) return;
//         _index->search(nq, static_cast<float*>(query), reinterpret_cast<idx_t*>(ids), distances, k, L2_DISTANCE);
//         _search_ids = ids;
//         _search_distances = distances;
//    }

//    template <typename T>
//    T* DB_<T>::get(idx_t& n, idx_t* keys) const {
//        if(!_index) return nullptr;
//        return _index->get(n, keys);
//    }
}