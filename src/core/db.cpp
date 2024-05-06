#include "db.h"
#include "storage.h"
#include "utils.h"
#include "filesystem.h"

#include "faiss_flat_index.h"
#include "spann_index.h"
#include "annoy_index.h"
#include "flat_index.h"

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
        _index->serialize_(out);
        _storage->serialize_(out);
    }

    template <typename T>
    void DB_<T>::deserialize_(std::istream &in) {
        _path = deserialize_string(in);
        _index_path = deserialize_string(in);
        _storage_path = deserialize_string(in);
        _dims = deserialize_numeric<idx_t>(in);

        _storage = std::make_unique<Storage>(_storage_path, true, false);
        _index = std::make_unique<index::FaissFlatIndex<T>>();

        _index->deserialize_(in);
        _storage->deserialize_(in);
    }

//    template <typename T>
//    DB_<T>::~DB_(){
//         delete[] _add_ids;
//         delete[] _search_ids;
//         delete[] _search_distances;
//    }

    template <typename T>
    Status* DB_<T>::status() const {
        return status_.get();
    }

    template <typename T>
    bool DB_<T>::open(const std::string &path) {
        std::ifstream file(path);
        if (!file) {
            std::cerr << "Error opening file for reading: \"" + path + "\"\n";
            return false;
        }
        _path = path;
        deserialize_(file);
        file.close();
        return true;
    }

    template<typename T>
    bool DB_<T>::create(index::IndexType index_type, const uint64_t &dims, const std::string &path,
                        const std::string &initial_data_path, const T *initial_data,
                        const uint64_t &initial_data_size, const NamedArgs *args) {
        _path = path;
        if(fs::exists(_path) || path.empty())
            throw::std::runtime_error("invalid path, \"" + path + "\" is blank or already exists");

        _storage_path = path + fs::preferred_separator + KV_STORE_EXT;
        if(fs::exists(_storage_path))
            throw::std::runtime_error("invalid path, \"" + _storage_path + "\" already exist");

        _index_path = path + fs::preferred_separator + INDEX_EXT;
        if(fs::exists(_index_path))
            throw::std::runtime_error("invalid path, \"" + _index_path + "\" already exist");

        if (!fs::create_directory(_path))
            throw::std::runtime_error("path, \"" + _path + "\" creation failed");

        _storage = std::make_unique<Storage>(_storage_path, true, false);

        if(index_type == index::IndexType::FAISS_FLAT)
            _index = std::make_unique<index::FaissFlatIndex<T>>();
        else if(index_type == index::IndexType::SPANN)
            _index = std::make_unique<index::SPANNIndex<T>>();
        else if(index_type == index::IndexType::ANNOY)
            _index = std::make_unique<index::AnnoyIndex<T>>();
//        else
//            _index = std::make_unique<index::FlatIndex<T>>();

        _index->build(dims, _index_path, initial_data_path, initial_data, initial_data_size, args);

        return true;
    }

    template <typename T>
    void DB_<T>::_save(const std::string& save_path) {
//        _index->save_(_index_path);
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

    template class DB_<int8_t>;
    template class DB_<int16_t>;
    template class DB_<int32_t>;
    template class DB_<int64_t>;
    template class DB_<uint8_t>;
    template class DB_<uint16_t>;
    template class DB_<uint32_t>;
    template class DB_<uint64_t>;
    template class DB_<float>;
    template class DB_<double>;
}