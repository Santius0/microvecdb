#ifdef FAISS

#include "faiss_flat_index.h"
#include "exception.h"
#include "filesystem.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <type_traits>
#include <sys/resource.h>

namespace mvdb::index {

    template <typename T>
    IndexType FaissFlatIndex<T>::type() const {
        return FAISS_FLAT;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const FaissFlatIndex<T>& obj){
        return os   << "dims_: " << obj.dims_ << std::endl
                    << "index_type_: " << obj.type() << std::endl;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const FaissFlatIndex<T>* obj){
        return os << "*(" << *obj << ")";
    }

    template <typename T>
    void FaissFlatIndex<T>::serialize_(std::ostream& out) const {
        serialize_numeric<unsigned char>(out, type());
        serialize_numeric<idx_t>(out, this->dims_);
        serialize_numeric<bool>(out, this->built_);
    }

    template <typename T>
    void FaissFlatIndex<T>::deserialize_(std::istream& in) {
        auto type = deserialize_numeric<unsigned char>(in);
        if (type != FAISS_FLAT) throw std::runtime_error("Unexpected index type: " + std::to_string(type));
        this->dims_ = deserialize_numeric<idx_t>(in);
        this->built_ = deserialize_numeric<bool>(in);
    }

    template <typename T>
    void FaissFlatIndex<T>::build(const idx_t &dims,
                                  const std::string& path,
                                  const T* v,
                                  idx_t* ids,
                                  const uint64_t& n,
                                  const NamedArgs* args) {

        if (path.empty())
            throw std::invalid_argument("Index path cannot be empty.");
        if (fs::exists(path))
            throw std::invalid_argument("Path '" + path + "' already exists. Please specify a new path.");
        if (!args)
            throw std::invalid_argument("NamedArgs pointer cannot be null.");

        std::cout << "Building FaissFlatIndex at " << path << std::endl;
        this->dims_ = dims;
        faiss_index_ = std::make_unique<faiss::IndexFlatL2>(this->dims_);
        std::cout << "Internal faiss_index_ built" << std::endl;

        if(path.empty())
            throw std::runtime_error("index path cannot be empty");

        if(fs::exists(path))
            throw std::runtime_error("'" + path + "' is invalid. make sure no file or dir exists at that location");

        if(n > 0) {
            if (!v) throw std::invalid_argument("Initial data pointer cannot be null when n > 0.");
            faiss_index_->add((faiss::idx_t)n, (float*)n);
        }
        this->save_(path);
        this->built_ = true;
    }

    template <typename T>
    bool FaissFlatIndex<T>::add(const idx_t& n, T* data, idx_t* ids) {
        if(n <= 0) return false;
        if(data == nullptr) return false;
        try {
            if(!faiss_index_) throw std::runtime_error("faiss_index_ == nullptr. Use index->open() before running add");
            if(ids) {
                for (int i = 0; i < n; i++) ids[i] = faiss_index_->ntotal + i;
            }
            faiss_index_->add(static_cast<long>(n), (float*)(data));
        } catch (const std::exception& e) {
            std::cerr << "Error adding data to index: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    template <typename T>
//    bool FaissFlatIndex::remove(const size_t& n, const faiss::IDSelector& ids) const {
    bool FaissFlatIndex<T>::remove(const idx_t& n, const idx_t* ids) {
        throw not_implemented("FaissFlatIndex<T, D>::remove(const idx_t& n, idx_t* ids) const not implemented");
//        try {
//            faiss_index_->remove_ids(ids);
//            save();
//            return true;
//        } catch (const std::exception& e) {
//            std::cerr << "Error removing data from index: " << e.what() << std::endl;
//            return false;
//        }
    }

    template <typename T>
    void FaissFlatIndex<T>::save_(const std::string& path) const {
//        std::ofstream file(path + INDEX_META_EXT, std::ios::binary | std::ios::out);;
//        if (!file) {
//            std::cerr << "Error opening file for reading: \"" + path + INDEX_META_EXT + "\"\n";
//            return;
//        }
//        serialize_(file);
//        file.close();
        faiss::write_index(faiss_index_.get(), path.c_str());
    }

    template <typename T>
    void FaissFlatIndex<T>::open(const std::string& path) {
        if(path.empty() || !fs::exists(path) || fs::is_directory(path))
            throw std::runtime_error("index path, '" + path + "' is either empty or invalid");
//        std::string meta_path = path + INDEX_META_EXT;
//        if(!fs::exists(meta_path) || fs::is_directory(meta_path))
//            throw std::runtime_error("index path, '" + path + "' is invalid");
//        std::ifstream in_file(meta_path, std::ios::binary | std::ios::in);
//        deserialize_(in_file);
//        in_file.close();
        if(!faiss_index_) faiss_index_ = std::make_unique<faiss::IndexFlatL2>(this->dims_);
        faiss_index_.reset(faiss::read_index(path.c_str()));
    }

    template <typename T>
    void FaissFlatIndex<T>::topk(const idx_t& nq,
                                 T* query,
                                 idx_t* ids,
                                 T* distances,
                                 double& peak_wss_mb,
                                 const int64_t& k,
                                 const NamedArgs* args) const {

        faiss_index_->search(static_cast<long>(nq), (float*)(query), (faiss::idx_t)k, (float*)(distances), reinterpret_cast<faiss::idx_t*>(ids));

        peak_wss_mb = peakWSS();
    }

    template <typename T>
    faiss::Index* FaissFlatIndex<T>::faiss_index(){
        return faiss_index_.get();
    }

    template <typename T>
    T* FaissFlatIndex<T>::get(idx_t& n, idx_t* keys) const {
        if (faiss_index_->is_trained) {
            if(keys != nullptr){
                for(size_t i = 0; i < n; i++) {
                    if(faiss_index_->ntotal <= keys[i]){
                        std::cerr << "ID '" << keys[i] << "' is out of index range." << std::endl;
                        return nullptr;
                    }
                }
            }
        }
        else {
            std::cerr << "Index does not support reconstruction." << std::endl;
            return nullptr;
        }
    }

    template <typename T>
    T* FaissFlatIndex<T>::get_all() const {
        idx_t n = faiss_index_->ntotal;
        auto *keys = new idx_t[n];
        auto* reconstructed_vec = new float[n * this->dims_];
        for(faiss::idx_t i = 0; i < n; i++){
            auto key = static_cast<faiss::idx_t>(keys[i]);
            faiss_index_->reconstruct(key, reconstructed_vec + (i * this->dims_));
        }
        return (T*)reconstructed_vec;
    }

    template <typename T>
    idx_t FaissFlatIndex<T>::dims() const {
        return this->dims_;
    }

    template <typename T>
    idx_t FaissFlatIndex<T>::ntotal() const {
        return faiss_index_->ntotal;
    }

    template <typename T>
    T* FaissFlatIndex<T>::index() const {
        return get_all();
    }

    template <typename T>
    bool FaissFlatIndex<T>::built() const {
        return this->built_;
    }

    template class FaissFlatIndex<int8_t>;
    template class FaissFlatIndex<int16_t>;
    template class FaissFlatIndex<uint8_t>;
    template class FaissFlatIndex<float>;

} // namespace mvdb

#endif