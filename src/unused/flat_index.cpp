#include "unused/flat_index.h"
#include "unused/knn.h"
#include "unused/distances.h"
#include "filesystem.h"
#include "constants.h"

#include <iostream>
#include <fstream>

namespace mvdb::index {

    template <typename T>
    IndexType FlatIndex<T>::type() const {
        return IndexType::FLAT;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const FlatIndex<T>& obj){
        return os   << "this->dims_: " << obj->dims_ << std::endl
                    << "index_type_: " << obj.type() << std::endl;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const FlatIndex<T>* obj){
        return os << "*(" << *obj << ")";
    }

    // save everything except index_path_ as to load the index you'll need to know the path ahead of the load anyway
    template <typename T>
    void FlatIndex<T>::serialize_(std::ostream& out) const {
        serialize_numeric<unsigned char>(out, type());
        serialize_numeric<idx_t>(out, this->dims_);
        serialize_numeric<idx_t>(out, this->ntotal_);
        out.write(reinterpret_cast<const char*>(index_->data()), this->ntotal_ * this->dims_ * sizeof(T));
    }

    // load everything except index_path as that should be known ahead of time to trigger this load in the first place
    template <typename T>
    void FlatIndex<T>::deserialize_(std::istream& in) {
        auto type = deserialize_numeric<unsigned char>(in);
        if (type != IndexType::FLAT) throw std::runtime_error("Unexpected index type: " + std::to_string(type));
        this->dims_ = deserialize_numeric<idx_t>(in);
        this->ntotal_ = deserialize_numeric<idx_t>(in);
        index_->reserve(this->ntotal_ * this->dims_);
        in.read(reinterpret_cast<char*>(index_->data()), this->ntotal_ * this->dims_ * sizeof(T));
    }

    template <typename T>
    void FlatIndex<T>::build(const mvdb::idx_t &dims, const std::string &path) {
        this->dims_ = dims;
        this->index_ = std::make_shared<std::vector<T>>();
        this->save_(path);
        this->index_->reserve(100 * dims);
    }

    template <typename T>
    void FlatIndex<T>::save_(const std::string& path) const {
        std::ofstream out_file(path, std::ios::binary | std::ios::out);
        serialize_(out_file);
        out_file.close();
    }

    template <typename T>
    void FlatIndex<T>::open(const std::string& path) {
        if(path.empty() || !fs::exists(path) || fs::is_directory(path))
            throw std::runtime_error("index path, '" + path + "' is either empty or invalid");
        if(!this->index_) this->index_ = std::make_shared<std::vector<T>>();
        std::ifstream in_file(path, std::ios::binary | std::ios::in);
        deserialize_(in_file);
        in_file.close();
        this->index_->reserve(100 * this->dims_);
    }

    template <typename T>
    bool FlatIndex<T>::add(const idx_t &n, T *data, idx_t *ids) {
        if (n <= 0 || !data) return false;
        if(!index_) throw std::runtime_error("index_ == nullptr. Use index->open() before running add");

        try {
            index_->reserve(index_->size() + n * this->dims_);    // ensure we have enough space to hold new elements
            for (idx_t i = 0; i < n; ++i) {
                for (idx_t j = 0; j < this->dims_; ++j) {
                    index_->push_back(data[i * this->dims_ + j]);
                }
                this->ntotal_++;
                if(ids) ids[i] = this->ntotal_;
            }
            if (this->ntotal_ != (index_->size() / this->dims_))    // num vectors must be equal to num elements/num dimensions
                std::cerr << "ERROR: this->ntotal_ != (index_->size()/this->dims_) => " << this->ntotal_ << " != " << index_->size() / this->dims_ << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error adding data to index: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    template <typename T>
    bool FlatIndex<T>::remove(const mvdb::idx_t &n, const mvdb::idx_t *ids) {
        return false;
    }
    
    template <typename T>
    void FlatIndex<T>::search(const idx_t &nq, T *queries, idx_t *ids, T *distances, const idx_t &k, const DISTANCE_METRIC& distance_metric) const {
        if(!index_ || !index_->data()) throw std::runtime_error("index_ not properly initialized. index_ == nullptr || index_data == nullptr");
        knn(index_->data(), this->ntotal_, this->dims_, queries, nq, k, ids, distances, distance_metric);
    }

    template <typename T>
    T* FlatIndex<T>::get(idx_t &n, idx_t *keys) const {
        if(n <= 0 || !keys) return nullptr;
        auto *vecs = new T[n * this->dims_];
        #pragma omp parallel for collapse(2)
        for(idx_t i = 0; i < n; ++i) {
            for (idx_t j = 0; j < this->dims_; ++j) {
                vecs[i * this->dims_ + j] = index_->at(keys[i] * this->dims_ + j);
            }
        }
        return vecs;
    }

    template <typename T>
    T* FlatIndex<T>::get_all() const {
        return index_->data();
    }

    template <typename T>
    idx_t FlatIndex<T>::dims() const {
        return this->dims_;
    }

    template <typename T>
    idx_t FlatIndex<T>::ntotal() const {
        return this->ntotal_;
    }

    template <typename T>
    T* FlatIndex<T>::index() const {
        return index_->data();
    }

    template class FlatIndex<float>;
}