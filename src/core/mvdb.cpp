#include "mvdb.h"
#include "utils.h"

namespace mvdb {

    template <typename T>
    MVDB<T>::MVDB() {
        db_ = std::make_unique<DB_<T>>();
    }

    template<typename T>
    void MVDB<T>::create(index::IndexType index_type,
                         const uint64_t &dims,
                         const std::string &path,
                         const T *v,
                         const std::string& bin,
                         const size_t *bin_sizes,
                         const uint64_t& n,
                         const NamedArgs *args) const {
        db_->create(index_type, dims, const_cast<std::string&>(path), v, bin, bin_sizes, n, args);
    }

    template<typename T>
    void MVDB<T>::open(const std::string& path) const {
        db_->open(const_cast<std::string&>(path));
    }

    template <typename T>
    bool MVDB<T>::insert(const idx_t &n,
                         const T *v,
                         const std::string &bin,
                         size_t *bin_sizes) const {
        return db_->insert(n, v, bin, bin_sizes);
    }

    template <typename T>
    void MVDB<T>::get_obj(const uint64_t& n, const uint64_t* keys, std::string* values) const {
        db_->get_obj(n, keys, values);
    }

    template <typename T>
    void MVDB<T>::remove(const idx_t& n,
                         const uint64_t &dims,
                         const T *v) const {
        std::cout << "Not Implemented" << std::endl;
    }

    template <typename T>
    void MVDB<T>::knn(const idx_t& nq,
                            T* query,
                            std::string query_path,
                            std::string result_path,
                            idx_t* ids,
                            T* distances,
                            double& peak_wss_mb,
                            const int64_t& k,
                            const NamedArgs* args) const {
        remove_trailing_slashes(query_path);
        remove_trailing_slashes(result_path);
        db_->index()->knn(nq, query, ids, distances, peak_wss_mb, k, args);
    }

    template <typename T>
    DB_<T>* MVDB<T>::get_db_() const {
        return db_.get();
    }

    template class MVDB<int8_t>;
    template class MVDB<int16_t>;
    template class MVDB<float>;
}