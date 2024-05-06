#include "annoy_index.h"

namespace mvdb::index {

    template <typename T>
    void AnnoyIndex<T>::build(const idx_t &dims, const std::string& path, const std::string& initial_data_path, const T* initial_data, const uint64_t& initial_data_size, const NamedArgs* args) {
        if(args) {
            if (const auto * spann_args = dynamic_cast<const AnnoyIndexNamedArgs *>(args)) {

            } else {
                throw std::runtime_error("failed to dynamic_cast from NamedArgs to AnnoyIndexNamedArgs");
            }
        }

        std::cout << "BUILDING AnnoyIndex at " << path << std::endl;
    }

    template<typename T>
    void AnnoyIndex<T>::save_(const std::string &path) const {

    }

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const AnnoyIndex<T> &obj) {
        return os << "";
    }

    template<typename T>
    std::ostream &operator<<(std::ostream &os, const AnnoyIndex<T> *obj) {
        return os << "";
    }

    template<typename T>
    void AnnoyIndex<T>::serialize_(std::ostream &out) const {

    }

    template<typename T>
    void AnnoyIndex<T>::deserialize_(std::istream &in) {

    }

    template<typename T>
    IndexType AnnoyIndex<T>::type() const {
        return IndexType::ANNOY;
    }

    template<typename T>
    void AnnoyIndex<T>::open(const std::string &path) {

    }

    template<typename T>
    bool AnnoyIndex<T>::add(const idx_t &n, T *data, idx_t *ids) {
        return false;
    }

    template<typename T>
    bool AnnoyIndex<T>::remove(const idx_t &n, const idx_t *ids) {
        return false;
    }

    template<typename T>
    void AnnoyIndex<T>::topk(const idx_t &nq, T *query, idx_t *ids, T *distances, const idx_t &k,
                               const DISTANCE_METRIC &distance_metric, const float& c) const {

    }

    template<typename T>
    T *AnnoyIndex<T>::get(idx_t &n, idx_t *keys) const {
        return nullptr;
    }

    template<typename T>
    T *AnnoyIndex<T>::get_all() const {
        return nullptr;
    }

    template<typename T>
    idx_t AnnoyIndex<T>::dims() const {
        return 0;
    }

    template<typename T>
    idx_t AnnoyIndex<T>::ntotal() const {
        return 0;
    }

    template class AnnoyIndex<int8_t>;
    template class AnnoyIndex<int16_t>;
    template class AnnoyIndex<int32_t>;
    template class AnnoyIndex<int64_t>;
    template class AnnoyIndex<uint8_t>;
    template class AnnoyIndex<uint16_t>;
    template class AnnoyIndex<uint32_t>;
    template class AnnoyIndex<uint64_t>;
    template class AnnoyIndex<float>;
    template class AnnoyIndex<double>;
}