#include "spann_index.h"
#include "exception.h"

#include "Helper/VectorSetReader.h"
#include "Core/VectorIndex.h"
#include "Core/Common.h"
#include "Helper/SimpleIniReader.h"

#include <memory>
#include <iostream>
#include <Core/Common/DistanceUtils.h>

namespace mvdb::index {

    template <typename T>
    IndexType SPANNIndex<T>::type() const {
        return IndexType::SPANN;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const SPANNIndex<T>& obj){
        return os   << "dims_: " << obj.dims_ << std::endl
                    << "index_type_: " << obj.type() << std::endl;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const SPANNIndex<T>* obj){
        return os << "*(" << *obj << ")";
    }

    template <typename T>
    void SPANNIndex<T>::serialize_(std::ostream& out) const {
        serialize_numeric<unsigned char>(out, type());
        serialize_numeric<idx_t>(out, this->dims_);
    }

    template <typename T>
    void SPANNIndex<T>::deserialize_(std::istream& in) {
        auto type = deserialize_numeric<unsigned char>(in);
        if (type != SPANN) throw std::runtime_error("Unexpected index type: " + std::to_string(type));
        this->dims_ = deserialize_numeric<idx_t>(in);
    }

    template <typename T>
    void SPANNIndex<T>::build(const idx_t &dims, const std::string& path, const std::string& initial_data_path, const T* initial_data, const uint64_t& initial_data_size, const NamedArgs* args) {
        throw not_implemented("SPANNIndex::build not implemented");
    }

    template<typename T>
    void SPANNIndex<T>::save_(const std::string &path) const {

    }

    template<typename T>
    void SPANNIndex<T>::open(const std::string &path) {

    }

    template<typename T>
    bool SPANNIndex<T>::add(const idx_t &n, T *data, idx_t *ids) {
        return false;
    }

    template<typename T>
    bool SPANNIndex<T>::remove(const idx_t &n, const idx_t *ids) {
        return false;
    }

    template<typename T>
    void SPANNIndex<T>::topk(const idx_t &nq, T *query, idx_t *ids, T *distances, const idx_t &k,
                               const DISTANCE_METRIC &distance_metric, const float& c) const {

    }

    template<typename T>
    T *SPANNIndex<T>::get(idx_t &n, idx_t *keys) const {
        return nullptr;
    }

    template<typename T>
    T *SPANNIndex<T>::get_all() const {
        return nullptr;
    }

    template<typename T>
    idx_t SPANNIndex<T>::dims() const {
        return 0;
    }

    template<typename T>
    idx_t SPANNIndex<T>::ntotal() const {
        return 0;
    }

    template class SPANNIndex<int8_t>;
    template class SPANNIndex<int16_t>;
    template class SPANNIndex<int32_t>;
    template class SPANNIndex<int64_t>;
    template class SPANNIndex<uint8_t>;
    template class SPANNIndex<uint16_t>;
    template class SPANNIndex<uint32_t>;
    template class SPANNIndex<uint64_t>;
    template class SPANNIndex<float>;
    template class SPANNIndex<double>;
}