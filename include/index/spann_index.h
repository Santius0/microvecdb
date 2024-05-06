#ifndef MICROVECDB_SPANN_INDEX_H
#define MICROVECDB_SPANN_INDEX_H

#include "index.h"

namespace mvdb::index {

    struct SPANNIndexNamedArgs final : NamedArgs {
        int option1 = 0;
        SPANNIndexNamedArgs() = default;
        ~SPANNIndexNamedArgs() override = default;
    };

    template <typename T = float>
    class SPANNIndex final : public Index<T> {
        friend std::ostream& operator<<(std::ostream& os, const SPANNIndex<T>& obj);
        friend std::ostream& operator<<(std::ostream& os, const SPANNIndex<T>* obj);
    protected:
        void save_(const std::string& path) const override;
    public:
        void serialize_(std::ostream &out) const override;
        void deserialize_(std::istream &in) override;
        SPANNIndex() = default;
        ~SPANNIndex() override = default;
        SPANNIndex(const SPANNIndex&) = delete;
        SPANNIndex& operator=(const SPANNIndex&) = delete;
        [[nodiscard]] IndexType type() const override;
        void build(const idx_t &dims, const std::string& path, const std::string& initial_data_path, const T* initial_data, const uint64_t& initial_data_size, const NamedArgs* args) override;
        void open(const std::string& path) override;
        [[nodiscard]] bool add(const idx_t& n, T* data, idx_t* ids) override;
        [[nodiscard]] bool remove(const idx_t& n, const idx_t* ids) override;
        void topk(const idx_t& nq, T* query, idx_t* ids, T* distances, const idx_t& k,
                  const DISTANCE_METRIC& distance_metric, const float& c) const override;
        T* get(idx_t& n, idx_t* keys) const override;
        [[nodiscard]] T* get_all() const override;
        [[nodiscard]] idx_t dims() const override;
        [[nodiscard]] idx_t ntotal() const override;
    };

    extern template class SPANNIndex<int8_t>;
    extern template class SPANNIndex<int16_t>;
    extern template class SPANNIndex<int32_t>;
    extern template class SPANNIndex<int64_t>;
    extern template class SPANNIndex<uint8_t>;
    extern template class SPANNIndex<uint16_t>;
    extern template class SPANNIndex<uint32_t>;
    extern template class SPANNIndex<uint64_t>;
    extern template class SPANNIndex<float>;
    extern template class SPANNIndex<double>;
}

#endif //MICROVECDB_SPANN_INDEX_H
