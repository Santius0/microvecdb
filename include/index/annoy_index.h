#ifndef MICROVECDB_ANNOY_INDEX_H
#define MICROVECDB_ANNOY_INDEX_H

#include "index.h"
#include "spann_index.h"

namespace mvdb::index {

    struct AnnoyIndexNamedArgs final : NamedArgs {
        int option1 = 0;
        AnnoyIndexNamedArgs() = default;
        ~AnnoyIndexNamedArgs() override = default;
    };

    template <typename T = float>
    class AnnoyIndex final : public Index<T> {
        friend std::ostream& operator<<(std::ostream& os, const AnnoyIndex<T>& obj);
        friend std::ostream& operator<<(std::ostream& os, const AnnoyIndex<T>* obj);
    protected:
        void save_(const std::string& path) const override;
    public:
        void serialize_(std::ostream &out) const override;
        void deserialize_(std::istream &in) override;
        AnnoyIndex() = default;
        ~AnnoyIndex() override = default;
        AnnoyIndex(const AnnoyIndex&) = delete;
        AnnoyIndex& operator=(const AnnoyIndex&) = delete;
        [[nodiscard]] IndexType type() const override;
        void build(const idx_t &dims, const std::string& path, const std::string& initial_data_path,
                   const T* initial_data, const uint64_t& initial_data_size, const NamedArgs* args) override;
        void open(const std::string& path) override;
        [[nodiscard]] bool add(const idx_t& n, T* data, idx_t* ids) override;
        [[nodiscard]] bool remove(const idx_t& n, const idx_t* ids) override;
        void search(const idx_t& nq, T* query, idx_t* ids, T* distances, const idx_t& k, const DISTANCE_METRIC& distance_metric) const override;
        T* get(idx_t& n, idx_t* keys) const override;
        [[nodiscard]] T* get_all() const override;
        [[nodiscard]] idx_t dims() const override;
        [[nodiscard]] idx_t ntotal() const override;
    };

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
    void AnnoyIndex<T>::search(const idx_t &nq, T *query, idx_t *ids, T *distances, const idx_t &k,
                               const DISTANCE_METRIC &distance_metric) const {

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

    extern template class AnnoyIndex<int8_t>;
    extern template class AnnoyIndex<int16_t>;
    extern template class AnnoyIndex<int32_t>;
    extern template class AnnoyIndex<int64_t>;
    extern template class AnnoyIndex<uint8_t>;
    extern template class AnnoyIndex<uint16_t>;
    extern template class AnnoyIndex<uint32_t>;
    extern template class AnnoyIndex<uint64_t>;
    extern template class AnnoyIndex<float>;
    extern template class AnnoyIndex<double>;
}

#endif //MICROVECDB_ANNOY_INDEX_H
