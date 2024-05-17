#ifndef MICROVEC_DB_FAISS_FLAT_INDEX_H
#define MICROVEC_DB_FAISS_FLAT_INDEX_H

#include "index.h"
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

namespace mvdb::index {

    struct FaissFlatIndexNamedArgs final : NamedArgs {
        FaissFlatIndexNamedArgs() = default;
        ~FaissFlatIndexNamedArgs() override = default;
    };

    template <typename T = float>
    class FaissFlatIndex final : public Index<T> {
        std::shared_ptr<faiss::Index> faiss_index_;    // The actual FAISS index
        friend std::ostream& operator<<(std::ostream& os, const FaissFlatIndex<T>& obj);
        friend std::ostream& operator<<(std::ostream& os, const FaissFlatIndex<T>* obj);
    protected:
        void save_(const std::string& path) const override;
    public:
        void serialize_(std::ostream &out) const override;
        void deserialize_(std::istream &in) override;
        FaissFlatIndex() = default;
        ~FaissFlatIndex() override = default;
        FaissFlatIndex(const FaissFlatIndex&) = delete;
        FaissFlatIndex& operator=(const FaissFlatIndex&) = delete;
        [[nodiscard]] IndexType type() const override;
        void build(const idx_t &dims, const std::string& path, const std::string& initial_data_path, const T* initial_data, const uint64_t& initial_data_size, const NamedArgs* args) override;
        void open(const std::string& path) override;
        [[nodiscard]] bool add(const idx_t& n, T* data, idx_t* ids) override;
        [[nodiscard]] bool remove(const idx_t& n, const idx_t* ids) override;
        void topk(const idx_t& nq, T* query, const std::string& query_path,
                  const std::string& result_path, idx_t* ids, T* distances, const idx_t& k,
                  const DISTANCE_METRIC& distance_metric, const float& c) const override;
        T* get(idx_t& n, idx_t* keys) const override;
        [[nodiscard]] T* get_all() const override;
        [[nodiscard]] idx_t dims() const override;
        [[nodiscard]] idx_t ntotal() const override;
        [[nodiscard]] bool built() const override;
        [[nodiscard]] T* index() const;
        faiss::Index* faiss_index();
    };

    extern template class FaissFlatIndex<int8_t>;
    extern template class FaissFlatIndex<int16_t>;
    extern template class FaissFlatIndex<uint8_t>;
    extern template class FaissFlatIndex<float>;

} // namespace mvdb

#endif // MICROVEC_DB_FAISS_FLAT_INDEX_H