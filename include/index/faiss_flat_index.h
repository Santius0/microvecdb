#ifndef MICROVEC_DB_FAISS_FLAT_INDEX_H
#define MICROVEC_DB_FAISS_FLAT_INDEX_H

#include "index.h"
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

namespace mvdb::index {

    template <typename T = float>
    class FaissFlatIndex final : public Index<T> {
        std::shared_ptr<faiss::Index> faiss_index_;    // The actual FAISS index
        friend std::ostream& operator<<(std::ostream& os, const FaissFlatIndex<T>& obj);
        friend std::ostream& operator<<(std::ostream& os, const FaissFlatIndex<T>* obj);
    public:
        void serialize_(std::ostream &out) const override;
        void deserialize_(std::istream &in) override;
        explicit FaissFlatIndex(const idx_t& dims, const std::string& path = "");
        ~FaissFlatIndex() override = default;
        FaissFlatIndex(const FaissFlatIndex&) = delete;
        FaissFlatIndex& operator=(const FaissFlatIndex&) = delete;
        [[nodiscard]] IndexType type() const override;
        void save(const std::string& path) const override;
        void load(const std::string& path) override;
        [[nodiscard]] bool add(const idx_t& n, T* data, idx_t* ids) override;
        [[nodiscard]] bool remove(const idx_t& n, const idx_t* ids) override;
        void search(const idx_t& nq, T* query, idx_t* ids, T* distances, const idx_t& k, const DISTANCE_METRIC& distance_metric) const override;
        T* get(idx_t& n, idx_t* keys) const override;
        [[nodiscard]] T* get_all() const override;
        [[nodiscard]] idx_t dims() const override;
        [[nodiscard]] idx_t ntotal() const override;
        [[nodiscard]] T* index() const;
        faiss::Index* faiss_index();
    };

} // namespace mvdb

#endif // MICROVEC_DB_FAISS_FLAT_INDEX_H