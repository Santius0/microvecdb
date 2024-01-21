#ifndef MICROVEC_DB_FAISS_FLAT_INDEX_H
#define MICROVEC_DB_FAISS_FLAT_INDEX_H

#include "index.h"
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

namespace mvdb {
    class FaissFlatIndex final : public Index {
        std::unique_ptr<faiss::Index> faiss_index_;    // The actual FAISS index
        friend std::ostream& operator<<(std::ostream& os, const FaissFlatIndex& obj);
        friend std::ostream& operator<<(std::ostream& os, const FaissFlatIndex* obj);
        void init() override;
    protected:
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;
    public:
        explicit FaissFlatIndex(const std::string& index_path, const idx_t& dims_);
        ~FaissFlatIndex() override = default;
        FaissFlatIndex(const FaissFlatIndex&) = delete;
        FaissFlatIndex& operator=(const FaissFlatIndex&) = delete;
        [[nodiscard]] bool add(const idx_t& n, value_t* data, idx_t* ids) override;
        [[nodiscard]] bool remove(const idx_t& n, const idx_t* ids) override;
        void save() const override;
        void load() override;
        void close() override;
        void search(const idx_t& nq, value_t* query, idx_t* ids, value_t* distances, const idx_t& k) const override;
        faiss::Index* faiss_index();
        value_t* get(idx_t& n, idx_t* keys) const override;
    };
} // namespace mvdb

#endif // MICROVEC_DB_FAISS_FLAT_INDEX_H