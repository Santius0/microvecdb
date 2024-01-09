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
    protected:
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;
    public:
        explicit FaissFlatIndex(const std::string& index_path, const uint64_t& dims_);
        ~FaissFlatIndex() override = default;

        // Non-copyable and non-movable
        FaissFlatIndex(const FaissFlatIndex&) = delete;
        FaissFlatIndex& operator=(const FaissFlatIndex&) = delete;

        void init() override;

        // Add data to the index
        [[nodiscard]] bool add(const size_t& n, const float* data, uint64_t* keys) const override;

        // Remove data from the index
        [[nodiscard]] bool remove(const size_t& n, const faiss::IDSelector& ids) const override;

        void save() const override;

        void load() override;

        void close() override;

        void search(const int& n, float* query, int64_t* ids, float* distances, const long& k) const override;

        faiss::Index* faiss_index();

    };

} // namespace mvdb

#endif // MICROVEC_DB_FAISS_FLAT_INDEX_H