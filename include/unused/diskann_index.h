#ifndef MICROVECDB_DISKANN_INDEX_H
#define MICROVECDB_DISKANN_INDEX_H

#include "index.h"
#include <disk_utils.h>

namespace mvdb::index {

    template <typename T>
    class DiskANNIndex final : public Index<T> {
//        std::shared_ptr<faiss::Index> faiss_index_;    // The actual FAISS index
        friend std::ostream& operator<<(std::ostream& os, const DiskANNIndex<T>& obj);
        friend std::ostream& operator<<(std::ostream& os, const DiskANNIndex<T>* obj);
    protected:
        void save_(const std::string& path) const override;
    public:
        void serialize_(std::ostream &out) const override;
        void deserialize_(std::istream &in) override;
        DiskANNIndex() = default;
        ~DiskANNIndex() override = default;
        DiskANNIndex(const DiskANNIndex&) = delete;
        DiskANNIndex& operator=(const DiskANNIndex&) = delete;
        [[nodiscard]] IndexType type() const override;
        void build(const idx_t& dims, const std::string& path) override;
        void open(const std::string& path) override;
        [[nodiscard]] bool add(const idx_t& n, value_t* data, idx_t* ids) override;
        [[nodiscard]] bool remove(const idx_t& n, const idx_t* ids) override;
        void search(const idx_t& nq, value_t* query, idx_t* ids, value_t* distances, const idx_t& k, const DISTANCE_METRIC& distance_metric) const override;
        value_t* get(idx_t& n, idx_t* keys) const override;
        [[nodiscard]] value_t* get_all() const override;
        [[nodiscard]] idx_t dims() const override;
        [[nodiscard]] idx_t ntotal() const override;
        [[nodiscard]] value_t* index() const;
//        faiss::Index* faiss_index();
    };
} // namespace mvdb

#endif //MICROVECDB_DISKANN_INDEX_H
