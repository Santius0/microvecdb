#ifndef MICROVECDB_DISKANN_INDEX_H
#define MICROVECDB_DISKANN_INDEX_H

namespace mvdb {
    class DiskANNIndex final : public Index {
        std::shared_ptr<faiss::Index> faiss_index_;    // The actual FAISS index
        friend std::ostream& operator<<(std::ostream& os, const FaissFlatIndex& obj);
        friend std::ostream& operator<<(std::ostream& os, const FaissFlatIndex* obj);
    protected:
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;
    public:
        explicit FaissFlatIndex(const idx_t& dims, const std::string& path = "");
        ~FaissFlatIndex() override = default;
        FaissFlatIndex(const FaissFlatIndex&) = delete;
        FaissFlatIndex& operator=(const FaissFlatIndex&) = delete;
        [[nodiscard]] IndexType type() const override;
        void save(const std::string& path) const override;
        void load(const std::string& path) override;
        [[nodiscard]] bool add(const idx_t& n, value_t* data, idx_t* ids) override;
        [[nodiscard]] bool remove(const idx_t& n, const idx_t* ids) override;
        void search(const idx_t& nq, value_t* query, idx_t* ids, value_t* distances, const idx_t& k, const DISTANCE_METRIC& distance_metric) const override;
        value_t* get(idx_t& n, idx_t* keys) const override;
        [[nodiscard]] value_t* get_all() const override;
        [[nodiscard]] idx_t dims() const override;
        [[nodiscard]] idx_t ntotal() const override;
        [[nodiscard]] value_t* index() const;
        faiss::Index* faiss_index();
    };
} // namespace mvdb

#endif //MICROVECDB_DISKANN_INDEX_H
