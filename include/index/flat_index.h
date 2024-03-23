#ifndef MICROVECDB_FLAT_INDEX_H
#define MICROVECDB_FLAT_INDEX_H

#include "index.h"

namespace mvdb {

    class FlatIndex final : public Index {
        std::shared_ptr<std::vector<value_t>> index_;    // The actual index
        friend std::ostream& operator<<(std::ostream& os, const FlatIndex& obj);
        friend std::ostream& operator<<(std::ostream& os, const FlatIndex* obj);
    protected:
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;
    public:
        explicit FlatIndex(const idx_t& dims, const std::string& path = "");
        ~FlatIndex() override = default;
        FlatIndex(const FlatIndex&) = delete;
        FlatIndex& operator=(const FlatIndex&) = delete;
        [[nodiscard]] IndexType type() const override;
        void save(const std::string& path) const override;
        void load(const std::string& path) override;
        [[nodiscard]] bool add(const idx_t& n, value_t* data, idx_t* ids) override;
        [[nodiscard]] bool remove(const idx_t& n, const idx_t* ids) override;
        void search(const idx_t& nq, value_t* query, idx_t* ids, value_t* distances, const idx_t& k, const DISTANCE_METRIC& distance_metric) const override;
        value_t* get(idx_t& n, idx_t* keys) const override;
        value_t* get_all() const override;
        [[nodiscard]] idx_t dims() const override;
        [[nodiscard]] idx_t ntotal() const override;
        [[nodiscard]] value_t* index() const;
    };

} // namespace mvdb


#endif //MICROVECDB_FLAT_INDEX_H
