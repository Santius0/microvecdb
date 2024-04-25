#ifndef MICROVECDB_FLAT_INDEX_H
#define MICROVECDB_FLAT_INDEX_H

#include "index.h"

namespace mvdb::index {

    template <typename T = float>
    class FlatIndex final : public index::Index<T> {
        std::shared_ptr<std::vector<T>> index_;    // The actual index
        friend std::ostream& operator<<(std::ostream& os, const FlatIndex<T>& obj);
        friend std::ostream& operator<<(std::ostream& os, const FlatIndex<T>* obj);
    public:
        void serialize_(std::ostream &out) const override;
        void deserialize_(std::istream &in) override;
        explicit FlatIndex(const idx_t& dims, const std::string& path = "");
        ~FlatIndex() override = default;
        FlatIndex(const FlatIndex&) = delete;
        FlatIndex& operator=(const FlatIndex&) = delete;
        [[nodiscard]] index::IndexType type() const override;
        void save(const std::string& path) const override;
        void load(const std::string& path) override;
        [[nodiscard]] bool add(const idx_t& n, T* data, idx_t* ids) override;
        [[nodiscard]] bool remove(const idx_t& n, const idx_t* ids) override;
        void search(const idx_t& nq, T* query, idx_t* ids, T* distances, const idx_t& k, const DISTANCE_METRIC& distance_metric) const override;
        T* get(idx_t& n, idx_t* keys) const override;
        T* get_all() const override;
        [[nodiscard]] idx_t dims() const override;
        [[nodiscard]] idx_t ntotal() const override;
        [[nodiscard]] T* index() const;
    };

} // namespace mvdb


#endif //MICROVECDB_FLAT_INDEX_H
