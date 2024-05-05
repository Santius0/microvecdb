#ifndef MICROVECDB_PQ_LOOKUP_H
#define MICROVECDB_PQ_LOOKUP_H

#include "constants.h"
#include "index.h"
#include <memory>


namespace mvdb {

    class PQ_Lookup final : public Index {
        std::shared_ptr <std::vector<value_t>> index_;    // The actual index
        friend std::ostream &operator<<(std::ostream &os, const PQ_Lookup &obj);

        friend std::ostream &operator<<(std::ostream &os, const PQ_Lookup *obj);

    protected:
        void serialize(std::ostream &out) const override;

        void deserialize(std::istream &in) override;

    public:
        explicit FlatIndex(const std::string &index_path, const idx_t &dims_);

        ~FlatIndex() override = default;

        FlatIndex(const FlatIndex &) = delete;

        FlatIndex &operator=(const FlatIndex &) = delete;

        IndexType type() const override;

        void save() const override;

        void load() override;

        void open() override;

        void close(bool force = false) override;

        [[nodiscard]] bool add(const idx_t &n, value_t *data, idx_t *ids) override;

        [[nodiscard]] bool remove(const idx_t &n, const idx_t *ids) override;

        void search(const idx_t &nq, value_t *query, idx_t *ids, value_t *distances, const idx_t &k) const override;

        value_t *get(idx_t &n, idx_t *keys) const override;

        bool is_open() const override;

        idx_t dims() const override;

        idx_t ntotal() const override;

        [[nodiscard]] std::vector <value_t> index() const;
    };
}

#endif //MICROVECDB_PQ_LOOKUP_H
