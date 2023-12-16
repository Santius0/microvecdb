#ifndef MICROVECDB_FLAT_INDEX_H
#define MICROVECDB_FLAT_INDEX_H

#pragma once

#include "index.h"

namespace mvdb {
    class FlatIndex final : public Index {
    protected:
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;
    public:
        FlatIndex() = default;
        FlatIndex(std::string obj_name, const DBObjectType& obj_type, std::string obj_col_name,
                  const mvdb::IndexType &idx_type, const uint64_t &idx_dims);
        ~FlatIndex() override = default;
        void add(const mvdb::vec_count_t& n, const float* __restrict x) override;
        void search(const vec_count_t& k, const vec_count_t& n, const float* __restrict x, const pkey_t* __restrict pkeys, float* __restrict distances) const override;
    };
}

#endif //MICROVECDB_FLAT_INDEX_H
