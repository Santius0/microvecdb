#ifndef MICROVECDB_INDEX_H
#define MICROVECDB_INDEX_H

#pragma once

#include "constants.hpp"
#include "db_object.h"
#include <vector>

#include <faiss/Index.h>

namespace mvdb {
    enum class IndexType {
        FLAT = 0,
        IVF = 1
    };

    class Index: public DBObject {
    protected:
        std::vector<float> data_{};
        IndexType idx_type_{};
        uint64_t idx_dims_{};
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;
    public:
        Index() = default;
        Index(std::string obj_name, const DBObjectType& obj_type, std::string obj_col_name,
              const mvdb::IndexType &idx_type, const uint64_t &idx_dims);
        ~Index() override = default;

        virtual void add(const vec_count_t& n, const float* __restrict x) = 0;
//        virtual void remove(vec_count_t n, const vec_count_t * labels) = 0;
        virtual void search(const vec_count_t& k, const vec_count_t& n, const float* __restrict x,
                            const pkey_t* __restrict pkeys, float* __restrict distances) const = 0;
//        virtual void range_search(vec_count_t n, const float* x, float radius
//                                  , RangeSearchResult* result, const SearchParameters* params = nullptr
//                                          ) const = 0;


    };
}

#endif //MICROVECDB_INDEX_H
