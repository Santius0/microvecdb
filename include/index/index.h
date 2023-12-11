#ifndef MICROVECDB_INDEX_H
#define MICROVECDB_INDEX_H

#pragma once

#include "db_object.h"
#include <vector>

namespace mvdb {

    enum class IndexType {
        FLAT = 0,
        IVF = 1
    };

    class Index final : public DBObject {
        std::vector<float> data_{};
        IndexType idx_type_{};
        uint64_t idx_dims_{};
    protected:
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;
    public:
        Index() = default;
        Index(std::string obj_name, const DBObjectType& obj_type, std::string obj_col_name,
              const mvdb::IndexType &idx_type, const uint64_t &idx_dims);
        ~Index() override = default;
    };
}

#endif //MICROVECDB_INDEX_H
