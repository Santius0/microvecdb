#include "index.h"

#include <utility>
#include "exception.h"

namespace mvdb {

    Index::Index(std::string obj_name, const DBObjectType& obj_type, std::string obj_col_name,
                      const mvdb::IndexType &idx_type, const uint64_t &idx_dims) :
            DBObject(std::move(obj_name), obj_type, std::move(obj_col_name)),
            idx_type_(idx_type), idx_dims_(idx_dims) {
        switch (idx_type_) {
            case IndexType::FLAT:
                data_.reserve(idx_dims_);
                break;
            case IndexType::IVF:
                break;
        }
    }

    void Index::serialize(std::ostream &out) const {
        DBObject::serialize(out);
        serialize_numeric<int>(out, static_cast<int>(idx_type_));
        serialize_numeric<uint64_t>(out, idx_dims_);
    }

    void Index::deserialize(std::istream &in) {
        DBObject::deserialize(in);
        idx_type_ = static_cast<IndexType>(deserialize_numeric<int>(in));
        idx_dims_ = deserialize_numeric<int>(in);
    }

}