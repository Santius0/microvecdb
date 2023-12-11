#include "db_object.h"

#include <utility>

namespace mvdb {
    DBObject::DBObject(std::string obj_name, const DBObjectType& obj_type, std::string  obj_col_name):
            obj_name_(std::move(obj_name)), obj_type_(obj_type), obj_col_name_(std::move(obj_col_name)){}

    void DBObject::serialize(std::ostream &out) const {
        serialize_string(out, obj_name_);
        serialize_numeric<int>(out, static_cast<int>(obj_type_));
        serialize_string(out, obj_col_name_);
    }

    void DBObject::deserialize(std::istream &in) {
        obj_name_ = deserialize_string(in);
        obj_type_ = static_cast<DBObjectType>(deserialize_numeric<int>(in));
        obj_col_name_ = deserialize_string(in);
    }
}