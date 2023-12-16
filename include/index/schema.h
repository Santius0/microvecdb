#ifndef MICROVECDB_SCHEMA_H
#define MICROVECDB_SCHEMA_H

#include "db_object.h"
#include <variant>
#include <unordered_map>
#include <typeinfo>
#include <typeindex>
#include <stdexcept>
#include <vector>
#include <utility>

namespace mvdb {
    using AttributeValue = std::variant<int8_t, int16_t, int32_t, int64_t,
                                        uint8_t, uint16_t, uint32_t, uint64_t,
                                        float, char, std::string>;
    using AttributeSet = std::unordered_map<std::string, AttributeValue>;
    using AttributeSchema = std::unordered_map<std::string, std::type_index>;

    struct Schema final : public DBObject {
        AttributeSchema attribute_schema_ = {};
        Schema() = default;
        Schema(std::string obj_name, std::string  obj_col_name,
               const std::vector<std::pair<std::string, std::type_index>>& attribute_list) :
               DBObject(obj_name, DBObjectType::SCHEMA, obj_col_name) {
            for (const auto& attribute : attribute_list) {
                attribute_schema_.insert(attribute);
            }
        }
        Schema(std::string obj_name, std::string  obj_col_name, const AttributeSchema& attributeSchema) :
        DBObject(obj_name, DBObjectType::SCHEMA, obj_col_name),
               attribute_schema_(attributeSchema) {}
        ~Schema() = default;

        inline void add_attribute(const std::pair<std::string, std::type_index>& attribute){
            attribute_schema_.insert(attribute);
        }

        inline void add_attribute(const std::string& attr_name, const std::type_index& attr_value){
            attribute_schema_.insert({attr_name, attr_value});
        }

        inline void remove_attribute(const std::string& attr_name){
            attribute_schema_.erase(attr_name);
        }

        inline void reset(){
            attribute_schema_.clear();
        }
    protected:
        inline void serialize(std::ostream &out) const override {
            DBObject::serialize(out);
            serialize_numeric(out, attribute_schema_.size()); // Serialize the number of attributes
            for (const auto& pair : attribute_schema_) {
                serialize_string(out, pair.first); // Serialize the attribute name
                serialize_string(out, pair.second.name()); // Serialize the type name
            }
        }

        inline void deserialize(std::istream& in) override {
            DBObject::deserialize(in);
            size_t numAttributes = deserialize_numeric<size_t>(in);
            for (size_t i = 0; i < numAttributes; ++i) {
                std::string name = deserialize_string(in);
                std::string typeName = deserialize_string(in);
                attribute_schema_.insert({name, std::type_index(typeid(typeName))}); // Deserialize type (requires additional handling)
            }
        }
    };
}

#endif //MICROVECDB_SCHEMA_H
