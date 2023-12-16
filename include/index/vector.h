#ifndef MICROVECDB_VECTOR_H
#define MICROVECDB_VECTOR_H

#include "constants.hpp"
#include "schema.h"
#include <vector>
#include <unordered_map>
#include <variant>
#include <string>
#include <typeinfo>
#include <typeindex>
#include <iostream>


namespace mvdb {

//    using AttributeValue = std::variant<int, float, char, std::string>;
//    using AttributeSet = std::unordered_map<std::string, AttributeValue>;
//    using AttributeSchema = std::unordered_map<std::string, std::type_index>;
//
//    bool setAttribute(AttributeSet& attributes, const AttributeSchema& schema, const std::string& key, const AttributeValue& value) {
//        auto it = schema.find(key);
//        if (it != schema.end() && it->second == typeid(value)) {
//            attributes[key] = value;
//            return true;
//        }
//        return false; // Invalid attribute or type mismatch
//    }

//    AttributeValue getAttribute(const AttributeSet& attributes, const std::string& key) {
//        auto it = attributes.find(key);
//        if (it != attributes.end()) {
//            return it->second;
//        }
//        // Handle the case where the attribute doesn't exist
//    }


    class Vector {
        vec_count_t d = 0;
        pkey_t pkey_ = 0;
        const Schema* schema_ = nullptr;
        const AttributeSet* attrs_ = nullptr;
        float* vector_ = nullptr;
        int8_t* vector_quantized_ = nullptr;
    public:
        Vector(const vec_count_t& d, const pkey_t& pkey, const Schema* schema, const AttributeSet* __restrict attrs, float* __restrict vector) :
        d(d), pkey_(pkey), schema_(schema), attrs_(attrs), vector_(vector) { };
        ~Vector(){
            if(vector_ != nullptr) delete vector_;
            if(attrs_ != nullptr) delete attrs_;
            if(vector_ != nullptr) delete vector_;
            if(vector_quantized_ != nullptr) delete vector_quantized_;
        }
    };
}

#endif //MICROVECDB_VECTOR_H
