#ifndef MICROVECDB_VECTOR_H
#define MICROVECDB_VECTOR_H

#include "serializable.hpp"
#include <vector>
#include <unordered_map>
#include <variant>
#include <string>
#include <typeinfo>
#include <typeindex>
#include <iostream>


namespace mvdb {

    using AttributeValue = std::variant<int, float, char, std::string>;
    using AttributeSet = std::unordered_map<std::string, AttributeValue>;
    using AttributeSchema = std::unordered_map<std::string, std::type_index>;

    bool setAttribute(AttributeSet& attributes, const AttributeSchema& schema, const std::string& key, const AttributeValue& value) {
        auto it = schema.find(key);
        if (it != schema.end() && it->second.hash_code() == typeid(value).hash_code()) {
            attributes[key] = value;
            return true;
        }
        return false; // Invalid attribute or type mismatch
    }

    AttributeValue getAttribute(const AttributeSet& attributes, const std::string& key) {
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            return it->second;
        }
        // Handle the case where the attribute doesn't exist
    }


    class Vector {
        uint64_t key_{};
        std::vector<float> vector_{};
    public:
        Vector(){
            std::cout << "MicroVector Init" << std::endl;
        };
        inline explicit Vector(uint64_t key){
            std::cout << typeid(int).hash_code() << std::endl;
            std::cout << typeid(122312312).hash_code() << std::endl;
            std::cout << (typeid(int).hash_code() == typeid(122312312).hash_code()) << std::endl << std::endl;

            std::cout << typeid(float).hash_code() << std::endl;
            std::cout << std::type_index(typeid(55.12)).hash_code() << std::endl;
            std::cout << (typeid(float).hash_code() == typeid(55.12).hash_code()) << std::endl << std::endl;

            std::cout << typeid(char).hash_code() << std::endl;
            std::cout << typeid('f').hash_code() << std::endl;
            std::cout << (typeid(char).hash_code() == typeid('f').hash_code()) << std::endl << std::endl;

            std::cout << typeid(std::string).hash_code() << std::endl;
            std::cout << typeid("hello").hash_code() << std::endl;
            std::cout << (typeid(std::string).hash_code() == typeid("hello").hash_code()) << std::endl << std::endl;
        };
        ~Vector() = default;
    };
}

#endif //MICROVECDB_VECTOR_H
