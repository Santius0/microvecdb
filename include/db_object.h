#ifndef MICROVECDB_DB_OBJECT_H
#define MICROVECDB_DB_OBJECT_H

#pragma once

#include "serializable.hpp"
#include <string>
#include <iostream>
#include <fstream>


namespace mvdb {

    enum class DBObjectType {
        INDEX = 0,
        VECTORIZER = 1,
        KV_STORE = 2,
        COLLECTION = 3
    };

    inline std::ostream& operator<<(std::ostream& os, DBObjectType objectType) {
        switch (objectType) {
            case DBObjectType::INDEX:           os << "INDEX";   break;
            case DBObjectType::VECTORIZER:      os << "VECTORIZER"; break;
            case DBObjectType::KV_STORE:        os << "KV_STORE";  break;
            case DBObjectType::COLLECTION:      os << "COLLECTION";  break;
            default:                            os << "Unknown Database Object Type";
        }
        return os;
    }

    class DBObject : Serializable {
        std::string obj_name_{};
        DBObjectType obj_type_{};
        std::string obj_col_name_{};
    protected:
        void serialize(std::ostream &out) const override;
        void deserialize(std::istream &in) override;

    public:
        DBObject() = default;
        DBObject(std::string obj_name, const DBObjectType& obj_type, std::string  obj_col_name);
        ~DBObject() override = default;
    };
}
#endif //MICROVECDB_DB_OBJECT_H
