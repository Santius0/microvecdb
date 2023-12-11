#ifndef SERIALIZABLE_H
#define SERIALIZABLE_H

#pragma once

#include <ostream>
#include <istream>

namespace mvdb {

    class Serializable {
    protected:
        virtual void serialize(std::ostream &out) const = 0;
        virtual void deserialize(std::istream &in) = 0;
    public:
        virtual ~Serializable() = default;
    };

    inline void serialize_string(std::ostream& out, const std::string& str) {
        const size_t len = str.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(str.data(), static_cast<long>(len));
    }

    inline std::string deserialize_string(std::istream& in) {
        size_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string str(len, '\0');
        in.read(&str[0], static_cast<long>(len));
        return str;
    }

    template<typename T>
    inline void serialize_numeric(std::ostream& out, const T& value) {
        static_assert(std::is_arithmetic<T>::value, "Numeric type required");
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    template<typename T>
    inline T deserialize_numeric(std::istream& in) {
        static_assert(std::is_arithmetic<T>::value, "Numeric type required");
        T value;
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }

}

#endif //SERIALIZABLE_H
