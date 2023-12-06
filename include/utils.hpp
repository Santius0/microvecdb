#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>

namespace mvdb {

    std::string trim(const std::string& str);
    void serializeString(std::ostream& out, const std::string& str);
    void serializeSizeT(std::ostream& out, size_t value);
    std::string deserializeString(std::istream& in);
    size_t deserializeSizeT(std::istream& in);
    std::string getCurrentTimeStamp();

}

#endif //UTILS_H
