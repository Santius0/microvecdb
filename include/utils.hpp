#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>

namespace mvdb {

    std::string trim(const std::string& str);
    void serializeString(std::ostream& out, const std::string& str);
    std::string deserializeString(std::istream& in);
    void serializeUInt64T(std::ostream& out, uint64_t value);
    uint64_t deserializeUInt64T(std::istream& in);
    void serializeInt64T(std::ostream& out, int64_t value);
    int64_t deserializeInt64T(std::istream& in);
    std::string getCurrentTimeStamp();

}

#endif //UTILS_H
