#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>

namespace mvdb {

    std::string trim(const std::string& str);
    void serializeString(std::ostream& out, const std::string& str);
    std::string deserializeString(std::istream& in);
    void serializeNumeric(std::ostream& out, int64_t value);
    int64_t deserializeNumeric(std::istream& in);
    std::string getCurrentTimeStamp();

}

#endif //UTILS_H
