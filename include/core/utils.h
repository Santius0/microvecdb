#ifndef MICROVEC_DB_CORE_UTILS_H
#define MICROVEC_DB_CORE_UTILS_H

#include "constants.h"

#include <string>
#include <vector>

namespace mvdb {

    std::string trim(const std::string &str);

    std::string getCurrentTimeStamp();

    int next_closest_multiple(int n, int val);

    void remove_trailing_slashes(std::string& path);

    double peakWSS();

}

#endif // MICROVEC_DB_CORE_UTILS_H
