#ifndef MICROVEC_DB_CORE_UTILS_H
#define MICROVEC_DB_CORE_UTILS_H

#include "constants.h"

#include <string>
#include <vector>

namespace mvdb {

    std::string trim(const std::string &str);

    std::string getCurrentTimeStamp();

    int next_closest_multiple(int n, int val);

    template <typename T>
    std::vector<T> read_vector(std::ifstream &file, const int& dims = 0);

    template <typename T>
    void read_fvecs(const std::string& filePath, std::vector<T>& data, std::vector<size_t>& startIndexes, int num_vecs = -1);

    template <typename T>
    int fvecs_num_vecs(const std::string& path);

    void remove_trailing_slashes(std::string& path);

}

#endif // MICROVEC_DB_CORE_UTILS_H
