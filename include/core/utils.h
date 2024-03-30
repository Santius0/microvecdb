#ifndef MICROVEC_DB_CORE_UTILS_H
#define MICROVEC_DB_CORE_UTILS_H

#include "constants.h"

#include <string>
#include <vector>

namespace mvdb {

    std::string trim(const std::string &str);

    std::string getCurrentTimeStamp();
//    template<typename Func, typename Callback>
//    auto run_task_with_callback(Func task, Callback callback);

    int next_closest_multiple(int n, int val);

    std::vector<float> read_vector(std::ifstream &file);

    void read_vectors(const std::string &filename, const int &num_vecs, const int &dims, value_t *vecs);

}

#endif // MICROVEC_DB_CORE_UTILS_H
