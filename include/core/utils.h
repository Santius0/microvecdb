#ifndef MICROVEC_DB_CORE_UTILS_H
#define MICROVEC_DB_CORE_UTILS_H

#include <string>

namespace mvdb {

    std::string trim(const std::string &str);

    std::string getCurrentTimeStamp();
//    template<typename Func, typename Callback>
//    auto run_task_with_callback(Func task, Callback callback);
}
#endif // MICROVEC_DB_CORE_UTILS_H
