#include "utils.h"
#include <chrono>
#include <iomanip>
#include <sstream>

namespace mvdb {

    std::string trim(const std::string& str) {
        const std::string chars = " \t\n\r\f\v/\\";
        const size_t start = str.find_first_not_of(chars);
        if (start == std::string::npos) return ""; // String contains only whitespace or specified characters
        const size_t end = str.find_last_not_of(chars);
        return str.substr(start, end - start + 1);
    }

    std::string getCurrentTimeStamp() {
        std::ostringstream timestampStream;
        const auto now = std::chrono::system_clock::now();
        const auto timeT = std::chrono::system_clock::to_time_t(now);
        timestampStream << std::put_time(std::localtime(&timeT), "%Y-%m-%dT%H:%M:%S");
        return timestampStream.str();
    }

//    template<typename Func, typename Callback>
//    auto run_task_with_callback(Func task, Callback callback) {
//        return [task, callback]() {
//            auto result = task(); // Execute the task
//            callback(result); // Execute the callback with the result
//        };
//    }

}