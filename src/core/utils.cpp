#include "utils.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <sys/resource.h>

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

    /**
    * Finds the next closest multiple of `multiple` for the number `n`.
    *
    * @param n The given number.
    * @param val The multiple to find the next closest of.
    * @return The next closest multiple of `val` for `n`.
    */
    int next_closest_multiple(int n, int val) {
        if (val == 0) return n; // if the multiple is zero returning n
        if (n % val == 0) return n;     // if n is already a multiple of 'multiple', no need to find the next one

        // Adding 'val' to n and then subtracting the remainder of n / val
        // This effectively rounds n up to the next multiple
        int nextMultiple = n + val - n % val;
        return nextMultiple;
    }

    void remove_trailing_slashes(std::string& path) {
        // Removes all trailing '/' and '\' from the end of the string
        path.erase(std::find_if(path.rbegin(), path.rend(), [](char ch) {
            return ch != '/' && ch != '\\';
        }).base(), path.end());
    }

    double peakWSS(){
        #ifndef _MSC_VER
        struct rusage rusage{};
        getrusage(RUSAGE_SELF, &rusage);
        double peak_wss = (double)(rusage.ru_maxrss) / (double)1024;
        #else
        PROCESS_MEMORY_COUNTERS pmc;
        GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
        unsigned long long peakWSS = pmc.PeakWorkingSetSize / 1000000000;
        #endif
        return peak_wss;
    }
}