#include "utils.hpp"
#include <string>
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
//
//    // Helper function to write a std::string to the stream
//    void serializeString(std::ostream& out, const std::string& str) {
//        const size_t len = str.size();
//        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
//        out.write(str.data(), static_cast<long>(len));
//    }
//
//    std::string deserializeString(std::istream& in) {
//        size_t len;
//        in.read(reinterpret_cast<char*>(&len), sizeof(len));
//        std::string str(len, '\0');
//        in.read(&str[0], static_cast<long>(len));
//        return str;
//    }
//
//    // Helper function to write a size_t to the stream
//    void serializeUInt64T(std::ostream& out, const uint64_t value) {
//        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
//    }
//
//    // Helper function to read a size_t from the stream
//    uint64_t deserializeUInt64T(std::istream& in) {
//        uint64_t value;
//        in.read(reinterpret_cast<char*>(&value), sizeof(value));
//        return value;
//    }
//
//    // Helper function to write a size_t to the stream
//    void serializeInt64T(std::ostream& out, const int64_t value) {
//        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
//    }
//
//    // Helper function to read a size_t from the stream
//    int64_t deserializeInt64T(std::istream& in) {
//        int64_t value;
//        in.read(reinterpret_cast<char*>(&value), sizeof(value));
//        return value;
//    }


    std::string getCurrentTimeStamp() {
        std::ostringstream timestampStream;
        const auto now = std::chrono::system_clock::now();
        const auto timeT = std::chrono::system_clock::to_time_t(now);
        timestampStream << std::put_time(std::localtime(&timeT), "%Y-%m-%dT%H:%M:%S");
        return timestampStream.str();
    }

}