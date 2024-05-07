#include "utils.h"
#include "constants.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>

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

    template <typename T>
    std::vector<T> read_vector(std::ifstream& file, const int& dims) { // reads a single vector from the .fvecs file
        std::vector<T> vec;
        int dimension;
        if (file.read(reinterpret_cast<char*>(&dimension), sizeof(int))) { // Read the dimension of the vector
//            if(dims > 0 && dims != dimension) throw std::runtime_error("specified dimensionality(" + std::to_string(dims) + ") does not match fvecs file dimensionality(" + std::to_string(dimension) + ")");
            vec.resize(dimension); // Resize the vector to hold all components
            for (int i = 0; i < dimension; ++i) {
                file.read(reinterpret_cast<char*>(&vec[i]), sizeof(T)); // Read each component of the vector
            }
        }
        return vec;
    }
    template std::vector<int8_t> read_vector<int8_t>(std::ifstream& file, const int& dims);
    template std::vector<int16_t> read_vector<int16_t>(std::ifstream& file, const int& dims);
    template std::vector<int32_t> read_vector<int32_t>(std::ifstream& file, const int& dims);
    template std::vector<int64_t> read_vector<int64_t>(std::ifstream& file, const int& dims);
    template std::vector<uint8_t> read_vector<uint8_t>(std::ifstream& file, const int& dims);
    template std::vector<uint16_t> read_vector<uint16_t>(std::ifstream& file, const int& dims);
    template std::vector<uint32_t> read_vector<uint32_t>(std::ifstream& file, const int& dims);
    template std::vector<uint64_t> read_vector<uint64_t>(std::ifstream& file, const int& dims);
    template std::vector<float> read_vector<float>(std::ifstream& file, const int& dims);
    template std::vector<double> read_vector<double>(std::ifstream& file, const int& dims);

    template<typename T>
    void read_fvecs(const std::string& file_path, std::vector<T>& data, std::vector<size_t>& start_indexes, int num_vecs) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        int dimension;
        if (!file.read(reinterpret_cast<char*>(&dimension), sizeof(int))) {
            throw std::runtime_error("Failed to read the dimension of the first vector.");
        }

        // Rewind to start to read dimensions again with each vector
        file.seekg(0, std::ios::beg);

        std::vector<T> vec(dimension);
        size_t currentStartIndex = 0;
        int vectorCount = 0;

        while (file.read(reinterpret_cast<char*>(&dimension), sizeof(int))) {
            if (dimension != vec.size()) {
                throw std::runtime_error("Inconsistent vector dimensions found.");
            }

            start_indexes.push_back(currentStartIndex);
            for (int i = 0; i < dimension; ++i) {
                if (!file.read(reinterpret_cast<char*>(&vec[i]), sizeof(T))) {
                    throw std::runtime_error("Failed to read all elements of a vector.");
                }
                data.push_back(vec[i]);
            }
            currentStartIndex += dimension;
            vectorCount++;

            // Check if we have read the required number of vectors
            if (num_vecs != -1 && vectorCount >= num_vecs) {
                break;
            }
        }

        file.close();
    }

    template void read_fvecs<int8_t>(const std::string& file_path, std::vector<int8_t>& data, std::vector<size_t>& start_indexes, int num_vecs = -1);
    template void read_fvecs<int16_t>(const std::string& file_path, std::vector<int16_t>& data, std::vector<size_t>& start_indexes, int num_vecs = -1);
    template void read_fvecs<int32_t>(const std::string& file_path, std::vector<int32_t>& data, std::vector<size_t>& start_indexes, int num_vecs = -1);
    template void read_fvecs<int64_t>(const std::string& file_path, std::vector<int64_t>& data, std::vector<size_t>& start_indexes, int num_vecs = -1);
    template void read_fvecs<uint8_t>(const std::string& file_path, std::vector<uint8_t>& data, std::vector<size_t>& start_indexes, int num_vecs = -1);
    template void read_fvecs<uint16_t>(const std::string& file_path, std::vector<uint16_t>& data, std::vector<size_t>& start_indexes, int num_vecs = -1);
    template void read_fvecs<uint32_t>(const std::string& file_path, std::vector<uint32_t>& data, std::vector<size_t>& start_indexes, int num_vecs = -1);
    template void read_fvecs<uint64_t>(const std::string& file_path, std::vector<uint64_t>& data, std::vector<size_t>& start_indexes, int num_vecs = -1);
    template void read_fvecs<float>(const std::string& file_path, std::vector<float>& data, std::vector<size_t>& start_indexes, int num_vecs = -1);
    template void read_fvecs<double>(const std::string& file_path, std::vector<double>& data, std::vector<size_t>& start_indexes, int num_vecs = -1);

    void remove_trailing_slashes(std::string& path) {
        // Removes all trailing '/' and '\' from the end of the string
        path.erase(std::find_if(path.rbegin(), path.rend(), [](char ch) {
            return ch != '/' && ch != '\\';
        }).base(), path.end());
    }

    template <typename T>
    int fvecs_num_vecs(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file, '" + path + "'" << std::endl;
            return -1;
        }
        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        int dimension;
        file.read(reinterpret_cast<char*>(&dimension), sizeof(int));
        size_t vector_size = dimension * sizeof(T) + sizeof(int);
        file.close();
        return fileSize / vector_size;
    }
    template int fvecs_num_vecs<int8_t>(const std::string& path);
    template int fvecs_num_vecs<int16_t>(const std::string& path);
    template int fvecs_num_vecs<int32_t>(const std::string& path);
    template int fvecs_num_vecs<int64_t>(const std::string& path);
    template int fvecs_num_vecs<uint8_t>(const std::string& path);
    template int fvecs_num_vecs<uint16_t>(const std::string& path);
    template int fvecs_num_vecs<uint32_t>(const std::string& path);
    template int fvecs_num_vecs<uint64_t>(const std::string& path);
    template int fvecs_num_vecs<float>(const std::string& path);
    template int fvecs_num_vecs<double>(const std::string& path);

}