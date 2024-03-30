#include "utils.h"
#include "constants.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>

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

    // reads a single vector from the .fvecs file
    std::vector<float> read_vector(std::ifstream& file) {
        std::vector<float> vec;
        int dimension;
        if (file.read(reinterpret_cast<char*>(&dimension), sizeof(int))) { // Read the dimension of the vector
            vec.resize(dimension); // Resize the vector to hold all components
            for (int i = 0; i < dimension; ++i) {
                file.read(reinterpret_cast<char*>(&vec[i]), sizeof(float)); // Read each component of the vector
            }
        }
        return vec;
    }

    // reads num_vecs vectors from a fvecs file
    void read_vectors(const std::string& filename, const int& num_vecs, const int& dims, value_t *vecs){
        std::ifstream file(filename, std::ios::binary);
        if(file){
            for (int i = 0; i < num_vecs; ++i) {
                if (file.eof()) {
                    std::cout << "Reached the end of file before reading " << num_vecs << " records.\n";
                    break;
                }
                std::vector<float> vec = read_vector(file);
                for (size_t j = 0; j < vec.size(); j++) {
                    vecs[i * dims + j] = vec[j];
                }
            }
            file.close();
        } else {
            std::cerr << "Error opening file: " << filename << std::endl;
        }
    }
}