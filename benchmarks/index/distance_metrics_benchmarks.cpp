#include "distances.h"
#include <benchmark/benchmark.h>
#include <fstream>
#include <cstdlib>

/**
 * Finds the next closest multiple of `multiple` for the number `n`.
 *
 * @param n The given number.
 * @param val The multiple to find the next closest of.
 * @return The next closest multiple of `val` for `n`.
 */
int nextClosestMultiple(int n, int val) {
    if (val == 0) return n; // if the multiple is zero returning n
    if (n % val == 0) return n;     // if n is already a multiple of 'multiple', no need to find the next one

    // Adding 'val' to n and then subtracting the remainder of n / val
    // This effectively rounds n up to the next multiple
    int nextMultiple = n + val - n % val;
    return nextMultiple;
}

// reads a single vector from the .fvecs file
std::vector<float> readVector(std::ifstream& file) {
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

void read_vectors(const std::string& filename, const int& num_vecs, const int& dims, mvdb::value_t *vecs){
    std::ifstream file(filename, std::ios::binary);
    if(file){
        for (int i = 0; i < num_vecs; ++i) {
            if (file.eof()) {
                std::cout << "Reached the end of file before reading " << num_vecs << " records.\n";
                break;
            }
            std::vector<float> vec = readVector(file);
            for (size_t j = 0; j < vec.size(); j++) {
                vecs[i * dims + j] = vec[j];
            }
        }
        file.close();
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
}

static void BM_L2_distance_naive(benchmark::State& state) {
    const int num_vecs = 10000, num_queries = 10, dims = 96;
    const std::string vector_file = "../../benchmarks/data/deep10M.fvecs", query_file = "../../benchmarks/data/deep10M.fvecs";
    auto *vecs = new mvdb::value_t[num_vecs * dims], *queries = new mvdb::value_t[num_queries * dims];
    read_vectors(vector_file, num_vecs, dims, vecs);
    read_vectors(query_file, num_queries, dims, queries);
    float *distances = nullptr;
    for (auto _ : state) {
        delete[] distances;
        distances = mvdb::l2_distance_naive(vecs, num_vecs, dims, queries, num_queries);
    }
//    std::cout << "L2 Naive: ";
//    for(int i = 0; i < 10; i++)
//        std::cout << distances[i] << " ";
//    std::cout << std::endl;
    delete[] vecs;
    delete[] queries;
    delete[] distances;
}

static void BM_L2_distance_blas(benchmark::State& state) {
    const int num_vecs = 10000, num_queries = 10, dims = 96;
    const std::string vector_file = "../../benchmarks/data/deep10M.fvecs", query_file = "../../benchmarks/data/deep10M.fvecs";
    auto *vecs = new mvdb::value_t[num_vecs * dims], *queries = new mvdb::value_t[num_queries * dims];
    read_vectors(vector_file, num_vecs, dims, vecs);
    read_vectors(query_file, num_queries, dims, queries);
    float *distances = nullptr;
    for (auto _ : state)
        distances = mvdb::l2_distance_blas(vecs, num_vecs, dims, queries, num_queries);
//    std::cout << "L2 OpenBLAS: ";
//    for(int i = 0; i < 10; i++)
//        std::cout << distances[i] << " ";
//    std::cout << std::endl;
    delete[] vecs;
    delete[] queries;
    delete[] distances;
}

static void BM_L2_distance_intrinsics(benchmark::State& state) {
    const int num_vecs = 10000, num_queries = 10, dims = 96;
    const std::string vector_file = "../../benchmarks/data/deep10M.fvecs", query_file = "../../benchmarks/data/deep10M.fvecs";
    #if defined(__AVX2__) || defined(__AVX__) // if using avx will be using __mm256
        auto *vecs = static_cast<mvdb::value_t*>(aligned_alloc(32, nextClosestMultiple(sizeof(mvdb::value_t) * num_vecs * dims, 32)));
        auto *queries = static_cast<mvdb::value_t*>(aligned_alloc(32, nextClosestMultiple(sizeof(mvdb::value_t) * num_queries * dims, 32)));
    #elif defined(__ARM_NEON)                // if using neon will be using float32x4
        auto *vecs = static_cast<mvdb::value_t*>(aligned_alloc(16, nextClosestMultiple(sizeof(mvdb::value_t) * num_vecs * dims, 16)));
        auto *queries = static_cast<mvdb::value_t*>(aligned_alloc(16, nextClosestMultiple(sizeof(mvdb::value_t) * num_queries * dims, 16)));
    #endif
    read_vectors(vector_file, num_vecs, dims, vecs);
    read_vectors(query_file, num_queries, dims, queries);
    float *distances = nullptr;
    for (auto _ : state) {
        delete[] distances;
        distances = mvdb::l2_distance_optimised(vecs, num_vecs, dims, queries, num_queries);
    }
//    std::cout << "L2 Intrinsics: ";
//    for(int i = 0; i < 10; i++){
//        std::cout << distances[i] << " ";
//    }
//    std::cout << std::endl;
    if(vecs) free(vecs);
    if(queries) free(queries);
    delete[] distances;
}

BENCHMARK(BM_L2_distance_naive);
BENCHMARK(BM_L2_distance_blas);
BENCHMARK(BM_L2_distance_intrinsics);

BENCHMARK_MAIN();
