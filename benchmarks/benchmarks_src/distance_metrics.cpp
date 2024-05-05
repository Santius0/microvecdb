#include "unused/distances.h"
#include "utils.h"

#include <benchmark/benchmark.h>
#include <cstdlib>


static void BM_L2_distance_naive(benchmark::State& state) {
    const int num_vecs = 10000, num_queries = 10, dims = 96;
    const std::string vector_file = "../../benchmarks/data/deep10M.fvecs", query_file = "../../benchmarks/data/deep1B_queries.fvecs";
    auto *vecs = new mvdb::value_t[num_vecs * dims], *queries = new mvdb::value_t[num_queries * dims];
    mvdb::read_vectors(vector_file, num_vecs, dims, vecs);
    mvdb::read_vectors(query_file, num_queries, dims, queries);
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
    const std::string vector_file = "../../benchmarks/data/deep10M.fvecs", query_file = "../../benchmarks/data/deep1B_queries.fvecs";
    auto *vecs = new mvdb::value_t[num_vecs * dims], *queries = new mvdb::value_t[num_queries * dims];
    mvdb::read_vectors(vector_file, num_vecs, dims, vecs);
    mvdb::read_vectors(query_file, num_queries, dims, queries);
    float *distances = nullptr;
    for (auto _ : state)
//        distances = mvdb::l2_distance_blas(vecs, num_vecs, dims, queries, num_queries);
//    std::cout << "L2 OpenBLAS: ";
//    for(int i = 0; i < 10; i++)
//        std::cout << distances[i] << " ";
//    std::cout << std::endl;
    delete[] vecs;
    delete[] queries;
    delete[] distances;
}

static void BM_L2_distance_optimised(benchmark::State& state) {
    const int num_vecs = 10000, num_queries = 10, dims = 96;
    const std::string vector_file = "../../benchmarks/data/deep10M.fvecs", query_file = "../../benchmarks/data/deep1B_queries.fvecs";
    #if defined(__AVX2__) || defined(__AVX__) // if using avx will be using __mm256
        auto *vecs = static_cast<mvdb::value_t*>(aligned_alloc(32, mvdb::next_closest_multiple(
                sizeof(mvdb::value_t) * num_vecs * dims, 32)));
        auto *queries = static_cast<mvdb::value_t*>(aligned_alloc(32, mvdb::next_closest_multiple(
                sizeof(mvdb::value_t) * num_queries * dims, 32)));
    #elif defined(__ARM_NEON)                // if using neon will be using float32x4
        auto *vecs = static_cast<mvdb::value_t*>(aligned_alloc(16, mvdb::next_closest_multiple(sizeof(mvdb::value_t) * num_vecs * dims, 16)));
        auto *queries = static_cast<mvdb::value_t*>(aligned_alloc(16, mvdb::next_closest_multiple(sizeof(mvdb::value_t) * num_queries * dims, 16)));
    #endif
    mvdb::read_vectors(vector_file, num_vecs, dims, vecs);
    mvdb::read_vectors(query_file, num_queries, dims, queries);
    float *distances = nullptr;
    for (auto _ : state) {
        delete[] distances;
        distances = mvdb::l2_distance_optimised(vecs, num_vecs, dims, queries, num_queries);
    }
//    std::cout << "L2 Optimised: ";
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
BENCHMARK(BM_L2_distance_optimised);