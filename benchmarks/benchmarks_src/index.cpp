#include "utils.h"
#include "unused/flat_index.h"
#include "faiss_flat_index.h"

#include <benchmark/benchmark.h>


static void BM_flat_index(benchmark::State& state) {
    const int num_queries = 4, dims = 96, k = 5;
    auto *index = new mvdb::index::FlatIndex<float>();
    index->open("../../benchmarks/flat_index_test.index");
    const std::string query_file = "../../benchmarks/data/deep1B_queries.fvecs";
    auto *queries = new mvdb::value_t[num_queries * dims];
    auto *ids = new mvdb::idx_t[num_queries * k];
    auto *distances = new mvdb::value_t[num_queries * k];
    mvdb::read_vectors(query_file, num_queries, dims, queries);

    for (auto _ : state) {
        index->search(num_queries, queries, ids, distances, k, mvdb::L2_DISTANCE);
    }

//    std::cout << "FlatIndex:" << std::endl;
//    for(int i = 0; i < k; i++)
//        std::cout << "id: " << ids[i] << " => distance: " << distances[i] << std::endl;

    delete index;
    delete[] queries;
    delete[] ids;
    delete[] distances;
}

static void BM_faiss_flat_index(benchmark::State& state) {
    const int num_queries = 4, dims = 96, k = 5;
    auto *index = new mvdb::index::FaissFlatIndex<float>();
    index->open("../../benchmarks/faiss_flat_index_test.index");
    const std::string query_file = "../../benchmarks/data/deep1B_queries.fvecs";
    auto *queries = new mvdb::value_t[num_queries * dims];
    auto *ids = new mvdb::idx_t[num_queries * k];
    auto *distances = new mvdb::value_t[num_queries * k];
    mvdb::read_vectors(query_file, num_queries, dims, queries);

    for (auto _ : state) {
        index->search(num_queries, queries, ids, distances, k, mvdb::L2_DISTANCE);
    }

//    std::cout << "FaissFlatIndex:" << std::endl;
//    for(int i = 0; i < k; i++)
//        std::cout << "id: " << ids[i] << " => distance: " << distances[i] << std::endl;

    delete index;
    delete[] queries;
    delete[] ids;
    delete[] distances;
}

//BENCHMARK(BM_flat_index)->Iterations(1);
BENCHMARK(BM_flat_index);
BENCHMARK(BM_faiss_flat_index);
