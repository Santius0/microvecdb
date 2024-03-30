#include "utils.h"
#include "flat_index.h"

#include <benchmark/benchmark.h>


static void BM_flat_index(benchmark::State& state) {
    const int num_queries = 10, dims = 96, k = 10;
    auto *index = new mvdb::FlatIndex(96, "../../benchmarks/testing.index");
    const std::string query_file = "../../benchmarks/data/deep1B_queries.fvecs";
    auto *queries = new mvdb::value_t[num_queries * dims];
    auto *ids = new mvdb::idx_t[num_queries * k];
    auto *distances = new mvdb::value_t[num_queries * k];
    mvdb::read_vectors(query_file, num_queries, dims, queries);

    for (auto _ : state) {
        index->search(num_queries, queries, ids, distances, k, mvdb::L2_DISTANCE);
    }

    delete index;
    delete[] queries;
    delete[] ids;
    delete[] distances;
}

//BENCHMARK(BM_flat_index)->Iterations(1);
BENCHMARK(BM_flat_index);
