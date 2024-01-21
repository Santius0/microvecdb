#include "constants.h"
#include "faiss_flat_index.h"
#include <gtest/gtest.h>
#include <random>
#include <algorithm>

// Function to generate random data with a fixed seed for consistency
void generateRandomData(float* data, size_t dataSize, int dimensions) {
    std::mt19937 gen(42); // Fixed seed for consistent results
    std::uniform_real_distribution<> dis(0, 1);

    for (size_t i = 0; i < dataSize * dimensions; ++i) {
        data[i] = static_cast<float>(dis(gen));
    }
}


// Function to pick nq random vectors from data
void pickRandomVectors(const float* data, size_t totalVectors, int dims, float* query, size_t nq, mvdb::idx_t* query_ids) {
    std::vector<int> indices(totalVectors);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ..., totalVectors - 1

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen); // Shuffle the indices

    for (size_t i = 0; i < nq; ++i) {
        std::copy_n(&data[indices[i] * dims], dims, &query[i * dims]);
        query_ids[i] = indices[i];
    }
}

class FaissFlatIndexTest : public ::testing::Test {
protected:
    mvdb::FaissFlatIndex* index{};

    void SetUp() override {
        index = new mvdb::FaissFlatIndex("./faiss_flat_index_test.index", 128);
        index->open();
    }

    void TearDown() override {
        delete index;
    }
};

TEST_F(FaissFlatIndexTest, AddNormalCase) {
    const size_t n = 10; // Number of data points
    const size_t dims = 128; // Number of dimensions
    float data[n * dims];
    mvdb::idx_t ids[n];
    generateRandomData(data, n, dims);
    EXPECT_TRUE(index->add(n, data, ids));
}

TEST_F(FaissFlatIndexTest, AddZeroElements) {
    mvdb::idx_t n = 0;
    mvdb::value_t* data = nullptr; // No data
    mvdb::idx_t* ids = nullptr; // No IDs
    EXPECT_FALSE(index->add(n, data, ids));
}

//TEST_F(FaissFlatIndexTest, AddManyElements) {
//    mvdb::idx_t n = 1000000; // An excessively large number
//    const size_t dims = 128; // Number of dimensions
//    auto* data = new mvdb::value_t[n * dims];
//    auto* ids = new mvdb::idx_t[n];
//    generateRandomData(data, n, dims);
//    EXPECT_TRUE(index->add(n, data, ids));
//    delete[] data;
//    delete[] ids;
//}

TEST_F(FaissFlatIndexTest, AddInvalidData) {
    mvdb::idx_t n = 10;
    const size_t dims = 128;
    auto* data = new mvdb::value_t[n * dims];
    mvdb::idx_t ids[10];
    mvdb::value_t* bad_data = nullptr;
    mvdb::idx_t* bad_ids = nullptr;
    EXPECT_FALSE(index->add(n, bad_data, ids));
    EXPECT_FALSE(index->add(n, data, bad_ids));
    EXPECT_FALSE(index->add(n, bad_data, bad_ids));
    delete[] data;
}


TEST_F(FaissFlatIndexTest, SearchTest) {
    mvdb::idx_t n = 1000; // An excessively large number
    const size_t dims = 128; // Number of dimensions
    auto* data = new mvdb::value_t[n * dims];
    auto* ids = new mvdb::idx_t[n];
    generateRandomData(data, n, dims);
    EXPECT_TRUE(index->add(n, data, ids));
    mvdb::idx_t nq = 5;
    const mvdb::idx_t k = 10;       // Number of nearest neighbors to find
    mvdb::value_t query[nq * dims]; // 5 queries of 128 dimensions each
    mvdb::idx_t query_ids[nq];      // 5 queries of 128 dimensions each
    mvdb::idx_t search_ids[nq * k]; // 5 queries done in parallel with 10 results each
    mvdb::value_t distances[nq * k];
    pickRandomVectors(data, n, dims, query, nq, query_ids);
    EXPECT_NO_THROW(index->search(nq, query, search_ids, distances, k));
    std::cout << "Query IDs:" << std::endl;
    for(int i = 0; i < nq; i++) {
        std::cout << query_ids[i] << std::endl;
    }
    std::cout << "Query Search IDs:" << std::endl;
    for(int i = 0; i < nq * k; i++){
        if(i == 0 || i % k == 0) std::cout << "Query " << i/k << "=> " << query_ids[i/k] << std::endl;
        std::cout << "id = " << search_ids[i] << " => distance = " << distances[i] << std::endl;
    }

    //TODO: properly run and test these two
}
