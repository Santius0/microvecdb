#include <gtest/gtest.h>
#include <random>

#include <index.h>
#include <quantization.h>
#include <faiss_flat_index.h>


void populate_random_vector(const size_t& n, float* v) {
    const unsigned int seed = 42;
//    std::random_device rd;  // Seed for the random number engine
    std::mt19937 gen(seed); // Standard mersenne_twister_engine
    std::uniform_real_distribution<> dis(1.0, 100.0); // Range of generated values
    for(size_t i = 0; i < n; i++) {
        if (i < n / 2) v[i] = dis(gen); // Generate positive values for the first half
        else v[i] = -dis(gen);          // Generate negative values for the second half
    }
}

//float* generateRandomVector(size_t n) {
//    auto* vec = new float[n];
//    for (size_t i = 0; i < n; ++i)
//        vec[i] = (rand() * 1.0)/RAND_MAX;
//    return vec;
//}
//// Test suite for FlatFaissIndex class
//TEST(FaissFlatIndexTests, BasicUsage) {
//    int dims = 300;
////    int n = 1000;
//    auto* index = new mvdb::FaissFlatIndex("./test_index", dims);
//    index->init();
//    EXPECT_EQ(true, index != nullptr);
//    EXPECT_EQ(true, index->faiss_index() != nullptr);
////    for(int i = 0; i < n; i++){
////        float *vec = generateRandomVector(dims);
////        index->add(dims, vec);
////        delete vec;
////    }
////    EXPECT_EQ(300, index->faiss_index()->ntotal);
//    int ids_to_remove[] = {1, 101, 201, 199};
//    delete index;
//    EXPECT_EQ(true, index == nullptr);
//}

TEST(ScalerQuantizationTest, ScalerQuanitzation) {
    size_t n = 1000;
    float *v = new float[n];
    mvdb::value_t *q = new mvdb::value_t[n];
    populate_random_vector(n, v);
    mvdb::scaler_quantization(n, v, q);
    for(size_t i = 0; i < n; i++)
        EXPECT_EQ(q[i], round(q[i]));
    delete[] v;
    delete[] q;
}

// FaissFlatIndex - Start ---------------------------------------------
//class FaissFlatIndexTest : public ::testing::Test {
//protected:
//    int n = 10000, d = 1000, nq = 100; // n = num input vectors, d = num dimensions, nq = num query vectors
//    mvdb::FaissFlatIndex* index;
//    float* input;
//    float* query;
//
//    void SetUp() override {
//        index = new mvdb::FaissFlatIndex("./test_index", 1000);
//        index->open();
//    }
//
//    void TearDown() override {
//        index->close();
//        delete index;
//    }
//};
// FaissFlatIndex - End ---------------------------------------------



// main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
