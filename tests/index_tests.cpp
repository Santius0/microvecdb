#include <gtest/gtest.h>
#include <index.hpp>
//#include <faiss_flat_index.h>

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

// main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
