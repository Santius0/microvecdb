//#include "flat_index.h"
//#include "filesystem.h"
//#include <gtest/gtest.h>
//#include <vector>
//#include <cmath>
//
//
//class FlatIndexTest : public ::testing::Test {
//protected:
//    mvdb::FlatIndex* index = nullptr;
//    std::string indexPath = "test_index.bin";
//    uint64_t dimensions = 3; // 3-dimensional vectors for simplicity
//
//    void SetUp() override {
//        // Ensure any existing index file is removed before a test starts
//        if (fs::exists(indexPath))
//            fs::remove(indexPath);
//
//        // Initialize the FlatIndex object
//        index = new mvdb::FlatIndex(indexPath, dimensions);
//
//        // Open the index (creates a new index if it doesn't exist)
//        index->open();
//        ASSERT_TRUE(index->is_open()) << "index->is_open == false => Index should be open after calling open()";
//    }
//
//    void TearDown() override {
//        if (index) {
//            index->close();
//            delete index;
//            index = nullptr;
//        }
//        if (fs::exists(indexPath))
//            fs::remove(indexPath);
//    }
//};
//
//TEST_F(FlatIndexTest, IsOpen) {
//    ASSERT_TRUE(index->is_open()) << "Index should be open after calling open()";
//    ASSERT_EQ(index->ntotal(), 0) << "Index should be open after calling open()";
//    ASSERT_EQ(index->dims(), dimensions) << "Index should be open after calling open()";
//}
//// Test for Adding Data to the Index
//TEST_F(FlatIndexTest, AddData) {
//    std::vector<mvdb::value_t> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
//    std::vector<mvdb::idx_t> ids(data.size()/3);
//    ASSERT_TRUE(index->add(data.size()/3, data.data(), ids.data())) << "Adding data to the index should succeed";
//}
//
//// Test for Saving and Reloading the Index
//TEST_F(FlatIndexTest, SaveAndReload) {
//    GTEST_LOG_(INFO) << "Testing saving and reloading a FlatIndex";
//    // index should already be open at this point
//    SCOPED_TRACE("Adding data to empty index");
//    std::vector<mvdb::value_t> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
//    std::vector<mvdb::idx_t> ids(data.size()/dimensions);
//    ASSERT_TRUE(index->add(data.size()/dimensions, data.data(), ids.data()));
//    mvdb::idx_t ntotal = index->ntotal();
//    ASSERT_EQ(ntotal, data.size()/dimensions);
//    SCOPED_TRACE("Saving index");
//    index->save();
//    SCOPED_TRACE("Deleting internal index object and resetting the values of all member variables");
//    index->close();
//    ASSERT_FALSE(index->is_open()) << "Index should be open after calling open()";
//    ASSERT_EQ(index->ntotal(), 0) << "Nt should be open after calling open()";
//    ASSERT_EQ(index->dims(), 0) << "Index should be open after calling open()";
//    SCOPED_TRACE("Reloading saved index");
//    index->open();
//    SCOPED_TRACE("Verifying is data was successfully loaded");
//    ASSERT_TRUE(index->is_open()) << "Index should be open after calling open()";
//    ASSERT_EQ(index->ntotal(), ntotal) << "Nt should be open after calling open()";
//    ASSERT_EQ(index->dims(), dimensions) << "Index should be open after calling open()";
//}
//
//// Test for Searching in the Index
//TEST_F(FlatIndexTest, Search) {
//    // index should already be open at this point
//    std::vector<mvdb::value_t> data = {4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 3.0f};
//    std::vector<mvdb::idx_t> ids(data.size()/dimensions);
//    ASSERT_TRUE(index->add(data.size()/dimensions, data.data(), ids.data()));
//
//    mvdb::value_t query[] = {2.0f, 3.0f, 4.0f};
//    mvdb::idx_t query_id = 1;
//    mvdb::value_t query_distance = 1.73205f;
//    mvdb::idx_t query_ids[1];
//    mvdb::value_t distances[1];
//    index->search(1, query, query_ids, distances, 1);
//
//    // We know index at id 1 with value {1.0f, 2.0f, 3.0f} should be the closest value, and because this is a flat
//    // index we can test for an exact match.
//    // Search result should:
//    //  query_ids[0] == 0
//    //  distances[0] == 1.73205
//    float scale = 10000.0f;
//    ASSERT_GE(distances[0], 0.0f) << "Search should return a non-negative distance";
//    EXPECT_EQ(query_ids[0], query_id) << "Search should result return ID 0 as closest vector index. Id returned = " << query_ids[0];
//    EXPECT_EQ(std::round(distances[0]*scale)/scale, std::round(query_distance*scale)/scale) << "Search should return distance of " << query_distance << ". Distance returned = " << distances[0];
//}
//
//int main(int argc, char **argv) {
//    ::testing::InitGoogleTest(&argc, argv);
//    return RUN_ALL_TESTS();
//}