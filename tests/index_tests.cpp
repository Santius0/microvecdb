#include <gtest/gtest.h>
#include <index.hpp>

// Test suite for integer functions
TEST(IntegerFunctionsTest, PositiveNos) {
    EXPECT_EQ(6, (3+3));
    EXPECT_EQ(12, (6+6));
}

TEST(IntegerFunctionsTest, NegativeNos) {
    EXPECT_EQ(-6, (-3-3));
    EXPECT_EQ(-12, (-6-6));
}

// main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
