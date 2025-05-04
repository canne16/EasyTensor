#include <gtest/gtest.h>
#include "tensor.hh"

using namespace etc;

TEST(TensorTest, Addition) {
    Tensor a({{1,2},{3,4}});
    Tensor b({{5,6},{7,8}});
    Tensor c = a + b;
    EXPECT_DOUBLE_EQ(c.values[0][0], 6);
    EXPECT_DOUBLE_EQ(c.values[1][1], 12);
}

TEST(TensorTest, Subtraction) {
    Tensor a({{5,6},{7,8}});
    Tensor b({{1,2},{3,4}});
    Tensor c = a - b;
    EXPECT_DOUBLE_EQ(c.values[0][1], 4);
    EXPECT_DOUBLE_EQ(c.values[1][0], 4);
}

TEST(TensorTest, Multiplication) {
    Tensor a({{1,2},{3,4}});
    Tensor b({{2,0},{1,2}});
    Tensor c = a * b;
    EXPECT_DOUBLE_EQ(c.values[0][0], 2);
    EXPECT_DOUBLE_EQ(c.values[1][1], 8);
}

TEST(TensorTest, Division) {
    Tensor a({{4,6},{8,10}});
    Tensor b({{2,3},{4,5}});
    Tensor c = a / b;
    EXPECT_DOUBLE_EQ(c.values[0][0], 2);
    EXPECT_DOUBLE_EQ(c.values[1][1], 2);
}

TEST(TensorTest, MatMul) {
    Tensor a({{1,2,3},{4,5,6}});
    Tensor b({{7,8},{9,10},{11,12}});
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.values.size(), 2);
    EXPECT_EQ(c.values[0].size(), 2);
    EXPECT_DOUBLE_EQ(c.values[0][0], 58);
    EXPECT_DOUBLE_EQ(c.values[1][1], 154);
}
