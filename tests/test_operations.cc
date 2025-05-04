#include <gtest/gtest.h>
#include "operations.hh"
#include "tensor.hh"
#include <cmath>

using namespace etc;

TEST(OperationsTest, Relu) {
    Tensor t({{-1, 0, 2}, {3, -4, 5}});
    Tensor out = Relu(t);
    EXPECT_DOUBLE_EQ(out.values[0][0], 0);
    EXPECT_DOUBLE_EQ(out.values[0][2], 2);
    EXPECT_DOUBLE_EQ(out.values[1][0], 3);
    EXPECT_DOUBLE_EQ(out.values[1][1], 0);
}

TEST(OperationsTest, Softmax) {
    Tensor t({{1, 2, 3}});
    Tensor out = Softmax(t);
    double sum = out.values[0][0] + out.values[0][1] + out.values[0][2];
    EXPECT_NEAR(sum, 1.0, 1e-9);
    EXPECT_GT(out.values[0][2], out.values[0][0]);
}

TEST(OperationsTest, Convol) {
    Tensor input({{1,2,3},{4,5,6},{7,8,9}});
    Tensor kernel({{1,0},{0,1}});
    Tensor out = Convol(input, kernel);
    EXPECT_EQ(out.values.size(), 3);
    EXPECT_EQ(out.values[0].size(), 3);
    // Check a few values for expected convolution result
    EXPECT_NEAR(out.values[1][1], 6, 1e-9);
}
