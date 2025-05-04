#include <gtest/gtest.h>
#include "etc.hh"

using namespace etc;

TEST(EtcTest, InputDataEvaluate) {
    Tensor t({{1,2},{3,4}});
    InputData input(t);
    Tensor out = input.evaluate();
    EXPECT_EQ(out.values, t.values);
}

TEST(EtcTest, ScalarAddOperation) {
    auto a = std::make_shared<InputData>(Tensor({{1,2},{3,4}}));
    Tensor b({{5,6},{7,8}});
    ScalarAddOperation op(a, b);
    Tensor out = op.evaluate();
    EXPECT_EQ(out.values[0][0], 6);
    EXPECT_EQ(out.values[1][1], 12);
}

TEST(EtcTest, NeuralNetworkInfer) {
    NeuralNetwork nn;
    auto a = std::make_shared<InputData>(Tensor({{1,2},{3,4}}));
    Tensor b({{5,6},{7,8}});
    auto add = std::make_shared<ScalarAddOperation>(a, b);
    nn.addOp(add);
    Tensor out = nn.infer();
    EXPECT_EQ(out.values[0][1], 8);
    EXPECT_EQ(out.values[1][0], 10);
}
