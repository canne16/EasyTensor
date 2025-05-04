#include <gtest/gtest.h>
#include "etc.hh"
#include <memory>
#include <vector>

using namespace etc;

TEST(EndToEnd, FullNetwork) {
    // Build a network similar to main.cc, using all core operations
    Tensor input({{1.0, 2.0, 3.0},
                  {3.0, 4.0, 5.0},
                  {5.0, 6.0, 7.0}});
    Tensor weight1({{2, 1, 0},
                    {0, 1, 2},
                    {1, 0, 1}});
    Tensor weight2({{1, -1, 0},
                    {0, 1, -1},
                    {-1, 0, 1}});
    Tensor kernel({{1.0, 0.0},
                   {0.0, 1.0}});

    auto input_data = std::make_shared<InputData>(input);
    auto add1 = std::make_shared<ScalarAddOperation>(input_data, weight1);
    auto mul1 = std::make_shared<ScalarMulOperation>(add1, weight2);
    auto relu1 = std::make_shared<ReLUOperation>(mul1);
    auto conv1 = std::make_shared<ConvolOperation>(mul1, kernel);
    auto add3 = std::make_shared<ScalarAddOperation>(input_data, weight2);
    auto conv2 = std::make_shared<ConvolOperation>(relu1, kernel);
    auto add4 = std::make_shared<ScalarAddOperation>(conv2, conv1);
    auto add5 = std::make_shared<ScalarAddOperation>(add4, add3);
    auto soft1 = std::make_shared<SoftmaxOperation>(add5);

    NeuralNetwork nn;
    nn.addOp(add1);
    nn.addOp(mul1);
    nn.addOp(relu1);
    nn.addOp(conv1);
    nn.addOp(add3);
    nn.addOp(conv2);
    nn.addOp(add4);
    nn.addOp(add5);
    nn.addOp(soft1);

    Tensor output = nn.infer();

    ASSERT_EQ(output.values.size(), 3);
    ASSERT_EQ(output.values[0].size(), 3);
    for (const auto& row : output.values) {
        double sum = 0.0;
        for (double v : row) sum += v;
        EXPECT_NEAR(sum, 1.0, 1e-9);
    }

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}