#include "etc.hh"
#include <iostream>

using namespace etc;

int main() {
    NeuralNetwork nn;

    Tensor input({{1.0, 2.0}, {3.0, 4.0}});
    Tensor weight({{0.5, 0.5}, {0.5, 0.5}});
    Tensor kernel({{1.0, 0.0}, {0.0, 1.0}});

    auto input_data = std::make_shared<InputData>(input);

    // Build a simple computation graph
    auto add1 = std::make_shared<ScalarAddOperation>(input_data, weight);
    auto mul1 = std::make_shared<ScalarMulOperation>(add1, weight);
    // auto conv1 = std::make_shared<ConvolOperation>(mul1, kernel);
    // auto add3 = std::make_shared<ScalarAddOperation>(input_data, weight);
    // auto conv2 = std::make_shared<ConvolOperation>(add3, kernel);
    // auto add4 = std::make_shared<ScalarAddOperation>(conv2, weight);


    nn.addOp(add1);
    nn.addOp(mul1);
    // nn.addOp(conv1);
    // nn.addOp(add3);
    // nn.addOp(conv2);
    // nn.addOp(add4);

    nn.exportGraph("network_graph");

    const auto& output = nn.infer();

    std::cout << output;
    
}