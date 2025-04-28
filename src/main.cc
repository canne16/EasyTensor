#include "etc.hh"

using namespace etc;

int main() {
    NeuralNetwork nn;

    Tensor input{};
    Tensor weight{};
    Tensor kernel{};

    auto input_data = std::make_shared<InputData>(input);

    // Build a simple computation graph
    auto add1 = std::make_shared<ScalarAddOperation>(input_data, weight);
    auto mul1 = std::make_shared<ScalarMulOperation>(add1, weight);
    auto conv1 = std::make_shared<ConvolOperation>(mul1, kernel);
    auto add2 = std::make_shared<ScalarAddOperation>(conv1, weight);
    auto add3 = std::make_shared<ScalarAddOperation>(input_data, weight);
    auto conv2 = std::make_shared<ConvolOperation>(add3, kernel);
    auto add4 = std::make_shared<ScalarAddOperation>(conv2, weight);


    nn.addOp(add1);
    nn.addOp(mul1);
    nn.addOp(conv1);
    nn.addOp(add2);
    nn.addOp(add3);
    nn.addOp(conv2);
    nn.addOp(add4);

    nn.exportGraph("network_graph");

    // const auto& output = nn.infer();
}