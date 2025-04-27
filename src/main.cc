#include "etc.hh"

using namespace etc;

int main() {
    NeuralNetwork nn;

    Tensor input{};
    Tensor weight{};

    const auto& input_data = std::make_shared<InputData>(input);

    // nn.addOp(std::make_shared<ScalarAddOperation>(input_data, weight));
    // const auto& output = nn.infer();
}