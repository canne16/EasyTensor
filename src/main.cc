#include "etc.hh"
#include <iostream>

using namespace etc;

int main() {

    auto print_shape = [](const Tensor& t, const std::string& name) {
        std::cout << name << " shape: " << t.values.size();
        if (!t.values.empty()) std::cout << " x " << t.values[0].size();
        std::cout << std::endl;
    };

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
    nn.exportGraph("network_graph");

    std::cout << output << std::endl;
    
}