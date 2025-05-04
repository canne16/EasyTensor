# EasyTensor

EasyTensor is an educational C++ project for experimenting with tensor computations and building a simple tensor computation graph compiler. It provides basic tensor operations, a computation graph abstraction, and a minimal neural network inference engine.

## Features

- 2D tensor (matrix) operations: addition, subtraction, multiplication, division, matrix multiplication
- Common neural network operations: ReLU, Softmax, Convolution
- Computation graph with node-based operations
- Simple neural network inference API
- Extensible operation and node system

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/EasyTensor.git
cd EasyTensor
```

**Dependency:**  
To use the graph export/visualization feature (`NeuralNetwork::exportGraph`), you need [Graphviz](https://graphviz.gitlab.io/) installed and `dot` available in your PATH.

On Ubuntu/Debian:

```bash
sudo apt-get install graphviz
```

Build with CMake:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Running Tests

After building, you can run the tests using:

```bash
cd build
ctest
```

or directly:

```bash
./runTests
```

## Usage

A typical usage example (see `src/main.cc`):

```cpp
#include "etc.hh"
using namespace etc;

int main() {
    Tensor input({{1.0, 2.0, 3.0}, {3.0, 4.0, 5.0}, {5.0, 6.0, 7.0}});
    Tensor weight1({{2, 1, 0}, {0, 1, 2}, {1, 0, 1}});
    Tensor weight2({{1, -1, 0}, {0, 1, -1}, {-1, 0, 1}});
    Tensor kernel({{1.0, 0.0}, {0.0, 1.0}});

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
    nn.exportGraph("network_graph");

    const auto& output = nn.infer();
    std::cout << output << std::endl;
}
```

## Project Structure

- `include/`: Header files for tensors, operations, and computation graph
- `src/`: Implementation files
- `main.cc`: Example usage and entry point
