#include <vector>
#include <memory>

namespace etc {

// stores multi dimensional data in NCHW format
class Tensor {};

class INode {
public:
    virtual Tensor evaluate() const = 0;
};

class IOperation : public INode {
public:
    virtual void setArgs(const std::vector<INode*>& args) = 0;
    virtual const std::vector<INode*>& getArgs() const = 0;
};


class BinaryOperation : public IOperation {
public:
    BinaryOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs);
};

class ScalarAddOperation : public BinaryOperation {};
class ScalarSubOperation : public BinaryOperation {};
class ScalarMulOperation : public BinaryOperation {};
class MatMulOperation    : public BinaryOperation {};
class ConvolOperatopn    : public BinaryOperation {};

class UnaryOperation : public IOperation {
public:
    UnaryOperation(const std::shared_ptr<INode> arg);
};

class ReLUOperation      : public UnaryOperation {};
class SoftmaxOperation   : public UnaryOperation {};

class InputData : public INode {
public:
    InputData(const Tensor& tensor);
};

class NeuralNetwork {
public:
    std::shared_ptr<IOperation> addOp(std::shared_ptr<IOperation> op);
    Tensor infer();
};

}

using namespace etc;

int main() {
    NeuralNetwork nn;

    Tensor input{};
    Tensor weight{};

    const auto& input_data = std::make_shared<InputData>(input);

    nn.addOp(std::make_shared<ScalarAddOperation>(input_data, weight));
    const auto& output = nn.infer();
}
