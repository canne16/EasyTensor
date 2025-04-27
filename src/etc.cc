#include <vector>
#include <memory>

#include "etc.hh"

namespace etc {

BinaryOperation::BinaryOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs) {
    args_.push_back(lhs.get());
    rhs_ = rhs;
}

BinaryOperation::BinaryOperation(const std::shared_ptr<INode> lhs, const std::shared_ptr<INode> rhs) {
    args_.push_back(lhs.get());
    args_.push_back(rhs.get());
}

UnaryOperation::UnaryOperation(const std::shared_ptr<INode> arg) {
    arg_.push_back(arg.get());
}

NeuralNetwork::addOp(std::shared_ptr<IOperation> op) {
    
    if (root_ == nullptr) {
        root_ = op;
    } else {
        root_.setArgs({root_.get(), op.get()});
    }
    return op;

}


}