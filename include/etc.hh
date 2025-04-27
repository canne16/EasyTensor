#pragma once

#include <vector>
#include <memory>

namespace etc {

// stores multi dimensional data in NCHW format
class Tensor {};

class INode {
    Tensor result_;
    
public:
    virtual Tensor evaluate() const = 0;
    const Tensor& getResult() const {
        return result_;
    }
};

class IOperation : public INode {
public:
    virtual void setArgs(const std::vector<INode*>& args) = 0;
    virtual const std::vector<INode*>& getArgs() const = 0;
};

class BinaryOperation : public IOperation {
    std::vector<INode*> args_;
    Tensor rhs_;
public:
    BinaryOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs);
    BinaryOperation(const std::shared_ptr<INode> lhs, const std::shared_ptr<INode> rhs);

    void setArgs(const std::vector<INode*>& args) override {
        args_ = args;
    }

    const std::vector<INode*>& getArgs() const override {
        return args_;
    }
};

class ScalarAddOperation : public BinaryOperation {};
class ScalarSubOperation : public BinaryOperation {};
class ScalarMulOperation : public BinaryOperation {};
class MatMulOperation    : public BinaryOperation {};
class ConvolOperatopn    : public BinaryOperation {};

class UnaryOperation : public IOperation {
    std::vector<INode*> arg_;
public:
    UnaryOperation(const std::shared_ptr<INode> arg);
    
    void setArgs(const std::vector<INode*>& args) override {
        arg_[0] = args[0];
    }
    const std::vector<INode*>& getArgs() const override {
        return arg_;
    }
};

class ReLUOperation      : public UnaryOperation {};
class SoftmaxOperation   : public UnaryOperation {};


class InputData : public INode {
    Tensor tensor_;
public:
    InputData(const Tensor& tensor) : tensor_(tensor) {};
    Tensor evaluate() const override {
        return tensor_;
    }
};

class NeuralNetwork {
    std::shared_ptr<INode> root_ = nullptr;
public:
    std::shared_ptr<IOperation> addOp(std::shared_ptr<IOperation> op);
    Tensor infer();
    INode* getRoot() const {
        return root_.get();
    }
};

}