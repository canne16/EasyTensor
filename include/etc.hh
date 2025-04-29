#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <atomic>
#include <string>

#include "tensor.hh"

namespace etc {

class INode {
    static std::atomic<size_t> global_id_;
    size_t id_;
    std::vector<const INode*> children_;
public:
    INode() : id_(global_id_++) {}
    size_t                            getId() const { return id_; }
    virtual Tensor                    evaluate() const = 0;
    virtual std::vector<const INode*> getChildren() const { return children_; }
    virtual std::string               getOpName() const { return "INode"; }
    void                              addChild(const INode* child) { children_.push_back(child); }
};

class IOperation : public INode {
public:
    virtual void setArgs(const std::vector<INode*>& args) = 0;
    virtual const std::vector<INode*>& getArgs() const = 0;
};

// BINARY OPERATIONS

class BinaryOperation : public IOperation {
    std::vector<INode*> args_;
    Tensor rhs_;
public:
    BinaryOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs);
    BinaryOperation(const std::shared_ptr<INode> lhs, const std::shared_ptr<INode> rhs);

    void                       setArgs(const std::vector<INode*>& args) override {args_ = args;}
    const std::vector<INode*>& getArgs() const override {return args_;}
    std::string                getOpName() const override { return "BinaryOp"; }
    Tensor                     evaluate() const override { return Tensor{}; }
};

class ScalarAddOperation : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    std::string getOpName() const override { return "ScalarAdd"; }
    Tensor evaluate() const override { 
        return getArgs()[0]->evaluate() + getArgs()[1]->evaluate();
    }
};
class ScalarSubOperation : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    std::string getOpName() const override { return "ScalarSub"; }
    Tensor evaluate() const override { 
        return getArgs()[0]->evaluate() - getArgs()[1]->evaluate();
    }
};
class ScalarMulOperation : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    std::string getOpName() const override { return "ScalarMul"; }
    Tensor evaluate() const override { 
        return getArgs()[0]->evaluate() * getArgs()[1]->evaluate();
    }
};
class MatMulOperation    : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    std::string getOpName() const override { return "MatMul"; }
    Tensor evaluate() const override { 
        return getArgs()[0]->evaluate().matmul(getArgs()[1]->evaluate());
    }
};
class ConvolOperation    : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    std::string getOpName() const override { return "Convol"; }
    Tensor evaluate() const override { 
        return Tensor{};
    }
};

// UNARY OPERATIONS

class UnaryOperation : public IOperation {
protected:
    std::vector<INode*> arg_;
public:
    UnaryOperation(const std::shared_ptr<INode> arg) { arg_.push_back(arg.get()); }
    
    void setArgs(const std::vector<INode*>& args) override { arg_ = args; }
    const std::vector<INode*>& getArgs() const override { return arg_; }
    std::string getOpName() const override { return "UnaryOp"; }
    Tensor evaluate() const override { return Tensor{}; }
};

class ReLUOperation      : public UnaryOperation {
public:
    using UnaryOperation::UnaryOperation;
    std::string getOpName() const override { return "ReLU"; }
    Tensor evaluate() const override { return Tensor{}; }
};
class SoftmaxOperation   : public UnaryOperation {
public:
    using UnaryOperation::UnaryOperation;
    std::string getOpName() const override { return "Softmax"; }
    Tensor evaluate() const override { return Tensor{}; }
};

class InputData : public INode {
    Tensor tensor_;
public:
    InputData(const Tensor& tensor) : tensor_(tensor) {};
    std::string getOpName() const override { return "Input"; }
    Tensor evaluate() const override { return tensor_; }
};

class NeuralNetwork {
    std::shared_ptr<INode> root_ = nullptr;
public:
    std::shared_ptr<IOperation> addOp(std::shared_ptr<IOperation> op);
    Tensor infer();
    INode* getRoot() const { return root_.get(); }
    void exportGraph(const std::string& filename) const;
};

}