#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <atomic>
#include <string> // <-- Add this line

namespace etc {

// stores multi dimensional data in NCHW format
class Tensor {};

class INode {
    Tensor result_;
    static std::atomic<size_t> global_id_;
    size_t id_;
public:
    INode() : id_(global_id_++) {}
    virtual Tensor evaluate() const = 0;
    const Tensor& getResult() const {
        return result_;
    }
    size_t getId() const { return id_; }
    virtual std::vector<const INode*> getChildren() const = 0;
    virtual std::string getOpName() const { return "INode"; } // <-- Ensure signature matches everywhere
};

class IOperation : public INode {
public:
    virtual void setArgs(const std::vector<INode*>& args) = 0;
    virtual const std::vector<INode*>& getArgs() const = 0;
    std::vector<const INode*> getChildren() const override {
        const auto& args = getArgs();
        return std::vector<const INode*>(args.begin(), args.end());
    }
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

    std::string getOpName() const override { return "BinaryOp"; }
    // Implement evaluate to avoid abstract class error
    Tensor evaluate() const override { return Tensor{}; }
};

class ScalarAddOperation : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    std::string getOpName() const override { return "ScalarAdd"; }
    Tensor evaluate() const override { return Tensor{}; }
};
class ScalarSubOperation : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    std::string getOpName() const override { return "ScalarSub"; }
    Tensor evaluate() const override { return Tensor{}; }
};
class ScalarMulOperation : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    std::string getOpName() const override { return "ScalarMul"; }
    Tensor evaluate() const override { return Tensor{}; }
};
class MatMulOperation    : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    std::string getOpName() const override { return "MatMul"; }
    Tensor evaluate() const override { return Tensor{}; }
};
class ConvolOperation    : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    std::string getOpName() const override { return "Convol"; }
    Tensor evaluate() const override { return Tensor{}; }
};

class UnaryOperation : public IOperation {
protected:
    std::vector<INode*> arg_;
public:
    UnaryOperation(const std::shared_ptr<INode> arg) { arg_.push_back(arg.get()); }
    
    void setArgs(const std::vector<INode*>& args) override {
        arg_ = args;
    }
    const std::vector<INode*>& getArgs() const override {
        return arg_;
    }
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
    Tensor evaluate() const override {
        return tensor_;
    }
    std::vector<const INode*> getChildren() const override { return {}; }
    std::string getOpName() const override { return "Input"; }
};

class NeuralNetwork {
    std::shared_ptr<INode> root_ = nullptr;
public:
    std::shared_ptr<IOperation> addOp(std::shared_ptr<IOperation> op);
    Tensor infer();
    INode* getRoot() const {
        return root_.get();
    }
    void exportGraph(const std::string& filename) const;
};

}