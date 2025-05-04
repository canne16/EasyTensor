#include <vector>
#include <memory>
#include <fstream>
#include <unordered_set>
#include <string>
#include <iostream>
#include <atomic>
#include <functional>
#include <stack>
#include <unordered_map>


#include "etc.hh"

namespace etc {

std::atomic<size_t> INode::global_id_{0};

BinaryOperation::BinaryOperation(const std::shared_ptr<INode> lhs, const Tensor& rhs) {
    args_.push_back(lhs.get());
    rhs_ = rhs;
}

BinaryOperation::BinaryOperation(const std::shared_ptr<INode> lhs, const std::shared_ptr<INode> rhs) {
    args_.push_back(lhs.get());
    args_.push_back(rhs.get());
}

Tensor NeuralNetwork::infer() {
    #ifndef TESTING
    std::cout << "[infer] Called infer()\n";
    #endif
    if (!root_) {
        #ifndef TESTING
        std::cout << "[infer] root_ is null\n";
        #endif
        return Tensor{};
    }
    #ifndef TESTING
    std::cout << "[infer] root_ is not null, calling evaluate() on node id " << root_->getId() << "\n";
    #endif
    return root_->evaluate();
}

std::shared_ptr<IOperation> NeuralNetwork::addOp(std::shared_ptr<IOperation> op) {
    #ifndef TESTING
    std::cout << "[addOp] Adding operation: " << op->getOpName() << " (id=" << op->getId() << ")\n";
    #endif
    const auto& args = op->getArgs();
    if (!root_) {
        #ifndef TESTING
        std::cout << "[addOp] root_ is null, setting root_ to op id " << op->getId() << "\n";
        #endif
        root_ = op;
    }
    for (const INode* arg : args) {
        #ifndef TESTING
        std::cout << "[addOp]   arg ptr: " << arg << "\n";
        #endif
        if (arg) {
            #ifndef TESTING
            std::cout << "[addOp]   adding child to arg id " << arg->getId() << "\n";
            #endif
            const_cast<INode*>(arg)->addChild(op.get());
        } else {
            #ifndef TESTING
            std::cout << "[addOp]   arg is null\n";
            #endif
        }
    }
    root_ = op;
    return op;
}

void NeuralNetwork::exportGraph(const std::string& filename) const {
    std::cout << "[exportGraph] Exporting graph to " << filename << ".dot\n";
    std::ofstream file(filename + ".dot");
    file << "digraph NeuralNetwork {\n";
    file << "    rankdir=LR;\n";
    file << "    node [shape=box, style=filled, fillcolor=lightgray];\n";

    std::unordered_set<const INode*> all_nodes;
    std::function<void(const INode*)> collect = [&](const INode* node) {
        if (!node || all_nodes.count(node)) return;
        std::cout << "[exportGraph]   collecting node id " << node->getId() << "\n";
        all_nodes.insert(node);
        for (const INode* child : node->getChildren()) {
            collect(child);
        }
        if (auto op = dynamic_cast<const IOperation*>(node)) {
            for (const INode* arg : op->getArgs()) {
                collect(arg);
            }
        }
    };
    collect(root_.get());

    for (const INode* node : all_nodes) {
        file << "    node" << node->getId()
             << " [label=\"" << node->getOpName() << "\\n(id=" << node->getId() << ")\"];\n";
    }

    for (const INode* parent : all_nodes) {
        for (const INode* child : parent->getChildren()) {
            if (child && all_nodes.count(child)) {
                file << "    node" << parent->getId() << " -> node" << child->getId() << ";\n";
            }
        }
    }

    file << "}\n";
    file.close();

    std::string command1 = "dot -Tsvg " + filename + ".dot -o " + filename + ".svg";
    std::cout << "[exportGraph] Running: " << command1 << "\n";
    system(command1.c_str());
    std::cout << "[exportGraph] Generated: " << filename << ".svg\n";
    std::string command2 = "xdg-open " + filename + ".svg";
    system(command2.c_str());
}


}