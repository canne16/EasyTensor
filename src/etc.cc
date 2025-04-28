#include <vector>
#include <memory>
#include <fstream>
#include <unordered_set>
#include <string>
#include <iostream>
#include <atomic>
#include <functional>

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

std::shared_ptr<IOperation> NeuralNetwork::addOp(std::shared_ptr<IOperation> op) {
    // Set the parent node as the first argument of the operation, if any
    const auto& args = op->getArgs();
    if (!args.empty()) {
        root_ = std::shared_ptr<INode>(const_cast<INode*>(args[0]), [](INode*){}); // non-owning
    }
    root_ = op;
    return op;
}

void NeuralNetwork::exportGraph(const std::string& filename) const {
    std::ofstream file(filename + ".dot");
    file << "digraph NeuralNetwork {\n";
    file << "    rankdir=LR;\n";
    file << "    node [shape=box, style=filled, fillcolor=lightgray];\n";

    std::unordered_set<size_t> visited;
    std::function<void(const INode*)> dfs = [&](const INode* node) {
        if (!node || visited.count(node->getId())) return;
        visited.insert(node->getId());
        // Use unique node name: node<ID>
        file << "    node" << node->getId() << " [label=\"" << node->getOpName() << "\"];\n";
        for (const INode* child : node->getChildren()) {
            if (child) {
                // Use unique node names for edges as well
                file << "    node" << child->getId() << " -> node" << node->getId() << ";\n";
                dfs(child);
            }
        }
    };
    dfs(root_.get());

    file << "}\n";
    file.close();

    std::string command = "dot -Tsvg " + filename + ".dot -o " + filename + ".svg";
    system(command.c_str());
    std::cout << "Generated: " << filename << ".svg\n";
}

}