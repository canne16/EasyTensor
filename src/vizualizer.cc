#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>

#include "etc.hh"

void export_graph_to_dot(const etc::NeuralNetwork& nn, const std::string& filename) {
    std::ofstream file(filename + ".dot");
    file << "digraph NeuralNetwork {\n";
    file << "    rankdir=LR;\n";
    file << "    node [shape=box, style=filled, fillcolor=lightgray];\n";

    // for (const auto& edge : edges) {
    //     file << "    " << edge.from << " -> " << edge.to;

    //     // Add style based on relationship type
    //     if (edge.relationship == "inherits") {
    //         file << " [label=\"inherits\", arrowhead=empty];\n";  // Hollow arrow for inheritance
    //     } else if (edge.relationship == "uses") {
    //         file << " [label=\"uses\", style=dashed];\n";
    //     } else {
    //         file << " [label=\"" << edge.relationship << "\"];\n";
    //     }
    // }

    file << "}\n";
    file.close();

    std::string command = "dot -Tsvg " + filename + ".dot -o " + filename + ".svg";
    system(command.c_str());
    std::cout << "Generated: " << filename << ".svg\n";
}
