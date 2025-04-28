#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>

#include "etc.hh"

namespace etc {
        
void export_graph_to_dot(const etc::NeuralNetwork& nn, const std::string& filename) {
    nn.exportGraph(filename);
    
    std::string command = "dot -Tsvg " + filename + ".dot -o " + filename + ".svg";
    system(command.c_str());
    std::cout << "Generated: " << filename << ".svg\n";
}

}