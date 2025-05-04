#pragma once

#include "tensor.hh"

namespace etc {    
    Tensor Relu(const Tensor& inp);
    Tensor Softmax(const Tensor& inp);
    Tensor Convol(const Tensor& input, const Tensor& kernel);
}