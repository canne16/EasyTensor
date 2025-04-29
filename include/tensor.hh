#pragma once
#include <vector>
#include <iostream>

namespace etc {

class Tensor {
public:
    std::vector<std::vector<double>> values;

    Tensor();
    Tensor(const std::vector<std::vector<double>>& vals);

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

};

}