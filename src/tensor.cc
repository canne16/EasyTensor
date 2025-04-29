#include "tensor.hh"
#include <stdexcept>
#include <iostream>

namespace etc {

Tensor::Tensor() = default;

Tensor::Tensor(const std::vector<std::vector<double>>& vals) : values(vals) {}

Tensor Tensor::operator+(const Tensor& other) const {
    if (values.size() != other.values.size() || (values.size() && values[0].size() != other.values[0].size()))
        throw std::invalid_argument("Tensor sizes must match for addition");
    std::vector<std::vector<double>> result(values.size(), std::vector<double>(values[0].size()));
    for (size_t i = 0; i < values.size(); ++i)
        for (size_t j = 0; j < values[0].size(); ++j)
            result[i][j] = values[i][j] + other.values[i][j];
    return Tensor(result);
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (values.size() != other.values.size() || (values.size() && values[0].size() != other.values[0].size()))
        throw std::invalid_argument("Tensor sizes must match for subtraction");
    std::vector<std::vector<double>> result(values.size(), std::vector<double>(values[0].size()));
    for (size_t i = 0; i < values.size(); ++i)
        for (size_t j = 0; j < values[0].size(); ++j)
            result[i][j] = values[i][j] - other.values[i][j];
    return Tensor(result);
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (values.size() != other.values.size() || (values.size() && values[0].size() != other.values[0].size()))
        throw std::invalid_argument("Tensor sizes must match for multiplication");
    std::vector<std::vector<double>> result(values.size(), std::vector<double>(values[0].size()));
    for (size_t i = 0; i < values.size(); ++i)
        for (size_t j = 0; j < values[0].size(); ++j)
            result[i][j] = values[i][j] * other.values[i][j];
    return Tensor(result);
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (values.size() != other.values.size() || (values.size() && values[0].size() != other.values[0].size()))
        throw std::invalid_argument("Tensor sizes must match for division");
    std::vector<std::vector<double>> result(values.size(), std::vector<double>(values[0].size()));
    for (size_t i = 0; i < values.size(); ++i)
        for (size_t j = 0; j < values[0].size(); ++j) {
            if (other.values[i][j] == 0)
                throw std::domain_error("Division by zero in tensor");
            result[i][j] = values[i][j] / other.values[i][j];
        }
    return Tensor(result);
}

// Standard matrix multiplication
Tensor Tensor::matmul(const Tensor& other) const {
    if (values.empty() || other.values.empty() || values[0].size() != other.values.size())
        throw std::invalid_argument("Invalid shapes for matrix multiplication");
    size_t m = values.size();
    size_t n = values[0].size();
    size_t p = other.values[0].size();
    std::vector<std::vector<double>> result(m, std::vector<double>(p, 0.0));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < p; ++j)
            for (size_t k = 0; k < n; ++k)
                result[i][j] += values[i][k] * other.values[k][j];
    return Tensor(result);
}


std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "[";
    for (size_t i = 0; i < t.values.size(); ++i) {
        os << "[";
        for (size_t j = 0; j < t.values[i].size(); ++j) {
            os << t.values[i][j];
            if (j + 1 < t.values[i].size()) os << ", ";
        }
        os << "]";
        if (i + 1 < t.values.size()) os << ", ";
    }
    os << "]";
    return os;
}

}