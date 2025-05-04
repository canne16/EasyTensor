#include "etc.hh"
#include <cmath>
#include <algorithm>

namespace etc {

Tensor Relu(const Tensor& inp) {
    Tensor out = inp;
    for (auto& row : out.values)
        for (auto& v : row)
            v = v < 0 ? 0 : v;
    return out;
}

Tensor Softmax(const Tensor& inp) {
    Tensor out = inp;
    for (auto& row : out.values) {
        if (row.empty()) continue;
        double max_val = *std::max_element(row.begin(), row.end());
        double sum = 0.0;
        for (auto& v : row) {
            v = std::exp(v - max_val);
            sum += v;
        }
        if (sum != 0) {
            for (auto& v : row)
                v /= sum;
        }
    }
    return out;
}

Tensor Convol(const Tensor& input, const Tensor& kernel) {
    if (input.values.empty() || kernel.values.empty())
        return Tensor{};
    size_t in_rows = input.values.size();
    size_t in_cols = input.values[0].size();
    size_t k_rows = kernel.values.size();
    size_t k_cols = kernel.values[0].size();

    // Calculate padding for 'same' convolution (handle even-sized kernels)
    size_t pad_top    = k_rows / 2;
    size_t pad_left   = k_cols / 2;
    size_t pad_bottom = k_rows - pad_top - 1;
    size_t pad_right  = k_cols - pad_left - 1;

    size_t padded_rows = in_rows + pad_top + pad_bottom;
    size_t padded_cols = in_cols + pad_left + pad_right;
    std::vector<std::vector<double>> padded(padded_rows, std::vector<double>(padded_cols, 0.0));
    for (size_t i = 0; i < in_rows; ++i)
        for (size_t j = 0; j < in_cols; ++j)
            padded[i + pad_top][j + pad_left] = input.values[i][j];

    // Output size matches input size for 'same' convolution
    std::vector<std::vector<double>> out(in_rows, std::vector<double>(in_cols, 0.0));
    for (size_t i = 0; i < in_rows; ++i) {
        for (size_t j = 0; j < in_cols; ++j) {
            double sum = 0.0;
            for (size_t ki = 0; ki < k_rows; ++ki) {
                for (size_t kj = 0; kj < k_cols; ++kj) {
                    sum += padded[i + ki][j + kj] * kernel.values[ki][kj];
                }
            }
            out[i][j] = sum;
        }
    }
    return Tensor(out);
}

}