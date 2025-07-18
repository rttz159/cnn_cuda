#pragma once
#include "tensor.h"

#define ALPHA 0.001

namespace Activation {

    inline Tensor<4> leaky_relu(const Tensor<4>& x)
    {
        const auto shape = x.get_shape(); // [N, C, H, W]
        Tensor<4> result(shape);
        for (size_t n = 0; n < shape[0]; ++n)
            for (size_t c = 0; c < shape[1]; ++c)
                for (size_t h = 0; h < shape[2]; ++h)
                    for (size_t w = 0; w < shape[3]; ++w)
                        result(n, c, h, w) = x(n, c, h, w) > 0.0f ? x(n, c, h, w) : ALPHA * x(n, c, h, w);
        return result;
    }

    inline Tensor<2> sigmoid(const Tensor<2>& x)
    {
        auto shape = x.get_shape();
        Tensor<2> result(shape);
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                result(i, j) = 1.0f / (1.0f + std::exp(-x(i, j)));
        return result;
    }

    inline float mean_squared_error(const Tensor<2>& prediction, const Tensor<2>& target)
    {
        auto shape = prediction.get_shape();
        float sum = 0.0f;
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
            {
                float diff = prediction(i, j) - target(i, j);
                sum += diff * diff;
            }
        return sum / (shape[0] * shape[1]);
    }

    inline Tensor<2> sigmoid_derivative(const Tensor<2>& x)
    {
        auto shape = x.get_shape();
        Tensor<2> grad(shape);
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
            {
                float s = 1.0f / (1.0f + std::exp(-x(i, j)));
                grad(i, j) = s * (1.0f - s);
            }
        return grad;
    }

    inline Tensor<4> leaky_relu_derivative(const Tensor<4>& x)
    {
        const auto shape = x.get_shape(); // [N, C, H, W]
        Tensor<4> grad(shape);
        for (size_t n = 0; n < shape[0]; ++n)
            for (size_t c = 0; c < shape[1]; ++c)
                for (size_t h = 0; h < shape[2]; ++h)
                    for (size_t w = 0; w < shape[3]; ++w)
                        grad(n, c, h, w) = x(n, c, h, w) > 0.0f ? 1.0f : ALPHA;
        return grad;
    }

    inline Tensor<2> mse_derivative(const Tensor<2>& prediction, const Tensor<2>& target)
    {
        auto shape = prediction.get_shape();
        Tensor<2> grad(shape);
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                grad(i, j) = 2.0f * (prediction(i, j) - target(i, j)) / shape[0];
        return grad;
    }
    
    inline float compute_mse_loss(const Tensor<2>& prediction, const Tensor<2>& target) {
        float loss = 0.0f;
        auto shape = prediction.get_shape();
        size_t batch_size = shape[0];
        size_t num_outputs = shape[1];

        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < num_outputs; ++j) {
                float diff = prediction(i, j) - target(i, j);
                loss += diff * diff;
            }
        }

        return loss / (batch_size * num_outputs);
    }

}
