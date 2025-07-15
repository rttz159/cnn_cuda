#pragma once
#include "tensor.h"

enum class ActivationFunction
{
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    None
};

enum class LossFunction
{
    MSE,
    CrossEntropy
};

namespace Activation {

    inline Tensor<2> relu(const Tensor<2>& x)
    {
        auto shape = x.get_shape();
        Tensor<2> result(shape);
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                result(i, j) = std::max(0.0f, x(i, j));
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

    inline Tensor<2> tanh_act(const Tensor<2>& x)
    {
        auto shape = x.get_shape();
        Tensor<2> result(shape);
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                result(i, j) = std::tanh(x(i, j));
        return result;
    }

    inline Tensor<2> softmax(const Tensor<2>& x)
    {
        auto shape = x.get_shape();
        Tensor<2> result(shape);
        for (size_t i = 0; i < shape[0]; ++i)
        {
            float max_val = x(i, 0);
            for (size_t j = 1; j < shape[1]; ++j)
                max_val = std::max(max_val, x(i, j));

            float sum = 0.0f;
            for (size_t j = 0; j < shape[1]; ++j)
            {
                result(i, j) = std::exp(x(i, j) - max_val);
                sum += result(i, j);
            }

            for (size_t j = 0; j < shape[1]; ++j)
                result(i, j) /= sum;
        }
        return result;
    }

    inline Tensor<2> apply_activation(const Tensor<2>& x, ActivationFunction func)
    {
        switch (func)
        {
        case ActivationFunction::ReLU:
            return relu(x);
        case ActivationFunction::Sigmoid:
            return sigmoid(x);
        case ActivationFunction::Tanh:
            return tanh_act(x);
        case ActivationFunction::Softmax:
            return softmax(x);
        case ActivationFunction::None:
            return x;
        }
        return x;
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

    inline float cross_entropy_loss(const Tensor<2>& prediction, const Tensor<2>& target)
    {
        auto shape = prediction.get_shape();
        float loss = 0.0f;
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
            {
                float p = prediction(i, j);
                float t = target(i, j);
                loss -= t * std::log(std::max(p, 1e-9f));
            }
        return loss / shape[0];
    }

    inline float compute_loss(const Tensor<2>& prediction,
        const Tensor<2>& target,
        LossFunction loss_func)
    {
        switch (loss_func)
        {
        case LossFunction::MSE:
            return mean_squared_error(prediction, target);
        case LossFunction::CrossEntropy:
            return cross_entropy_loss(prediction, target);
        }
        return 0.0f;
    }

    inline Tensor<2> relu_derivative(const Tensor<2>& x)
    {
        auto shape = x.get_shape();
        Tensor<2> grad(shape);
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                grad(i, j) = x(i, j) > 0.0f ? 1.0f : 0.0f;
        return grad;
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

    inline Tensor<2> tanh_derivative(const Tensor<2>& x)
    {
        auto shape = x.get_shape();
        Tensor<2> grad(shape);
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
            {
                float t = std::tanh(x(i, j));
                grad(i, j) = 1.0f - t * t;
            }
        return grad;
    }

    inline Tensor<2> mse_derivative(const Tensor<2>& prediction, const Tensor<2>& target)
    {
        auto shape = prediction.get_shape();
        Tensor<2> grad(shape);
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                grad(i, j) = 2.0f * (prediction(i, j) - target(i, j)) / (shape[0] * shape[1]);
        return grad;
    }

    inline Tensor<2> softmax_cross_entropy_derivative(const Tensor<2>& prediction,
        const Tensor<2>& target)
    {
        auto shape = prediction.get_shape();
        Tensor<2> grad(shape);
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                grad(i, j) = (prediction(i, j) - target(i, j)) / shape[0];
        return grad;
    }

    inline Tensor<2> activation_derivative(const Tensor<2>& pre_activation,
        ActivationFunction func)
    {
        switch (func)
        {
        case ActivationFunction::ReLU:
            return relu_derivative(pre_activation);
        case ActivationFunction::Sigmoid:
            return sigmoid_derivative(pre_activation);
        case ActivationFunction::Tanh:
            return tanh_derivative(pre_activation);
        case ActivationFunction::Softmax:
            return Tensor<2>(pre_activation.get_shape());
        case ActivationFunction::None:
            return Tensor<2>(pre_activation.get_shape());
        }
        return Tensor<2>(pre_activation.get_shape());
    }

    inline Tensor<2> loss_derivative(const Tensor<2>& prediction,
        const Tensor<2>& target,
        LossFunction loss_func,
        ActivationFunction output_activation)
    {
        if (loss_func == LossFunction::CrossEntropy && output_activation == ActivationFunction::Softmax)
        {
            return softmax_cross_entropy_derivative(prediction, target);
        }
        else if (loss_func == LossFunction::MSE)
        {
            return mse_derivative(prediction, target);
        }
        return Tensor<2>(prediction.get_shape());
    }

    inline float softmax_cross_entropy_loss(const Tensor<2>& prediction, const Tensor<2>& target)
    {
        float loss = 0.0f;
        auto shape = prediction.get_shape();
        size_t batch_size = shape[0];
        size_t num_classes = shape[1];

        for (size_t i = 0; i < batch_size; ++i)
        {
            for (size_t j = 0; j < num_classes; ++j)
            {
                float p = prediction(i, j);
                float t = target(i, j);

                if (t > 0)
                    loss -= t * std::log(std::max(p, 1e-7f));
            }
        }

        return loss / static_cast<float>(batch_size);
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

    inline float compute_loss(const Tensor<2>& prediction,
                          const Tensor<2>& target,
                          LossFunction loss_func,
                          ActivationFunction output_activation)
    {
        if (loss_func == LossFunction::CrossEntropy && output_activation == ActivationFunction::Softmax)
        {
            return softmax_cross_entropy_loss(prediction, target);
        }
        else if (loss_func == LossFunction::MSE)
        {
            return compute_mse_loss(prediction, target);
        }
        return 0.0f;
    }


}
