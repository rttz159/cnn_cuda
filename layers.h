#pragma once
#include "tensor.h"

template <typename T, size_t N>
class Layer {
public:
    Layer() = default;
    virtual ~Layer() = default;

    // Pure virtual functions for forward and backward propagation
    virtual Tensor<T, N> forward(const Tensor<T, N>& input) = 0;
    virtual Tensor<T, N> backward(const Tensor<T, N>& output_gradient) = 0;

    // Method to get layer type (for debugging/identification)
    virtual std::string get_type() const = 0;

    // Method to initialize weights
    virtual void initialize_weights() {}

    // Method to update weights
    virtual void update_weights(double learning_rate) {}

protected:
    Tensor<T, N> input_cache;  // Stores input for backward pass
    Tensor<T, N> output_cache; // Stores output for backward pass (e.g., for activation derivatives)
};