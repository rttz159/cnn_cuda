#pragma once
#include "Tensor.h"
#include <vector>
#include <cmath>
#include <iostream>

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

class MultiLayerPerceptron
{
public:
    double learning_rate = 0.001;
    int batch_size = 1;
    LossFunction lossfunction;

    std::vector<int> layer_sizes;
    std::vector<ActivationFunction> activation_functions;

    Tensor<float, 2> input;
    Tensor<float, 2> desire_output;

    std::vector<Tensor<float, 2>> weight;
    std::vector<Tensor<float, 2>> biases;
    std::vector<Tensor<float, 2>> activations;
    std::vector<Tensor<float, 2>> pre_activations;
    std::vector<Tensor<float, 2>> deltas;

    MultiLayerPerceptron(std::vector<int> layers_size, std::vector<ActivationFunction> activation_functions, LossFunction lossfunction, double learning_rate, int batch_size)
    {
        this->learning_rate = learning_rate;
        this->layer_sizes = layers_size;
        this->batch_size = batch_size;
        this->activation_functions = activation_functions;
        this->lossfunction = lossfunction;
        initialize_input_output();
        initialize_weights_and_biases();
        initialize_activations();
        initialize_deltas();
    }

    void forward();
    void backward();

    void set_input(const std::vector<std::vector<float>>& input_batch);
    void set_desire_output(const std::vector<std::vector<float>>& output_batch);

    void train(const std::vector<std::vector<float>>& X_train,
        const std::vector<std::vector<float>>& y_train,
        int epochs);

    std::vector<std::vector<float>> predict_batch(const std::vector<std::vector<float>>& input_batch);

    void print_network_details() const;

private:
    void initialize_activations();
    void initialize_deltas();
    void initialize_weights_and_biases();
    void initialize_input_output();

    static Tensor<float, 2> relu(const Tensor<float, 2>& x);
    static Tensor<float, 2> sigmoid(const Tensor<float, 2>& x);
    static Tensor<float, 2> tanh_act(const Tensor<float, 2>& x);
    static Tensor<float, 2> softmax(const Tensor<float, 2>& x);
    static Tensor<float, 2> apply_activation(const Tensor<float, 2>& x, ActivationFunction func);

    static float mean_squared_error(const Tensor<float, 2>& prediction, const Tensor<float, 2>& target);
    static float cross_entropy_loss(const Tensor<float, 2>& prediction, const Tensor<float, 2>& target);
    static float compute_loss(const Tensor<float, 2>& prediction,
        const Tensor<float, 2>& target,
        LossFunction loss_func);

    static Tensor<float, 2> relu_derivative(const Tensor<float, 2>& x);
    static Tensor<float, 2> sigmoid_derivative(const Tensor<float, 2>& x);
    static Tensor<float, 2> tanh_derivative(const Tensor<float, 2>& x);
    static Tensor<float, 2> mse_derivative(const Tensor<float, 2>& prediction, const Tensor<float, 2>& target);
    static Tensor<float, 2> softmax_cross_entropy_derivative(const Tensor<float, 2>& prediction,
        const Tensor<float, 2>& target);
    static Tensor<float, 2> activation_derivative(const Tensor<float, 2>& pre_activation,
        ActivationFunction func);
    static Tensor<float, 2> loss_derivative(const Tensor<float, 2>& prediction,
        const Tensor<float, 2>& target,
        LossFunction loss_func,
        ActivationFunction output_activation);
};