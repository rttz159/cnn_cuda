#pragma once
#include "tensor.h"
#include "cuda_tensor.cuh"
#include "activation_utils.h"
#include "activation_utils.cuh"
#include <vector>
#include <cmath>
#include <iostream>
#include <memory>

class MultiLayerPerceptron
{
public:
    bool cudaEnabled = false;
    double learning_rate = 0.001;
    int batch_size = 1;
    LossFunction lossfunction;

    std::vector<int> layer_sizes;
    std::vector<ActivationFunction> activation_functions;

    Tensor<2> input;
    Tensor<2> desire_output;

    std::vector<CudaTensor<2>> weight_cuda;
    std::vector<CudaTensor<2>> biases_cuda;
    std::vector<CudaTensor<2>> activations_cuda;
    std::vector<CudaTensor<2>> pre_activations_cuda;
    std::vector<CudaTensor<2>> deltas_cuda;

    CudaTensor<2> input_cuda;
    CudaTensor<2> desire_output_cuda;

    std::vector<Tensor<2>> weight;
    std::vector<Tensor<2>> biases;
    std::vector<Tensor<2>> activations;
    std::vector<Tensor<2>> pre_activations;
    std::vector<Tensor<2>> deltas;

    MultiLayerPerceptron(std::vector<int> layers_size, std::vector<ActivationFunction> activation_functions, LossFunction lossfunction, double learning_rate, int batch_size, bool cudaEnabled = false)
    {
        this->learning_rate = learning_rate;
        this->layer_sizes = layers_size;
        this->batch_size = batch_size;
        this->activation_functions = activation_functions;
        this->lossfunction = lossfunction;
        this->cudaEnabled = cudaEnabled;
        initialize_input_output();
        initialize_weights_and_biases();
        initialize_activations();
        initialize_deltas();
    }

    void forward();
    void backward();

    void set_input(const std::vector<std::vector<float>> &input_batch);
    void set_desire_output(const std::vector<std::vector<float>> &output_batch);

    void train(const std::vector<std::vector<float>> &X_train,
               const std::vector<std::vector<float>> &y_train,
               int epochs);

    std::vector<std::vector<float>> predict_batch(const std::vector<std::vector<float>> &input_batch);

    void print_network_details() const;

    float evaluate_accuracy(const std::vector<std::vector<float>> &X, const std::vector<std::vector<float>> &y);

private:
    void initialize_activations();
    void initialize_deltas();
    void initialize_weights_and_biases();
    void initialize_input_output();
};