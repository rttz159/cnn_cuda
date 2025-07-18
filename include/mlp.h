#pragma once
#include "tensor.h"
#include "cuda_tensor.cuh"
#include "activation_utils.h"
#include "activation_utils.cuh"
#include <vector>
#include <cmath>
#include <iostream>
#include <memory>

/*
    Follow source code, implement MSE loss and Sigmoid activation function
*/

class MultiLayerPerceptron
{
public:
    bool cudaEnabled;
    double learning_rate = 0.1;
    int batch_size;
    float last_loss_gpu = 0.0f;

    std::vector<int> layer_sizes;

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

    MultiLayerPerceptron(std::vector<int> layers_size, double learning_rate, int batch_size, bool cudaEnabled = false)
    {
        this->learning_rate = learning_rate;
        this->layer_sizes = layers_size;
        this->batch_size = batch_size;
        this->cudaEnabled = cudaEnabled;
        initialize_network();
    }

    void forward();
    void backward();
    Tensor<2> fw(Tensor<2>& input);
    Tensor<2> bp(Tensor<2>& grad_output);
    CudaTensor<2> fw_cuda(CudaTensor<2> &input);
    CudaTensor<2> bp_cuda(CudaTensor<2> &grad_output);

    void set_input(const std::vector<std::vector<float>> &input_batch);
    void set_desire_output(const std::vector<std::vector<float>> &output_batch);

    void train(const std::vector<std::vector<float>> &X_train,
               const std::vector<std::vector<float>> &y_train,
               int epochs);

    std::vector<std::vector<float>> predict_batch(const std::vector<std::vector<float>> &input_batch);

    void print_network_details() const;

    float evaluate_accuracy(const std::vector<std::vector<float>> &X, const std::vector<std::vector<float>> &y);

    void initialize_network();

private:
    void initialize_activations();
    void initialize_deltas();
    void initialize_weights_and_biases();
    void initialize_input_output();
};