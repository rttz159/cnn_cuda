#pragma once

#include <vector>

#define BETA1 0.9
#define BETA2 0.999
#define EPS 1e-7

class MLP_CUDA{
public:
    MLP_CUDA(std::vector<int> num_of_neurons, float learning_rate, int batch_size, int time_step, float bias);
    ~MLP_CUDA();

    void run(const std::vector<std::vector<float>>& inputs);
    void bp(const std::vector<std::vector<float>>& error);

    std::vector<std::vector<float>> get_outputs();

    std::vector<std::vector<float>>& get_input_gradients();

    void resize_batch(int new_batch_size);

    std::vector<float*> weights;
    std::vector<float*> biases;
    
    std::vector<float*> activations;
    std::vector<float*> preactivations;

    std::vector<float*> adam_first_weight;
    std::vector<float*> adam_second_weight;

    std::vector<float*> adam_first_bias;
    std::vector<float*> adam_second_bias;
    
    std::vector<float*> deltas;

    std::vector<int> num_of_neurons;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<float>> d_input;

    float* d_error_buf = nullptr;
    float* d_sig_buf   = nullptr;
    float* d_tmp_buf   = nullptr;
    float* d_input_buf = nullptr;
    std::vector<float*> d_AT_buf;    
    std::vector<float*> d_gradW_buf; 
    std::vector<float*> d_gradb_buf;

    float learning_rate;
    float bias;

    int batch_size;
    int time_step;
};