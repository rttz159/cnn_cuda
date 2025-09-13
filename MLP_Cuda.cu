#include "MLP_Cuda.h"
#include "utilities_kernel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cstdlib>

#define CUDA_CHECK(call) do { cudaError_t err = call;  if (err != cudaSuccess) {  std::cout << "CUDA Error: " << cudaGetErrorString(err)  << " at " << __FILE__ << ":" << __LINE__ << std::endl;  exit(EXIT_FAILURE);  }  } while(0)

float cuda_rand(){
	return (float)(rand() % 100)/1000 ;
}

MLP_CUDA::MLP_CUDA(std::vector<int> num_of_neurons,
                   float learning_rate, int batch_size,
                   int time_step, float bias) {
    this->num_of_neurons = num_of_neurons;
    this->learning_rate  = learning_rate;
    this->batch_size     = batch_size;
    this->time_step      = time_step;
    this->bias           = bias;

    float *input_activations;
    CUDA_CHECK(cudaMalloc(&input_activations, batch_size * num_of_neurons[0] * sizeof(float)));
    activations.push_back(input_activations);
    
    CUDA_CHECK(cudaStreamCreate(&s_grad));
    CUDA_CHECK(cudaStreamCreate(&s_adamA));
    CUDA_CHECK(cudaStreamCreate(&s_adamB));

    CUDA_CHECK(cudaEventCreate(&ev_grads_done));

    cudaStream_t s_copy, s_zero;
    CUDA_CHECK(cudaStreamCreate(&s_copy));
    CUDA_CHECK(cudaStreamCreate(&s_zero));

    for (int i = 1; i < num_of_neurons.size(); i++) {
        int weight_size = num_of_neurons[i-1] * num_of_neurons[i];
        int bias_size   = num_of_neurons[i];

        float *temp_weight, *temp_bias, *temp_activations, *temp_preactivations;
        float *temp_m_w, *temp_v_w, *temp_m_b, *temp_v_b;

        CUDA_CHECK(cudaMalloc(&temp_weight, weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&temp_bias,   bias_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&temp_m_w,    weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&temp_v_w,    weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&temp_m_b,    bias_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&temp_v_b,    bias_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&temp_activations,   batch_size * bias_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&temp_preactivations, batch_size * bias_size * sizeof(float)));

        CUDA_CHECK(cudaMemsetAsync(temp_m_w, 0, weight_size * sizeof(float), s_zero));
        CUDA_CHECK(cudaMemsetAsync(temp_v_w, 0, weight_size * sizeof(float), s_zero));
        CUDA_CHECK(cudaMemsetAsync(temp_m_b, 0, bias_size * sizeof(float),   s_zero));
        CUDA_CHECK(cudaMemsetAsync(temp_v_b, 0, bias_size * sizeof(float),   s_zero));

        std::vector<float> h_weights(weight_size);
        std::vector<float> h_biases(bias_size);
        for (int j = 0; j < weight_size; j++) h_weights[j] = (float)cuda_rand();
        for (int j = 0; j < bias_size; j++)   h_biases[j]  = bias;

        CUDA_CHECK(cudaMemcpyAsync(temp_weight, h_weights.data(),
                                   weight_size * sizeof(float),
                                   cudaMemcpyHostToDevice, s_copy));
        CUDA_CHECK(cudaMemcpyAsync(temp_bias, h_biases.data(),
                                   bias_size * sizeof(float),
                                   cudaMemcpyHostToDevice, s_copy));

        weights.push_back(temp_weight);
        biases.push_back(temp_bias);
        activations.push_back(temp_activations);
        preactivations.push_back(temp_preactivations);
        adam_first_weight.push_back(temp_m_w);
        adam_second_weight.push_back(temp_v_w);
        adam_first_bias.push_back(temp_m_b);
        adam_second_bias.push_back(temp_v_b);
    }

    int max_width = 0;
    for (int i = 1; i < num_of_neurons.size(); i++) {
        max_width = std::max(max_width, num_of_neurons[i]);
    }
    CUDA_CHECK(cudaMalloc(&d_error_buf, batch_size * num_of_neurons.back() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sig_buf,   batch_size * max_width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp_buf,   batch_size * max_width * sizeof(float)));

    d_AT_buf.resize(num_of_neurons.size());
    d_gradW_buf.resize(num_of_neurons.size());
    d_gradb_buf.resize(num_of_neurons.size());

    for (int l = 1; l < num_of_neurons.size(); l++) {
        int n_lm1 = num_of_neurons[l-1];
        int n_l   = num_of_neurons[l];
        CUDA_CHECK(cudaMalloc(&d_AT_buf[l],    n_lm1 * batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gradW_buf[l], n_lm1 * n_l * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gradb_buf[l], n_l * sizeof(float)));
    }

    CUDA_CHECK(cudaStreamSynchronize(s_copy));
    CUDA_CHECK(cudaStreamSynchronize(s_zero));
    cudaStreamDestroy(s_copy);
    cudaStreamDestroy(s_zero);
}

MLP_CUDA::~MLP_CUDA() {
    int num_layers = num_of_neurons.size() - 1;

    for (int i = 0; i < num_layers; i++) {
        if (weights[i]) CUDA_CHECK(cudaFree(weights[i]));
        if (biases[i]) CUDA_CHECK(cudaFree(biases[i]));
        if (activations[i]) CUDA_CHECK(cudaFree(activations[i]));
        if (preactivations[i]) CUDA_CHECK(cudaFree(preactivations[i]));
        if (adam_first_weight[i]) CUDA_CHECK(cudaFree(adam_first_weight[i]));
        if (adam_second_weight[i]) CUDA_CHECK(cudaFree(adam_second_weight[i]));
        if (adam_first_bias[i]) CUDA_CHECK(cudaFree(adam_first_bias[i]));
        if (adam_second_bias[i]) CUDA_CHECK(cudaFree(adam_second_bias[i]));
        if (deltas[i]) CUDA_CHECK(cudaFree(deltas[i]));
    }

    if (d_error_buf) cudaFree(d_error_buf);
    if (d_sig_buf)   cudaFree(d_sig_buf);
    if (d_tmp_buf)   cudaFree(d_tmp_buf);
    for (int l = 1; l < d_AT_buf.size(); l++) {
        if (d_AT_buf[l])    cudaFree(d_AT_buf[l]);
        if (d_gradW_buf[l]) cudaFree(d_gradW_buf[l]);
        if (d_gradb_buf[l]) cudaFree(d_gradb_buf[l]);
    }

    weights.clear();
    biases.clear();
    activations.clear();
    preactivations.clear();
    adam_first_weight.clear();
    adam_second_weight.clear();
    adam_first_bias.clear();
    adam_second_bias.clear();

    cudaStreamDestroy(s_grad);
    cudaStreamDestroy(s_adamA);
    cudaStreamDestroy(s_adamB);
    cudaEventDestroy(ev_grads_done);
}

std::vector<std::vector<float>> MLP_CUDA::get_outputs(){
    int batch_size = this->batch_size;
    int output_size = num_of_neurons.back();
    
    std::vector<float> h_output(batch_size * output_size);
    
    CUDA_CHECK(cudaMemcpy(h_output.data(), 
                      activations.back(),
                      batch_size * output_size * sizeof(float), 
                      cudaMemcpyDeviceToHost));
                      
    std::vector<std::vector<float>> outputs(batch_size, std::vector<float>(output_size));
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < output_size; j++) {
            outputs[i][j] = static_cast<float>(h_output[i * output_size + j]);
        }
    }

    return outputs;
}

std::vector<std::vector<float>>& MLP_CUDA::get_input_gradients() {
    return d_input;
}

void MLP_CUDA::run(const std::vector<std::vector<float>>& inputs){
    int new_batch_size = inputs.size();
    if (new_batch_size != batch_size) {
        resize_batch(new_batch_size);
    }
    std::vector<float> flatten_input;
    flatten_input.reserve(inputs.size() * inputs[0].size());
    for (const auto& row : inputs) {
        flatten_input.insert(flatten_input.end(), row.begin(), row.end());
    }

    float *d_input = activations[0];
    CUDA_CHECK(cudaMemcpy(d_input, flatten_input.data(), flatten_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    int num_layers = num_of_neurons.size() - 1;
    for (int i = 0; i < num_layers; i++) { 
        float *A_prev = activations[i];

        device_matrix_mul(A_prev, weights[i], preactivations[i], batch_size, num_of_neurons[i], num_of_neurons[i+1]);
        add_bias(preactivations[i], biases[i], batch_size, num_of_neurons[i+1]);
        apply_sigmoid(preactivations[i], activations[i+1], batch_size * num_of_neurons[i+1]);
    }
}

void MLP_CUDA::bp(const std::vector<std::vector<float>>& error) {
    const int L = static_cast<int>(num_of_neurons.size()) - 1;
    if (L <= 0) return;

    if ((int)deltas.size() != L + 1) {
        deltas.assign(L + 1, nullptr);
        for (int l = 1; l <= L; l++) {
            CUDA_CHECK(cudaMalloc(&deltas[l], batch_size * num_of_neurons[l] * sizeof(float)));
        }
    }

    int out_size = num_of_neurons[L];
    std::vector<float> h_error(batch_size * out_size, 0.0f);
    for (int b = 0; b < batch_size; b++) {
        const auto& eb = (b < (int)error.size()) ? error[b] : std::vector<float>{};
        for (int j = 0; j < out_size && j < (int)eb.size(); j++) {
            h_error[b * out_size + j] = eb[j];
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(d_error_buf, h_error.data(),
                               h_error.size() * sizeof(float),
                               cudaMemcpyHostToDevice, s_grad));

    {
        int sz = batch_size * out_size;
        sigmoid_derivative(activations[L], d_sig_buf, sz, s_grad);
        elementwise_multiply(d_error_buf, d_sig_buf, deltas[L], sz, s_grad);
    }

    for (int l = L - 1; l >= 1; l--) {
        int n_l   = num_of_neurons[l];
        int n_lp1 = num_of_neurons[l + 1];

        float* d_WT = nullptr;
        CUDA_CHECK(cudaMalloc(&d_WT, n_lp1 * n_l * sizeof(float)));

        device_matrix_transpose(weights[l], d_WT, n_l, n_lp1, s_grad);
        device_matrix_mul(deltas[l+1], d_WT, d_tmp_buf,
                          batch_size, n_lp1, n_l, s_grad);

        sigmoid_derivative(activations[l], d_sig_buf, batch_size * n_l, s_grad);
        elementwise_multiply(d_tmp_buf, d_sig_buf, deltas[l], batch_size * n_l, s_grad);

        CUDA_CHECK(cudaFree(d_WT));
    }

    cudaEventRecord(ev_grads_done, s_grad);

    ++time_step;

    cudaStreamWaitEvent(s_adamA, ev_grads_done, 0);
    cudaStreamWaitEvent(s_adamB, ev_grads_done, 0);

    for (int l = 1; l <= L; l++) {
        int n_lm1 = num_of_neurons[l - 1];
        int n_l   = num_of_neurons[l];

        device_matrix_transpose(activations[l - 1], d_AT_buf[l],
                                batch_size, n_lm1, s_adamA);
        device_matrix_mul(d_AT_buf[l], deltas[l], d_gradW_buf[l],
                          n_lm1, batch_size, n_l, s_adamA);
        adam_update(weights[l - 1], d_gradW_buf[l],
                    adam_first_weight[l - 1], adam_second_weight[l - 1],
                    learning_rate, BETA1, BETA2, EPS,
                    time_step, n_lm1 * n_l, s_adamA);

        mean_across_batch_reduction(deltas[l], d_gradb_buf[l],
                                    batch_size, n_l, s_adamB);
        adam_update(biases[l - 1], d_gradb_buf[l],
                    adam_first_bias[l - 1], adam_second_bias[l - 1],
                    learning_rate, BETA1, BETA2, EPS,
                    time_step, n_l, s_adamB);
    }

    int input_dim = num_of_neurons[0];
    float* d_W0T = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input_buf, batch_size * input_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W0T, input_dim * num_of_neurons[1] * sizeof(float)));

    cudaStreamWaitEvent(s_grad, ev_grads_done, 0);
    device_matrix_transpose(weights[0], d_W0T, input_dim, num_of_neurons[1], s_grad);
    device_matrix_mul(deltas[1], d_W0T, d_input_buf,
                      batch_size, num_of_neurons[1], input_dim, s_grad);
    CUDA_CHECK(cudaFree(d_W0T));

    std::vector<float> h_d_input(batch_size * input_dim);
    CUDA_CHECK(cudaMemcpyAsync(h_d_input.data(), d_input_buf,
                               h_d_input.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, s_grad));
    cudaStreamSynchronize(s_grad);

    d_input.resize(batch_size, std::vector<float>(input_dim));

    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < input_dim; i++) {
            d_input[b][i] = h_d_input[b * input_dim + i];
        }
    }
}


void MLP_CUDA::resize_batch(int new_batch_size) {
    if (new_batch_size == batch_size) return;
    batch_size = new_batch_size;

    for (size_t l = 0; l < activations.size(); l++) {
        if (activations[l]) CUDA_CHECK(cudaFree(activations[l]));

        int act_size = batch_size * num_of_neurons[l];
        CUDA_CHECK(cudaMalloc(&activations[l], act_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(activations[l], 0, act_size * sizeof(float)));

        if (l > 0) {
            if (preactivations[l-1]) CUDA_CHECK(cudaFree(preactivations[l-1]));
            CUDA_CHECK(cudaMalloc(&preactivations[l-1], act_size * sizeof(float)));
            CUDA_CHECK(cudaMemset(preactivations[l-1], 0, act_size * sizeof(float))); 
        }
    }

    int max_width = 0;
    for (int i = 1; i < num_of_neurons.size(); i++) max_width = std::max(max_width, num_of_neurons[i]);

    if (d_error_buf) CUDA_CHECK(cudaFree(d_error_buf));
    if (d_sig_buf)   CUDA_CHECK(cudaFree(d_sig_buf));
    if (d_tmp_buf)   CUDA_CHECK(cudaFree(d_tmp_buf));

    CUDA_CHECK(cudaMalloc(&d_error_buf, batch_size * num_of_neurons.back() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sig_buf, batch_size * max_width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp_buf, batch_size * max_width * sizeof(float)));
}
