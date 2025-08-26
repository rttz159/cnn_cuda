#include "Conv_Cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

#define CUDA_CHECK(call) do { cudaError_t err = call;  if (err != cudaSuccess) {  std::cout << "CUDA Error: " << cudaGetErrorString(err)  << " at " << __FILE__ << ":" << __LINE__ << std::endl;  exit(EXIT_FAILURE);  }  } while(0)

Conv_CUDA::Conv_CUDA(int in_channels, int in_H, int in_W,
                     int num_kernels, int filter_size,
                     int stride_, int padding_,
                     float learning_rate_, int batch_size_, int time_step_,
                     float bias)
{
    this->C = in_channels;
    this->H = in_H;
    this->W = in_W;
    this->K = num_kernels;
    this->F = filter_size;
    this->stride = stride_;
    this->padding = padding_;
    this->learning_rate = learning_rate_;
    this->batch_size = batch_size_;
    this->time_step = time_step_;

    this->H_out = (H + 2 * padding - F) / stride + 1;
    this->W_out = (W + 2 * padding - F) / stride + 1;
    int out_hw = H_out * W_out;

    size_t input_bytes = size_t(batch_size) * C * H * W * sizeof(float);
    size_t filters_bytes = size_t(K) * C * F * F * sizeof(float);
    size_t bias_bytes = size_t(K) * sizeof(float);
    size_t input_col_bytes = size_t(C) * F * F * (batch_size * out_hw) * sizeof(float); 
    size_t output_col_bytes = size_t(K) * (batch_size * out_hw) * sizeof(float);
    size_t delta_bytes = output_col_bytes; 
    size_t grad_filters_bytes = filters_bytes;
    size_t grad_bias_bytes = bias_bytes;

    CUDA_CHECK(cudaMalloc((void**)&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, output_col_bytes)); 
    CUDA_CHECK(cudaMalloc((void**)&d_filters, filters_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_bias, bias_bytes));

    CUDA_CHECK(cudaMalloc((void**)&d_input_col, input_col_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_input_col_T, (batch_size * out_hw) * (C * F * F) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_filters_T, (C * F * F) * K * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&d_delta, delta_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_grad_filters, grad_filters_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_grad_bias, grad_bias_bytes));

    CUDA_CHECK(cudaMalloc((void**)&d_m_filters, grad_filters_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_v_filters, grad_filters_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_m_bias, grad_bias_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_v_bias, grad_bias_bytes));

    CUDA_CHECK(cudaMemset(d_output, 0, output_col_bytes));
    CUDA_CHECK(cudaMemset(d_input_col, 0, input_col_bytes));
    CUDA_CHECK(cudaMemset(d_delta, 0, delta_bytes));
    CUDA_CHECK(cudaMemset(d_grad_filters, 0, grad_filters_bytes));
    CUDA_CHECK(cudaMemset(d_grad_bias, 0, grad_bias_bytes));
    CUDA_CHECK(cudaMemset(d_m_filters, 0, grad_filters_bytes));
    CUDA_CHECK(cudaMemset(d_v_filters, 0, grad_filters_bytes));
    CUDA_CHECK(cudaMemset(d_m_bias, 0, grad_bias_bytes));
    CUDA_CHECK(cudaMemset(d_v_bias, 0, grad_bias_bytes));

    std::vector<float> h_filters(K * C * F * F);
    std::vector<float> h_bias(K);
    for (size_t i = 0; i < h_filters.size(); ++i) h_filters[i] = ((double) (rand() % 100))/1000;
    for (int i = 0; i < K; ++i) h_bias[i] = bias;

    CUDA_CHECK(cudaMemcpy(d_filters, h_filters.data(), filters_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_bytes, cudaMemcpyHostToDevice));
}

Conv_CUDA::~Conv_CUDA() {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filters);
    cudaFree(d_bias);
    cudaFree(d_input_col);
    cudaFree(d_input_col_T);
    cudaFree(d_filters_T);
    cudaFree(d_delta);
    cudaFree(d_grad_filters);
    cudaFree(d_grad_bias);
    cudaFree(d_m_filters);
    cudaFree(d_v_filters);
    cudaFree(d_m_bias);
    cudaFree(d_v_bias);
}

std::vector<std::vector<float>> Conv_CUDA::get_outputs() {
    int out_hw = H_out * W_out;
    int cols = batch_size * out_hw;
    size_t out_bytes = size_t(K) * cols * sizeof(float);
    std::vector<float> h_out(K * cols);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_output, out_bytes, cudaMemcpyDeviceToHost));

    std::vector<std::vector<float>> outs(batch_size, std::vector<float>(K * out_hw));
    for (int n = 0; n < batch_size; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int p = 0; p < out_hw; ++p) {
                int col = n * out_hw + p;
                outs[n][k * out_hw + p] = h_out[k * cols + col];
            }
        }
    }
    return outs;
}

std::vector<std::vector<float>>& Conv_CUDA::get_input_gradients() {
    return h_input_grad;
}

void Conv_CUDA::run(const std::vector<std::vector<float>>& inputs) {
    int new_batch_size = inputs.size();
    if (new_batch_size != batch_size) {
        resize_batch(new_batch_size);
    }
    int out_hw = this->H_out * this->W_out;
    size_t input_bytes = size_t(this->batch_size) * C * H * W * sizeof(float);
    std::vector<float> h_in(this->batch_size * C * H * W);
    for (int n = 0; n < this->batch_size; ++n) {
        if ((int)inputs[n].size() != C * H * W) {
            std::cerr << "Input size mismatch in conv run()." << std::endl;
            exit(EXIT_FAILURE);
        }
        std::copy(inputs[n].begin(), inputs[n].end(), h_in.begin() + n * (C * H * W));
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_in.data(), input_bytes, cudaMemcpyHostToDevice));

    im2col_wrapper(d_input, d_input_col, this->batch_size, C, H, W, F, padding, stride);

    int Kc = C * F * F;
    int cols = this->batch_size * out_hw;
    device_matrix_mul(d_filters, d_input_col, d_output, K, Kc, cols);

    add_bias_per_filter(d_output, d_bias, this->batch_size, K, out_hw);

    leaky_relu_host(d_output, K * cols, 0.01f);
}

void Conv_CUDA::bp(const std::vector<std::vector<float>>& error) {
    int out_hw = H_out * W_out;
    int cols = this->batch_size * out_hw;
    if ((int)error.size() != this->batch_size) {
        std::cerr << "Error batch size mismatch in bp()." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<float> h_delta(K * cols, 0.0f);
    for (int n = 0; n < this->batch_size; ++n) {
        if ((int)error[n].size() != K * out_hw) {
            std::cerr << "Error inner size mismatch in bp()." << std::endl;
            exit(EXIT_FAILURE);
        }
        for (int k = 0; k < K; ++k) {
            for (int p = 0; p < out_hw; ++p) {
                int col = n * out_hw + p;
                h_delta[k * cols + col] = error[n][k * out_hw + p];
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(d_delta, h_delta.data(), K * cols * sizeof(float), cudaMemcpyHostToDevice));

    compute_bias_grad(d_delta, d_grad_bias, this->batch_size, K, out_hw);

    int Kc = C * F * F;
    device_matrix_transpose(d_input_col, d_input_col_T, Kc, cols);

    device_matrix_mul(d_delta, d_input_col_T, d_grad_filters, K, cols, Kc);

    device_matrix_transpose(d_filters, d_filters_T, K, Kc); 
    device_matrix_mul(d_filters_T, d_delta, d_input_col, Kc, K, cols); 

    size_t input_bytes = size_t(batch_size) * C * H * W * sizeof(float);
    CUDA_CHECK(cudaMemset(d_input, 0, input_bytes));
    col2im_wrapper(d_input_col, d_input, batch_size, C, H, W, F, padding, stride);

    size_t input_size = size_t(batch_size) * C * H * W;
    std::vector<float> h_input(input_size);
    CUDA_CHECK(cudaMemcpy(h_input.data(), d_input,
                        input_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

    h_input_grad.resize(batch_size);
    for (int n = 0; n < batch_size; ++n) {
        h_input_grad[n].assign(h_input.begin() + n * (C*H*W),
                            h_input.begin() + (n+1) * (C*H*W));
    }

    time_step += 1;
    int filters_size = K * Kc;
    int bias_size = K;
    adam_update(d_filters, d_grad_filters, d_m_filters, d_v_filters, learning_rate, BETA1, BETA2, EPS, time_step, filters_size);
    adam_update(d_bias, d_grad_bias, d_m_bias, d_v_bias, learning_rate, BETA1, BETA2, EPS, time_step, bias_size);
}

void Conv_CUDA::resize_batch(int new_batch_size) {
    batch_size = new_batch_size;
    size_t input_bytes = size_t(batch_size) * C * H * W * sizeof(float);
    size_t out_hw = H_out * W_out;
    size_t output_col_bytes = size_t(K) * (batch_size * out_hw) * sizeof(float);
    size_t input_col_bytes = size_t(C) * F * F * (batch_size * out_hw) * sizeof(float);

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_input_col); cudaFree(d_input_col_T);

    CUDA_CHECK(cudaMalloc((void**)&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, output_col_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_input_col, input_col_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_input_col_T, (batch_size * out_hw) * (C * F * F) * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_input, 0, input_bytes));
    CUDA_CHECK(cudaMemset(d_output, 0, output_col_bytes));
    CUDA_CHECK(cudaMemset(d_input_col, 0, input_col_bytes));
    CUDA_CHECK(cudaMemset(d_input_col_T, 0, input_col_bytes));
}