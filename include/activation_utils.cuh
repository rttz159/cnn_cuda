#pragma once
#include "activation_utils.h" 
#include "cuda_tensor.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define LEAKY_RELU_ALPHA 0.001

__device__ inline float leaky_relu_cuda(float x) {
    return x > 0.0f ? x : LEAKY_RELU_ALPHA * x;
}

__device__ inline float sigmoid_cuda(float x) {
    x = fmaxf(fminf(x, 40.0f), -40.0f);
    return 1.0f / (1.0f + expf(-x));
}

__device__ inline float sigmoid_derivative_cuda(float x) {
    float s = sigmoid_cuda(x);
    return s * (1.0f - s);
}

__device__ inline float leaky_relu_derivative_cuda(float x) {
    return x > 0.0f ? 1.0f : LEAKY_RELU_ALPHA;
}

__global__ void activation_kernel(float* __restrict__ input, float* __restrict__ output, size_t size);

__global__ void activation_derivative_kernel(float* __restrict__ input, float* __restrict__ output, size_t size);

__global__ void reduce_rows_kernel(float* __restrict__ input, float* __restrict__ output, size_t rows, size_t cols);

__global__ void broadcast_to_rows_kernel(const float* __restrict__ src, float* __restrict__ dst, size_t rows, size_t cols);

__global__ void update_weights_kernel(float* __restrict__ W, const float* __restrict__ grad, float lr, size_t size);

__global__ void kernel_mse_loss(const float* __restrict__ prediction, const float* __restrict__ target, float *loss_buffer, size_t size);

__global__ void broadcast_bias_to_matrix_kernel(const float* __restrict__ bias, float* __restrict__ out, size_t F, size_t N);

__global__ void reduce_columns_kernel(const float* __restrict__ input, float* __restrict__ output, size_t rows, size_t cols);

void apply_leaky_ReLu_cuda(CudaTensor<2>& input, CudaTensor<2>& output);
void apply_leaky_ReLu_derivative_cuda(CudaTensor<2>& input, CudaTensor<2>& output);
void apply_sigmoid_cuda(CudaTensor<2>& input, CudaTensor<2>& output);
void apply_sigmoid_derivative_cuda(CudaTensor<2>& input, CudaTensor<2>& output);
void broadcast_to_rows_cuda(const CudaTensor<2>& src, CudaTensor<2>& dst);
void broadcast_bias_cuda(const CudaTensor<1>& bias, CudaTensor<2>& out);
void reduce_rows_cuda(CudaTensor<2>& input, CudaTensor<2>& output);
void reduce_columns_cuda(const CudaTensor<2>& input, CudaTensor<1>& output);
void update_weights_cuda(CudaTensor<2>& W, const CudaTensor<2>& grad, float lr, float batch_size_float) ;
float compute_loss_gpu(const CudaTensor<2>& prediction, const CudaTensor<2>& target);
