#pragma once
#include "activation_utils.h" 
#include "cuda_tensor.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ inline float sigmoid_cuda(float x) {
    x = fmaxf(fminf(x, 40.0f), -40.0f);
    return 1.0f / (1.0f + expf(-x));
}

__device__ inline float sigmoid_derivative_cuda(float x) {
    float s = sigmoid_cuda(x);
    return s * (1.0f - s);
}

__global__ void activation_kernel(float* input, float* output, size_t size);

__global__ void activation_derivative_kernel(float* input, float* output, size_t size);

__global__ void reduce_rows_kernel(float* input, float* output, size_t rows, size_t cols);

__global__ void broadcast_to_rows_kernel(const float* src, float* dst, size_t rows, size_t cols);

__global__ void update_weights_kernel(float* W, const float* grad, float lr, size_t size);

__global__ void kernel_mse_loss(const float *prediction, const float *target, float *loss_buffer, size_t size);

void apply_activation_cuda(CudaTensor<2>& input, CudaTensor<2>& output);
void apply_activation_derivative_cuda(CudaTensor<2>& input, CudaTensor<2>& output);
void broadcast_to_rows_cuda(const CudaTensor<2>& src, CudaTensor<2>& dst);
void reduce_rows_cuda(CudaTensor<2>& input, CudaTensor<2>& output);
void update_weights_cuda(CudaTensor<2>& W, const CudaTensor<2>& grad, float lr, float batch_size_float) ;
float compute_loss_gpu(const CudaTensor<2>& prediction, const CudaTensor<2>& target);
