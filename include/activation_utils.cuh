#pragma once
#include "activation_utils.h" 
#include "cuda_tensor.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define LEAKY_RELU_ALPHA 0.01f

__device__ inline float relu_cuda(float x) {
    return x > 0.0f ? x : LEAKY_RELU_ALPHA * x;
}

__device__ inline float sigmoid_cuda(float x) {
    x = fmaxf(fminf(x, 40.0f), -40.0f);
    return 1.0f / (1.0f + expf(-x));
}

__device__ inline float tanh_act_cuda(float x) {
    return tanhf(x);
}

__device__ inline float sigmoid_derivative_cuda(float x) {
    float s = sigmoid_cuda(x);
    return s * (1.0f - s);
}

__device__ inline float tanh_derivative_cuda(float x) {
    x = fmaxf(fminf(x, 40.0f), -40.0f);
    float t = tanhf(x);
    return 1.0f - t * t;
}

__device__ inline float relu_derivative_cuda(float x) {
    return x > 0.0f ? 1.0f : LEAKY_RELU_ALPHA;
}

__global__ void softmax_kernel(float* input, float* output, size_t rows, size_t cols);

__global__ void activation_kernel(float* input, float* output, size_t size, ActivationFunction func);

__global__ void activation_derivative_kernel(float* input, float* output, size_t size, ActivationFunction func);

__global__ void softmax_cross_entropy_grad_kernel(float* prediction, float* target, float* grad, size_t batch, size_t classes);

__global__ void reduce_rows_kernel(float* input, float* output, size_t rows, size_t cols);

__global__ void broadcast_to_rows_kernel(const float* src, float* dst, size_t rows, size_t cols);

__global__ void update_weights_kernel(float* W, const float* grad, float lr, size_t size);

__global__ void kernel_mse_loss(const float *prediction, const float *target, float *loss_buffer, size_t size);

__global__ void kernel_softmax_cross_entropy_loss(const float *logits, const float *target, float *loss_buffer, int batch_size, int num_classes);

void apply_activation_cuda(CudaTensor<2>& input, CudaTensor<2>& output, ActivationFunction func);

void apply_activation_derivative_cuda(CudaTensor<2>& input, CudaTensor<2>& output, ActivationFunction func);
void apply_softmax_cross_entropy_grad_cuda(CudaTensor<2>& prediction, CudaTensor<2>& target, CudaTensor<2>& grad);
void broadcast_to_rows_cuda(const CudaTensor<2>& src, CudaTensor<2>& dst);
void reduce_rows_cuda(CudaTensor<2>& input, CudaTensor<2>& output);
void update_weights_cuda(CudaTensor<2>& W, const CudaTensor<2>& grad, float lr, float batch_size_float) ;

float compute_loss_gpu(const CudaTensor<2>& prediction,
                              const CudaTensor<2>& target,
                              LossFunction loss_func,
                              ActivationFunction output_activation);
