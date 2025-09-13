#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

#include <cuda_runtime.h>

void device_matrix_mul(const float *A, const float *B, float *C,
                       int N1, int N2, int N3, cudaStream_t stream = 0);

void device_matrix_transpose(const float *input, float *output,
                             int rows, int cols, cudaStream_t stream = 0);

void add_bias(float* d_Z, const float* d_b,
              int batch_size, int layer_size, cudaStream_t stream = 0);

void apply_sigmoid(const float* d_input, float* d_output,
                   int size, cudaStream_t stream = 0);

void sigmoid_derivative(const float* d_input, float* d_output,
                        int size, cudaStream_t stream = 0);

void elementwise_multiply(const float* a, const float* b, float* out,
                          int size, cudaStream_t stream = 0);

void adam_update(float* param, const float* grad, float* m, float* v,
                 float lr, float beta1, float beta2,
                 float epsilon, int timestep, int size,
                 cudaStream_t stream = 0);

void mean_across_batch_reduction(const float* delta, float* grad_bias,
                                 int batch_size, int layer_size, cudaStream_t stream = 0);

void im2col_wrapper(const float* d_input, float* d_output,
                    int N, int C, int H, int W, int F, int pad, int stride,
                    cudaStream_t stream = 0);

void col2im_wrapper(const float* d_input_col, float* d_output,
                    int N, int C, int H, int W, int F, int pad, int stride,
                    cudaStream_t stream = 0);

void leaky_relu_host(float* d_data, int size, float alpha,
                     cudaStream_t stream = 0);

void leaky_relu_derivative(const float* d_input, float* d_output,
                           int size, float alpha, cudaStream_t stream = 0);

void add_bias_per_filter(float* Z, const float* b,
                         int N, int K, int out_hw, cudaStream_t stream = 0);

void add_inplace(float* dst, const float* src, int size,
                 cudaStream_t stream = 0);

void compute_bias_grad(const float* delta, float* grad_bias,
                       int N, int K, int out_hw, cudaStream_t stream = 0);
