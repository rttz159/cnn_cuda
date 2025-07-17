#pragma once

#include "tensor.h"
#include "activation_utils.h"
#include "cuda_tensor.cuh"
#include <vector>
#include <cmath>
#include <cassert>

/*
    Images in the dimension of [Batch_size, Channels, Height, Width]
*/

__global__ void im2col_batch_kernel(
    const float* __restrict__ input,  // [B, C, H, W]
    float* __restrict__ output_col,   // [C*KH*KW, B*OH*OW]
    int B, int C, int H, int W,
    int KH, int KW,
    int OH, int OW,
    int stride, int padding);

__global__ void col2im_batch_kernel(
    const float* __restrict__ col,  // [C*KH*KW, B*OH*OW]
    float* __restrict__ input_grad, // [B, C, H, W]
    int B, int C, int H, int W,
    int KH, int KW,
    int OH, int OW,
    int stride, int padding);

class ConvBlock
{
public:
    int batch_size;
    int in_channels, out_channels;
    int kernel_size, stride, padding;

    Tensor<2> weights; // [F, C×KH×KW]
    Tensor<1> biases;  // [F]

    Tensor<4> input_cache;     // [B, C, H, W]
    Tensor<2> input_cols;      // [C×KH×KW, B×OH×OW]
    Tensor<4> pre_activations; // [B, F, OH, OW]
    Tensor<4> activations;     // [B, F, OH, OW]

    Tensor<2> d_weights; // [F, C×KH×KW]
    Tensor<1> d_biases;  // [F]

    Tensor<4> forward(const Tensor<4> &input_batch);
    Tensor<4> backward(const Tensor<4> &d_out);

private:
    void im2col_batch(const Tensor<4> &input, Tensor<2> &out);
    void col2im_batch(const Tensor<2> &col, Tensor<4> &out);
    void im2col_batch_cuda(const CudaTensor<4>& input, CudaTensor<2>& output_col);
    void col2im_batch_cuda(const CudaTensor<2>& col, CudaTensor<4>& input_grad);
};
