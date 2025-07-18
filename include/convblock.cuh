#pragma once

#include "tensor.h"
#include "cuda_tensor.cuh"
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <ctime>

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
    ConvBlock(int batch_size, int in_channels, int out_channels,
            int kernel_size, int stride = 1, int padding = 0,
            bool use_cuda = false)
        : batch_size(batch_size), in_channels(in_channels), out_channels(out_channels),
        kernel_size(kernel_size), stride(stride), padding(padding), cudaEnabled(use_cuda)
    {
        // [out_channels, in_channels * kernel_size * kernel_size]
        int fan_in = in_channels * kernel_size * kernel_size;

        weights = Tensor<2>({static_cast<size_t>(out_channels), static_cast<size_t>(fan_in)});
        biases = Tensor<1>({static_cast<size_t>(out_channels)});
        d_weights = Tensor<2>({static_cast<size_t>(out_channels), static_cast<size_t>(fan_in)});
        d_biases = Tensor<1>({static_cast<size_t>(out_channels)});

        initialize_tensor_he(weights, fan_in);

        if (cudaEnabled) {
            weights_cuda = CudaTensor<2>({static_cast<size_t>(out_channels), static_cast<size_t>(fan_in)});
            weights_cuda.copy_from_host(weights.raw_data_arr());
            biases_cuda = CudaTensor<1>({static_cast<size_t>(out_channels)});
            biases_cuda.copy_from_host(biases.raw_data_arr());
            d_weights_cuda = CudaTensor<2>({static_cast<size_t>(out_channels), static_cast<size_t>(fan_in)});
            d_weights_cuda.copy_from_host(d_weights.raw_data_arr());
            d_biases_cuda = CudaTensor<1>({static_cast<size_t>(out_channels)});
            d_biases_cuda.copy_from_host(d_biases.raw_data_arr());
        }
    }

    bool cudaEnabled;
    int batch_size;
    int in_channels, out_channels;
    int kernel_size, stride, padding;
    float learning_rate = 0.1;

    Tensor<2> weights; // [F, C×KH×KW]
    Tensor<1> biases;  // [F]

    Tensor<4> input_cache;     // [B, C, H, W]
    Tensor<2> input_cols;      // [C×KH×KW, B×OH×OW]
    Tensor<4> pre_activations; // [B, F, OH, OW]
    Tensor<4> activations;     // [B, F, OH, OW]

    Tensor<2> d_weights; // [F, C×KH×KW]
    Tensor<1> d_biases;  // [F]

    CudaTensor<2> weights_cuda; // [F, C×KH×KW]
    CudaTensor<1> biases_cuda;  // [F]

    CudaTensor<4> input_cache_cuda;     // [B, C, H, W]
    CudaTensor<2> input_cols_cuda;      // [C×KH×KW, B×OH×OW]
    CudaTensor<2> pre_activations_cuda; // [B, F, OH, OW]
    CudaTensor<4> activations_cuda;     // [B, F, OH, OW]

    CudaTensor<2> d_weights_cuda; // [F, C×KH×KW]
    CudaTensor<1> d_biases_cuda;  // [F]

    Tensor<4> fw(Tensor<4> &input_batch);
    Tensor<4> bp(Tensor<4> &d_out);
    CudaTensor<4> fw_cuda(CudaTensor<4> &input_batch);
    CudaTensor<4> bp_cuda(CudaTensor<4> &d_out);

private:
    void im2col_batch(const Tensor<4> &input, Tensor<2> &out);
    void col2im_batch(const Tensor<2> &col, Tensor<4> &out);
    void im2col_batch_cuda(const CudaTensor<4>& input, CudaTensor<2>& output_col);
    void col2im_batch_cuda(const CudaTensor<2>& col, CudaTensor<4>& input_grad);

    inline void initialize_tensor_he(Tensor<2>& tensor, int fan_in) {
    float scale = std::sqrt(2.0f / fan_in);
        std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
        std::normal_distribution<float> dist(0.0f, scale);

        for (size_t i = 0; i < tensor.size(); ++i) {
            tensor.raw_data()[i] = dist(rng);
        }
    }

};
