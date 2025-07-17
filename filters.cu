#include "tensor.h"
#include "filters.h"

__global__ void im2col_batch_kernel(
    const float* __restrict__ input,  // [B, C, H, W]
    float* __restrict__ output_col,   // [C*KH*KW, B*OH*OW]
    int B, int C, int H, int W,
    int KH, int KW,
    int OH, int OW,
    int stride, int padding)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cols = B * OH * OW;

    if (col_idx >= total_cols) return;

    int b = col_idx / (OH * OW);
    int out_y = (col_idx % (OH * OW)) / OW;
    int out_x = (col_idx % (OH * OW)) % OW;

    int row_idx = 0;
    for (int c = 0; c < C; ++c)
    {
        for (int i = 0; i < KH; ++i)
        {
            for (int j = 0; j < KW; ++j)
            {
                int in_y = out_y * stride + i - padding;
                int in_x = out_x * stride + j - padding;

                float val = 0.0f;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
                {
                    int input_idx = ((b * C + c) * H + in_y) * W + in_x;
                    val = input[input_idx];
                }

                int out_index = row_idx * total_cols + col_idx;
                output_col[out_index] = val;
                ++row_idx;
            }
        }
    }
}

__global__ void col2im_batch_kernel(
    const float* __restrict__ col,  // [C*KH*KW, B*OH*OW]
    float* __restrict__ input_grad, // [B, C, H, W]
    int B, int C, int H, int W,
    int KH, int KW,
    int OH, int OW,
    int stride, int padding)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cols = B * OH * OW;

    if (col_idx >= total_cols) return;

    int b = col_idx / (OH * OW);
    int out_y = (col_idx % (OH * OW)) / OW;
    int out_x = (col_idx % (OH * OW)) % OW;

    int row_idx = 0;
    for (int c = 0; c < C; ++c)
    {
        for (int i = 0; i < KH; ++i)
        {
            for (int j = 0; j < KW; ++j)
            {
                int in_y = out_y * stride + i - padding;
                int in_x = out_x * stride + j - padding;

                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
                {
                    int input_idx = ((b * C + c) * H + in_y) * W + in_x;
                    int col_index = row_idx * total_cols + col_idx;
                    atomicAdd(&input_grad[input_idx], col[col_index]);
                }
                ++row_idx;
            }
        }
    }
}

void ConvBlock::im2col_batch_cuda(const CudaTensor<4>& input, CudaTensor<2>& output_col)
{
    const auto shape = input.get_shape();
    int B = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int KH = kernel_size, KW = kernel_size;
    int OH = (H + 2 * padding - KH) / stride + 1;
    int OW = (W + 2 * padding - KW) / stride + 1;

    output_col.resize({static_cast<size_t>(C * KH * KW),static_cast<size_t>(B * OH * OW)});

    int total_cols = B * OH * OW;
    int threads = 256;
    int blocks = (total_cols + threads - 1) / threads;

    im2col_batch_kernel<<<blocks, threads>>>(
        input.data, output_col.data,
        B, C, H, W,
        KH, KW,
        OH, OW,
        stride, padding
    );

    cudaDeviceSynchronize(); 
}

void ConvBlock::col2im_batch_cuda(const CudaTensor<2>& col, CudaTensor<4>& input_grad)
{
    const auto shape = input_grad.get_shape();
    int B = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int KH = kernel_size, KW = kernel_size;
    int OH = (H + 2 * padding - KH) / stride + 1;
    int OW = (W + 2 * padding - KW) / stride + 1;

    int total_cols = B * OH * OW;
    int threads = 256;
    int blocks = (total_cols + threads - 1) / threads;

    col2im_batch_kernel<<<blocks, threads>>>(
        col.data, input_grad.data,
        B, C, H, W,
        KH, KW,
        OH, OW,
        stride, padding
    );

    cudaDeviceSynchronize(); 
}

void ConvBlock::im2col_batch(const Tensor<4>& input, Tensor<2>& output_col)
{
    const size_t B = input.get_shape()[0];
    const size_t C = input.get_shape()[1];
    const size_t H = input.get_shape()[2];
    const size_t W = input.get_shape()[3];

    const size_t KH = kernel_size;
    const size_t KW = kernel_size;

    const size_t OH = (H + 2 * padding - KH) / stride + 1;
    const size_t OW = (W + 2 * padding - KW) / stride + 1;

    output_col = Tensor<2>({C * KH * KW, B * OH * OW});

    size_t col_idx = 0;

    for (size_t b = 0; b < B; ++b)
    {
        for (size_t y = 0; y < OH; ++y)
        {
            for (size_t x = 0; x < OW; ++x)
            {
                size_t row_idx = 0;
                for (size_t c = 0; c < C; ++c)
                {
                    for (size_t i = 0; i < KH; ++i)
                    {
                        for (size_t j = 0; j < KW; ++j)
                        {
                            int in_y = y * stride + i - padding;
                            int in_x = x * stride + j - padding;

                            float val = 0.0f;
                            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
                                val = input(b, c, in_y, in_x);

                            output_col(row_idx++, col_idx) = val;
                        }
                    }
                }
                ++col_idx;
            }
        }
    }
}


void ConvBlock::col2im_batch(const Tensor<2>& col, Tensor<4>& input_grad)
{
    const size_t B = input_grad.get_shape()[0];
    const size_t C = input_grad.get_shape()[1];
    const size_t H = input_grad.get_shape()[2];
    const size_t W = input_grad.get_shape()[3];

    const size_t KH = kernel_size;
    const size_t KW = kernel_size;
    const size_t OH = (H + 2 * padding - KH) / stride + 1;
    const size_t OW = (W + 2 * padding - KW) / stride + 1;

    input_grad.fill(0.0f);

    size_t col_idx = 0;

    for (size_t b = 0; b < B; ++b)
    {
        for (size_t y = 0; y < OH; ++y)
        {
            for (size_t x = 0; x < OW; ++x)
            {
                size_t row_idx = 0;
                for (size_t c = 0; c < C; ++c)
                {
                    for (size_t i = 0; i < KH; ++i)
                    {
                        for (size_t j = 0; j < KW; ++j)
                        {
                            int in_y = y * stride + i - padding;
                            int in_x = x * stride + j - padding;

                            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
                            {
                                input_grad(b, c, in_y, in_x) += col(row_idx, col_idx);
                            }
                            ++row_idx;
                        }
                    }
                }
                ++col_idx;
            }
        }
    }
}
