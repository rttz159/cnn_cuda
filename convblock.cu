#include "tensor.h"
#include "convblock.cuh"
#include "activation_utils.h"
#include "activation_utils.cuh"

__global__ void im2col_batch_kernel(
    const float* __restrict__ input,  // [B, C, H, W]
    float* __restrict__ output_col,   // [C*KH*KW, B*OH*OW]
    int B, int C, int H, int W, // Image
    int KH, int KW, // Kernel
    int OH, int OW, // Output
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
    int B, int C, int H, int W, // Image
    int KH, int KW, // Kernel
    int OH, int OW, // Output
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

void print_tensor2D(const CudaTensor<2>& tensor, const std::string& name = "Tensor") {
    const size_t* shape = tensor.get_shape();
    size_t rows = shape[0];
    size_t cols = shape[1];

    std::vector<float> host_data(rows * cols);
    tensor.copy_to_host(host_data.data());

    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "[ ";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << host_data[i * cols + j] << " ";
        }
        std::cout << "]\n";
    }
}

void print_tensor4D(const CudaTensor<4>& tensor, const std::string& name = "Tensor4D") {
    const size_t* shape = tensor.get_shape();
    size_t B = shape[0], C = shape[1], H = shape[2], W = shape[3];

    std::vector<float> host_data(B * C * H * W);
    tensor.copy_to_host(host_data.data());

    std::cout << name << " (" << B << "x" << C << "x" << H << "x" << W << "):\n";
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < C; ++c) {
            std::cout << "Batch " << b << ", Channel " << c << ":\n";
            for (size_t h = 0; h < H; ++h) {
                std::cout << "[ ";
                for (size_t w = 0; w < W; ++w) {
                    std::cout << host_data[((b * C + c) * H + h) * W + w] << " ";
                }
                std::cout << "]\n";
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

Tensor<4> ConvBlock::fw(Tensor<4>& input_batch) {
    input_cache = input_batch;

    int B = input_batch.get_shape()[0];
    int C = input_batch.get_shape()[1];
    int H = input_batch.get_shape()[2];
    int W = input_batch.get_shape()[3];

    int KH = kernel_size;
    int KW = kernel_size;

    int OH = (H + 2 * padding - KH) / stride + 1;
    int OW = (W + 2 * padding - KW) / stride + 1;

    input_cols = Tensor<2>({static_cast<size_t>(C * KH * KW), static_cast<size_t>(B * OH * OW)});
    pre_activations = Tensor<4>({static_cast<size_t>(B), static_cast<size_t>(out_channels), static_cast<size_t>(OH), static_cast<size_t>(OW)});
    activations = Tensor<4>({static_cast<size_t>(B), static_cast<size_t>(out_channels), static_cast<size_t>(OH), static_cast<size_t>(OW)});

    // im2col:
    im2col_batch(input_batch, input_cols);

    // weights [F, C*KH*KW] × input_cols [C*KH*KW, B*OH*OW] = [F, B*OH*OW]
    Tensor<2> pre_act_flat = Tensor<2>::matmul(weights, input_cols);

    // Add bias by broadcasting
    for (int f = 0; f < out_channels; ++f)
        for (int i = 0; i < B * OH * OW; ++i)
            pre_act_flat(f, i) += biases(f);

    // Reshape flat to 4D tensor [B, F, OH, OW] -> output height, output weight
    for (int f = 0; f < out_channels; ++f)
        for (int b = 0; b < B; ++b)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow)
                    pre_activations(b, f, oh, ow) = pre_act_flat(f, b * OH * OW + oh * OW + ow);

    activations = Activation::leaky_relu(pre_activations);
    return activations;
}

Tensor<4> ConvBlock::bp(Tensor<4>& d_out) {
    int B = input_cache.get_shape()[0];
    int C = input_cache.get_shape()[1];
    int H = input_cache.get_shape()[2];
    int W = input_cache.get_shape()[3];

    int KH = kernel_size;
    int KW = kernel_size;

    int OH = (H + 2 * padding - KH) / stride + 1;
    int OW = (W + 2 * padding - KW) / stride + 1;

    // d_pre_activations = d_out (elemental_wise multiplication) leaky_relu_derivative(pre_activations)
    Tensor<4> act_deriv = Activation::leaky_relu_derivative(pre_activations);
    Tensor<4> d_pre_activations = d_out * act_deriv;

    // Flatten d_pre_activations to [F, B*OH*OW]
    Tensor<2> d_pre_acts_flat({static_cast<size_t>(out_channels),
                           static_cast<size_t>(B * OH * OW)});
    for (int f = 0; f < out_channels; ++f)
        for (int b = 0; b < B; ++b)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow)
                    d_pre_acts_flat(f, b * OH * OW + oh * OW + ow) = d_pre_activations(b, f, oh, ow);

    // d_weights = d_pre_acts_flat * input_cols.T
    d_weights = Tensor<2>::matmul(d_pre_acts_flat, Tensor<2>::transpose(input_cols));

    // d_biases = sum over B*OH*OW
    d_biases = Tensor<1>({static_cast<size_t>(out_channels)});
    for (int f = 0; f < out_channels; ++f) {
        float sum = 0.0f;
        for (int i = 0; i < B * OH * OW; ++i)
            sum += d_pre_acts_flat(f, i);
        d_biases(f) = sum;
    }

    // d_input_cols = Transposed Weight * d_pre_acts_flat
    Tensor<2> d_input_cols = Tensor<2>::matmul(Tensor<2>::transpose(weights), d_pre_acts_flat);

    Tensor<4> d_input({static_cast<size_t>(B),
                   static_cast<size_t>(C),
                   static_cast<size_t>(H),
                   static_cast<size_t>(W)});
    col2im_batch(d_input_cols, d_input);

    return d_input;
}

CudaTensor<4> ConvBlock::fw_cuda(CudaTensor<4> &input_batch) {
    input_cache_cuda = input_batch;

    int B = input_batch.get_shape()[0];
    int C = input_batch.get_shape()[1];
    int H = input_batch.get_shape()[2];
    int W = input_batch.get_shape()[3];
    int K = kernel_size;
    int F = out_channels;

    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;

    // Resize temporary tensors
    input_cols_cuda.resize({static_cast<size_t>(C * K * K), static_cast<size_t>(B * OH * OW)});
    pre_activations_cuda.resize({static_cast<size_t>(F), static_cast<size_t>(B*OH*OW)});
    activations_cuda.resize({static_cast<size_t>(B), static_cast<size_t>(F), static_cast<size_t>(OH), static_cast<size_t>(OW)});

    // im2col
    im2col_batch_cuda(input_cache_cuda, input_cols_cuda);

    // Matrix multiplication: pre_act = W × cols
    CudaTensor<2> pre_acts_flat({static_cast<size_t>(F), static_cast<size_t>(B * OH * OW)});
    CudaTensor<2>::matmul_device(weights_cuda, input_cols_cuda, pre_acts_flat, false, false);

    // Broadcast biases
    CudaTensor<2> bias_expanded({static_cast<size_t>(F), static_cast<size_t>(B * OH * OW)});
    broadcast_bias_cuda(biases_cuda, bias_expanded);
    pre_acts_flat.elementwise_add(pre_acts_flat, bias_expanded);

    pre_activations_cuda = pre_acts_flat;

    // Apply leaky ReLU
    CudaTensor<2> acts_flat({static_cast<size_t>(F), static_cast<size_t>(B * OH * OW)});
    apply_leaky_ReLu_cuda(pre_acts_flat, acts_flat);

    // Reshape from flat to 4D
    CudaTensor<4> acts_4d({static_cast<size_t>(B), static_cast<size_t>(F), static_cast<size_t>(OH), static_cast<size_t>(OW)});
    acts_4d.reshape_from(acts_flat); 
    activations_cuda = acts_4d;

    return acts_4d;
}

CudaTensor<4> ConvBlock::bp_cuda(CudaTensor<4>& d_out) {
    int B = batch_size;
    int C = in_channels;
    int K = kernel_size;
    int F = out_channels;
    int H = input_cache_cuda.get_shape()[2];
    int W = input_cache_cuda.get_shape()[3];
    int OH = (H + 2 * padding - K) / stride + 1;
    int OW = (W + 2 * padding - K) / stride + 1;

    // Flatten d_out
    CudaTensor<2> d_out_flat({static_cast<size_t>(F), static_cast<size_t>(B * OH * OW)});
    d_out_flat.reshape_from(d_out);

    // d_pre_acts = d_out_flat * act_deriv
    CudaTensor<2> d_pre_acts(d_out_flat);
    apply_leaky_ReLu_derivative_cuda(pre_activations_cuda, d_pre_acts);

    // Gradient for weights: dW = d_pre_acts * input_cols^T
    CudaTensor<2> dW({static_cast<size_t>(F), static_cast<size_t>(C * K * K)});
    CudaTensor<2>::matmul_device(d_pre_acts, input_cols_cuda, dW, false, true);
    d_weights_cuda = dW;

    // Gradient for bias: reduce across columns
    CudaTensor<1> db({static_cast<size_t>(F)});
    reduce_columns_cuda(d_pre_acts, db);
    d_biases_cuda = db;

    // Gradient for input_cols: dX = weights^T * d_pre_acts
    CudaTensor<2> d_cols({static_cast<size_t>(C * K * K), static_cast<size_t>(B * OH * OW)});
    CudaTensor<2>::matmul_device(weights_cuda, d_pre_acts, d_cols, true, false);

    // Convert col2im
    CudaTensor<4> d_input({static_cast<size_t>(B), static_cast<size_t>(C), static_cast<size_t>(H), static_cast<size_t>(W)});
    col2im_batch_cuda(d_cols, d_input);

    return d_input;
}
