#include "activation_utils.cuh"
#include <iostream>

__global__ void activation_sigmoid_kernel(float* __restrict__ input, float* __restrict__ output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];
    float y = x;

    y = sigmoid_cuda(x);
            
    output[idx] = y;
}

__global__ void activation_sigmoid_derivative_kernel(float* __restrict__ input, float* __restrict__ output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];
    float grad = 1.0f;

    grad = sigmoid_derivative_cuda(x);

    output[idx] *= grad;
}

__global__ void activation_leaky_relu_kernel(float* __restrict__ input, float* __restrict__ output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];
    float y = x;

    y = leaky_relu_cuda(x);
            
    output[idx] = y;
}

__global__ void activation_leaky_relu_derivative_kernel(float* __restrict__ input, float* __restrict__ output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];
    float grad = 1.0f;

    grad = leaky_relu_derivative_cuda(x);

    output[idx] *= grad;
}

__global__ void reduce_rows_kernel(float* __restrict__ input, float* __restrict__ output, size_t rows, size_t cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float sum = 0.0f;
    for (size_t i = 0; i < rows; ++i)
        sum += input[i * cols + col];
    output[col] = sum;
}

__global__ void broadcast_to_rows_kernel(const float* __restrict__ src, float* __restrict__ dst, size_t rows, size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * cols;

    if (idx >= total) return;

    size_t row = idx / cols;
    size_t col = idx % cols;

    dst[row * cols + col] = src[col];
}

__global__ void update_weights_kernel(float* __restrict__ W, const float* __restrict__ grad, float lr, size_t size, float batch_size_float) { 
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        W[idx] -= (lr / batch_size_float) * grad[idx]; 
    }
}

__global__ void kernel_mse_loss(const float*__restrict__ prediction, const float* __restrict__ target, float *loss_buffer, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float diff = prediction[idx] - target[idx];
        loss_buffer[idx] = diff * diff;
    }
}

__global__ void broadcast_bias_to_matrix_kernel(
    const float* __restrict__ bias, float* __restrict__ out,
    size_t F, size_t N) // N = B * OH * OW
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < F * N) {
        int f = idx / N;
        out[idx] = bias[f];
    }
}

__global__ void reduce_columns_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    size_t rows, size_t cols)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= rows) return;

    float sum = 0.0f;
    for (size_t j = 0; j < cols; ++j) {
        sum += input[f * cols + j];
    }
    output[f] = sum;
}

__global__ void apply_mask_to_rows(float* data, const float* mask, size_t B, size_t C) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;

    float m = mask[row];
    for (size_t col = 0; col < C; ++col) {
        data[row * C + col] *= m;
    }
}

__global__ void update_bias_kernel(float* bias, const float* grad, float lr, int batch_size, int F) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < F) {
        bias[i] -= lr * grad[i] / batch_size;
    }
}

void apply_leaky_ReLu_cuda(CudaTensor<2>& input, CudaTensor<2>& output) {
    size_t size = input.size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    activation_leaky_relu_kernel<<<blocks, threads>>>(input.raw(), output.raw(), size);
    cudaDeviceSynchronize();
}

void apply_leaky_ReLu_derivative_cuda(CudaTensor<2>& input, CudaTensor<2>& output) {
    size_t size = input.size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    activation_leaky_relu_derivative_kernel<<<blocks, threads>>>(input.raw(), output.raw(), size);
    cudaDeviceSynchronize();
}

void apply_sigmoid_cuda(CudaTensor<2>& input, CudaTensor<2>& output) {
    size_t size = input.size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    activation_sigmoid_kernel<<<blocks, threads>>>(input.raw(), output.raw(), size);
    cudaDeviceSynchronize();
}

void apply_sigmoid_derivative_cuda(CudaTensor<2>& input, CudaTensor<2>& output) {
    size_t size = input.size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    activation_sigmoid_derivative_kernel<<<blocks, threads>>>(input.raw(), output.raw(), size);
    cudaDeviceSynchronize();
}

void broadcast_bias_cuda(const CudaTensor<1>& bias, CudaTensor<2>& out) {
    const size_t F = bias.get_shape()[0];
    const size_t N = out.get_shape()[1];

    if (out.get_shape()[0] != F)
        throw std::invalid_argument("broadcast_bias_cuda: shape mismatch");

    int threads = 128;
    int blocks = (F * N + threads - 1) / threads;
    broadcast_bias_to_matrix_kernel<<<blocks, threads>>>(bias.raw(), out.raw(), F, N);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error("broadcast_bias_cuda kernel failed: " + std::string(cudaGetErrorString(err)));
}

void broadcast_to_rows_cuda(const CudaTensor<2>& src, CudaTensor<2>& dst) {
    if (src.get_shape()[0] != 1)
        throw std::invalid_argument("broadcast_to_rows: source must have 1 row");

    size_t rows = dst.get_shape()[0];
    size_t cols = dst.get_shape()[1];

    if (src.get_shape()[1] != cols)
        throw std::invalid_argument("broadcast_to_rows: column mismatch");

    size_t total = rows * cols;
    int threads = 128;
    int blocks = (total + threads - 1) / threads;

    broadcast_to_rows_kernel<<<blocks, threads>>>(src.raw(), dst.raw(), rows, cols);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

}

void reduce_columns_cuda(const CudaTensor<2>& input, CudaTensor<1>& output) {
    const size_t F = input.get_shape()[0];
    const size_t N = input.get_shape()[1];

    if (F == 0 || N == 0) {
        std::cerr << "Warning: reduce_columns_cuda called with empty tensor ("
                << F << "x" << N << "). Skipping.\n";
        return;
    }

    if (output.get_shape()[0] != F)
        throw std::invalid_argument("reduce_columns_cuda: shape mismatch");

    int threads = 128;
    int blocks = (F + threads - 1) / threads;

    reduce_columns_kernel<<<blocks, threads>>>(input.raw(), output.raw(), F, N);
    cudaError_t sync_err = cudaDeviceSynchronize();

    if (sync_err != cudaSuccess)
        throw std::runtime_error("reduce_columns_cuda failed: " + std::string(cudaGetErrorString(sync_err)));
}

void reduce_rows_cuda(CudaTensor<2>& input, CudaTensor<2>& output) {
    const size_t* shape_in = input.get_shape();
    const size_t rows = shape_in[0];
    const size_t cols = shape_in[1];

    const size_t* shape_out = output.get_shape();
    if (shape_out[0] != 1 || shape_out[1] != cols) {
        throw std::invalid_argument("reduce_rows: output tensor must have shape [1, cols]");
    }

    int threads = 256;
    int blocks = (cols + threads - 1) / threads;

    reduce_rows_kernel<<<blocks, threads>>>(input.raw(), output.raw(), rows, cols);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("reduce_rows kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void update_weights_cuda(CudaTensor<2>& W, CudaTensor<2>& grad, float lr, float batch_size_float) {
    if (W.size() != grad.size()) {
        throw std::invalid_argument("update_weights: size mismatch between weight and gradient tensors");
    }

    size_t size = W.size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    update_weights_kernel<<<blocks, threads>>>(W.raw(), grad.raw(), lr, size, batch_size_float);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("update_weights kernel failed: ") + cudaGetErrorString(err));
    }
}

float compute_loss_gpu(const CudaTensor<2>& prediction,
                              const CudaTensor<2>& target)
{
    const size_t* shape = prediction.get_shape();
    int batch_size = static_cast<int>(shape[0]);
    int output_size = static_cast<int>(shape[1]);

    float* d_loss_buffer = nullptr;
    float* h_loss_buffer = nullptr;
    size_t loss_buffer_size = batch_size;

    cudaMalloc(&d_loss_buffer, loss_buffer_size * sizeof(float));
    h_loss_buffer = new float[loss_buffer_size];

    int threads = 256;
    int blocks = (loss_buffer_size + threads - 1) / threads;

    kernel_mse_loss<<<blocks, threads>>>( prediction.raw(), target.raw(), d_loss_buffer, batch_size * output_size);

    cudaMemcpy(h_loss_buffer, d_loss_buffer, loss_buffer_size * sizeof(float), cudaMemcpyDeviceToHost);

    float total_loss = 0.0f;
    for (size_t i = 0; i < loss_buffer_size; ++i)
        total_loss += h_loss_buffer[i];

    float average_loss = total_loss / loss_buffer_size;

    cudaFree(d_loss_buffer);
    delete[] h_loss_buffer;

    return average_loss;
}

void apply_mask_to_rows_cuda(CudaTensor<2>& tensor, CudaTensor<1>& mask) {
    size_t B = tensor.get_shape()[0];
    size_t C = tensor.get_shape()[1];

    dim3 blockSize(128);
    dim3 gridSize((B + blockSize.x - 1) / blockSize.x);

    apply_mask_to_rows<<<gridSize, blockSize>>>(
        tensor.data, mask.data, B, C
    );
    cudaDeviceSynchronize();
}

void update_bias_cuda(CudaTensor<1>& bias, const CudaTensor<1>& grad, float lr, int B) {
    int F = bias.get_shape()[0];
    int blockSize = 256;
    int gridSize = (F + blockSize - 1) / blockSize;
    update_bias_kernel<<<gridSize, blockSize>>>(bias.data, grad.data, lr, B, F);
}