#include "utilities_kernel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(call) do { cudaError_t err = call;  if (err != cudaSuccess) {  std::cout << "CUDA Error: " << cudaGetErrorString(err)  << " at " << __FILE__ << ":" << __LINE__ << std::endl;  exit(EXIT_FAILURE);  }  } while(0)

__global__ void tiled_mat_mul_kernel(const float *A, const float *B, float *C,
                                     int N1, int N2, int N3)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by * TILE_WIDTH + ty;
    const int col = bx * TILE_WIDTH + tx;

    __shared__ float shA[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float shB[TILE_WIDTH][TILE_WIDTH + 1];

    float acc = 0.0f;
    const int numPhases = (N2 + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < numPhases; ++phase) {
        int a_col = phase * TILE_WIDTH + tx;
        int b_row = phase * TILE_WIDTH + ty;

        if (row < N1 && a_col < N2)
            shA[ty][tx] = A[row * N2 + a_col];
        else
            shA[ty][tx] = 0.0f;

        if (b_row < N2 && col < N3)
            shB[ty][tx] = B[b_row * N3 + col];
        else
            shB[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k)
            acc += shA[ty][k] * shB[k][tx];

        __syncthreads();
    }

    if (row < N1 && col < N3)
        C[row * N3 + col] = acc;
}

__global__ void transpose_kernel(const float *input, float *output, int rows, int cols)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int col = bx * TILE_WIDTH + tx;  
    const int row = by * TILE_WIDTH + ty;  

    __shared__ float tile[TILE_WIDTH][TILE_WIDTH + 1];

    if (row < rows && col < cols) {
        tile[ty][tx] = input[row * cols + col];
    }

    __syncthreads();

    int transposed_col = by * TILE_WIDTH + tx; 
    int transposed_row = bx * TILE_WIDTH + ty; 

    if (transposed_row < cols && transposed_col < rows) {
        output[transposed_row * rows + transposed_col] = tile[tx][ty];
    }
}

__global__ void add_bias_kernel(float* Z, const float* b, int batch_size, int layer_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * layer_size;

    if (idx < total) {
        int col = idx % layer_size;
        Z[idx] += b[col];
    }
}

__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void kernel_sigmoid_derivative(const float* d_input, float* d_output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float a = d_input[idx];         
        d_output[idx] = a * (1.0f - a);
    }
}

__global__ void kernel_elementwise_multiply(const float* a, const float* b, float* out, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void kernel_mean_across_batch_reduction(const float* delta, float* grad_bias, int batch_size, int layer_size) {

    extern __shared__ float sdata[];
    int neuron_idx = blockIdx.x;  
    int tid = threadIdx.x;      
    int threads = blockDim.x;
    float sum = 0.0f;

    for (int i = tid; i < batch_size; i += threads) {
        sum += delta[i * layer_size + neuron_idx];  
    }

    sdata[tid] = sum;

    __syncthreads();

    for (unsigned int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_bias[neuron_idx] = sdata[0] / batch_size; 
    }
}

__global__ void kernel_adam_update(float* param, const float* grad,
                                   float* m, float* v,
                                   float lr, float beta1, float beta2,
                                   float epsilon,
                                   float bias_correction1, float bias_correction2,
                                   int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad[idx] * grad[idx];

        float m_hat = m[idx] / bias_correction1;
        float v_hat = v[idx] / bias_correction2;

        param[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

__global__ void im2col_kernel(const float* input, float* output,
                              int N, int C, int H, int W,
                              int F, int pad, int stride,
                              int out_h, int out_w)
{
    int Kc = C * F * F;
    int out_hw = out_h * out_w;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row >= Kc || col >= N * out_hw) return;

    int n = col / out_hw;
    int out_index = col % out_hw;
    int out_y = out_index / out_w;
    int out_x = out_index % out_w;

    int c = row / (F*F);
    int ky = (row / F) % F;
    int kx = row % F;

    int in_y = out_y * stride - pad + ky;
    int in_x = out_x * stride - pad + kx;

    float val = 0.0f;
    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
        val = input[n * (C*H*W) + c*(H*W) + in_y*W + in_x];
    }

    output[row * (N*out_hw) + col] = val;
}

__global__ void col2im_kernel(const float* input_col, float* output,
                              int N, int C, int H, int W,
                              int F, int pad, int stride,
                              int out_h, int out_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int tmp = idx;
    int x = tmp % W; tmp /= W;
    int y = tmp % H; tmp /= H;
    int c = tmp % C; tmp /= C;
    int n = tmp;

    float acc = 0.0f;

    for (int ky = 0; ky < F; ++ky) {
        for (int kx = 0; kx < F; ++kx) {
            int out_y = (y + pad - ky);
            int out_x = (x + pad - kx);

            if (out_y % stride != 0 || out_x % stride != 0) continue;

            out_y /= stride;
            out_x /= stride;

            if (out_y < 0 || out_y >= out_h || out_x < 0 || out_x >= out_w) continue;

            int out_hw = out_h * out_w;
            int col = n * out_hw + out_y * out_w + out_x;
            int row = c * F * F + ky * F + kx;

            acc += input_col[row * (N*out_hw) + col];
        }
    }

    output[idx] = acc;
}

__global__ void leaky_relu(float* data, int size, float alpha = 0.01f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = (x > 0) ? x : alpha * x;
    }
}

__global__ void leaky_relu_derivative_kernel(const float* d_input, float* d_output, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = (d_input[idx] > 0.0f) ? 1.0f : alpha;
    }
}

__global__ void add_bias_per_filter_kernel(float* Z, const float* b, int N, int K, int out_hw) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = N * K * out_hw;
    if (idx >= total) return;

    int tmp = idx / out_hw;
    int k = tmp % K;
    Z[idx] += b[k];
}

__global__ void add_inplace_kernel(float* dst, const float* src, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) dst[idx] += src[idx];
}

__global__ void compute_bias_grad_kernel(const float* delta, float* grad_bias, int N, int K, int out_hw) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const int total_elements = N * out_hw;
    const float inv_denom = 1.0f / (float(N) * float(out_hw));

    float sum = 0.0f;

    const int stride = K;
    const float* delta_k = delta + k;

    int i = 0;
    const int unroll_limit = total_elements - 3;

    for (; i < unroll_limit; i += 4) {
        sum += delta_k[i * stride];
        sum += delta_k[(i + 1) * stride];
        sum += delta_k[(i + 2) * stride];
        sum += delta_k[(i + 3) * stride];
    }

    for (; i < total_elements; ++i) {
        sum += delta_k[i * stride];
    }

    grad_bias[k] = sum * inv_denom;
}

__global__ void compute_bias_grad_kernel_shared(const float* delta, float* grad_bias, int N, int K, int out_hw) {
    extern __shared__ float sdata[];

    int k = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (k >= K) return;

    const int total_elements = N * out_hw;
    const float inv_denom = 1.0f / (float(N) * float(out_hw));

    float sum = 0.0f;
    const int stride = K;
    const float* delta_k = delta + k;

    for (int i = tid; i < total_elements; i += blockSize) {
        sum += delta_k[i * stride];
    }

    sdata[tid] = sum;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }

    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        __syncwarp();
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        __syncwarp();
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        __syncwarp();
        if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
        __syncwarp();
        if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
        __syncwarp();
        if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
        __syncwarp();
    }

    if (tid == 0) {
        grad_bias[k] = sdata[0] * inv_denom;
    }
}

/* ===================== Wrappers (stream-aware) ===================== */

void device_matrix_mul(const float *A, const float *B, float *C,
                       int N1, int N2, int N3, cudaStream_t stream)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N3 + TILE_WIDTH - 1) / TILE_WIDTH,
              (N1 + TILE_WIDTH - 1) / TILE_WIDTH);

    if (N1 <= 0 || N2 <= 0 || N3 <= 0) {
        std::cerr << "Invalid dims! N1=" << N1 << " N2=" << N2 << " N3=" << N3 << "\n";
        exit(EXIT_FAILURE);
    }

    tiled_mat_mul_kernel<<<grid, block, 0, stream>>>(A, B, C, N1, N2, N3);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "MatMul Kernel Launch Error: " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
    // async: caller decides when to synchronize
}

void device_matrix_transpose(const float *input, float *output, int rows, int cols, cudaStream_t stream)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((cols + TILE_WIDTH - 1) / TILE_WIDTH, (rows + TILE_WIDTH - 1) / TILE_WIDTH);

    transpose_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Transpose Kernel Launch Error: " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

void add_bias(float* d_Z, const float* d_b, int batch_size, int layer_size, cudaStream_t stream) {
    int total = batch_size * layer_size;
    int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    add_bias_kernel<<<blocks, threadsPerBlock, 0, stream>>>(d_Z, d_b, batch_size, layer_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA add_bias_kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void apply_sigmoid(const float* d_input, float* d_output, int size, cudaStream_t stream) {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocks, threadsPerBlock, 0, stream>>>(d_input, d_output, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA sigmoid_kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void sigmoid_derivative(const float* d_input, float* d_output, int size, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    kernel_sigmoid_derivative<<<numBlocks, blockSize, 0, stream>>>(d_input, d_output, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA sigmoid_derivative launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void elementwise_multiply(const float* a, const float* b, float* out, int size, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    kernel_elementwise_multiply<<<numBlocks, blockSize, 0, stream>>>(a, b, out, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA elementwise_multiply launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void adam_update(float* param, const float* grad, float* m, float* v,
                 float lr, float beta1, float beta2,
                 float epsilon, int timestep, int size, cudaStream_t stream) {
    float bias_correction1 = 1.0f - powf(beta1, timestep);
    float bias_correction2 = 1.0f - powf(beta2, timestep);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    kernel_adam_update<<<numBlocks, blockSize, 0, stream>>>(
        param, grad, m, v, lr, beta1, beta2, epsilon,
        bias_correction1, bias_correction2, size
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA adam_update launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void mean_across_batch_reduction(const float* delta, float* grad_bias, int batch_size, int layer_size, cudaStream_t stream) {
    int threads = 256;
    size_t shared_mem = threads * sizeof(float);

    kernel_mean_across_batch_reduction<<<layer_size, threads, shared_mem, stream>>>(delta, grad_bias, batch_size, layer_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA mean_across_batch_reduction launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void im2col_wrapper(const float* d_input, float* d_output,
                    int N, int C, int H, int W, int F, int pad, int stride, cudaStream_t stream)
{
    int out_h = (H + 2*pad - F) / stride + 1;
    int out_w = (W + 2*pad - F) / stride + 1;

    dim3 block(16,16);
    dim3 grid((N*out_h*out_w + block.x - 1)/block.x,
              (C*F*F + block.y - 1)/block.y);

    im2col_kernel<<<grid, block, 0, stream>>>(d_input, d_output,
                                             N,C,H,W,F,pad,stride,out_h,out_w);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA im2col_kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void col2im_wrapper(const float* d_input_col, float* d_output,
                    int N, int C, int H, int W, int F, int pad, int stride, cudaStream_t stream)
{
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1)/threads;

    int out_h = (H + 2*pad - F) / stride + 1;
    int out_w = (W + 2*pad - F) / stride + 1;

    col2im_kernel<<<blocks, threads, 0, stream>>>(d_input_col, d_output,
                                                  N,C,H,W,F,pad,stride,out_h,out_w);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA col2im_kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void leaky_relu_host(float* d_data, int size, float alpha, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    leaky_relu<<<blocks, threads, 0, stream>>>(d_data, size, alpha);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA leaky_relu launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void leaky_relu_derivative(const float* d_input, float* d_output, int size, float alpha, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    leaky_relu_derivative_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, size, alpha);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA leaky_relu_derivative launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void add_bias_per_filter(float* Z, const float* b, int N, int K, int out_hw, cudaStream_t stream) {
    int total = N * K * out_hw;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_per_filter_kernel<<<blocks, threads, 0, stream>>>(Z, b, N, K, out_hw);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA add_bias_per_filter launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void add_inplace(float* dst, const float* src, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_inplace_kernel<<<blocks, threads, 0, stream>>>(dst, src, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA add_inplace launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void compute_bias_grad(const float* delta, float* grad_bias, int N, int K, int out_hw, cudaStream_t stream) {
    const int total_work = N * out_hw;

    if (total_work <= 1024) {
        int threads = std::min(256, K);
        int blocks = (K + threads - 1) / threads;
        compute_bias_grad_kernel<<<blocks, threads, 0, stream>>>(delta, grad_bias, N, K, out_hw);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA compute_bias_grad_kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        int threads = std::min(512, total_work);
        threads = ((threads + 31) / 32) * 32;  // round up to warp multiple
        size_t shared_mem = threads * sizeof(float);

        compute_bias_grad_kernel_shared<<<K, threads, shared_mem, stream>>>(delta, grad_bias, N, K, out_hw);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA compute_bias_grad_kernel_shared launch failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}
