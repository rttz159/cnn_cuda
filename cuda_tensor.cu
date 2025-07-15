#include "cuda_tensor.h"
#define TILE_WIDTH 32


__global__ void tiled_mat_mul_kernel(float* A, float* B, float* C,
    int N1, int N2, int N3,
    bool transpose_A, bool transpose_B)
{
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);

    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = TILE_WIDTH * by + ty;
    int j = TILE_WIDTH * bx + tx;

    // Allocating shared memory
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    // Parallel mat mul
    float value = 0;
    for (int phase = 0; phase < ((N2 + TILE_WIDTH - 1) / TILE_WIDTH); phase++) {
        if (!transpose_A) {
            sh_A[ty][tx] = (i < N1 && (phase * TILE_WIDTH + tx) < N2)
                ? A[i * N2 + phase * TILE_WIDTH + tx]
                : 0.0f;
        }
        else {
            sh_A[ty][tx] = ((phase * TILE_WIDTH + tx) < N2 && i < N1)
                ? A[(phase * TILE_WIDTH + tx) * N1 + i]
                : 0.0f;
        }

        if (!transpose_B) {
            sh_B[ty][tx] = ((phase * TILE_WIDTH + ty) < N2 && j < N3)
                ? B[(phase * TILE_WIDTH + ty) * N3 + j]
                : 0.0f;
        }
        else {
            sh_B[ty][tx] = (j < N2 && (phase * TILE_WIDTH + ty) < N3)
                ? B[j * N2 + (phase * TILE_WIDTH + ty)]
                : 0.0f;
        }

        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++)
            value += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }
    if ((i < N1) && (j < N3))
        C[i * N3 + j] = value;
}


__global__ void kernel_add(const float* A, const float* B, float* C, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = A[idx] + B[idx];
}

__global__ void kernel_scalar_mul(const float* A, float* C, float scalar, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = A[idx] * scalar;
}

template<int N>
void CudaTensor<N>::elementwise_add(const CudaTensor<N>& A, const CudaTensor<N>& B) {
    if (A.total_size != B.total_size || A.total_size != total_size)
        throw std::invalid_argument("Size mismatch for addition");

    size_t threads = 256;
    size_t blocks = (total_size + threads - 1) / threads;
    kernel_add << <blocks, threads >> > (A.data, B.data, this->data, total_size);
}

template<int N>
void CudaTensor<N>::scalar_multiply(const CudaTensor<N>& A, float scalar) {
    if (A.total_size != total_size)
        throw std::invalid_argument("Size mismatch for scalar multiplication");

    size_t threads = 256;
    size_t blocks = (total_size + threads - 1) / threads;
    kernel_scalar_mul << <blocks, threads >> > (A.data, this->data, scalar, total_size);
}

template<int N>
void CudaTensor<N>::matmul_device(const CudaTensor<2>& A, const CudaTensor<2>& B, CudaTensor<2>& C, bool transpose_A, bool transpose_B) {
    size_t N1 = transpose_A ? A.get_shape()[1] : A.get_shape()[0];
    size_t N2 = transpose_A ? A.get_shape()[0] : A.get_shape()[1];
    size_t N3 = transpose_B ? B.get_shape()[0] : B.get_shape()[1];

    size_t B_inner = transpose_B ? B.get_shape()[1] : B.get_shape()[0];
    if (N2 != B_inner) {
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
    }

    C = CudaTensor<2>({ N1, N3 });

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N3 + TILE_WIDTH - 1) / TILE_WIDTH,
        (N1 + TILE_WIDTH - 1) / TILE_WIDTH);

    tiled_mat_mul_kernel << <gridDim, blockDim >> > (
        A.data, B.data, C.data,
        static_cast<int>(N1),
        static_cast<int>(N2),
        static_cast<int>(N3),
        transpose_A,
        transpose_B
        );
}