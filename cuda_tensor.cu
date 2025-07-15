#include "cuda_tensor.cuh"

__global__ void tiled_mat_mul_kernel(float *A, float *B, float *C,
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
    for (int phase = 0; phase < ((N2 + TILE_WIDTH - 1) / TILE_WIDTH); phase++)
    {
        if (!transpose_A)
        {
            sh_A[ty][tx] = (i < N1 && (phase * TILE_WIDTH + tx) < N2)
                               ? A[i * N2 + phase * TILE_WIDTH + tx]
                               : 0.0f;
        }
        else
        {
            sh_A[ty][tx] = ((phase * TILE_WIDTH + tx) < N2 && i < N1)
                               ? A[(phase * TILE_WIDTH + tx) * N1 + i]
                               : 0.0f;
        }

        if (!transpose_B)
        {
            sh_B[ty][tx] = ((phase * TILE_WIDTH + ty) < N2 && j < N3)
                               ? B[(phase * TILE_WIDTH + ty) * N3 + j]
                               : 0.0f;
        }
        else
        {
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

__global__ void kernel_add(const float *A, const float *B, float *C, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        C[idx] = A[idx] + B[idx];
}

__global__ void kernel_scalar_mul(const float *A, float *C, float scalar, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        C[idx] = A[idx] * scalar;
}

__global__ void kernel_subtract(const float *A, const float *B, float *C, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] - B[idx];
    }
}
