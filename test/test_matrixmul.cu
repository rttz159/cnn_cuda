#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_KERNEL_CHECK() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "Kernel Launch Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
    CUDA_CHECK(cudaDeviceSynchronize()); \
} while(0)

// ---------------- KERNEL ----------------
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

void device_matrix_mul(const float *A, const float *B, float *C,
                       int N1, int N2, int N3)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N3 + TILE_WIDTH - 1) / TILE_WIDTH,
              (N1 + TILE_WIDTH - 1) / TILE_WIDTH);

    tiled_mat_mul_kernel<<<grid, block>>>(A, B, C, N1, N2, N3);
    CUDA_KERNEL_CHECK();
}

// ---------------- CPU Reference ----------------
void cpu_matrix_mul(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int N1, int N2, int N3)
{
    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N3; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N2; ++k) {
                sum += A[i * N2 + k] * B[k * N3 + j];
            }
            C[i * N3 + j] = sum;
        }
    }
}

// ---------------- MAIN TEST ----------------
int main() {
    int N1 = 16, N2 = 72, N3 = 72; // Matrix dimensions: A(N1 x N2), B(N2 x N3), C(N1 x N3)

    std::cout << "Testing CUDA MatMul with dimensions: "
              << N1 << "x" << N2 << " * " << N2 << "x" << N3 << "\n";

    size_t sizeA = N1 * N2 * sizeof(float);
    size_t sizeB = N2 * N3 * sizeof(float);
    size_t sizeC = N1 * N3 * sizeof(float);

    std::vector<float> h_A(N1 * N2);
    std::vector<float> h_B(N2 * N3);
    std::vector<float> h_C(N1 * N3, 0);
    std::vector<float> h_ref(N1 * N3, 0);

    // Initialize random data
    for (auto& x : h_A) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto& x : h_B) x = static_cast<float>(rand()) / RAND_MAX;

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice));

    // Launch GPU kernel
    device_matrix_mul(d_A, d_B, d_C, N1, N2, N3);

    // Copy result back to CPU
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));

    // Compute reference result on CPU
    cpu_matrix_mul(h_A, h_B, h_ref, N1, N2, N3);

    // Validate
    float maxError = 0.0f;
    for (int i = 0; i < N1 * N3; ++i) {
        maxError = std::max(maxError, fabs(h_C[i] - h_ref[i]));
    }

    std::cout << "Max absolute error: " << maxError << std::endl;
    if (maxError < 1e-3)
        std::cout << "✅ MatMul is correct!\n";
    else
        std::cout << "❌ MatMul mismatch!\n";

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
