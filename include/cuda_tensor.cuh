#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <array>

#define TILE_WIDTH 32

__global__ void tiled_mat_mul_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C,
                                     int N1, int N2, int N3,
                                     bool transpose_A, bool transpose_B);

__global__ void kernel_add(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, size_t size);

__global__ void kernel_subtract(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, size_t size);

__global__ void kernel_scalar_mul(const float* __restrict__ A, float* __restrict__ C, float scalar, size_t size);

__global__ void kernel_elementwise_multiply(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, size_t size);

template <int N>
class CudaTensor
{
public:
    float *data = nullptr;
    size_t shape[N];
    size_t strides[N];
    size_t total_size = 1;

    CudaTensor()
    {
        for (int i = 0; i < N; ++i) {
            shape[i] = 0;
            strides[i] = 0;
        }
    }

    CudaTensor(const size_t (&dims)[N])
    {
        for (int i = 0; i < N; ++i)
        {
            shape[i] = dims[i];
            total_size *= dims[i];
        }
        compute_strides();

        cudaMalloc(&data, total_size * sizeof(float));
        cudaMemset(data, 0, total_size * sizeof(float));
    }

    CudaTensor(const CudaTensor& other)
    {
        for (int i = 0; i < N; ++i)
            shape[i] = other.shape[i];

        total_size = other.total_size;
        compute_strides();

        cudaMalloc(&data, total_size * sizeof(float));
        cudaMemcpy(data, other.data, total_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    CudaTensor(const std::array<size_t, N>& shape_array)
    {
        for (int i = 0; i < N; ++i)
        {
            shape[i] = shape_array[i];
            total_size *= shape[i];
        }

        compute_strides();

        cudaMalloc(&data, total_size * sizeof(float));
        cudaMemset(data, 0, total_size * sizeof(float));
    }

    ~CudaTensor()
    {
        if (data)
            cudaFree(data);
    }

    void copy_from_host(const float *host_data)
    {
        cudaMemcpy(data, host_data, total_size * sizeof(float), cudaMemcpyHostToDevice);
    }

    void copy_to_host(float *host_data) const
    {
        cudaMemcpy(host_data, data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void resize(const std::array<size_t, N>& new_shape) {
        size_t new_total = 1;
        for (int i = 0; i < N; ++i) new_total *= new_shape[i];

        if (new_total != total_size) {
            if (data) cudaFree(data);
            cudaMalloc(&data, new_total * sizeof(float));
            total_size = new_total;
        }

        std::copy(new_shape.begin(), new_shape.end(), shape);
        compute_strides();
    }

    void reshape_from(const CudaTensor<2>& flat) {
        if (flat.size() != this->size()) {
            throw std::runtime_error("reshape_from: total size mismatch between flat and 4D tensor");
        }

        this->data = flat.data;
        this->total_size = flat.size();
        compute_strides(); 
    }

    void reshape_from(const CudaTensor<4>& source) {
        if (source.size() != this->size())
            throw std::runtime_error("reshape_from: mismatched size");

        this->data = source.data; 
        this->total_size = source.size();
        compute_strides(); 
    }

    std::vector<size_t> get_shape_vector() const {
        return std::vector<size_t>(shape, shape + N);
    }

    size_t flat_index(const size_t (&indices)[N]) const
    {
        size_t idx = 0;
        for (int i = 0; i < N; ++i)
        {
            if (indices[i] >= shape[i])
                throw std::out_of_range("Index out of bounds");
            idx += indices[i] * strides[i];
        }
        return idx;
    }

    float *raw() { return data; }
    const float *raw() const { return data; }

    size_t size() const { return total_size; }

    const size_t *get_shape() const { return shape; }
    const size_t *get_strides() const { return strides; }

    void elementwise_add(const CudaTensor<N> &A, const CudaTensor<N> &B)
    {
        if (A.total_size != B.total_size || A.total_size != total_size)
            throw std::invalid_argument("Size mismatch for addition");

        size_t threads = 256;
        size_t blocks = (total_size + threads - 1) / threads;
        kernel_add<<<blocks, threads>>>(A.data, B.data, this->data, total_size);
        cudaDeviceSynchronize();
    }

    void elementwise_subtract(const CudaTensor<N> &A, const CudaTensor<N> &B)
    {
        if (A.total_size != B.total_size || A.total_size != total_size)
            throw std::invalid_argument("Size mismatch for subtraction");

        size_t threads = 256;
        size_t blocks = (total_size + threads - 1) / threads;
        kernel_subtract<<<blocks, threads>>>(A.data, B.data, this->data, total_size);
        cudaDeviceSynchronize();
    }

    void elementwise_multiply(const CudaTensor<N> &A, const CudaTensor<N> &B)
    {
        if (A.total_size != B.total_size || A.total_size != total_size)
            throw std::invalid_argument("Size mismatch for multiplication");

        size_t threads = 256;
        size_t blocks = (total_size + threads - 1) / threads;
        kernel_elementwise_multiply<<<blocks, threads>>>(A.data, B.data, this->data, total_size);
        cudaDeviceSynchronize();
    }

    void scalar_multiply(const CudaTensor<N> &A, float scalar)
    {
        if (A.total_size != total_size)
            throw std::invalid_argument("Size mismatch for scalar multiplication");

        size_t threads = 256;
        size_t blocks = (total_size + threads - 1) / threads;
        kernel_scalar_mul<<<blocks, threads>>>(A.data, this->data, scalar, total_size);
        cudaDeviceSynchronize();
    }

    static void matmul_device(const CudaTensor<2> &A, const CudaTensor<2> &B, CudaTensor<2> &C, bool transpose_A = false, bool transpose_B = false)
    {
        size_t N1 = transpose_A ? A.get_shape()[1] : A.get_shape()[0];
        size_t N2 = transpose_A ? A.get_shape()[0] : A.get_shape()[1];
        size_t N3 = transpose_B ? B.get_shape()[0] : B.get_shape()[1];

        size_t B_inner = transpose_B ? B.get_shape()[1] : B.get_shape()[0];
        if (N2 != B_inner)
        {
            throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
        }

        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((N3 + TILE_WIDTH - 1) / TILE_WIDTH,
                     (N1 + TILE_WIDTH - 1) / TILE_WIDTH);

        tiled_mat_mul_kernel<<<gridDim, blockDim>>>(
            A.data, B.data, C.data,
            static_cast<int>(N1),
            static_cast<int>(N2),
            static_cast<int>(N3),
            transpose_A,
            transpose_B);
        cudaDeviceSynchronize();
    }

    CudaTensor& operator=(const CudaTensor& other)
    {
        if (this == &other) return *this; 

        if (data)
            cudaFree(data);

        for (int i = 0; i < N; ++i)
            shape[i] = other.shape[i];

        total_size = other.total_size;
        compute_strides();

        cudaMalloc(&data, total_size * sizeof(float));
        cudaMemcpy(data, other.data, total_size * sizeof(float), cudaMemcpyDeviceToDevice);

        return *this;
    }


private:
    void compute_strides()
    {
        strides[N - 1] = 1;
        for (int i = N - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];
    }
};
