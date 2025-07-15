#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>

template <int N>
class CudaTensor {
public:
    float* data = nullptr;
    size_t shape[N];
    size_t strides[N];
    size_t total_size = 1;

    CudaTensor(const size_t(&dims)[N]) {
        for (int i = 0; i < N; ++i) {
            shape[i] = dims[i];
            total_size *= dims[i];
        }
        compute_strides();

        cudaMalloc(&data, total_size * sizeof(float));
        cudaMemset(data, 0, total_size * sizeof(float));
    }

    ~CudaTensor() {
        if (data) cudaFree(data);
    }

    void copy_from_host(const float* host_data) {
        cudaMemcpy(data, host_data, total_size * sizeof(float), cudaMemcpyHostToDevice);
    }

    void copy_to_host(float* host_data) const {
        cudaMemcpy(host_data, data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    size_t flat_index(const size_t(&indices)[N]) const {
        size_t idx = 0;
        for (int i = 0; i < N; ++i) {
            if (indices[i] >= shape[i]) throw std::out_of_range("Index out of bounds");
            idx += indices[i] * strides[i];
        }
        return idx;
    }

    float* raw() { return data; }
    const float* raw() const { return data; }

    size_t size() const { return total_size; }

    const size_t* get_shape() const { return shape; }
    const size_t* get_strides() const { return strides; }

    // Element-wise addition: this = A + B
    void elementwise_add(const CudaTensor<N>& A, const CudaTensor<N>& B);

    // Scalar multiplication: this = A * scalar
    void scalar_multiply(const CudaTensor<N>& A, float scalar);

    // Matrix multiplication: C = A * B
    void matmul_device(const CudaTensor<2>& A, const CudaTensor<2>& B, CudaTensor<2>& C, bool transpose_A = false, bool transpose_B = false);

private:
    void compute_strides() {
        strides[N - 1] = 1;
        for (int i = N - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];
    }
};
