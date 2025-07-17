#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <stdexcept>
#include <cmath>

template <size_t N>
class Tensor
{
    /*
        Example usage:
            Tensor<4> tensor({2, 3, 4, 5});

        The indicies is passed in with the fastest changing indicies or the lowest rank in the last.
        For example:
            a rank 2 tensor or 2d array (3, 2):
                [[1,2],
                [4,5],
                [7,8]]
    */
public:
    Tensor()
    {
        shape.fill(0);
        strides.fill(0);
    }

    Tensor(const std::array<size_t, N> &shape) : shape(shape)
    {
        compute_strides();
        size_t total_size = 1;
        for (auto s : shape)
            total_size *= s;
        data.resize(total_size);
    }

    Tensor(const std::array<size_t, N> &shape, const std::vector<float> &values) : shape(shape)
    {
        compute_strides();
        size_t total_size = 1;
        for (auto s : shape)
            total_size *= s;
        if (values.size() != total_size)
        {
            throw std::invalid_argument("Data size does not match tensor shape");
        }
        data = values;
    }

    // Compute flat index
    size_t index(const std::array<size_t, N> &indices) const
    {
        size_t idx = 0;
        for (size_t i = 0; i < N; ++i)
        {
            if (indices[i] >= shape[i])
                throw std::out_of_range("Index out of bounds");
            idx += indices[i] * strides[i];
        }
        if (idx >= data.size())
        {
            throw std::out_of_range("Computed index exceeds data size");
        }
        return idx;
    }

    // Variadic operator()
    template <typename... Args>
    float &operator()(Args... args)
    {
        static_assert(sizeof...(args) == N, "Number of indices must match tensor rank");
        std::array<size_t, N> indices = {static_cast<size_t>(args)...};
        return data[index(indices)];
    }

    template <typename... Args>
    const float &operator()(Args... args) const
    {
        static_assert(sizeof...(args) == N, "Number of indices must match tensor rank");
        std::array<size_t, N> indices = {static_cast<size_t>(args)...};
        return data[index(indices)];
    }

    Tensor<N> &operator=(const Tensor<N> &other)
    {
        if (this != &other)
        {
            shape = other.shape;
            data = other.data;
            compute_strides();
        }
        return *this;
    }

    const std::array<size_t, N> &get_shape() const
    {
        return shape;
    }

    std::vector<float> &raw_data()
    {
        return data;
    }

    float *raw_data_arr()
    {
        return data.data();
    }

    size_t size() const
    {
        return data.size();
    }

    void fill(float x) {
        std::fill(data.begin(), data.end(), x);
    }

    // Matrix Multiplication for rank 2 tensor
    static Tensor<2> matmul(const Tensor<2> &A, const Tensor<2> &B)
    {
        auto a_shape = A.get_shape();
        auto b_shape = B.get_shape();

        size_t M = a_shape[0];
        size_t K = a_shape[1];
        size_t K2 = b_shape[0];
        size_t N_ = b_shape[1];

        if (K != K2)
            throw std::invalid_argument("Incompatible matrix dimensions");

        Tensor<2> result({M, N_});
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N_; ++j)
            {
                float sum = 0;
                for (size_t k = 0; k < K; ++k)
                    sum += A(i, k) * B(k, j);
                result(i, j) = sum;
            }

        return result;
    }

    static Tensor<2> transpose(const Tensor<2> &mat)
    {
        auto shape = mat.get_shape();
        Tensor<2> result({shape[1], shape[0]});
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                result(j, i) = mat(i, j);
        return result;
    }

    Tensor<2> add_bias_column() const
    {
        static_assert(N == 2, "add_bias_column() is only supported for 2D tensors");

        size_t rows = shape[0];
        size_t cols = shape[1];

        Tensor<2> result({rows, cols + 1});
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                result(i, j) = (*this)(i, j);
            }
            result(i, cols) = static_cast<float>(1.0);
        }
        return result;
    }

    Tensor<2> remove_bias_column() const
    {
        static_assert(N == 2, "remove_bias_column() is only supported for 2D tensors");

        size_t rows = shape[0];
        size_t cols = shape[1];

        if (cols < 2)
            throw std::invalid_argument("Cannot remove bias column from tensor with less than 2 columns");

        Tensor<2> result({rows, cols - 1});
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols - 1; ++j)
                result(i, j) = (*this)(i, j);

        return result;
    }

    // Element-wise multiplication
    Tensor<N> operator*(const Tensor<N> &other) const
    {
        if (shape != other.shape)
            throw std::invalid_argument("Tensors must have the same shape for element-wise multiplication");

        Tensor<N> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    // Scalar multiplication
    Tensor<N> operator*(float scalar) const
    {
        Tensor<N> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    // Tensor subtraction (for weight updates)
    Tensor<N> operator-(const Tensor<N> &other) const
    {
        if (shape != other.shape)
            throw std::invalid_argument("Tensors must have the same shape for subtraction");

        Tensor<N> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    // Broadcast a 1-row tensor to multiple rows (for bias expansion)
    Tensor<2> broadcast_to_rows(size_t num_rows) const
    {
        static_assert(N == 2, "broadcast_to_rows() is only supported for 2D tensors (row vectors)");
        if (shape[0] != 1)
        {
            throw std::invalid_argument("broadcast_to_rows() expects a 1-row tensor.");
        }
        size_t cols = shape[1];
        Tensor<2> result({num_rows, cols});
        for (size_t r = 0; r < num_rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                result(r, c) = (*this)(0, c); // Replicate the single row
            }
        }
        return result;
    }

    // Tensor addition
    Tensor<N> operator+(const Tensor<N> &other) const
    {
        if (shape != other.shape)
            throw std::invalid_argument("Tensors must have the same shape for addition");

        Tensor<N> result(shape);
        for (size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

private:
    std::array<size_t, N> shape;
    std::array<size_t, N> strides;
    std::vector<float> data;

    // Compute strides for row-major layout
    void compute_strides()
    {
        strides[N - 1] = 1;
        for (int i = N - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
};