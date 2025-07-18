#pragma once

#include "convblock.cuh"
#include "mlp.h" // MultiLayerPerceptron class
#include "tensor.h"
#include "cuda_tensor.cuh"
#include <vector>
#include <stdexcept>

/*
Input Image -> ConvBlock 1 -> ConvBlock 2 -> Flatten -> MLP (MultiLayerPerceptron) -> Output
*/

class CNN {
public:
    CNN(int batch_size,
        int in_channels,
        int in_height,
        int in_width,
        int num_classes,
        bool use_cuda = false)
        : batch_size(batch_size),
          in_channels(in_channels),
          in_height(in_height),
          in_width(in_width),
          num_classes(num_classes),
          use_cuda(use_cuda)
    {
        // Conv Layer 1
        conv1 = std::make_unique<ConvBlock>(batch_size, in_channels, 16, 3, 1, 1, use_cuda); // Conv1: 3x3, 16 filters
        out_h1 = (in_height + 2 * 1 - 3) / 1 + 1; // padding = 1, stride = 1
        out_w1 = (in_width  + 2 * 1 - 3) / 1 + 1;

        // Conv Layer 2
        conv2 = std::make_unique<ConvBlock>(batch_size, 16, 32, 3, 1, 1, use_cuda); // Conv2: 3x3, 32 filters
        out_h2 = (out_h1 + 2 * 1 - 3) / 1 + 1; // padding = 1, stride = 1
        out_w2 = (out_w1 + 2 * 1 - 3) / 1 + 1;

        // Flattened feature size
        flattened_dim = 32 * out_h2 * out_w2;

        // MLP
        std::vector<int> mlp_layers = {flattened_dim, 128, num_classes};
        mlp = std::make_unique<MultiLayerPerceptron>(mlp_layers, 0.01f, batch_size, use_cuda);
    }

    Tensor<2> forward(Tensor<4>& input) {
        std::cout << "[Forward] Input: ";
        input.print_shape();

        Tensor<4> out1 = conv1->fw(input);
        std::cout << "[Forward] After Conv1: ";
        out1.print_shape();

        Tensor<4> out2 = conv2->fw(out1);
        std::cout << "[Forward] After Conv2: ";
        out2.print_shape();

        Tensor<2> flattened = flatten(out2);
        std::cout << "[Forward] Flattened: ";
        flattened.print_shape();

        Tensor<2> out = mlp->fw(flattened);
        std::cout << "[Forward] MLP Output: ";
        out.print_shape();

        return out;
    }

    Tensor<2> backward(Tensor<2>& d_out) {
        std::cout << "[Backward] Loss gradient d_out: ";
        d_out.print_shape();

        Tensor<2> d_flatten = mlp->bp(d_out);
        std::cout << "[Backward] After MLP BP: ";
        d_flatten.print_shape();

        Tensor<4> d_flattened = unflatten(d_flatten);
        std::cout << "[Backward] After Unflatten: ";
        d_flattened.print_shape();

        Tensor<4> d_conv2 = conv2->bp(d_flattened);
        std::cout << "[Backward] After Conv2 BP: ";
        d_conv2.print_shape();

        Tensor<4> d_conv1 = conv1->bp(d_conv2);
        std::cout << "[Backward] After Conv1 BP: ";
        d_conv1.print_shape();

        std::array<size_t, 2> flat_shape = {d_conv1.get_shape()[0], d_conv1.size() / d_conv1.get_shape()[0]};
        Tensor<2> final_out = d_conv1.reshape<2>(flat_shape);
        std::cout << "[Backward] Final reshaped output: ";
        final_out.print_shape();

        return final_out;
    }

    CudaTensor<2> forward_cuda(CudaTensor<4>& input) {
        CudaTensor<4> out1 = conv1->fw_cuda(input);
        CudaTensor<4> out2 = conv2->fw_cuda(out1);
        std::array<size_t, 2> flat_shape = {static_cast<size_t>(batch_size), static_cast<size_t>(flattened_dim)};
        CudaTensor<2> flattened(flat_shape);
        flattened.reshape_from(out2);
        return mlp->fw_cuda(flattened);
    }

    CudaTensor<2> backward_cuda(CudaTensor<2>& d_out) {
        CudaTensor<2> d_hidden = mlp->bp_cuda(d_out);
        CudaTensor<4> d_flattened(std::array<size_t, 4>{
            static_cast<size_t>(batch_size),
            32,                   
            static_cast<size_t>(out_h2),
            static_cast<size_t>(out_w2)
        });
        d_flattened.reshape_from(d_hidden);
        CudaTensor<4> d_conv2 = conv2->bp_cuda(d_flattened);
        CudaTensor<2> output(std::array<size_t, 2>{
            static_cast<size_t>(batch_size),
            static_cast<size_t>(in_channels * in_height * in_width)
        });
        output.reshape_from(conv1->bp_cuda(d_conv2));
        return output;
    }

private:
    int batch_size, in_channels, in_height, in_width, num_classes;
    int out_h1, out_w1, out_h2, out_w2, flattened_dim;
    bool use_cuda;

    std::unique_ptr<ConvBlock> conv1;
    std::unique_ptr<ConvBlock> conv2;
    std::unique_ptr<MultiLayerPerceptron> mlp;

    Tensor<2> flatten(const Tensor<4>& input) {
        const auto shape = input.get_shape(); // [B, C, H, W]
        Tensor<2> output({shape[0], shape[1] * shape[2] * shape[3]});
        for (size_t b = 0; b < shape[0]; ++b)
            for (size_t c = 0; c < shape[1]; ++c)
                for (size_t h = 0; h < shape[2]; ++h)
                    for (size_t w = 0; w < shape[3]; ++w)
                        output(b, c * shape[2] * shape[3] + h * shape[3] + w) = input(b, c, h, w);
        return output;
    }

    Tensor<4> unflatten(const Tensor<2>& input) {
        size_t B = input.get_shape()[0]; 
        size_t flat_dim = input.get_shape()[1]; 
        size_t C = flat_dim / (out_h2 * out_w2);

        if (C * out_h2 * out_w2 != flat_dim) {
            throw std::runtime_error("Invalid unflatten dimensions: shape mismatch.");
        }

        Tensor<4> output({static_cast<size_t>(B), static_cast<size_t>(C),
                  static_cast<size_t>(out_h2), static_cast<size_t>(out_w2)});
        for (size_t b = 0; b < B; ++b)
            for (size_t c = 0; c < C; ++c)
                for (size_t h = 0; h < out_h2; ++h)
                    for (size_t w = 0; w < out_w2; ++w)
                        output(b, c, h, w) = input(b, c * out_h2 * out_w2 + h * out_w2 + w);
        return output;
    }

};
