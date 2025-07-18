#include <iostream>
#include "convblock.cuh"

int main() {
    // Dummy configuration
    int batch_size = 2;
    int in_channels = 3;
    int out_channels = 1;
    int height = 4;
    int width = 4;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;

    ConvBlock conv(batch_size, in_channels, out_channels, kernel_size, stride, padding, true);

    // ===== Create dummy input =====
    Tensor<4> input({(size_t)batch_size, (size_t)in_channels, (size_t)height, (size_t)width});
    for (size_t i = 0; i < input.size(); ++i) {
        input.raw_data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    CudaTensor<4> input_cuda(input.get_shape());
    input_cuda.copy_from_host(input.raw_data_arr());

    std::cout << "\n\nStart Forward Pass \n\n";

    // ===== Forward Pass =====
    CudaTensor<4> output_cuda = conv.fw_cuda(input_cuda);

    Tensor<4> output_host = Tensor<4>::from_shape_vector(output_cuda.get_shape_vector());
    output_cuda.copy_to_host(output_host.raw_data_arr());

    std::cout << "Forward pass output shape: ";
    output_host.print_shape();

    // ===== Dummy gradient for backward =====
    Tensor<4> grad_output(output_host.get_shape());
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_output.raw_data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    CudaTensor<4> grad_output_cuda(grad_output.get_shape());
    grad_output_cuda.copy_from_host(grad_output.raw_data_arr());

    std::cout << "\n\nStart Back Propagation \n\n";

    // ===== Backward Pass =====
    CudaTensor<4> grad_input_cuda = conv.bp_cuda(grad_output_cuda);

    Tensor<4> grad_input_host = Tensor<4>::from_shape_vector(grad_input_cuda.get_shape_vector());
    grad_input_cuda.copy_to_host(grad_input_host.raw_data_arr());

    std::cout << "Backward pass output (d_input) shape: ";
    grad_input_host.print_shape();

    return 0;
}
