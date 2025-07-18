#include "cnn.cuh"
#include <iostream>
#include <cassert>
#include <array>

int main()
{
    const int batch_size = 2;
    const int in_channels = 3;
    const int in_height = 8;
    const int in_width = 8;
    const int num_classes = 10;

    CNN cnn(batch_size, in_channels, in_height, in_width, num_classes, true);

    // Create dummy input
    Tensor<4> input({batch_size, in_channels, in_height, in_width});
    for (size_t i = 0; i < input.size(); ++i)
    {
        input.raw_data()[i] = static_cast<float>(i % 255) / 255.0f; // Normalized pattern
    }

    std::array<size_t, 4> input_shape = {batch_size, in_channels, in_height, in_width};
    CudaTensor<4> input_cuda(input_shape);
    input_cuda.copy_from_host(input.raw_data_arr());

    // Forward pass
    CudaTensor<2> output = cnn.forward_cuda(input_cuda);
    std::cout << "Forward output shape: (" << output.get_shape()[0] << ", " << output.get_shape()[1] << ")\n";
    assert(output.get_shape()[0] == batch_size);
    assert(output.get_shape()[1] == num_classes);

    // Create dummy gradient (same shape as output)
    Tensor<2> d_out({batch_size, num_classes});
    for (size_t i = 0; i < d_out.size(); ++i)
    {
        d_out.raw_data()[i] = 0.01f;
    }

    std::array<size_t, 2> output_shape = {batch_size, num_classes};
    CudaTensor<2> d_out_cuda(output_shape);
    d_out_cuda.copy_from_host(d_out.raw_data_arr());

    // Backward pass
    CudaTensor<2> d_input = cnn.backward_cuda(d_out_cuda);
    std::cout << "Backward output shape: (" << d_input.get_shape()[0] << ", " << d_input.get_shape()[1] << ")\n";
    assert(d_input.get_shape()[0] == batch_size);
    assert(d_input.get_shape()[1] == in_channels * in_height * in_width);

    std::cout << "Test passed successfully!\n";
    return 0;
}
