#pragma once

#include <iostream>
#include <vector>
#include "tensor.h"
#include "cuda_tensor.cuh"

using namespace std;

class Convolutional
{
public:
    int _image_dim[3] = {1, 16, 16}; // image specification
    int _specs[4] = {2, 3, 3, 3};    // filter specifications
    int _out_dim[3] = {2, 13, 13};   // convoluted output dimensions
    int _padding = 1;
    int _stride = 2;

    Tensor<4> filter; //[num_kernels, kernel_height, kernel_width, input_depth]
    Tensor<3> cache;  // for img after padding

    inline int compute_conv_output_dim(int input_size, int filter_size, int padding, int stride)
    {
        return ((input_size - filter_size + 2 * padding) / stride) + 1;
    }

    void _out_dimension();
    void padding(Tensor<3> &original_img, Tensor<3> &out_img);
    void backward(Tensor<3> d_out_vol, Tensor<3> &d_input);
    void forward(Tensor<3> &image, Tensor<3> &out);

private:
    vector<float> bias;
};