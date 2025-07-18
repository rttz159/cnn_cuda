#pragma once
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include "tensor.h"
#include "cuda_tensor.cuh"

int count_total_images_(const std::string &path);

class OpenCVLoader
{
public:

    OpenCVLoader(){}
    
    OpenCVLoader(int batch_size, int target_height, int target_width, bool cudaEnabled = false, int target_channels = 3)
    {
        this->batch_size = batch_size;
        this->target_channels = target_channels;
        this->target_height = target_height;
        this->target_width = target_width;
        this->cudaEnabled = cudaEnabled;
    }

    // Path to the train, test and validation folders
    void load_from_folders();
    std::map<std::string, int> local_class_to_label;

    void load_images_into_batches(
                        const std::string &dataset_path,
                        std::vector<Tensor<4>> &batches,
                        std::vector<Tensor<2>> &label_batches,
                        std::map<std::string, int> &class_to_index,
                        int batch_size,
                        int target_channels,
                        int target_height,
                        int target_width);

    int target_width;
    int target_height;
    int target_channels;
    int batch_size;
    bool cudaEnabled;

    std::vector<Tensor<1>> mask_batches;
    
    std::vector<Tensor<4>> training_batches;
    std::vector<Tensor<2>> training_label_batches;
    std::vector<Tensor<4>> testing_batches;
    std::vector<Tensor<2>> testing_label_batches;
    std::vector<Tensor<4>> validation_batches;
    std::vector<Tensor<2>> validation_label_batches;

    std::vector<CudaTensor<4>> training_batches_cuda;
    std::vector<CudaTensor<2>> training_label_batches_cuda;
    std::vector<CudaTensor<4>> testing_batches_cuda;
    std::vector<CudaTensor<2>> testing_label_batches_cuda;
    std::vector<CudaTensor<4>> validation_batches_cuda;
    std::vector<CudaTensor<2>> validation_label_batches_cuda;

};