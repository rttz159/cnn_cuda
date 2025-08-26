#pragma once
#include "Conv_CUDA.h"
#include "MLP_CUDA.h"
#include <vector>
#include <memory>

struct TrainResult {
    float loss;
    float accuracy;
};

class CNN_CUDA {
public:
    CNN_CUDA(std::vector<Conv_CUDA*> conv_layers, MLP_CUDA* mlp_layer);

    ~CNN_CUDA();

    void run(const std::vector<std::vector<float>>& inputs);

    void bp(const std::vector<std::vector<float>>& error);

    TrainResult train_batch(const std::vector<std::vector<float>>& inputs,
                                  const std::vector<int>& labels);

    std::vector<std::vector<float>> get_outputs();

private:
    std::vector<Conv_CUDA*> conv_layers;
    MLP_CUDA* mlp_layer;

    std::vector<std::vector<float>> flatten(const std::vector<std::vector<float>>& x);

    std::vector<std::vector<float>> unflatten(
        const std::vector<std::vector<float>>& x, int C, int H, int W);
        
};
