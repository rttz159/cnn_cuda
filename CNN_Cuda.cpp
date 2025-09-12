#include "CNN_Cuda.h"
#include <iostream>

CNN_CUDA::CNN_CUDA(std::vector<Conv_CUDA*> conv_layers, MLP_CUDA* mlp_layer)
    : conv_layers(conv_layers), mlp_layer(mlp_layer) {}

CNN_CUDA::~CNN_CUDA() {
    for (auto* conv : conv_layers) {
        delete conv;
    }
    delete mlp_layer;
}

void CNN_CUDA::run(const std::vector<std::vector<float>>& inputs) {
    std::vector<std::vector<float>> x = inputs;

    for (auto* conv : conv_layers) {
        conv->run(x);
        x = conv->get_outputs();
    }

    std::vector<std::vector<float>> flat_x = flatten(x);
    mlp_layer->run(flat_x);
}

void CNN_CUDA::bp(const std::vector<std::vector<float>>& error) {
    mlp_layer->bp(error);
    std::vector<std::vector<float>> grad = mlp_layer->get_input_gradients();

    grad = unflatten(grad, conv_layers.back()->K, 
                        conv_layers.back()->H_out, 
                        conv_layers.back()->W_out);

    for (int i = (int)conv_layers.size() - 1; i >= 0; i--) {
        conv_layers[i]->bp(grad);
        grad = conv_layers[i]->get_input_gradients();
    }
}

std::vector<std::vector<float>> CNN_CUDA::get_outputs() {
    return mlp_layer->get_outputs();
}

std::vector<std::vector<float>> CNN_CUDA::flatten(const std::vector<std::vector<float>>& x) {
    std::vector<std::vector<float>> flat(x.size());
    #pragma omp parallel for
    for (size_t n = 0; n < x.size(); n++) {
        flat[n].reserve(x[n].size());
        for (auto v : x[n]) flat[n].push_back(v);
    }
    return flat;
}

std::vector<std::vector<float>> CNN_CUDA::unflatten(
    const std::vector<std::vector<float>>& x, int C, int H, int W) 
{
    std::vector<std::vector<float>> unflat(x.size(), std::vector<float>(C*H*W));
    #pragma omp parallel for
    for (size_t n = 0; n < x.size(); n++) {
        for (size_t i = 0; i < x[n].size(); i++) {
            unflat[n][i] = x[n][i];
        }
    }
    return unflat;
}

TrainResult CNN_CUDA::train_batch(const std::vector<std::vector<float>>& inputs,
                                  const std::vector<int>& labels) 
{
    size_t current_batch_size = inputs.size();
    run(inputs);
    
    size_t num_classes = mlp_layer->get_outputs()[0].size();
    std::vector<std::vector<float>> target(current_batch_size, std::vector<float>(num_classes, 0.0f));

    #pragma omp parallel for
    for (size_t i = 0; i < current_batch_size; i++) {
        target[i][labels[i]] = 1.0f;
    }

    std::vector<std::vector<float>> grad(current_batch_size, std::vector<float>(num_classes));
    float loss = 0.0f;
    int correct = 0;
    auto outputs = mlp_layer->get_outputs();

    #pragma omp parallel for reduction(+:loss, correct)
    for (size_t i = 0; i < current_batch_size; i++) {
        int pred_label = 0;
        float max_val = outputs[i][0];
        for (size_t j = 0; j < num_classes; j++) {
            float diff = outputs[i][j] - target[i][j];
            grad[i][j] = 2.0f * diff / current_batch_size; 
            loss += diff * diff;

            if (outputs[i][j] > max_val) {
                max_val = outputs[i][j];
                pred_label = j;
            }
        }
        if (pred_label == labels[i]) correct++;
    }

    bp(grad);

    TrainResult result;
    result.loss = loss / current_batch_size;
    result.accuracy = static_cast<float>(correct) / current_batch_size;
    return result;
}