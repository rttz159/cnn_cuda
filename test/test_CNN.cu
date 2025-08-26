#include <iostream>
#include <vector>
#include <iomanip>
#include "CNN_CUDA.h"

int main() {
    int batch_size = 2;       // larger batch
    int C = 1;
    int H = 4;
    int W = 4;
    float learning_rate = 0.01f;
    int time_step = 0;
    float bias_init = 0.0f;

    // --- Create conv1 ---
    int K1 = 2;
    int F1 = 3;
    int stride1 = 1;
    int pad1 = 1;
    Conv_CUDA* conv1 = new Conv_CUDA(C, H, W, K1, F1, stride1, pad1, learning_rate, batch_size, time_step, bias_init);

    // --- Create conv2 ---
    int K2 = 3;
    int F2 = 3;
    int stride2 = 1;
    int pad2 = 1;
    Conv_CUDA* conv2 = new Conv_CUDA(conv1->K, conv1->H_out, conv1->W_out, K2, F2, stride2, pad2, learning_rate, batch_size, time_step, bias_init);

    std::vector<Conv_CUDA*> conv_layers = {conv1, conv2};

    int flatten_size = conv2->K * conv2->H_out * conv2->W_out;
    std::vector<int> mlp_neurons = {flatten_size, 8, 2};
    MLP_CUDA* mlp = new MLP_CUDA(mlp_neurons, learning_rate, batch_size, time_step, bias_init);

    CNN_CUDA cnn(conv_layers, mlp);

    // --- Prepare input ---
    std::vector<float> sample;
    sample.reserve(C*H*W);
    for (int i = 1; i <= C*H*W; ++i) sample.push_back(float(i));
    std::vector<std::vector<float>> inputs(batch_size, sample);

    std::cout << std::fixed << std::setprecision(6);

    // --- Dummy target for testing ---
    std::vector<std::vector<float>> target(batch_size, std::vector<float>(mlp_neurons.back(), 0.0f));
    // For example, first output should be 1, second 0
    target[0][0] = 1.0f; target[0][1] = 0.0f;
    target[1][0] = 0.0f; target[1][1] = 1.0f;

    int epochs = 100;
    for (int e = 0; e < epochs; ++e) {
        // Forward pass
        cnn.run(inputs);
        auto outputs = cnn.get_outputs();

        // Compute simple MSE loss
        float loss = 0.0f;
        for (int n = 0; n < batch_size; ++n)
            for (int i = 0; i < mlp_neurons.back(); ++i)
                loss += (outputs[n][i] - target[n][i]) * (outputs[n][i] - target[n][i]);
        loss /= batch_size;
        
        // Print loss every 10 epochs
        if (e % 10 == 0) std::cout << "Epoch " << e << ", Loss = " << loss << std::endl;

        // Compute gradient for output (dL/dY)
        std::vector<std::vector<float>> grad(batch_size, std::vector<float>(mlp_neurons.back(), 0.0f));
        for (int n = 0; n < batch_size; ++n)
            for (int i = 0; i < mlp_neurons.back(); ++i)
                grad[n][i] = 2.0f * (outputs[n][i] - target[n][i]) / batch_size;

        // Backward pass + update weights
        cnn.bp(grad);
    }

    // Final outputs
    auto final_outputs = cnn.get_outputs();
    std::cout << "\nFinal MLP outputs:" << std::endl;
    for (int n = 0; n < batch_size; ++n) {
        std::cout << " sample " << n << ":";
        for (float v : final_outputs[n]) std::cout << " " << v;
        std::cout << std::endl;
    }

    return 0;
}