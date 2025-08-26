#include "MLP_Cuda.h"
#include <iostream>
#include <vector>
#include <cmath>

// helper: sigmoid on CPU
float sigmoid_cpu(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// helper: mean squared error
float mse_loss(const std::vector<std::vector<float>>& preds,
               const std::vector<std::vector<float>>& labels) {
    float loss = 0.0f;
    int total = 0;
    for (int i = 0; i < preds.size(); i++) {
        for (int j = 0; j < preds[i].size(); j++) {
            float diff = preds[i][j] - labels[i][j];
            loss += diff * diff;
            total++;
        }
    }
    return loss / total;
}

int main() {
    // --- 1. Define a tiny XOR network: 2 input, 2 hidden, 1 output ---
    std::vector<int> layers = {2, 4, 1};
    float learning_rate = 0.1f;
    int batch_size = 4;   // XOR has 4 samples
    int time_step = 0;
    float bias_init = 0.0f;

    MLP_CUDA mlp(layers, learning_rate, batch_size, time_step, bias_init);

    // --- 2. XOR input/output dataset ---
    std::vector<std::vector<float>> inputs = {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };
    std::vector<std::vector<float>> labels = {
        {0},
        {1},
        {1},
        {0}
    };

    // --- 3. Training loop ---
    for (int epoch = 1; epoch <= 2000; epoch++) {
        // Forward
        mlp.run(inputs);
        auto outputs = mlp.get_outputs();

        // Compute error = prediction - target
        std::vector<std::vector<float>> error(batch_size, std::vector<float>(1, 0.0f));
        for (int i = 0; i < batch_size; i++) {
            error[i][0] = outputs[i][0] - labels[i][0];
        }

        // Backward (update weights)
        mlp.bp(error);

        // Occasionally print loss and outputs
        if (epoch % 200 == 0) {
            float loss = mse_loss(outputs, labels);
            std::cout << "Epoch " << epoch << " Loss = " << loss << "\n";
            for (int i = 0; i < batch_size; i++) {
                std::cout << inputs[i][0] << " XOR " << inputs[i][1]
                          << " => pred=" << outputs[i][0]
                          << " target=" << labels[i][0] << "\n";
            }
            std::cout << "----------------------------\n";
        }
    }

    return 0;
}
