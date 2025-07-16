#include <iostream>
#include <vector>
#include "mlp.h"
#include "activation_utils.h"

using namespace std;

void print_vector(const vector<vector<float>>& vec, const string& label) {
    cout << label << ":\n";
    for (const auto& row : vec) {
        for (float v : row) cout << v << " ";
        cout << endl;
    }
}

int main() {
    std::vector<int> layer_sizes = {2, 128, 2}; 
    double lr = 0.01;
    int batch_size = 2;
    int training_epoch = 50;

    // Simple XOR-style sample: input = [1, 0], label = [0, 1]
    vector<vector<float>> X = {{1.0f, 0.0f},{1.0f,1.0f}};
    vector<vector<float>> Y = {{0.0f, 1.0f},{1.0f,0.0f}};

    cout << "\n==== CPU MODE ====" << endl;
    MultiLayerPerceptron cpu_mlp(layer_sizes, lr, batch_size, false);
    auto cpu_pred_before = cpu_mlp.predict_batch(X);
    print_vector(cpu_pred_before, "CPU Prediction BEFORE");

    cpu_mlp.train(X, Y, training_epoch);
    auto cpu_pred_after = cpu_mlp.predict_batch(X);
    print_vector(cpu_pred_after, "CPU Prediction AFTER");

    cout << "\n==== GPU MODE ====" << endl;
    MultiLayerPerceptron gpu_mlp(layer_sizes, lr, batch_size, true);
    auto gpu_pred_before = gpu_mlp.predict_batch(X);
    print_vector(gpu_pred_before, "GPU Prediction BEFORE");

    gpu_mlp.train(X, Y, training_epoch);
    auto gpu_pred_after = gpu_mlp.predict_batch(X);
    print_vector(gpu_pred_after, "GPU Prediction AFTER");

    cout << "\n==== Compare Difference ====" << endl;
    for (size_t i = 0; i < cpu_pred_after[0].size(); ++i) {
        float diff = abs(cpu_pred_after[0][i] - gpu_pred_after[0][i]);
        cout << "Output[" << i << "] diff: " << diff << endl;
    }

    return 0;
}
