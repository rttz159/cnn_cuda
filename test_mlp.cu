#include <iostream>
#include <vector>
#include <chrono>
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
    std::vector<int> layer_sizes = {2, 240, 240, 240, 2}; 
    double lr = 0.01;
    int batch_size = 2;
    int training_epoch = 100;

    // Simple XOR-style sample: input = [1, 0], label = [0, 1]
    vector<vector<float>> X = {{1.0f, 0.0f},{1.0f,1.0f}};
    vector<vector<float>> Y = {{0.0f, 1.0f},{1.0f,0.0f}};

    auto start_cpu_time = std::chrono::high_resolution_clock::now();

    cout << "\n==== CPU MODE ====" << endl;
    MultiLayerPerceptron cpu_mlp(layer_sizes, lr, batch_size, false);
    auto cpu_pred_before = cpu_mlp.predict_batch(X);
    print_vector(cpu_pred_before, "CPU Prediction BEFORE");

    cpu_mlp.train(X, Y, training_epoch);
    auto cpu_pred_after = cpu_mlp.predict_batch(X);
    print_vector(cpu_pred_after, "CPU Prediction AFTER");

    auto end_cpu_time = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu_time - start_cpu_time).count();
    std::cout << "CPU ms = " << duration_cpu << "\n";

    auto start_gpu_time = std::chrono::high_resolution_clock::now();

    cout << "\n==== GPU MODE ====" << endl;
    MultiLayerPerceptron gpu_mlp(layer_sizes, lr, batch_size, true);
    auto gpu_pred_before = gpu_mlp.predict_batch(X);
    print_vector(gpu_pred_before, "GPU Prediction BEFORE");

    gpu_mlp.train(X, Y, training_epoch);
    auto gpu_pred_after = gpu_mlp.predict_batch(X);
    print_vector(gpu_pred_after, "GPU Prediction AFTER");

    auto end_gpu_time = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu_time - start_gpu_time).count();
    std::cout << "GPU ms = " << duration_gpu << "\n";

    cout << "\n==== Compare Difference ====" << endl;
    for (size_t i = 0; i < cpu_pred_after[0].size(); ++i) {
        float diff = abs(cpu_pred_after[0][i] - gpu_pred_after[0][i]);
        cout << "Output[" << i << "] diff: " << diff << endl;
    }

    return 0;
}
