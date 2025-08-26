#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include "CNN_CUDA.h"
#include "Datasets.h"

using namespace std;

int main() {
    srand(42);
    std::mt19937 rng(42);

    float bias_conv = 0.1f;
    float eta_conv = 0.01f;
    float bias_dense = 1.0f;
    float eta_dense = 0.01f;
    int batch_size = 32;  
    int epochs = 2;

    Conv_CUDA* conv1 = new Conv_CUDA(1, 28, 28, 8, 3, 2, 0, eta_conv, batch_size, 0, bias_conv);
    Conv_CUDA* conv2 = new Conv_CUDA(8, 13, 13, 2, 3, 2, 0, eta_conv, batch_size, 0, bias_conv);
    vector<Conv_CUDA*> conv_layers = {conv1, conv2};

    int flatten_size = conv2->K * conv2->H_out * conv2->W_out;
    vector<int> hidden = {72};
    vector<int> mlp_layers = {flatten_size};
    mlp_layers.insert(mlp_layers.end(), hidden.begin(), hidden.end());
    mlp_layers.push_back(10); 
    MLP_CUDA* mlp = new MLP_CUDA(mlp_layers, eta_dense, batch_size, 0, bias_dense);

    CNN_CUDA network(conv_layers, mlp);

    MNIST mnist_loader;
    vector<vector<float>> Train_DS, Test_DS;
    vector<int> Train_L, Test_L;
    mnist_loader.get_mnist(Train_DS, Train_L, Test_DS, Test_L);

    cout << "Train dataset size: " << Train_DS.size() << "\n";
    cout << "Test dataset size: " << Test_DS.size() << "\n\n";

    for (int e = 0; e < epochs; ++e) {
        float epoch_loss = 0.0f;
        int epoch_correct = 0;
        size_t num_batches = 0;

        auto epoch_start = chrono::high_resolution_clock::now();

        for (size_t i = 0; i < Train_DS.size(); i += batch_size) {
            int current_batch_size = std::min(batch_size, static_cast<int>(Train_DS.size() - i));

            vector<vector<float>> batch(current_batch_size);
            vector<int> labels(current_batch_size);
            for (int j = 0; j < current_batch_size; ++j) {
                batch[j] = Train_DS[i + j];
                labels[j] = Train_L[i + j];
            }

            auto result = network.train_batch(batch, labels);
            epoch_loss += result.loss;
            epoch_correct += static_cast<int>(result.accuracy * current_batch_size);
            num_batches++;
        }

        auto epoch_end = chrono::high_resolution_clock::now();
        chrono::duration<double> epoch_time = epoch_end - epoch_start;

        int correct = 0;
        int total = 0;
        auto test_start = chrono::high_resolution_clock::now();
        for (size_t i = 0; i < Test_DS.size(); i += batch_size) {
            int current_batch_size = std::min(batch_size, static_cast<int>(Test_DS.size() - i));
            vector<vector<float>> batch(current_batch_size);
            for (int j = 0; j < current_batch_size; ++j)
                batch[j] = Test_DS[i + j];

            network.run(batch);
            auto outputs = network.get_outputs();

            for (int j = 0; j < current_batch_size; ++j) {
                auto &out_vec = outputs[j];
                int pred_label = std::distance(out_vec.begin(), std::max_element(out_vec.begin(), out_vec.end()));
                if (pred_label == Test_L[i + j])
                    correct++;
                total++;
            }
        }
        auto test_end = chrono::high_resolution_clock::now();
        chrono::duration<double> test_time = test_end - test_start;

        float train_accuracy = static_cast<float>(epoch_correct) / Train_DS.size();
        float test_accuracy = static_cast<float>(correct) / total;

        cout << "========================\n";
        cout << "Epoch " << e + 1 << "/" << epochs << "\n";
        cout << "Training Loss: " << (epoch_loss / num_batches) << "\n";
        cout << "Training Accuracy: " << train_accuracy * 100.0f << "%\n";
        cout << "Test Accuracy: " << test_accuracy * 100.0f << "%\n";
        cout << "Training time: " << epoch_time.count() << " seconds\n";
        cout << "Test time: " << test_time.count() << " seconds\n";
        cout << "========================\n";
    }

    return 0;
}
