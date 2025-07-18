#include "Datasets.cuh"
#include "cnn.cuh"
#include <iostream>
#include <cmath>
#include <chrono>

constexpr int NUM_EPOCHS = 10;

Tensor<2> compute_loss_cpu(const Tensor<2>& predictions, const Tensor<2>& labels, float& loss_out) {
    size_t B = predictions.get_shape()[0], C = predictions.get_shape()[1];
    Tensor<2> grad({B, C});
    float total_loss = 0.0f;
    int valid_samples = 0;

    for (size_t i = 0; i < B; ++i) {
        int label = static_cast<int>(labels(i, 0));
        if (label < 0 || label >= C) continue;

        for (size_t j = 0; j < C; ++j) {
            float target = (j == label) ? 1.0f : 0.0f;
            float diff = predictions(i, j) - target;
            grad(i, j) = diff;
            total_loss += 0.5f * diff * diff;
        }
        valid_samples++;
    }

    loss_out = (valid_samples > 0) ? total_loss / valid_samples : 0.0f;
    return grad;
}

Tensor<2> softmax_cpu(const Tensor<2>& logits) {
    size_t B = logits.get_shape()[0];
    size_t C = logits.get_shape()[1];
    Tensor<2> probs({B, C});

    for (size_t i = 0; i < B; ++i) {
        float max_logit = logits(i, 0);
        for (size_t j = 1; j < C; ++j)
            if (logits(i, j) > max_logit) max_logit = logits(i, j);

        float sum_exp = 0.0f;
        for (size_t j = 0; j < C; ++j) {
            probs(i, j) = std::exp(logits(i, j) - max_logit);
            sum_exp += probs(i, j);
        }

        for (size_t j = 0; j < C; ++j)
            probs(i, j) /= sum_exp;
    }

    return probs;
}


float compute_accuracy_cpu(const Tensor<2>& predictions, const Tensor<2>& labels) {
    size_t B = predictions.get_shape()[0], C = predictions.get_shape()[1];
    int correct = 0;
    int valid_samples = 0;

    for (size_t i = 0; i < B; ++i) {
        int label = static_cast<int>(labels(i, 0));
        if (label < 0 || label >= static_cast<int>(C)) continue; 

        int pred = 0;
        float max_val = predictions(i, 0);
        for (size_t j = 1; j < C; ++j) {
            if (predictions(i, j) > max_val) {
                max_val = predictions(i, j);
                pred = j;
            }
        }

        if (pred == label) correct++;
        valid_samples++;
    }

    if (valid_samples == 0) return 0.0f;
    return static_cast<float>(correct) / valid_samples;
}


int main() {

    const int batch_size = 16;
    const int in_channels = 3;
    const int in_height = 32;
    const int in_width = 32;
    const int num_classes = 10;
    const bool use_cuda = true;

    // Load Dataset
    OpenCVLoader loader(batch_size, in_height, in_width, use_cuda);
    std::map<std::string, int> class_to_label;

    std::cout << "Loading training and validation data..." << std::endl;
    loader.load_from_folders();
    class_to_label = loader.local_class_to_label;

    // Initialize Model
    CNN model(batch_size, in_channels, in_height, in_width, num_classes, use_cuda);

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        size_t num_batches = loader.training_batches.size();

        for (size_t i = 0; i < num_batches; i++) {
            if (use_cuda) {
                // GPU path
                CudaTensor<4>& input = loader.training_batches_cuda[i];
                CudaTensor<2>& labels = loader.training_label_batches_cuda[i];

                CudaTensor<2> output = model.forward_cuda(input);
                Tensor<2> output_cpu = output.to_host();
                Tensor<2> labels_cpu = labels.to_host();

                float loss = 0.0f;
                Tensor<2> grad_cpu = compute_loss_cpu(output_cpu, labels_cpu, loss);
                for (size_t i = 0; i < grad_cpu.get_shape()[0]; ++i) {
                    int label = static_cast<int>(labels_cpu(i, 0));
                    if (label < 0) {
                        for (size_t j = 0; j < grad_cpu.get_shape()[1]; ++j) {
                            grad_cpu(i, j) = 0.0f;
                        }
                    }
                }

                CudaTensor<2> grad_cuda(grad_cpu.get_shape());
                grad_cuda.copy_from_host(grad_cpu.raw_data_arr());
                model.backward_cuda(grad_cuda,loader.mask_batches[i]);

                Tensor<2> probs = softmax_cpu(output_cpu);

                float acc = compute_accuracy_cpu(probs, labels_cpu);
                epoch_loss += loss;
                epoch_acc += acc;
            } else {
                // CPU path
                Tensor<4>& input = loader.training_batches[i];
                Tensor<2>& labels = loader.training_label_batches[i];

                Tensor<2> output = model.forward(input);

                float loss = 0.0f;
                Tensor<2> grad = compute_loss_cpu(output, labels, loss);
                model.backward(grad);

                Tensor<2> probs = softmax_cpu(output);
                float acc = compute_accuracy_cpu(probs, labels);
                epoch_loss += loss;
                epoch_acc += acc;
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start).count();

        std::cout << "[Epoch " << epoch + 1 << "] Loss: " << epoch_loss / num_batches
                << ", Train Accuracy: " << (epoch_acc / num_batches) * 100.0f << "%"
                << ", Time: " << duration << " ms" << std::endl;

        // Validation
        float val_acc = 0.0f;
        size_t val_batches = loader.validation_batches.size();

        for (size_t i = 0; i < val_batches; i++) {
            if (use_cuda) {
                CudaTensor<4>& input = loader.validation_batches_cuda[i];
                CudaTensor<2>& labels = loader.validation_label_batches_cuda[i];

                CudaTensor<2> output = model.forward_cuda(input);
                Tensor<2> output_cpu = output.to_host();
                Tensor<2> labels_cpu = labels.to_host();

                val_acc += compute_accuracy_cpu(output_cpu, labels_cpu);
            } else {
                Tensor<4>& input = loader.validation_batches[i];
                Tensor<2>& labels = loader.validation_label_batches[i];

                Tensor<2> output = model.forward(input);
                val_acc += compute_accuracy_cpu(output, labels);
            }
        }

        std::cout << "Validation Accuracy: "
                  << (val_acc / val_batches) * 100.0f << "%" << std::endl;
    }

    return 0;
}
