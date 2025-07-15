#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "mlp.h"
#include "activation_utils.h"

using namespace std;

void generate_spiral_data(std::vector<std::vector<float>> &X, std::vector<std::vector<float>> &y, int points_per_class, int num_classes)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.05f);

    for (int class_label = 0; class_label < num_classes; ++class_label)
    {
        for (int i = 0; i < points_per_class; ++i)
        {
            float r = static_cast<float>(i) / points_per_class;
            float theta = class_label * 3.14159f + r * 4.0f * 3.14159f;

            float x = r * cos(theta) + noise(gen);
            float y_coord = r * sin(theta) + noise(gen);

            X.push_back({x, y_coord});
            if (class_label == 0)
                y.push_back({1.0f, 0.0f}); // One-hot label
            else
                y.push_back({0.0f, 1.0f});
        }
    }
}

int main()
{
    std::cout << "--- Starting Spiral Classification Test ---" << std::endl;

    // Generate dataset
    std::vector<std::vector<float>> X_train;
    std::vector<std::vector<float>> y_train;
    generate_spiral_data(X_train, y_train, 100, 2); // 100 points per class

    // Define network architecture
    std::vector<int> layer_sizes = {2, 200, 200, 2}; // Input, hidden1, hidden2, output
    std::vector<ActivationFunction> activation_functions = {
        ActivationFunction::Tanh, ActivationFunction::Tanh, ActivationFunction::Softmax};

    double learning_rate = 0.1;
    int batch_size = 32;
    int epochs = 2500;

    // Create and train MLP
    MultiLayerPerceptron mlp(layer_sizes, activation_functions, LossFunction::CrossEntropy, learning_rate, batch_size, true);
    std::cout << "Training model on spiral dataset..." << std::endl;
    mlp.train(X_train, y_train, epochs);

    // Test predictions on sample points
    std::cout << "\n--- Testing sample predictions ---" << std::endl;
    std::vector<std::vector<float>> test_points = {
        {0.0f, 0.0f},
        {0.5f, 0.5f},
        {-0.5f, -0.5f},
        {1.0f, 0.0f},
        {-1.0f, 0.0f}};

    std::vector<std::vector<float>> predictions = mlp.predict_batch(test_points);

    for (size_t i = 0; i < test_points.size(); ++i)
    {
        std::cout << "Input (" << test_points[i][0] << ", " << test_points[i][1] << ") -> Output: (";
        for (size_t j = 0; j < predictions[i].size(); ++j)
        {
            std::cout << predictions[i][j] << (j == predictions[i].size() - 1 ? "" : ", ");
        }
        std::cout << ")" << std::endl;
    }

    std::cout << "--- Spiral Test Complete ---" << std::endl;
    return 0;
}
