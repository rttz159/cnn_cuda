#include <iostream>
#include <vector>
#include "mlp.h"
#include "activation_utils.h"

using namespace std;

int main()
{
    std::cout << "--- Starting MLP Test ---" << std::endl;

    // Define network architecture: 2 input features, 3 hidden neurons, 2 output neurons
    // Input features are 2, but with bias, it becomes 3 for the first layer's weight.
    // Output features are 2.
    std::vector<int> layer_sizes = { 2, 10, 10, 2 }; // Input layer (features), Hidden layer, Output layer
    std::vector<enum ActivationFunction> activation_function = { ActivationFunction::ReLU, ActivationFunction::ReLU, ActivationFunction::ReLU, ActivationFunction::Softmax };
    double learning_rate = 0.001;
    int batch_size = 4;

    // Create an MLP instance
    MultiLayerPerceptron mlp(layer_sizes, activation_function, LossFunction::CrossEntropy, learning_rate, batch_size);
    std::cout << "MLP initialized with " << layer_sizes.size() << " layers." << std::endl;

    // --- Test Case 1: Simple XOR-like data ---
    // Input: (feature1, feature2)
    // Output: (class0_prob, class1_prob) - One-hot encoded
    std::vector<std::vector<float>> X_train = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f} };

    std::vector<std::vector<float>> y_train = {
        {1.0f, 0.0f}, // Class 0
        {0.0f, 1.0f}, // Class 1
        {0.0f, 1.0f}, // Class 1
        {1.0f, 0.0f}  // Class 0
    };

    int epochs = 10000; // Train for a sufficient number of epochs

    std::cout << "\n--- Training MLP with XOR-like data ---" << std::endl;
    mlp.train(X_train, y_train, epochs);
    std::cout << "Training complete." << std::endl;

    // --- Test Case 2: Make predictions ---
    std::cout << "\n--- Making predictions ---" << std::endl;
    std::vector<std::vector<float>> X_test = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.5f, 0.5f} // A new input
    };

    std::vector<std::vector<float>> predictions = mlp.predict_batch(X_test);

    std::cout << "Predictions:" << std::endl;
    for (size_t i = 0; i < predictions.size(); ++i)
    {
        std::cout << "Input: (" << X_test[i][0] << ", " << X_test[i][1] << ") -> Output: (";
        for (size_t j = 0; j < predictions[i].size(); ++j)
        {
            std::cout << predictions[i][j] << (j == predictions[i].size() - 1 ? "" : ", ");
        }
        std::cout << ")" << std::endl;
    }

    // --- Basic Sanity Check ---
    // After training, we expect the network to have learned the XOR pattern.
    // The predictions should be close to the target outputs.
    // This is a qualitative check. For proper unit testing, you'd use assertions.
    std::cout << "\n--- Sanity Check (Qualitative) ---" << std::endl;
    // For (0,0), expected output is (1,0)
    // For (0,1), expected output is (0,1)
    // For (1,0), expected output is (0,1)
    // For (1,1), expected output is (1,0)
    // Look at the predictions above and see if they are close to these.

    std::cout << "\n--- MLP Test Complete ---" << std::endl;

    return 0;
}