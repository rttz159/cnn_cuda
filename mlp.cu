#include "mlp.h"
#include <random>
#include <iostream>
#include <numeric>
#include <string>

std::default_random_engine rng(std::random_device{}());

void print_tensor_shape(const std::string &name, const Tensor<2> &tensor)
{
    auto shape = tensor.get_shape();
    std::cout << name << " shape: (" << shape[0] << ", " << shape[1] << ")" << std::endl;
}

void print_tensor_content(const std::string &name, const Tensor<2> &tensor)
{
    auto shape = tensor.get_shape();
    std::cout << name << " content (" << shape[0] << "x" << shape[1] << "):" << std::endl;
    for (size_t i = 0; i < shape[0]; ++i)
    {
        std::cout << "[";
        for (size_t j = 0; j < shape[1]; ++j)
        {
            std::cout << tensor(i, j) << (j == shape[1] - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
}

void MultiLayerPerceptron::print_network_details() const
{
    std::cout << "\n=== NEURON NETWORK DETAILS ===" << std::endl;
    std::cout << "The nn's weight size :" << this->weight.size() << std::endl;
    std::cout << "The nn's biases size :" << this->biases.size() << std::endl;
    std::cout << "The nn's acitvation size :" << this->activations.size() << std::endl;
    std::cout << "The nn's preacitvation size :" << this->pre_activations.size() << std::endl;
    std::cout << "The nn's deltas size :" << this->deltas.size() << std::endl;
    // Printing network structure (weights and biases)
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i)
    {
        std::cout << "--- Layer " << i << " to Layer " << i + 1 << " ---" << std::endl;
        print_tensor_content("Weight Matrix (weight[" + std::to_string(i) + "])", weight[i]);
        print_tensor_content("Bias Vector (biases[" + std::to_string(i) + "])", biases[i]);
    }

    std::cout << "=== END NETWORK DETAILS ===\n"
              << std::endl;
}

void MultiLayerPerceptron::forward()
{
    if (cudaEnabled)
    {
        CudaTensor<2> current = input_cuda;

        for (size_t i = 0; i < layer_sizes.size() - 1; ++i)
        {
            CudaTensor<2> z({static_cast<size_t>(batch_size), static_cast<size_t>(layer_sizes[i + 1])});
            CudaTensor<2>::matmul_device(current, weight_cuda[i], z, false, true);

            CudaTensor<2> z_biased({static_cast<size_t>(batch_size), static_cast<size_t>(layer_sizes[i + 1])});
            CudaTensor<2> broadcast_bias({static_cast<size_t>(batch_size), static_cast<size_t>(layer_sizes[i + 1])});
            broadcast_to_rows_cuda(biases_cuda[i], broadcast_bias);
            z_biased.elementwise_add(z, broadcast_bias);

            pre_activations_cuda[i] = z_biased;
            apply_activation_cuda(z_biased, activations_cuda[i], activation_functions[i]);

            current = activations_cuda[i];
        }
    }
    else
    {
        Tensor<2> current_activation = input;

        for (size_t i = 0; i < layer_sizes.size() - 1; ++i)
        {

            Tensor<2> transposed_weight = Tensor<2>::transpose(weight[i]);

            Tensor<2> z = Tensor<2>::matmul(current_activation, transposed_weight);

            pre_activations[i] = z + biases[i].broadcast_to_rows(batch_size);

            activations[i] = Activation::apply_activation(pre_activations[i], activation_functions[i]);

            current_activation = activations[i];
        }
    }
}

void MultiLayerPerceptron::backward()
{
    if (cudaEnabled)
    {
        apply_softmax_cross_entropy_grad_cuda(activations_cuda.back(), desire_output_cuda, deltas_cuda.back());

        for (int i = static_cast<int>(layer_sizes.size() - 3); i >= 0; --i)
        {
            CudaTensor<2>::matmul_device(deltas_cuda[i + 1], weight_cuda[i + 1], deltas_cuda[i]);
            apply_activation_derivative_cuda(pre_activations_cuda[i], deltas_cuda[i], activation_functions[i]);
        }

        for (size_t i = 0; i < layer_sizes.size() - 1; ++i)
        {
            CudaTensor<2> input_i = (i == 0) ? input_cuda : activations_cuda[i - 1];
            CudaTensor<2> weight_grad({static_cast<size_t>(layer_sizes[i + 1]), static_cast<size_t>(layer_sizes[i])});
            CudaTensor<2>::matmul_device(deltas_cuda[i], input_i, weight_grad, true, false);

            CudaTensor<2> bias_grad({1, static_cast<size_t>(layer_sizes[i + 1])});
            reduce_rows_cuda(deltas_cuda[i], bias_grad);

            update_weights_cuda(weight_cuda[i], weight_grad, learning_rate);
            update_weights_cuda(biases_cuda[i], bias_grad, learning_rate);
        }
    }
    else
    {
        // Calculate delta for the output layer
        deltas.back() = Activation::loss_derivative(activations.back(), desire_output, lossfunction, activation_functions[layer_sizes.size() - 1]);

        // Backpropagate deltas through hidden layers
        for (int i = static_cast<int>(layer_sizes.size() - 3); i >= 0; i--)
        {

            Tensor<2> next_delta;
            Tensor<2> next_weight;

            next_delta = deltas[i + 1];
            next_weight = weight[i + 1];

            // Calculate delta for current layer
            Tensor<2> error_prop = Tensor<2>::matmul(next_delta, next_weight);
            Tensor<2> act_deriv = Activation::activation_derivative(pre_activations[i], activation_functions[i]);

            deltas[i] = error_prop * act_deriv;
        }

        // Update weights and biases
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i)
        {

            Tensor<2> current_input_for_grad = (i == 0) ? input : activations[i - 1];

            // Weight gradient: dW = A_prev^T * delta
            Tensor<2> weight_gradient = Tensor<2>::matmul(Tensor<2>::transpose(deltas[i]), current_input_for_grad);

            // Bias gradient: sum over batch dimension
            Tensor<2> bias_gradient({1, static_cast<size_t>(layer_sizes[i + 1])});
            for (size_t col = 0; col < layer_sizes[i + 1]; ++col)
            {
                float sum_col = 0.0f;
                for (size_t row = 0; row < batch_size; ++row)
                    sum_col += deltas[i](row, col);
                bias_gradient(0, col) = sum_col;
            }

            weight[i] = weight[i] - (weight_gradient * learning_rate);
            biases[i] = biases[i] - (bias_gradient * learning_rate);
        }
    }
}

void MultiLayerPerceptron::set_input(const std::vector<std::vector<float>> &input_batch)
{
    if (input.get_shape()[0] != input_batch.size() || input.get_shape()[1] != static_cast<size_t>(layer_sizes[0]))
    {
        input = Tensor<2>({input_batch.size(), static_cast<size_t>(layer_sizes[0])});
    }

    for (size_t i = 0; i < input_batch.size(); ++i)
    {
        for (size_t j = 0; j < input_batch[i].size(); ++j)
        {
            input(i, j) = input_batch[i][j];
        }
    }

    if (cudaEnabled)
        input_cuda.copy_from_host(input.raw_data_arr());
}

void MultiLayerPerceptron::set_desire_output(const std::vector<std::vector<float>> &output_batch)
{
    auto shape = desire_output.get_shape();
    if (shape[0] != output_batch.size() || shape[1] != output_batch[0].size())
    {
        desire_output = Tensor<2>({output_batch.size(), output_batch[0].size()});
    }

    for (size_t i = 0; i < output_batch.size(); ++i)
        for (size_t j = 0; j < output_batch[i].size(); ++j)
            desire_output(i, j) = output_batch[i][j];

    if (cudaEnabled)
        desire_output_cuda.copy_from_host(desire_output.raw_data_arr());
}

void MultiLayerPerceptron::train(const std::vector<std::vector<float>> &X_train,
                                 const std::vector<std::vector<float>> &y_train,
                                 int epochs)
{
    size_t num_samples = X_train.size();

    for (int epoch = 0; epoch < epochs; ++epoch)
    {

        for (size_t i = 0; i < num_samples; i += batch_size)
        {
            size_t end = std::min(i + batch_size, num_samples);
            std::vector<std::vector<float>> batch_X(X_train.begin() + i, X_train.begin() + end);
            std::vector<std::vector<float>> batch_y(y_train.begin() + i, y_train.begin() + end);

            if (batch_X.size() != batch_size)
            {
                batch_size = static_cast<int>(batch_X.size());
                initialize_input_output();
                initialize_activations();
                initialize_deltas();
            }

            set_input(batch_X);
            set_desire_output(batch_y);
            forward();
            backward();
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " complete.\n";
    }
}

std::vector<std::vector<float>> MultiLayerPerceptron::predict_batch(const std::vector<std::vector<float>> &input_batch)
{
    size_t batch_len = input_batch.size();

    // Dynamically adjust batch_size for prediction if different from current
    if (batch_size != static_cast<int>(batch_len))
    {
        batch_size = static_cast<int>(batch_len);
        initialize_input_output();
        initialize_activations();
        initialize_deltas();
    }

    set_input(input_batch);
    forward();

    if (cudaEnabled)
    {
        const CudaTensor<2> &output_cuda = activations_cuda.back();
        auto shape = output_cuda.get_shape();

        std::vector<float> host_output(output_cuda.size());
        output_cuda.copy_to_host(host_output.data());

        std::vector<std::vector<float>> predictions(shape[0], std::vector<float>(shape[1]));
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                predictions[i][j] = host_output[i * shape[1] + j];

        return predictions;
    }
    else
    {
        const Tensor<2> &output = activations.back();
        auto shape = output.get_shape();

        std::vector<std::vector<float>> predictions(shape[0], std::vector<float>(shape[1]));
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                predictions[i][j] = output(i, j);

        return predictions;
    }
}

void MultiLayerPerceptron::initialize_activations()
{
    activations.clear();
    pre_activations.clear();
    activations_cuda.clear();
    pre_activations_cuda.clear();

    for (size_t i = 1; i < layer_sizes.size(); i++)
    {
        size_t size = static_cast<size_t>(layer_sizes[i]);
        activations.emplace_back(Tensor<2>({static_cast<size_t>(batch_size), size}));
        pre_activations.emplace_back(Tensor<2>({static_cast<size_t>(batch_size), size}));

        if (cudaEnabled)
        {
            activations_cuda.emplace_back(CudaTensor<2>({static_cast<size_t>(batch_size), size}));
            pre_activations_cuda.emplace_back(CudaTensor<2>({static_cast<size_t>(batch_size), size}));
        }
    }
}

void MultiLayerPerceptron::initialize_deltas()
{
    deltas.clear();
    deltas_cuda.clear();
    for (size_t i = 0; i < layer_sizes.size() - 1; i++)
    {
        deltas.emplace_back(Tensor<2>({static_cast<size_t>(batch_size), static_cast<size_t>(layer_sizes[i + 1])}));
        if (cudaEnabled)
        {
            deltas_cuda.emplace_back(CudaTensor<2>({static_cast<size_t>(batch_size), static_cast<size_t>(layer_sizes[i + 1])}));
        }
    }
}

void MultiLayerPerceptron::initialize_weights_and_biases()
{
    weight.clear();
    biases.clear();
    weight_cuda.clear();
    biases_cuda.clear();

    for (size_t i = 0; i < layer_sizes.size() - 1; i++)
    {
        size_t in_features = layer_sizes[i];
        size_t out_features = layer_sizes[i + 1];

        float limit = std::sqrt(6.0f / (in_features + out_features));
        std::uniform_real_distribution<float> dist(-limit, limit);

        Tensor<2> W({out_features, in_features});
        Tensor<2> B({1, out_features});

        for (size_t r = 0; r < out_features; ++r)
        {
            for (size_t c = 0; c < in_features; ++c)
            {
                W(r, c) = dist(rng);
            }
        }
        for (size_t c = 0; c < out_features; ++c)
        {
            B(0, c) = 0.0f;
        }

        weight.push_back(W);
        biases.push_back(B);

        if (cudaEnabled)
        {
            CudaTensor<2> W_cuda({out_features, in_features});
            CudaTensor<2> B_cuda({1, out_features});

            W_cuda.copy_from_host(W.raw_data_arr());
            B_cuda.copy_from_host(B.raw_data_arr());

            weight_cuda.push_back(W_cuda);
            biases_cuda.push_back(B_cuda);
        }
    }
}

void MultiLayerPerceptron::initialize_input_output()
{
    input = Tensor<2>({static_cast<size_t>(batch_size), static_cast<size_t>(layer_sizes[0])});
    desire_output = Tensor<2>({static_cast<size_t>(batch_size), static_cast<size_t>(layer_sizes.back())});

    if (cudaEnabled)
    {
        input_cuda = CudaTensor<2>({static_cast<size_t>(batch_size), static_cast<size_t>(layer_sizes[0])});
        desire_output_cuda = CudaTensor<2>({static_cast<size_t>(batch_size), static_cast<size_t>(layer_sizes.back())});
    }
}
