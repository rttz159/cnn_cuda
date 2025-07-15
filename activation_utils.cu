#include "activation_utils.cuh"
#include <iostream>

__global__ void activation_kernel(float* input, float* output, size_t size, ActivationFunction func) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];
    float y = x;

    switch (func) {
        case ActivationFunction::ReLU:
            y = relu_cuda(x);
            break;
        case ActivationFunction::Sigmoid:
            y = sigmoid_cuda(x);
            break;
        case ActivationFunction::Tanh:
            y = tanh_act_cuda(x);
            break;
        case ActivationFunction::None:
        default:
            y = x;
            break;
    }

    output[idx] = y;
}

__global__ void softmax_kernel(float* input, float* output, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float* input_row = input + row * cols;
    float* output_row = output + row * cols;

    float max_val = input_row[0];
    for (size_t j = 1; j < cols; ++j) {
        max_val = fmaxf(max_val, input_row[j]);
    }

    float sum_exp = 0.0f;
    for (size_t j = 0; j < cols; ++j) {
        output_row[j] = __expf(input_row[j] - max_val);
        sum_exp += output_row[j];
    }

    for (size_t j = 0; j < cols; ++j) {
        output_row[j] /= sum_exp;
    }
}


__global__ void activation_derivative_kernel(float* input, float* output, size_t size, ActivationFunction func) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];
    float grad = 1.0f;

    switch (func) {
        case ActivationFunction::ReLU:
            grad = relu_derivative_cuda(x);
            break;
        case ActivationFunction::Sigmoid:
            grad = sigmoid_derivative_cuda(x);
            break;
        case ActivationFunction::Tanh:
            grad = tanh_derivative_cuda(x);
            break;
        case ActivationFunction::None:
        default:
            grad = 1.0f;
            break;
    }

    output[idx] *= grad;
}

__global__ void softmax_cross_entropy_grad_kernel(float* prediction, float* target, float* grad, size_t batch, size_t classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * classes) return;

    size_t i = idx / classes;
    size_t j = idx % classes;

    grad[i * classes + j] = (prediction[i * classes + j] - target[i * classes + j]);
}

__global__ void reduce_rows_kernel(float* input, float* output, size_t rows, size_t cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float sum = 0.0f;
    for (size_t i = 0; i < rows; ++i)
        sum += input[i * cols + col];
    output[col] = sum;
}

__global__ void broadcast_to_rows_kernel(const float* src, float* dst, size_t rows, size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * cols;

    if (idx >= total) return;

    size_t row = idx / cols;
    size_t col = idx % cols;

    dst[row * cols + col] = src[col];
}

__global__ void update_weights_kernel(float* W, const float* grad, float lr, size_t size, float batch_size_float) { 
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        W[idx] -= (lr / batch_size_float) * grad[idx]; 
    }
}

__global__ void kernel_mse_loss(const float *prediction, const float *target, float *loss_buffer, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float diff = prediction[idx] - target[idx];
        loss_buffer[idx] = diff * diff;
    }
}

__global__ void kernel_softmax_cross_entropy_loss(const float *logits, const float *target, float *loss_buffer, int batch_size, int num_classes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    const float *logit_row = logits + i * num_classes;
    const float *target_row = target + i * num_classes;

    float max_logit = logit_row[0];
    for (int j = 1; j < num_classes; ++j)
        max_logit = fmaxf(max_logit, logit_row[j]);

    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; ++j)
        sum_exp += __expf(logit_row[j] - max_logit);

    float loss = 0.0f;
    for (int j = 0; j < num_classes; ++j)
    {
        float softmax = __expf(logit_row[j] - max_logit) / sum_exp;
        if (target_row[j] == 1.0f)
        {
            loss = -__logf(softmax + 1e-8f);
            break;
        }
    }

    loss_buffer[i] = loss;
}

void apply_activation_cuda(CudaTensor<2>& input, CudaTensor<2>& output, ActivationFunction func) {
    size_t size = input.size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    if (func == ActivationFunction::Softmax) {
        size_t rows = input.get_shape()[0];
        size_t cols = input.get_shape()[1];

        dim3 softmax_blocks((rows + threads - 1) / threads, 1); 
        dim3 softmax_threads(threads, 1);

        softmax_kernel<<<rows, threads>>>(input.raw(), output.raw(), rows, cols); 
    } else {
        activation_kernel<<<blocks, threads>>>(input.raw(), output.raw(), size, func);
    }
    cudaDeviceSynchronize();
}

void apply_activation_derivative_cuda(CudaTensor<2>& input, CudaTensor<2>& output, ActivationFunction func) {
    size_t size = input.size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    activation_derivative_kernel<<<blocks, threads>>>(input.raw(), output.raw(), size, func);
    cudaDeviceSynchronize();
}

void apply_softmax_cross_entropy_grad_cuda(CudaTensor<2>& prediction, CudaTensor<2>& target, CudaTensor<2>& grad) {
    size_t batch = prediction.get_shape()[0];
    size_t classes = prediction.get_shape()[1];
    int threads = 256;
    int blocks = (batch * classes + threads - 1) / threads;
    softmax_cross_entropy_grad_kernel<<<blocks, threads>>>(prediction.raw(), target.raw(), grad.raw(), batch, classes);
    cudaDeviceSynchronize();
}

void broadcast_to_rows_cuda(const CudaTensor<2>& src, CudaTensor<2>& dst) {
    if (src.get_shape()[0] != 1)
        throw std::invalid_argument("broadcast_to_rows: source must have 1 row");

    size_t rows = dst.get_shape()[0];
    size_t cols = dst.get_shape()[1];

    if (src.get_shape()[1] != cols)
        throw std::invalid_argument("broadcast_to_rows: column mismatch");

    size_t total = rows * cols;
    int threads = 128;
    int blocks = (total + threads - 1) / threads;

    broadcast_to_rows_kernel<<<blocks, threads>>>(src.raw(), dst.raw(), rows, cols);
    cudaDeviceSynchronize();
 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

}

void reduce_rows_cuda(CudaTensor<2>& input, CudaTensor<2>& output) {
    const size_t* shape_in = input.get_shape();
    const size_t rows = shape_in[0];
    const size_t cols = shape_in[1];

    const size_t* shape_out = output.get_shape();
    if (shape_out[0] != 1 || shape_out[1] != cols) {
        throw std::invalid_argument("reduce_rows: output tensor must have shape [1, cols]");
    }

    int threads = 256;
    int blocks = (cols + threads - 1) / threads;

    reduce_rows_kernel<<<blocks, threads>>>(input.raw(), output.raw(), rows, cols);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("reduce_rows kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void update_weights_cuda(CudaTensor<2>& W, const CudaTensor<2>& grad, float lr, float batch_size_float) {
    if (W.size() != grad.size()) {
        throw std::invalid_argument("update_weights: size mismatch between weight and gradient tensors");
    }

    size_t size = W.size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    update_weights_kernel<<<blocks, threads>>>(W.raw(), grad.raw(), lr, size, batch_size_float);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("update_weights kernel failed: ") + cudaGetErrorString(err));
    }
}

float compute_loss_gpu(const CudaTensor<2>& prediction,
                              const CudaTensor<2>& target,
                              LossFunction loss_func,
                              ActivationFunction output_activation)
{
    const size_t* shape = prediction.get_shape();
    int batch_size = static_cast<int>(shape[0]);
    int output_size = static_cast<int>(shape[1]);

    float* d_loss_buffer = nullptr;
    float* h_loss_buffer = nullptr;
    size_t loss_buffer_size = (loss_func == LossFunction::MSE)
        ? batch_size * output_size
        : batch_size;

    cudaMalloc(&d_loss_buffer, loss_buffer_size * sizeof(float));
    h_loss_buffer = new float[loss_buffer_size];

    int threads = 256;
    int blocks = (loss_buffer_size + threads - 1) / threads;

    if (loss_func == LossFunction::MSE)
    {
        kernel_mse_loss<<<blocks, threads>>>(
            prediction.raw(), target.raw(), d_loss_buffer, batch_size * output_size);
    }
    else if (loss_func == LossFunction::CrossEntropy &&
             output_activation == ActivationFunction::Softmax)
    {
        blocks = (batch_size + threads - 1) / threads;
        kernel_softmax_cross_entropy_loss<<<blocks, threads>>>(
            prediction.raw(), target.raw(), d_loss_buffer, batch_size, output_size);
    }
    else
    {
        cudaFree(d_loss_buffer);
        delete[] h_loss_buffer;
        return 0.0f;
    }

    cudaMemcpy(h_loss_buffer, d_loss_buffer, loss_buffer_size * sizeof(float), cudaMemcpyDeviceToHost);

    float total_loss = 0.0f;
    for (size_t i = 0; i < loss_buffer_size; ++i)
        total_loss += h_loss_buffer[i];

    float average_loss = total_loss / loss_buffer_size;

    cudaFree(d_loss_buffer);
    delete[] h_loss_buffer;

    return average_loss;
}