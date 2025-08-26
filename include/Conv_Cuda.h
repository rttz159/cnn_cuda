#include "utilities_kernel.h"
#include <vector>

#ifndef BETA1
#define BETA1 0.9
#endif

#ifndef BETA2
#define BETA2 0.999
#endif

#ifndef EPS
#define EPS 1e-7
#endif

class Conv_CUDA {
public:
    Conv_CUDA(int in_channels, int in_H, int in_W,
                int num_kernels, int filter_size,
                int stride_, int padding_,
                float learning_rate_, int batch_size_, int time_step_,
                float bias);

    ~Conv_CUDA();

    void run(const std::vector<std::vector<float>>& inputs);
    void bp(const std::vector<std::vector<float>>& error);
    
    std::vector<std::vector<float>>& get_input_gradients();

    std::vector<std::vector<float>> get_outputs();

    void resize_batch(int new_batch_size);

    int C, H, W;          
    int K, F;      
    int stride, padding;
    int H_out, W_out;
    int batch_size;
    int time_step;
    float learning_rate;

    float* d_input = nullptr;          
    float* d_output = nullptr;     
    float* d_filters;        
    float* d_bias;         

    float* d_input_col = nullptr;     
    float* d_input_col_T = nullptr;     
    float* d_filters_T;      

    float* d_delta;          
    float* d_grad_filters;  
    float* d_grad_bias;     

    float* d_m_filters;
    float* d_v_filters;
    float* d_m_bias;
    float* d_v_bias;

    float* d_act_deriv;

    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<float>> h_input_grad;
};
