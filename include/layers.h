#pragma once
#include "tensor.h"
#include "cuda_tensor.cuh"

template <int Rank>
class Layer
{
public:
    virtual Tensor<Rank> fw(const Tensor<Rank> &input) = 0;
    virtual Tensor<Rank> bp(const Tensor<Rank> &grad_output) = 0;
    virtual CudaTensor<Rank> fw_cuda(const CudaTensor<Rank> &input) = 0;
    virtual CudaTensor<Rank> bp_cuda(const CudaTensor<Rank> &grad_output) = 0;
    virtual ~Layer() {}
};