template <int Rank>
class Layer
{
public:
    virtual Tensor<Rank> fw(const Tensor<Rank> &input) = 0;
    virtual Tensor<Rank> bp(const Tensor<Rank> &grad_output) = 0;
    virtual void update_weights(float learning_rate) = 0;
    virtual ~Layer() {}
};
