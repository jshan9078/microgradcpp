#pragma once
#include "../autograd/Value.hpp"
#include <vector>
#include <random>

using namespace std;

namespace micrograd {

class Module {
public:
    virtual ~Module() = default;
    virtual vector<Val> parameters() const = 0;
    void zero_grad();
};

double get_random_uniform(double min, double max);

class Neuron : public Module {
private:
    // Vector of weights (synaptic connections) associated with each input dimension.
    vector<Val> weights;

    // Bias value added to the weighted input sum to adjust the activation threshold.
    Val bias;

    // Flag indicating whether to apply a non-linear activation function (ReLU) to the sum.
    bool non_linear;

public:
    Neuron(int number_of_inputs, bool non_linear = true);
    Val forward(const vector<Val>& inputs) const;
    vector<Val> parameters() const override;
    const vector<Val>& get_weights() const { return weights; }
    const Val& get_bias() const { return bias; }
    bool is_non_linear() const { return non_linear; }
};

class Layer : public Module {
private:
    // Vector of individual neurons representing the units of this fully connected layer.
    vector<Neuron> neurons;

public:
    Layer(int number_of_inputs, int number_of_outputs, bool non_linear = true);
    vector<Val> forward(const vector<Val>& inputs) const;
    vector<Val> parameters() const override;
    const vector<Neuron>& get_neurons() const { return neurons; }
};

class MLP : public Module {
private:
    // Vector of neural network layers that form the multi-layer perceptron.
    vector<Layer> layers;

public:
    MLP(int number_of_inputs, const vector<int>& number_of_outputs_list);
    vector<Val> forward(const vector<Val>& inputs) const;
    vector<Val> parameters() const override;
    const vector<Layer>& get_layers() const { return layers; }
};

}
