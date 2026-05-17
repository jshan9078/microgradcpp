#include "NN.hpp"

namespace micrograd {

// ============================================================================
// Neural Network Module Base Class
// ============================================================================

// Resets accumulated gradients to 0.0 before a new optimization step.
void Module::zero_grad() {
    for (auto& p : parameters()) {
        p->grad = 0.0;
    }
}


// ============================================================================
// Random Number Generator Implementation
// ============================================================================

// Generates thread-safe, deterministic random double values in the range [min, max].
double get_random_uniform(double min, double max) {
    static thread_local mt19937 gen(42);
    uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}


// ============================================================================
// Neuron Class Implementation
// ============================================================================

// Constructor: Initializes a single neuron with a vector of weights and a bias.
// Parameters:
//   - number_of_inputs: The quantity of weight connections. (Sample input: 3)
//   - non_linear: Flag indicating whether to use ReLU. (Sample input: true)
Neuron::Neuron(int number_of_inputs, bool non_linear) : bias(make_shared<Value>(0.0)), non_linear(non_linear) {
    weights.reserve(number_of_inputs);
    for (int i = 0; i < number_of_inputs; ++i) {
        weights.push_back(make_shared<Value>(get_random_uniform(-1.0, 1.0)));
    }
}

// Forward pass: f(inputs) = Activation( dot(weights, inputs) + bias ).
// Parameters:
//   - inputs: Input vector of Value nodes. (Sample input: {val_x0, val_x1, val_x2})
Val Neuron::forward(const vector<Val>& inputs) const {
    Val activation = bias;
    for (size_t i = 0; i < weights.size(); ++i) {
        activation = activation + (weights[i] * inputs[i]);
    }
    return non_linear ? activation->relu() : activation;
}

// Retrieves neuron parameters (weights + bias) for gradient updates.
vector<Val> Neuron::parameters() const {
    vector<Val> params = weights;
    params.push_back(bias);
    return params;
}


// ============================================================================
// Layer Class Implementation
// ============================================================================

// Constructor: Groups multiple neurons together representing a fully connected layer.
// Parameters:
//   - number_of_inputs: Input dimension of incoming features. (Sample input: 3)
//   - number_of_outputs: Neuron count representing layer outputs. (Sample input: 4)
//   - non_linear: Flag indicating whether to apply non-linearities. (Sample input: true)
Layer::Layer(int number_of_inputs, int number_of_outputs, bool non_linear) {
    neurons.reserve(number_of_outputs);
    for (int i = 0; i < number_of_outputs; ++i) {
        neurons.emplace_back(number_of_inputs, non_linear);
    }
}

// Forward pass: Evaluates feedforward pass for each neuron independently.
// Parameters:
//   - inputs: Vector of input nodes. (Sample input: {val_x0, val_x1, val_x2})
vector<Val> Layer::forward(const vector<Val>& inputs) const {
    vector<Val> layer_outputs;
    layer_outputs.reserve(neurons.size());
    for (const auto& neuron : neurons) {
        layer_outputs.push_back(neuron.forward(inputs));
    }
    return layer_outputs;
}

// Flattens parameter lists across all neurons inside this layer.
vector<Val> Layer::parameters() const {
    vector<Val> params;
    for (const auto& n : neurons) {
        auto neuron_params = n.parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}


// ============================================================================
// Multi-Layer Perceptron (MLP) Implementation
// ============================================================================

// Constructor: Chain of Layer objects mapping input sizes to succeeding layers.
// Parameters:
//   - number_of_inputs: Network's primary input features count.
//   - number_of_outputs_list: Target sizes for hidden + output layers. (Sample input: {4, 4, 1})
MLP::MLP(int number_of_inputs, const vector<int>& number_of_outputs_list) {
    vector<int> layer_sizes = {number_of_inputs};
    layer_sizes.insert(layer_sizes.end(), number_of_outputs_list.begin(), number_of_outputs_list.end());
    layers.reserve(number_of_outputs_list.size());
    for (size_t i = 0; i < number_of_outputs_list.size(); ++i) {
        bool non_linear = (i != number_of_outputs_list.size() - 1);
        // Constructing Layer instantiates the individual Neuron objects inside memory.
        layers.emplace_back(layer_sizes[i], layer_sizes[i+1], non_linear);
    }
}

// Forward pass: Chained layers pass outputs to succeeding inputs.
// Parameters:
//   - inputs: Initial network input feature values. (Sample input: {val_x0, val_x1, val_x2})
vector<Val> MLP::forward(const vector<Val>& inputs) const {
    vector<Val> current = inputs;
    for (const auto& layer : layers) {
        current = layer.forward(current);
    }
    return current;
}

// Consolidates parameter lists across all layers for network backpropagation updates.
vector<Val> MLP::parameters() const {
    vector<Val> params;
    for (const auto& layer : layers) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

}
