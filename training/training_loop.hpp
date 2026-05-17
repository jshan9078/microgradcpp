#pragma once
#include "../neural_network/NN.hpp"
#include <vector>

namespace micrograd {

// Executes gradient descent optimization to train the MLP model on a given dataset.
// Parameters:
//   - model: Reference to the Multi-Layer Perceptron model to optimize.
//   - inputs: Dataset input features (vector of input vectors).
//   - targets: Target label or scalar value for each input vector.
//   - epochs: Number of complete training epochs. (Sample: 100)
//   - learning_rate: Optimizer step size. (Sample: 0.05)
void train(MLP& model, 
           const std::vector<std::vector<Val>>& inputs, 
           const std::vector<Val>& targets, 
           int epochs = 100, 
           double learning_rate = 0.05);

}
