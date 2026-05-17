#include "training_loop.hpp"
#include <iostream>
#include <iomanip>

namespace micrograd {

// Executes gradient descent optimization to train the MLP model on a given dataset.
void train(MLP& model, 
           const std::vector<std::vector<Val>>& inputs, 
           const std::vector<Val>& targets, 
           int epochs, 
           double learning_rate) {
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::vector<Val> predictions;
        predictions.reserve(inputs.size());
        for (const auto& input : inputs) {
            predictions.push_back(model.forward(input)[0]);
        }

        // Compute the Mean Squared Error (MSE) loss.
        // Calculates the average of the squared differences between predictions and ground-truth targets,
        // rendering the total error metric scale-invariant to the dataset size.
        Val loss = std::make_shared<Value>(0.0);
        for (size_t i = 0; i < targets.size(); ++i) {
            loss = loss + (predictions[i] - targets[i])->pow(2);
        }
        loss = loss / static_cast<double>(targets.size());

        model.zero_grad();
        loss->backward_pass();

        for (auto& parameter : model.parameters()) {
            parameter->data -= learning_rate * parameter->grad;
        }

        if (epoch % 10 == 0 || epoch == 1) {
            std::cout << "  Epoch " << std::setw(3) << epoch << " | Loss: " << std::fixed << std::setprecision(6) 
                      << loss->data << std::endl;
        }
    }
}

}
