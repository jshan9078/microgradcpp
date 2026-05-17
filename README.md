# micrograd C++ (Split-File Modular Edition)

A self-contained, lightweight scalar autograd engine and neural network library in C++17, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). Designed purely for educational clarity, splitting the codebase into modular subdirectories representing different abstraction layers, while preserving a facade convenience header `micrograd.hpp` at the root.

This edition features fully unabbreviated variable and parameter names, using `weights`, `bias`, `number_of_inputs`, `number_of_outputs`, `number_of_outputs_list`, `inputs`, `activation`, `non_linear`, and explicit `.forward()` calls instead of standard shorthand symbols or C++ function operators (`operator()`) for ultimate readability.

## Repository Structure

```
microgradcpp/
├── autograd/
│   ├── Value.hpp       (Value node and overloaded math operators declarations)
│   └── Value.cpp       (Value backpropagation math and operator implementations)
├── neural_network/
│   ├── NN.hpp          (Neuron, Layer, and MLP declarations)
│   └── NN.cpp          (Weights, bias parameters, and forward neural layouts)
├── training/
│   ├── training_loop.hpp (MLP gradient descent optimization trainer declaration)
│   └── training_loop.cpp (Training optimization implementation with loss epoch logging)
├── test/
│   └── test_micrograd.cpp (Engine unit tests checking gradients against PyTorch outputs)
├── micrograd.hpp       (Convenience facade umbrella header for root inclusions)
├── demo.cpp            (A simple Multi-Layer Perceptron training demo)
├── Makefile            (Direct compiler rules using clang++)
└── README.md           (This guide)
```

---

## Detailed Developer Guide & Usage

This library is split into three main modules: the **Autograd engine**, the **Neural Network layers**, and the **Training loops**. Below are complete instructions and examples for using each of them.

### 1. Autograd Math Engine (`/autograd`)
The scalar autograd computational DAG is centered around the `Value` node (aliased as `Val` using `std::shared_ptr<Value>`). Nodes are linked dynamically through mathematical operators (+, -, *, /) and activation functions.

To compute local derivatives, build your algebraic expression and call `.backward_pass()` on the final output node:

```cpp
#include "micrograd.hpp"
#include <iostream>

using namespace micrograd;

int main() {
    // 1. Instantiate leaf inputs
    Val x = make_shared<Value>(-4.0);
    Val w = make_shared<Value>(2.0);
    
    // 2. Perform forward math operations
    Val y = x * w + 5.0;     // y = -4 * 2 + 5 = -3.0
    Val loss = y->relu();    // loss = max(0, -3) = 0.0
    
    // 3. Backpropagate derivatives
    loss->backward_pass();
    
    // 4. Access gradients
    std::cout << "d(loss)/dx: " << x->grad << std::endl;
    std::cout << "d(loss)/dw: " << w->grad << std::endl;
    return 0;
}
```

---

### 2. Neural Network Construction (`/neural_network`)
Define fully connected multi-layer perceptron (MLP) layouts. An MLP consists of nested `Layer` structures, which in turn contain `Neuron` nodes.

- **Neuron**: Accumulates inputs: $f(x) = \text{Activation}(\sum w_i x_i + bias)$.
- **Layer**: Represents a standard fully connected layer grouping multiple independent Neurons.
- **MLP**: A chained stack of layers processing inputs feedforward.

```cpp
#include "micrograd.hpp"
#include <vector>

using namespace micrograd;

int main() {
    // Instantiate MLP mapping: 3 inputs -> hidden layer 4 -> hidden layer 4 -> 1 output
    MLP model(3, {4, 4, 1});
    
    // Forward pass
    std::vector<Val> input_sample = {make_shared<Value>(2.0), make_shared<Value>(1.5), make_shared<Value>(-1.0)};
    std::vector<Val> prediction = model.forward(input_sample);
    
    std::cout << "Output: " << prediction[0]->data << std::endl;
    return 0;
}
```

---

### 3. Modular Gradient Descent Training (`/training`)
Utilize the centralized `train(...)` API to train an MLP model using standard Stochastic Gradient Descent (SGD) with step learning rate rules:

```cpp
#include "micrograd.hpp"
#include "training/training_loop.hpp"
#include <vector>

using namespace micrograd;

int main() {
    MLP model(3, {4, 4, 1});
    
    // Prepare binary classification dataset (4 samples)
    std::vector<std::vector<Val>> inputs = {
        {make_shared<Value>(2.0), make_shared<Value>(3.0), make_shared<Value>(-1.0)},
        {make_shared<Value>(3.0), make_shared<Value>(-1.0), make_shared<Value>(0.5)},
        {make_shared<Value>(0.5), make_shared<Value>(1.0), make_shared<Value>(1.0)},
        {make_shared<Value>(1.0), make_shared<Value>(1.0), make_shared<Value>(-1.0)}
    };
    std::vector<Val> targets = {
        make_shared<Value>(1.0),
        make_shared<Value>(-1.0),
        make_shared<Value>(-1.0),
        make_shared<Value>(1.0)
    };
    
    // Optimize model parameters (weights + bias) over 100 epochs, learning rate = 0.05
    train(model, inputs, targets, 100, 0.05);
    return 0;
}
```

---

## How to Build and Run

### Using the Makefile (Recommended)
Easily compile and execute targets using the simple `Makefile`:

```bash
# Compile and run the complete autograd and MLP unit test suite
make test

# Compile and execute the MLP network training loop converges showcase
make run-demo

# Build both binaries without running them
make build

# Remove compiled binary executables
make clean
```

### Manual Compilation
Alternatively, compile using standard `clang++` (or `g++`) compiler directives:

```bash
# Compile and execute the test suite
clang++ -O3 -std=c++17 test/test_micrograd.cpp autograd/Value.cpp neural_network/NN.cpp -I. -o test_micrograd && ./test_micrograd

# Compile and execute the MLP demo showcase
clang++ -O3 -std=c++17 demo.cpp autograd/Value.cpp neural_network/NN.cpp training/training_loop.cpp -I. -o demo && ./demo
```

## Exact Mathematical Correctness

### Sanity Assertions
Unit tests evaluate forward and backward passes against exact analytical values and PyTorch:
- `y->data` = `-20.0`
- `x->grad` = `46.0`
- `g->data` = `24.704081632653`
- `a->grad` = `138.833819241983`
- `b->grad` = `645.577259475219`

### MLP Optimization Loss
Optimizes an `MLP(3, {4, 4, 1})` binary classifier down to `< 0.29` loss:
```
Training the MLP (100 epochs)...
  Epoch   1 | Loss: 4.391651
  Epoch  50 | Loss: 0.542668
  Epoch 100 | Loss: 0.289267

Predictions:
  Input: [2, 3, -1]    | Predicted: +0.4951 (Target: +1.0)
  Input: [3, -1, 0.5]  | Predicted: -0.8467 (Target: -1.0)
  Input: [0.5, 1, 1]   | Predicted: -1.0810 (Target: -1.0)
  Input: [1, 1, -1]    | Predicted: +0.5782 (Target: +1.0)
```
