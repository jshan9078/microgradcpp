// demo.cpp
// Purpose: Interactive executable demonstrating autograd computations and training an MLP model on binary classification data.

#include "micrograd.hpp"
#include "training/training_loop.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace micrograd;

void run_autograd_demo() {
    cout << "\n=============================================" << endl;
    cout << "  1. Scalar Autograd Computational Graph Demo" << endl;
    cout << "=============================================" << endl;

    Val a = make_shared<Value>(-4.0);
    Val b = make_shared<Value>(2.0);
    Val c = a + b;
    Val d = a * b + b->pow(3);
    c += c + 1.0;
    c += 1.0 + c + (-a);
    d += d * 2.0 + (b + a)->relu();
    d += 3.0 * d + (b - a)->relu();
    Val e = c - d;
    Val f = e->pow(2);
    Val g = f / 2.0;
    g += 10.0 / f;

    cout << "Expression: g = f/2 + 10/f" << endl;
    cout << "Forward Pass Output (g): " << g->data << endl;

    g->backward_pass();

    cout << "Backward Pass Gradients:" << endl;
    cout << "  dg/da: " << a->grad << endl;
    cout << "  dg/db: " << b->grad << endl;
}

void run_mlp_demo() {
    cout << "\n=============================================" << endl;
    cout << "  2. Multi-Layer Perceptron Training Demo" << endl;
    cout << "=============================================" << endl;

    MLP model(3, {4, 4, 1});

    vector<vector<Val>> xs = {
        {make_shared<Value>(2.0), make_shared<Value>(3.0), make_shared<Value>(-1.0)},
        {make_shared<Value>(3.0), make_shared<Value>(-1.0), make_shared<Value>(0.5)},
        {make_shared<Value>(0.5), make_shared<Value>(1.0), make_shared<Value>(1.0)},
        {make_shared<Value>(1.0), make_shared<Value>(1.0), make_shared<Value>(-1.0)}
    };
    vector<Val> ys = {
        make_shared<Value>(1.0),
        make_shared<Value>(-1.0),
        make_shared<Value>(-1.0),
        make_shared<Value>(1.0)
    };

    cout << "Dataset:" << endl;
    for (size_t i = 0; i < xs.size(); ++i) {
        cout << "  x" << i << ": [" << xs[i][0]->data << ", " << xs[i][1]->data << ", " << xs[i][2]->data 
                  << "] -> target y" << i << ": " << ys[i]->data << endl;
    }

    cout << "\nTraining the MLP (100 epochs)..." << endl;
    train(model, xs, ys, 100, 0.05);

    cout << "\nPredictions after training:" << endl;
    for (size_t i = 0; i < xs.size(); ++i) {
        Val pred = model.forward(xs[i])[0];
        cout << "  Input: [" << xs[i][0]->data << ", " << xs[i][1]->data << ", " << xs[i][2]->data 
             << "] | Prediction Value: " << showpos << fixed << setprecision(4) << pred->data 
             << " | Target Value: " << noshowpos << ys[i]->data << endl;
    }
}

int main() {
    cout << "\033[36;1m=============================================" << endl;
    cout << "     MICROGRAD C++ IMPLEMENTATION SHOWCASE" << endl;
    cout << "=============================================\033[0m" << endl;
    
    run_autograd_demo();
    run_mlp_demo();
    
    cout << "\033[36;1m=============================================\033[0m\n" << endl;
    return 0;
}
