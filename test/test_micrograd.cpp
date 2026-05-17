// test_micrograd.cpp
// Purpose: Minimalist unit test suite validating autograd engine math and neural network structures.

#include "../micrograd.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace micrograd;

void test_sanity_check() {
    Val x = make_shared<Value>(-4.0);
    Val z = 2.0 * x + 2.0 + x;
    Val q = z->relu() + z * x;
    Val h = (z * z)->relu();
    Val y = h + q + q * x;
    y->backward_pass();

    assert(abs(y->data - (-20.0)) < 1e-9);
    assert(abs(x->grad - 46.0) < 1e-9);
    cout << "\033[32m[PASS] Sanity Check passed!\033[0m" << endl;
}

void test_more_ops() {
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
    g->backward_pass();

    assert(abs(g->data - 24.704081632653) < 1e-6);
    assert(abs(a->grad - 138.833819241983) < 1e-6);
    assert(abs(b->grad - 645.577259475219) < 1e-6);
    cout << "\033[32m[PASS] Extended operators test passed!\033[0m" << endl;
}

void test_nn() {
    MLP mlp(3, {4, 4, 1});
    vector<Val> x = {make_shared<Value>(2.0), make_shared<Value>(3.0), make_shared<Value>(-1.0)};
    vector<Val> out = mlp.forward(x);
    assert(out.size() == 1);
    
    auto params = mlp.parameters();
    assert(params.size() == (3 * 4 + 4) + (4 * 4 + 4) + (4 * 1 + 1)); 

    mlp.zero_grad();
    for (const auto& p : params) {
        assert(p->grad == 0.0);
    }
    cout << "\033[32m[PASS] Neural Network architecture test passed!\033[0m" << endl;
}

int main() {
    cout << "\033[34mRunning micrograd C++ test suite...\033[0m" << endl;
    try {
        test_sanity_check();
        test_more_ops();
        test_nn();
        cout << "\033[32;1mAll tests passed successfully!\033[0m" << endl;
    } catch (const exception& e) {
        cerr << "\033[31;1mTest suite failed with exception: " << e.what() << "\033[0m" << endl;
        return 1;
    }
    return 0;
}
