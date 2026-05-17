#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <memory>
#include <functional>

using namespace std;

namespace micrograd {

enum class Op { NONE, ADD, SUB, MUL, RELU, POW };

struct Value;
using Val = shared_ptr<Value>;

struct Value : public enable_shared_from_this<Value> {
    // ========================================================================
    // Member Variables
    // ========================================================================

    // The raw scalar value (floating-point number) representing the output of this computational node.
    double data;

    // The accumulated derivative (gradient) of the final loss with respect to this value.
    double grad = 0.0;

    // Pointers to the parent nodes in the computational graph that produced this node.
    vector<Val> prev;

    // The type of operation that produced this node.
    Op op = Op::NONE;

    // Stored scalar exponent used to calculate power rule derivative.
    double pow_exp = 0.0;

    // Constructor
    Value(double data, vector<Val> prev = {}, Op op = Op::NONE, double pow_exp = 0.0);

    Val relu();
    Val pow(double power);
    void backward_step();
    void backward_pass();
};

// Operator declarations
Val operator+(const Val& lhs, const Val& rhs);
Val operator+(const Val& lhs, double rhs);
Val operator+(double lhs, const Val& rhs);

Val operator*(const Val& lhs, const Val& rhs);
Val operator*(const Val& lhs, double rhs);
Val operator*(double lhs, const Val& rhs);

Val operator-(const Val& lhs, const Val& rhs);
Val operator-(const Val& lhs, double rhs);
Val operator-(double lhs, const Val& rhs);
Val operator-(const Val& val);

Val operator/(const Val& lhs, const Val& rhs);
Val operator/(const Val& lhs, double rhs);
Val operator/(double lhs, const Val& rhs);

Val& operator+=(Val& lhs, const Val& rhs);
Val& operator+=(Val& lhs, double rhs);
Val& operator-=(Val& lhs, const Val& rhs);
Val& operator-=(Val& lhs, double rhs);
Val& operator*=(Val& lhs, const Val& rhs);
Val& operator*=(Val& lhs, double rhs);
Val& operator/=(Val& lhs, const Val& rhs);
Val& operator/=(Val& lhs, double rhs);

ostream& operator<<(ostream& os, const Val& val);

}
