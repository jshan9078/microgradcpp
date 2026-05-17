#include "Value.hpp"

namespace micrograd {

// ============================================================================
// Value Constructor Implementation
// ============================================================================

// Constructor: Initializes value data, operational parents, operation enum, and exponent parameters.
// Parameters:
//   - data: Raw floating-point scalar value of this computational node.
//   - prev: Pointers to parent nodes in the computational graph. (Sample input: {val_a, val_b})
//   - op: Operation type that produced this node. (Sample input: Op::ADD)
//   - pow_exp: Optional stored exponent used in power derivatives.
Value::Value(double data, vector<Val> prev, Op op, double pow_exp)
    : data(data), prev(std::move(prev)), op(op), pow_exp(pow_exp) {}


// ============================================================================
// Autograd Operations
// ============================================================================

// Rectified Linear Unit (ReLU) activation: f(x) = max(0, x).
Val Value::relu() {
    auto self = shared_from_this();
    return make_shared<Value>(
        data < 0.0 ? 0.0 : data,
        vector<Val>{self},
        Op::RELU
    );
}

// Power operator: f(x) = x^power.
// Parameters:
//   - power: The scalar exponent.
Val Value::pow(double power) {
    auto self = shared_from_this();
    return make_shared<Value>(
        std::pow(data, power),
        vector<Val>{self},
        Op::POW,
        power
    );
}

// Local Gradient Propagation Step.
// Centralizes all derivative math rules in a single switch block.
void Value::backward_step() {
    switch (op) {
        case Op::NONE:
            // Leaf node: No parents to propagate gradients to.
            break;

        case Op::ADD:
            // Addition: f(x, y) = x + y.
            // Derivative: df/dx = 1.0, df/dy = 1.0.
            prev[0]->grad += grad;
            prev[1]->grad += grad;
            break;

        case Op::SUB:
            // Subtraction: f(x, y) = x - y.
            // Derivative: df/dx = 1.0, df/dy = -1.0.
            prev[0]->grad += grad;
            prev[1]->grad -= grad;
            break;

        case Op::MUL:
            // Multiplication: f(x, y) = x * y.
            // Derivative (Product Rule): df/dx = y, df/dy = x.
            prev[0]->grad += prev[1]->data * grad;
            prev[1]->grad += prev[0]->data * grad;
            break;

        case Op::RELU:
            // Rectified Linear Unit (ReLU): f(x) = max(0, x).
            // Derivative: d(ReLU)/dx = 1.0 if f(x) > 0 else 0.0.
            prev[0]->grad += (data > 0.0 ? 1.0 : 0.0) * grad;
            break;

        case Op::POW:
            // Power operator: f(x) = x^p.
            // Derivative (Power Rule): df/dx = p * x^(p-1).
            prev[0]->grad += (pow_exp * std::pow(prev[0]->data, pow_exp - 1)) * grad;
            break;
    }
}

// Topological Sort and Reverse Propagation (Backpropagation).
// Dynamically builds the computational DAG in topological order using depth-first search.
void Value::backward_pass() {
    vector<Val> topo;
    unordered_set<Val> visited;

    // Helper lambda to recursively traverse parents and insert them into sorted list.
    function<void(const Val&)> build_topo = [&](const Val& v) {
        if (v && visited.find(v) == visited.end()) {
            visited.insert(v);
            for (const auto& child : v->prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    // Build topological order starting from current node (typically the loss node).
    build_topo(shared_from_this());

    // Base case: set gradient of the output node to 1.0.
    grad = 1.0;
    
    // Backpropagate gradients in reverse topological order (loss -> outputs -> hidden layers -> inputs).
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->backward_step();
    }
}


// ============================================================================
// Operator Overloads
// ============================================================================

// Addition: f(x, y) = x + y.
Val operator+(const Val& lhs, const Val& rhs) {
    return make_shared<Value>(lhs->data + rhs->data, vector<Val>{lhs, rhs}, Op::ADD);
}

// Addition: f(x, scalar) = x + scalar.
Val operator+(const Val& lhs, double rhs) { return lhs + make_shared<Value>(rhs); }

// Addition: f(scalar, y) = scalar + y.
Val operator+(double lhs, const Val& rhs) { return make_shared<Value>(lhs) + rhs; }

// Multiplication: f(x, y) = x * y.
Val operator*(const Val& lhs, const Val& rhs) {
    return make_shared<Value>(lhs->data * rhs->data, vector<Val>{lhs, rhs}, Op::MUL);
}

// Multiplication: f(x, scalar) = x * scalar.
Val operator*(const Val& lhs, double rhs) { return lhs * make_shared<Value>(rhs); }

// Multiplication: f(scalar, y) = scalar + y.
Val operator*(double lhs, const Val& rhs) { return make_shared<Value>(lhs) * rhs; }

// Subtraction: f(x, y) = x - y.
Val operator-(const Val& lhs, const Val& rhs) {
    return make_shared<Value>(lhs->data - rhs->data, vector<Val>{lhs, rhs}, Op::SUB);
}

// Subtraction: f(x, scalar) = x - scalar.
Val operator-(const Val& lhs, double rhs) { return lhs - make_shared<Value>(rhs); }

// Subtraction: f(scalar, y) = scalar - y.
Val operator-(double lhs, const Val& rhs) { return make_shared<Value>(lhs) - rhs; }

// Unary negation: f(x) = -x.
Val operator-(const Val& val) {
    return val * -1.0;
}

// Division: f(x, y) = x / y.
Val operator/(const Val& lhs, const Val& rhs) {
    return lhs * rhs->pow(-1.0);
}

// Division: f(x, scalar) = x / scalar.
Val operator/(const Val& lhs, double rhs) { return lhs / make_shared<Value>(rhs); }

// Division: f(scalar, y) = scalar / y.
Val operator/(double lhs, const Val& rhs) { return make_shared<Value>(lhs) / rhs; }

// Compound addition-assignment.
Val& operator+=(Val& lhs, const Val& rhs) { lhs = lhs + rhs; return lhs; }

// Compound addition-assignment.
Val& operator+=(Val& lhs, double rhs) { lhs = lhs + rhs; return lhs; }

// Compound subtraction-assignment.
Val& operator-=(Val& lhs, const Val& rhs) { lhs = lhs - rhs; return lhs; }

// Compound subtraction-assignment.
Val& operator-=(Val& lhs, double rhs) { lhs = lhs - rhs; return lhs; }

// Compound multiplication-assignment.
Val& operator*=(Val& lhs, const Val& rhs) { lhs = lhs * rhs; return lhs; }

// Compound multiplication-assignment.
Val& operator*=(Val& lhs, double rhs) { lhs = lhs * rhs; return lhs; }

// Compound division-assignment.
Val& operator/=(Val& lhs, const Val& rhs) { lhs = lhs / rhs; return lhs; }

// Compound division-assignment.
Val& operator/=(Val& lhs, double rhs) { lhs = lhs / rhs; return lhs; }

// Stream insertion operator: prints value data and gradient values.
ostream& operator<<(ostream& os, const Val& val) {
    os << "Value(data=" << val->data << ", grad=" << val->grad << ")";
    return os;
}

}
