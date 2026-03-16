// activation_functions.cpp -- Neural Network Activation Functions
//
// This example demonstrates building common neural network activation
// functions from MTL5's transcendental building blocks:
//
//   1. Sigmoid:  sigma(x) = 1 / (1 + exp(-x))
//   2. Tanh:     tanh(x)  -- already an MTL5 primitive
//   3. ReLU:     max(0, x) via signum
//   4. Softmax:  exp(x_i) / Sigma exp(x_j)
//   5. GELU:     x * 0.5 * (1 + erf(x / sqrt(2)))
//
// Each activation is built entirely from element-wise operations on
// vectors and matrices -- the same building blocks used in real
// neural network frameworks.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <numbers>

using namespace mtl;

// -- Sigmoid: sigma(x) = 1 / (1 + exp(-x)) ----------------------------------
template <Vector V>
auto sigmoid(const V& x) {
    using T = typename V::value_type;
    auto neg_x = -T{1} * x;
    auto exp_neg = mtl::exp(neg_x);
    vec::dense_vector<T> result(x.size());
    for (typename V::size_type i = 0; i < x.size(); ++i) {
        result(i) = T{1} / (T{1} + exp_neg(i));
    }
    return result;
}

// -- ReLU: max(0, x) = x * (signum(x) + 1) / 2 -------------------------
// Using signum: when x>0 -> (1+1)/2=1 -> keep; x<0 -> (-1+1)/2=0 -> zero
template <Vector V>
auto relu(const V& x) {
    using T = typename V::value_type;
    auto sgn = signum(x);
    vec::dense_vector<T> result(x.size());
    for (typename V::size_type i = 0; i < x.size(); ++i) {
        result(i) = x(i) * (sgn(i) + T{1}) / T{2};
    }
    return result;
}

// -- Softmax: exp(x_i) / Sigma exp(x_j) -------------------------------------
template <Vector V>
auto softmax(const V& x) {
    using T = typename V::value_type;
    // Numerically stable: subtract max before exp
    T max_val = x(0);
    for (typename V::size_type i = 1; i < x.size(); ++i) {
        if (x(i) > max_val) max_val = x(i);
    }
    vec::dense_vector<T> shifted(x.size());
    for (typename V::size_type i = 0; i < x.size(); ++i) {
        shifted(i) = x(i) - max_val;
    }
    auto e = mtl::exp(shifted);
    T total{0};
    for (typename V::size_type i = 0; i < e.size(); ++i) {
        total += e(i);
    }
    vec::dense_vector<T> result(x.size());
    for (typename V::size_type i = 0; i < x.size(); ++i) {
        result(i) = e(i) / total;
    }
    return result;
}

// -- GELU: x * 0.5 * (1 + erf(x / sqrt(2))) -----------------------------
template <Vector V>
auto gelu(const V& x) {
    using T = typename V::value_type;
    const T inv_sqrt2 = T{1} / std::sqrt(T{2});
    auto scaled = inv_sqrt2 * x;
    auto erf_vals = mtl::erf(scaled);
    vec::dense_vector<T> result(x.size());
    for (typename V::size_type i = 0; i < x.size(); ++i) {
        result(i) = x(i) * T{0.5} * (T{1} + erf_vals(i));
    }
    return result;
}

void print_vec(const std::string& label, const auto& v) {
    std::cout << label << ": [";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << v(i);
    }
    std::cout << "]\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Neural Network Activation Functions with MTL5 ===\n\n";

    // Test input spanning negative, zero, and positive values
    dense_vector<double> x = {-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0};
    print_vec("Input x", x);
    std::cout << "\n";

    // -- 1. Sigmoid ------------------------------------------------------
    auto sig = sigmoid(x);
    print_vec("Sigmoid", sig);

    // Verify sigmoid properties
    std::cout << "  Properties:\n";
    std::cout << "    sigma(0) = " << sig(4) << " (should be 0.5)\n";
    std::cout << "    sigma(-3) + sigma(3) = " << sig(0) + sig(8) << " (should be 1.0)\n";
    std::cout << "    Range: [" << sig(0) << ", " << sig(8) << "] subset (0,1)\n\n";

    // -- 2. Tanh ---------------------------------------------------------
    auto th = mtl::tanh(x);
    print_vec("Tanh", th);
    std::cout << "  Properties:\n";
    std::cout << "    tanh(0) = " << th(4) << " (should be 0.0)\n";
    std::cout << "    Symmetric: tanh(-1) + tanh(1) = " << th(2) + th(6) << " (should be 0.0)\n\n";

    // -- 3. ReLU ---------------------------------------------------------
    auto re = relu(x);
    print_vec("ReLU", re);
    std::cout << "  Properties:\n";
    std::cout << "    ReLU(-3) = " << re(0) << " (should be 0.0)\n";
    std::cout << "    ReLU(0) = " << re(4) << " (should be 0.0)\n";
    std::cout << "    ReLU(3) = " << re(8) << " (should be 3.0)\n\n";

    // -- 4. Softmax ------------------------------------------------------
    dense_vector<double> logits = {2.0, 1.0, 0.1};
    auto probs = softmax(logits);
    print_vec("Logits", logits);
    print_vec("Softmax", probs);
    double prob_sum = 0.0;
    for (std::size_t i = 0; i < probs.size(); ++i) prob_sum += probs(i);
    std::cout << "  Sum of probabilities: " << prob_sum << " (should be 1.0)\n\n";

    // -- 5. GELU ---------------------------------------------------------
    auto ge = gelu(x);
    print_vec("GELU", ge);
    std::cout << "  Properties:\n";
    std::cout << "    GELU(0) = " << ge(4) << " (should be 0.0)\n";
    std::cout << "    GELU approaches ReLU for large positive x:\n";
    std::cout << "      GELU(3) = " << ge(8) << ", ReLU(3) = " << re(8) << "\n";
    std::cout << "    GELU is slightly negative for small negative x:\n";
    std::cout << "      GELU(-0.5) = " << ge(3) << "\n\n";

    // -- 6. Matrix activation: sigmoid on a weight matrix ----------------
    std::cout << "-- Matrix Activation --\n";
    mat::dense2D<double> W(2, 3);
    W(0,0) = -1.0; W(0,1) = 0.0; W(0,2) = 1.0;
    W(1,0) = -2.0; W(1,1) = 0.5; W(1,2) = 2.0;

    auto W_tanh = mtl::tanh(W);
    std::cout << "W (2x3):\n";
    for (std::size_t r = 0; r < 2; ++r) {
        std::cout << "  [";
        for (std::size_t c = 0; c < 3; ++c) {
            if (c > 0) std::cout << ", ";
            std::cout << W(r, c);
        }
        std::cout << "]\n";
    }
    std::cout << "tanh(W):\n";
    for (std::size_t r = 0; r < 2; ++r) {
        std::cout << "  [";
        for (std::size_t c = 0; c < 3; ++c) {
            if (c > 0) std::cout << ", ";
            std::cout << W_tanh(r, c);
        }
        std::cout << "]\n";
    }

    std::cout << "\nAll activation function demonstrations completed successfully.\n";
    return 0;
}
