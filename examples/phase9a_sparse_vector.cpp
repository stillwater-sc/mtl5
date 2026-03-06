// phase9a_sparse_vector.cpp - Sparse Vector Operations
//
// This example demonstrates:
//   1. Construction and population of sparse vectors
//   2. The inserter pattern for batch construction
//   3. Sparse dot product with a dense vector
//   4. The crop() operation for thresholding small entries
//   5. Memory efficiency: sparse vs dense for high-dimensional sparse data
//
// Key insight: Sparse vectors store only non-zero entries as sorted (index, value)
// pairs. For vectors with few non-zeros relative to their dimension, this provides
// both memory savings and computational speedups since operations skip zeros.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mtl;

/// Manual sparse-dense dot product: sum over stored entries only
double sparse_dense_dot(const vec::sparse_vector<double>& sv,
                        const vec::dense_vector<double>& dv) {
    double result = 0.0;
    for (auto [idx, val] : sv) {
        result += val * dv(idx);
    }
    return result;
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 9A: Sparse Vector Operations\n";
    std::cout << "=============================================================\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // 1. Basic Construction and Element Access
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 1. Basic Sparse Vector ===\n\n";

    vec::sparse_vector<double> sv(10);
    std::cout << "Created sparse vector of dimension " << sv.size()
              << " with " << sv.nnz() << " stored entries.\n\n";

    // Insert elements (in any order - sorted internally)
    sv.insert(7, 3.14);
    sv.insert(2, 1.41);
    sv.insert(5, 2.72);

    std::cout << "After inserting 3 elements:\n";
    std::cout << "  nnz = " << sv.nnz() << "\n";
    std::cout << "  Stored entries (sorted by index):\n";
    for (auto [idx, val] : sv) {
        std::cout << "    [" << idx << "] = " << val << "\n";
    }

    // Read access: present vs absent
    std::cout << "\n  sv(2) = " << sv(2) << "  (present)\n";
    std::cout << "  sv(3) = " << sv(3) << "  (absent - returns 0)\n";
    std::cout << "  sv(7) = " << sv(7) << "  (present)\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // 2. The Inserter Pattern
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 2. Inserter Pattern ===\n\n";

    std::cout << "The inserter pattern provides a convenient interface for\n";
    std::cout << "batch construction, mirroring the matrix inserter pattern.\n\n";

    vec::sparse_vector<double> counts(8);
    {
        // Using update_plus to accumulate values at same index
        vec::vec_inserter<vec::sparse_vector<double>,
                          mat::update_plus<double>> ins(counts);
        // Simulate word-count accumulation
        ins[1] << 1.0;  // word 1 seen once
        ins[3] << 1.0;  // word 3 seen once
        ins[1] << 1.0;  // word 1 seen again → accumulate to 2.0
        ins[5] << 1.0;  // word 5 seen
        ins[3] << 1.0;  // word 3 again → 2.0
        ins[3] << 1.0;  // word 3 again → 3.0
    }

    std::cout << "Word frequency vector (dimension=" << counts.size()
              << ", nnz=" << counts.nnz() << "):\n";
    for (auto [idx, val] : counts) {
        std::cout << "  word[" << idx << "] appeared " << val << " time(s)\n";
    }
    std::cout << "\n";

    // ══════════════════════════════════════════════════════════════════════
    // 3. Sparse-Dense Dot Product
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 3. Sparse-Dense Dot Product ===\n\n";

    const std::size_t N = 1000;
    vec::sparse_vector<double> sparse_v(N);
    vec::dense_vector<double> dense_v(N, 1.0);

    // Only 5 non-zero entries in a 1000-dim vector
    sparse_v.insert(10, 1.0);
    sparse_v.insert(100, 2.0);
    sparse_v.insert(500, 3.0);
    sparse_v.insert(750, 4.0);
    sparse_v.insert(999, 5.0);

    double dot_result = sparse_dense_dot(sparse_v, dense_v);
    std::cout << "Sparse vector: " << sparse_v.nnz() << " non-zeros in "
              << N << " dimensions\n";
    std::cout << "Dense vector:  all ones\n";
    std::cout << "Dot product:   " << dot_result
              << "  (= 1+2+3+4+5 = 15)\n";
    std::cout << "Operations:    " << sparse_v.nnz()
              << " multiply-adds (not " << N << ")\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // 4. Crop / Threshold
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 4. Threshold Cropping ===\n\n";

    vec::sparse_vector<double> noisy(20);
    noisy.insert(0, 5.0);
    noisy.insert(3, 0.001);
    noisy.insert(7, -3.2);
    noisy.insert(10, 0.0005);
    noisy.insert(15, 8.1);
    noisy.insert(18, -0.0001);

    std::cout << "Before crop (nnz=" << noisy.nnz() << "):\n";
    for (auto [idx, val] : noisy) {
        std::cout << "  [" << std::setw(2) << idx << "] = "
                  << std::setw(10) << val << "\n";
    }

    noisy.crop(0.01);
    std::cout << "\nAfter crop(threshold=0.01) - nnz=" << noisy.nnz() << ":\n";
    for (auto [idx, val] : noisy) {
        std::cout << "  [" << std::setw(2) << idx << "] = "
                  << std::setw(10) << val << "\n";
    }
    std::cout << "\n";

    // ══════════════════════════════════════════════════════════════════════
    // 5. Memory Comparison
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 5. Memory Efficiency ===\n\n";

    const std::size_t dim = 100000;
    const std::size_t nnz = 50;

    std::size_t dense_bytes = dim * sizeof(double);
    std::size_t sparse_bytes = nnz * (sizeof(std::size_t) + sizeof(double));

    std::cout << "Dimension: " << dim << ", Non-zeros: " << nnz << "\n";
    std::cout << "Dense vector:  " << dense_bytes << " bytes ("
              << dense_bytes / 1024 << " KB)\n";
    std::cout << "Sparse vector: " << sparse_bytes << " bytes ("
              << sparse_bytes << " B)\n";
    std::cout << "Ratio: " << std::fixed << std::setprecision(1)
              << static_cast<double>(dense_bytes) / static_cast<double>(sparse_bytes)
              << "x memory savings\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Key Takeaways
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. Sparse vectors store only non-zero entries as sorted\n";
    std::cout << "   (index, value) pairs with O(log n) lookup.\n";
    std::cout << "2. The inserter pattern (with update_plus) supports\n";
    std::cout << "   accumulation, useful for histogram/frequency tasks.\n";
    std::cout << "3. Sparse-dense operations only touch stored entries,\n";
    std::cout << "   giving O(nnz) instead of O(n) complexity.\n";
    std::cout << "4. crop() removes near-zero entries to maintain sparsity\n";
    std::cout << "   after numerical operations introduce fill-in.\n";

    return EXIT_SUCCESS;
}
