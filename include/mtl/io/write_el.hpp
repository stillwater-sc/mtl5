#pragma once
// MTL5 — Write dense/sparse matrices to CSV/whitespace-delimited files
#include <fstream>
#include <string>
#include <stdexcept>
#include <mtl/mat/compressed2D.hpp>

namespace mtl::io {

/// Write a dense matrix to a CSV or whitespace-delimited file.
/// One row per line, values separated by delimiter.
template <typename Matrix>
void write_dense(const std::string& filename, const Matrix& A, char delimiter = ',') {
    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("write_dense: cannot open file: " + filename);

    out.precision(17);
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        for (std::size_t j = 0; j < A.num_cols(); ++j) {
            if (j > 0) out << delimiter;
            out << A(i, j);
        }
        out << '\n';
    }
}

/// Write a sparse matrix to a triplet file (0-based indices).
/// Format: row col value, one triplet per line.
template <typename Value, typename Parameters>
void write_sparse(const std::string& filename,
                  const mat::compressed2D<Value, Parameters>& A) {
    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("write_sparse: cannot open file: " + filename);

    const auto& starts  = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data    = A.ref_data();

    out.precision(17);
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            out << i << ' ' << indices[k] << ' ' << data[k] << '\n';
}

} // namespace mtl::io
