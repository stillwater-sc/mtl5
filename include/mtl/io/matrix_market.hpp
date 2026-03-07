#pragma once
// MTL5 -- Matrix Market (.mtx) file format reader/writer
// Supports: real coordinate general, real coordinate symmetric,
//           real array general (dense).
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/coordinate2D.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::io {

/// Read a Matrix Market file into a compressed2D (for coordinate format)
/// or dense2D (for array format). Returns compressed2D for sparse matrices.
template <typename Value = double, typename Parameters = mat::parameters<>>
mat::compressed2D<Value, Parameters> mm_read(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open())
        throw std::runtime_error("mm_read: cannot open file: " + filename);

    std::string line;
    // Read banner line
    std::getline(in, line);

    // Parse banner: %%MatrixMarket matrix coordinate|array real|... general|symmetric|...
    bool is_coordinate = true;
    bool is_symmetric = false;

    // Lowercase the banner for case-insensitive matching
    std::string banner = line;
    std::transform(banner.begin(), banner.end(), banner.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (banner.find("array") != std::string::npos)
        is_coordinate = false;
    if (banner.find("symmetric") != std::string::npos)
        is_symmetric = true;

    // Skip comment lines
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
    }

    // Parse dimensions
    using size_type = typename Parameters::size_type;
    size_type nrows, ncols, nnz_entries;
    std::istringstream dim_stream(line);

    if (is_coordinate) {
        dim_stream >> nrows >> ncols >> nnz_entries;
    } else {
        dim_stream >> nrows >> ncols;
        nnz_entries = nrows * ncols;
    }

    if (is_coordinate) {
        mat::coordinate2D<Value, Parameters> coo(nrows, ncols);
        coo.reserve(is_symmetric ? 2 * nnz_entries : nnz_entries);

        for (size_type k = 0; k < nnz_entries; ++k) {
            size_type r, c;
            Value v;
            in >> r >> c >> v;
            r--; c--;  // 1-based to 0-based
            coo.insert(r, c, v);
            if (is_symmetric && r != c)
                coo.insert(c, r, v);
        }
        return coo.compress();
    } else {
        // Array (dense column-major) format
        mat::coordinate2D<Value, Parameters> coo(nrows, ncols);
        coo.reserve(nrows * ncols);

        for (size_type j = 0; j < ncols; ++j)
            for (size_type i = 0; i < nrows; ++i) {
                Value v;
                in >> v;
                if (v != math::zero<Value>())
                    coo.insert(i, j, v);
            }
        return coo.compress();
    }
}

/// Read a Matrix Market file into a dense2D.
template <typename Value = double>
mat::dense2D<Value> mm_read_dense(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open())
        throw std::runtime_error("mm_read_dense: cannot open file: " + filename);

    std::string line;
    std::getline(in, line);

    std::string banner = line;
    std::transform(banner.begin(), banner.end(), banner.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    bool is_coordinate = (banner.find("array") == std::string::npos);
    bool is_symmetric = (banner.find("symmetric") != std::string::npos);

    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
    }

    std::size_t nrows, ncols, nnz_entries = 0;
    std::istringstream dim_stream(line);

    if (is_coordinate) {
        dim_stream >> nrows >> ncols >> nnz_entries;
    } else {
        dim_stream >> nrows >> ncols;
    }

    mat::dense2D<Value> A(nrows, ncols);
    for (std::size_t i = 0; i < nrows; ++i)
        for (std::size_t j = 0; j < ncols; ++j)
            A(i, j) = math::zero<Value>();

    if (is_coordinate) {
        for (std::size_t k = 0; k < nnz_entries; ++k) {
            std::size_t r, c;
            Value v;
            in >> r >> c >> v;
            r--; c--;
            A(r, c) = v;
            if (is_symmetric && r != c)
                A(c, r) = v;
        }
    } else {
        // Column-major dense
        for (std::size_t j = 0; j < ncols; ++j)
            for (std::size_t i = 0; i < nrows; ++i)
                in >> A(i, j);
    }
    return A;
}

/// Write a dense matrix in Matrix Market array format.
template <typename Matrix>
void mm_write(const std::string& filename, const Matrix& A,
              const std::string& comment = "") {
    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("mm_write: cannot open file: " + filename);

    out << "%%MatrixMarket matrix array real general\n";
    if (!comment.empty())
        out << "% " << comment << "\n";
    out << A.num_rows() << " " << A.num_cols() << "\n";

    // Column-major output
    out.precision(17);
    for (std::size_t j = 0; j < A.num_cols(); ++j)
        for (std::size_t i = 0; i < A.num_rows(); ++i)
            out << A(i, j) << "\n";
}

/// Write a sparse matrix in Matrix Market coordinate format.
template <typename Value, typename Parameters>
void mm_write_sparse(const std::string& filename,
                     const mat::compressed2D<Value, Parameters>& A,
                     const std::string& comment = "") {
    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("mm_write_sparse: cannot open file: " + filename);

    out << "%%MatrixMarket matrix coordinate real general\n";
    if (!comment.empty())
        out << "% " << comment << "\n";
    out << A.num_rows() << " " << A.num_cols() << " " << A.nnz() << "\n";

    const auto& starts = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data = A.ref_data();

    out.precision(17);
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            out << (i + 1) << " " << (indices[k] + 1) << " " << data[k] << "\n";
}

} // namespace mtl::io
