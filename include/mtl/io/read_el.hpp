#pragma once
// MTL5 -- Read dense/sparse matrices from CSV/whitespace-delimited files
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/coordinate2D.hpp>

namespace mtl::io {

/// Read a dense matrix from a CSV or whitespace-delimited file.
/// One row per line, values separated by delimiter.
/// Dimensions are auto-detected from file content.
template <typename Value = double>
mat::dense2D<Value> read_dense(const std::string& filename, char delimiter = ',') {
    std::ifstream in(filename);
    if (!in.is_open())
        throw std::runtime_error("read_dense: cannot open file: " + filename);

    std::vector<std::vector<Value>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        // Replace delimiter with space for uniform parsing
        if (delimiter != ' ')
            for (auto& ch : line)
                if (ch == delimiter) ch = ' ';
        std::istringstream iss(line);
        std::vector<Value> row;
        Value v;
        while (iss >> v)
            row.push_back(v);
        if (!row.empty())
            rows.push_back(std::move(row));
    }

    if (rows.empty())
        throw std::runtime_error("read_dense: empty file: " + filename);

    std::size_t nrows = rows.size();
    std::size_t ncols = rows[0].size();
    mat::dense2D<Value> A(nrows, ncols);
    for (std::size_t i = 0; i < nrows; ++i)
        for (std::size_t j = 0; j < ncols && j < rows[i].size(); ++j)
            A(i, j) = rows[i][j];
    return A;
}

/// Read a sparse matrix from a triplet file (row col value per line, 0-based).
/// Requires explicit dimensions since the file may not contain all rows/cols.
template <typename Value = double>
mat::compressed2D<Value> read_sparse(const std::string& filename,
                                     std::size_t nrows, std::size_t ncols) {
    std::ifstream in(filename);
    if (!in.is_open())
        throw std::runtime_error("read_sparse: cannot open file: " + filename);

    mat::coordinate2D<Value> coo(nrows, ncols);
    std::size_t r, c;
    Value v;
    while (in >> r >> c >> v)
        coo.insert(r, c, v);
    return coo.compress();
}

} // namespace mtl::io
