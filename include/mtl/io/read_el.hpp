#pragma once
// MTL5 -- Read dense/sparse matrices from CSV/whitespace-delimited files
//
// Both readers check every line and throw std::runtime_error on input they
// cannot read exactly, instead of returning a partial or padded matrix (#843).
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <mtl/io/text_parse.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/coordinate2D.hpp>

namespace mtl::io {

namespace detail {

/// The fields of one line. With a space delimiter, any run of whitespace
/// separates fields. With another delimiter, the line is split on exactly that
/// character and each field is trimmed, so an empty field ("1,,2") is an error
/// rather than a dropped column.
inline bool split_fields(std::string_view line, char delimiter,
                         std::vector<std::string_view>& fields) {
    fields.clear();
    if (delimiter == ' ') {
        token_cursor cur(line);
        std::string_view tok;
        while (cur.next(tok)) fields.push_back(tok);
        return true;
    }
    std::size_t start = 0;
    for (;;) {
        const std::size_t end = line.find(delimiter, start);
        std::string_view f = line.substr(start, end == std::string_view::npos
                                                    ? std::string_view::npos : end - start);
        std::size_t a = 0, b = f.size();
        while (a < b && is_space(f[a])) ++a;
        while (b > a && is_space(f[b - 1])) --b;
        f = f.substr(a, b - a);
        if (f.empty()) return false;
        fields.push_back(f);
        if (end == std::string_view::npos) return true;
        start = end + 1;
    }
}

} // namespace detail

/// Read a dense matrix from a CSV or whitespace-delimited file.
/// One row per line, values separated by delimiter; blank lines are skipped.
/// Dimensions are taken from the file: every row must have the same number of
/// values. Throws std::runtime_error on an empty field, a value that does not
/// parse, a row of a different length, or a file with no rows.
template <typename Value = double>
mat::dense2D<Value> read_dense(const std::string& filename, char delimiter = ',') {
    std::ifstream in(filename);
    if (!in.is_open())
        throw std::runtime_error("read_dense: cannot open file: " + filename);

    std::vector<Value> values;
    std::vector<std::string_view> fields;
    std::size_t nrows = 0, ncols = 0, lineno = 0;
    std::string line;
    while (std::getline(in, line)) {
        ++lineno;
        if (detail::is_blank(line)) continue;
        auto where = [&] { return filename + ":" + std::to_string(lineno); };
        if (!detail::split_fields(line, delimiter, fields))
            throw std::runtime_error("read_dense: empty field at " + where());
        if (nrows == 0) ncols = fields.size();
        else if (fields.size() != ncols)
            throw std::runtime_error("read_dense: row with " + std::to_string(fields.size())
                                     + " values where the first row has "
                                     + std::to_string(ncols) + " at " + where());
        for (auto f : fields) {
            Value v{};
            if (!detail::parse_value(f, v))
                throw std::runtime_error("read_dense: malformed value '" + std::string(f)
                                         + "' at " + where());
            values.push_back(v);
        }
        ++nrows;
    }

    if (nrows == 0)
        throw std::runtime_error("read_dense: empty file: " + filename);

    mat::dense2D<Value> A(nrows, ncols);
    for (std::size_t i = 0; i < nrows; ++i)
        for (std::size_t j = 0; j < ncols; ++j)
            A(i, j) = values[i * ncols + j];
    return A;
}

/// Read a sparse matrix from a triplet file (row col value per line, 0-based).
/// Requires explicit dimensions since the file may not contain all rows/cols.
/// Blank lines are skipped. Throws std::runtime_error on a line that is not
/// exactly "row col value", and std::out_of_range on an index outside the
/// given dimensions. Duplicates are summed, as in coordinate2D::compress().
template <typename Value = double>
mat::compressed2D<Value> read_sparse(const std::string& filename,
                                     std::size_t nrows, std::size_t ncols) {
    std::ifstream in(filename);
    if (!in.is_open())
        throw std::runtime_error("read_sparse: cannot open file: " + filename);

    mat::coordinate2D<Value> coo(nrows, ncols);
    std::string line;
    std::size_t lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (detail::is_blank(line)) continue;
        auto where = [&] { return filename + ":" + std::to_string(lineno); };
        detail::token_cursor cur(line);
        std::string_view tok;
        std::size_t r = 0, c = 0;
        Value v{};
        if (!(cur.next(tok) && detail::parse_index(tok, r) &&
              cur.next(tok) && detail::parse_index(tok, c) &&
              cur.next(tok) && detail::parse_value(tok, v) && cur.done()))
            throw std::runtime_error("read_sparse: malformed line (expected 'row col value') "
                                     "at " + where() + ": '" + line + "'");
        if (r >= nrows || c >= ncols)
            throw std::out_of_range("read_sparse: index (" + std::to_string(r) + ", "
                                    + std::to_string(c) + ") outside a "
                                    + std::to_string(nrows) + " x " + std::to_string(ncols)
                                    + " matrix at " + where());
        coo.insert(r, c, v);
    }
    return coo.compress();
}

} // namespace mtl::io
