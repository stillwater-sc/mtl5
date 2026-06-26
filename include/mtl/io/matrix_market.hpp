#pragma once
// MTL5 -- Matrix Market (.mtx) file format reader/writer
// Supports: real coordinate general, real coordinate symmetric,
//           real array general (dense).
//
// Files ending in ".gz" are read transparently when MTL5 is built with zlib
// (configure with -DMTL5_WITH_ZLIB=ON, which defines MTL5_HAS_ZLIB). Without
// zlib, opening a ".gz" path throws a clear error.
//
// The sparse coordinate reader assembles CRS directly from one triplet buffer
// (sized from the header nnz) and sorts in place, avoiding the extra full-size
// copy that coordinate2D::compress() makes -- roughly halving transient peak
// memory on very large matrices (e.g. circuit5M) while producing identical
// output (same (row,col) order, same duplicate accumulation).
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <tuple>
#include <vector>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/coordinate2D.hpp>
#include <mtl/math/identity.hpp>
#ifdef MTL5_HAS_ZLIB
#include <zlib.h>
#endif

namespace mtl::io {

namespace detail {

/// Line source for Matrix Market input: reads from a plain file, or transparently
/// from a gzip-compressed ".gz" file when built with zlib. One small abstraction
/// so the parsers are agnostic to compression.
class mm_reader {
public:
    explicit mm_reader(const std::string& filename) {
        const bool gz = filename.size() >= 3 &&
                        filename.compare(filename.size() - 3, 3, ".gz") == 0;
        if (gz) {
#ifdef MTL5_HAS_ZLIB
            gz_ = gzopen(filename.c_str(), "rb");
            if (!gz_)
                throw std::runtime_error("mm_read: cannot open gzip file: " + filename);
            gzbuf_.resize(1u << 16);
            use_gz_ = true;
            return;
#else
            throw std::runtime_error(
                "mm_read: '" + filename + "' is gzip-compressed but MTL5 was built "
                "without zlib support (configure with -DMTL5_WITH_ZLIB=ON)");
#endif
        }
        plain_.open(filename);
        if (!plain_.is_open())
            throw std::runtime_error("mm_read: cannot open file: " + filename);
    }

#ifdef MTL5_HAS_ZLIB
    ~mm_reader() { if (gz_) gzclose(gz_); }
    mm_reader(const mm_reader&) = delete;
    mm_reader& operator=(const mm_reader&) = delete;
#endif

    /// Read one logical line (newline stripped, CRLF-safe). Returns false at EOF.
    bool next_line(std::string& out) {
#ifdef MTL5_HAS_ZLIB
        if (use_gz_) {
            out.clear();
            for (;;) {
                if (gzgets(gz_, gzbuf_.data(), static_cast<int>(gzbuf_.size())) == nullptr)
                    return !out.empty();              // trailing line without newline
                out += gzbuf_.data();
                if (!out.empty() && out.back() == '\n') {
                    out.pop_back();
                    if (!out.empty() && out.back() == '\r') out.pop_back();
                    return true;
                }
                // no newline yet -> long line, keep appending
            }
        }
#endif
        if (std::getline(plain_, out)) {
            if (!out.empty() && out.back() == '\r') out.pop_back();
            return true;
        }
        return false;
    }

private:
    std::ifstream plain_;
#ifdef MTL5_HAS_ZLIB
    gzFile gz_ = nullptr;
    std::vector<char> gzbuf_;
#endif
    bool use_gz_ = false;
};

} // namespace detail

/// Read a Matrix Market file into a compressed2D (for coordinate format)
/// or dense2D (for array format). Returns compressed2D for sparse matrices.
template <typename Value = double, typename Parameters = mat::parameters<>>
mat::compressed2D<Value, Parameters> mm_read(const std::string& filename) {
    detail::mm_reader in(filename);

    std::string line;
    if (!in.next_line(line))                       // banner line
        throw std::runtime_error("mm_read: empty file: " + filename);

    // Parse banner: %%MatrixMarket matrix coordinate|array real|... general|symmetric|...
    bool is_coordinate = true;
    bool is_symmetric = false;
    std::string banner = line;
    std::transform(banner.begin(), banner.end(), banner.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (banner.find("array") != std::string::npos)     is_coordinate = false;
    if (banner.find("symmetric") != std::string::npos) is_symmetric = true;

    // Skip comment / blank lines to the dimension line.
    while (in.next_line(line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
    }

    using size_type = typename Parameters::size_type;
    size_type nrows = 0, ncols = 0, nnz_entries = 0;
    {
        std::istringstream dim_stream(line);
        if (is_coordinate) dim_stream >> nrows >> ncols >> nnz_entries;
        else             { dim_stream >> nrows >> ncols; nnz_entries = nrows * ncols; }
    }

    if (is_coordinate) {
        // Direct CRS assembly: gather triplets into one buffer (sized from the
        // header nnz), sort in place, then merge duplicates into CRS. This matches
        // coordinate2D::compress() output exactly but avoids its extra full-size
        // copy -- the transient-peak-memory win for very large matrices.
        using triplet = std::tuple<size_type, size_type, Value>;
        std::vector<triplet> ents;
        ents.reserve(is_symmetric ? 2 * nnz_entries : nnz_entries);

        std::istringstream ss;
        for (size_type k = 0; k < nnz_entries; ++k) {
            if (!in.next_line(line))
                throw std::runtime_error("mm_read: unexpected EOF in entries (expected "
                                         + std::to_string(nnz_entries) + ")");
            if (line.empty()) { --k; continue; }   // tolerate stray blank lines
            ss.clear(); ss.str(line);
            size_type r, c; Value v;
            ss >> r >> c >> v;
            if (ss.fail())
                throw std::runtime_error("mm_read: malformed coordinate entry: '" + line + "'");
            --r; --c;                              // 1-based -> 0-based
            ents.emplace_back(r, c, v);
            if (is_symmetric && r != c) ents.emplace_back(c, r, v);
        }

        std::sort(ents.begin(), ents.end(),
            [](const triplet& a, const triplet& b) {
                if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
                return std::get<1>(a) < std::get<1>(b);
            });

        std::vector<size_type> starts(nrows + 1, size_type(0));
        std::vector<size_type> indices;
        std::vector<Value>     data;
        indices.reserve(ents.size());
        data.reserve(ents.size());
        for (size_type k = 0; k < ents.size(); ) {
            const size_type r = std::get<0>(ents[k]);
            const size_type c = std::get<1>(ents[k]);
            Value acc = std::get<2>(ents[k]);
            ++k;
            while (k < ents.size() &&                         // accumulate duplicates
                   std::get<0>(ents[k]) == r && std::get<1>(ents[k]) == c) {
                acc += std::get<2>(ents[k]);
                ++k;
            }
            indices.push_back(c);
            data.push_back(acc);
            starts[r + 1]++;
        }
        for (size_type i = 0; i < nrows; ++i) starts[i + 1] += starts[i];

        return mat::compressed2D<Value, Parameters>(
            nrows, ncols, data.size(), starts.data(), indices.data(), data.data());
    } else {
        // Array (dense column-major) format: one value per line.
        mat::coordinate2D<Value, Parameters> coo(nrows, ncols);
        coo.reserve(nrows * ncols);
        std::istringstream ss;
        for (size_type j = 0; j < ncols; ++j)
            for (size_type i = 0; i < nrows; ++i) {
                if (!in.next_line(line))
                    throw std::runtime_error("mm_read: unexpected EOF in array data");
                if (line.empty()) { --i; continue; }
                ss.clear(); ss.str(line);
                Value v; ss >> v;
                if (ss.fail())
                    throw std::runtime_error("mm_read: malformed array value: '" + line + "'");
                if (v != math::zero<Value>()) coo.insert(i, j, v);
            }
        return coo.compress();
    }
}

/// Read a Matrix Market file into a dense2D. Reads ".gz" transparently (zlib).
template <typename Value = double>
mat::dense2D<Value> mm_read_dense(const std::string& filename) {
    detail::mm_reader in(filename);

    std::string line;
    if (!in.next_line(line))
        throw std::runtime_error("mm_read_dense: empty file: " + filename);

    std::string banner = line;
    std::transform(banner.begin(), banner.end(), banner.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    bool is_coordinate = (banner.find("array") == std::string::npos);
    bool is_symmetric = (banner.find("symmetric") != std::string::npos);

    while (in.next_line(line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
    }

    std::size_t nrows = 0, ncols = 0, nnz_entries = 0;
    {
        std::istringstream dim_stream(line);
        if (is_coordinate) dim_stream >> nrows >> ncols >> nnz_entries;
        else               dim_stream >> nrows >> ncols;
    }

    mat::dense2D<Value> A(nrows, ncols);
    for (std::size_t i = 0; i < nrows; ++i)
        for (std::size_t j = 0; j < ncols; ++j)
            A(i, j) = math::zero<Value>();

    std::istringstream ss;
    if (is_coordinate) {
        for (std::size_t k = 0; k < nnz_entries; ++k) {
            if (!in.next_line(line))
                throw std::runtime_error("mm_read_dense: unexpected EOF in entries");
            if (line.empty()) { --k; continue; }
            ss.clear(); ss.str(line);
            std::size_t r, c; Value v;
            ss >> r >> c >> v;
            if (ss.fail())
                throw std::runtime_error("mm_read_dense: malformed entry: '" + line + "'");
            r--; c--;
            A(r, c) = v;
            if (is_symmetric && r != c) A(c, r) = v;
        }
    } else {
        // Column-major dense: one value per line.
        for (std::size_t j = 0; j < ncols; ++j)
            for (std::size_t i = 0; i < nrows; ++i) {
                if (!in.next_line(line))
                    throw std::runtime_error("mm_read_dense: unexpected EOF in array data");
                if (line.empty()) { --i; continue; }
                ss.clear(); ss.str(line);
                ss >> A(i, j);
                if (ss.fail())
                    throw std::runtime_error("mm_read_dense: malformed array value: '" + line + "'");
            }
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
