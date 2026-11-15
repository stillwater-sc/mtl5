#pragma once
// MTL5 -- Matrix Market (.mtx) file format reader/writer
// Reads coordinate and array files with a real, integer or complex field and
// general, symmetric, skew-symmetric or hermitian symmetry (pattern is not
// supported); writes coordinate and array files with the field taken from the
// value type. Malformed input throws std::runtime_error (#125, #843).
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
#include <algorithm>
#include <cctype>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <vector>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/io/text_parse.hpp>
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
        std::error_code ec;
        const auto bytes = std::filesystem::file_size(filename, ec);
        if (!ec) bytes_ = static_cast<std::uintmax_t>(bytes);
    }

    /// The most entry lines the file can hold: one takes at least 6 bytes
    /// ("1 1 1\n"). Unknown, so unbounded, for a gzip file.
    std::size_t max_entries() const noexcept {
        if (use_gz_ || bytes_ == 0) return std::numeric_limits<std::size_t>::max();
        return static_cast<std::size_t>(bytes_ / 6 + 1);
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
                if (gzgets(gz_, gzbuf_.data(), static_cast<int>(gzbuf_.size())) == nullptr) {
                    // End of stream, or an error: a corrupt or truncated file
                    // must not read as a short one (#843).
                    int err = Z_OK;
                    const char* msg = gzerror(gz_, &err);
                    if (err != Z_OK && err != Z_STREAM_END)
                        throw std::runtime_error(std::string("mm_read: corrupt or truncated "
                                                             "gzip stream: ") + (msg ? msg : ""));
                    return !out.empty();              // trailing line without newline
                }
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
    std::uintmax_t bytes_ = 0;
};

/// The banner `%%MatrixMarket matrix <format> <field> <symmetry>`.
struct mm_header {
    enum class field_t { real, integer, complex, pattern };
    enum class symmetry_t { general, symmetric, skew_symmetric, hermitian };
    bool coordinate = true;
    field_t field = field_t::real;
    symmetry_t symmetry = symmetry_t::general;

    bool mirrored() const noexcept { return symmetry != symmetry_t::general; }
};

/// Parse the banner. All five tokens are required and matched exactly (case
/// insensitive): `symmetric` is a substring of `skew-symmetric`, so substring
/// matching is unsafe, and a missing token used to default silently (#843).
/// `pattern` is not supported, and `hermitian` needs a `complex` field.
inline mm_header parse_mm_banner(const std::string& banner_line) {
    std::string b = banner_line;
    std::transform(b.begin(), b.end(), b.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    token_cursor cur(b);
    std::string_view tok[5];
    int n = 0;
    while (n < 5 && cur.next(tok[n])) ++n;
    auto bad = [&](const std::string& why) {
        return std::runtime_error("mm_read: " + why + " in banner: '" + banner_line + "'");
    };
    if (n < 5 || !cur.done())
        throw bad("expected '%%MatrixMarket matrix <format> <field> <symmetry>'");
    if (tok[0] != "%%matrixmarket") throw bad("unrecognized tag");
    if (tok[1] != "matrix")         throw bad("unsupported object (only 'matrix')");

    mm_header h;
    if (tok[2] == "coordinate")  h.coordinate = true;
    else if (tok[2] == "array")  h.coordinate = false;
    else throw bad("unrecognized format");

    using F = mm_header::field_t;
    if (tok[3] == "real")         h.field = F::real;
    else if (tok[3] == "integer") h.field = F::integer;
    else if (tok[3] == "complex") h.field = F::complex;
    else if (tok[3] == "pattern")
        throw bad("unsupported field 'pattern' (only real, integer and complex)");
    else throw bad("unrecognized field");

    using S = mm_header::symmetry_t;
    if (tok[4] == "general")             h.symmetry = S::general;
    else if (tok[4] == "symmetric")      h.symmetry = S::symmetric;
    else if (tok[4] == "skew-symmetric") h.symmetry = S::skew_symmetric;
    else if (tok[4] == "hermitian")      h.symmetry = S::hermitian;
    else throw bad("unrecognized symmetry");

    if (h.symmetry == S::hermitian && h.field != F::complex)
        throw bad("'hermitian' needs the 'complex' field");
    return h;
}

/// a * b, or false if it overflows `max`.
inline bool checked_mul(std::size_t a, std::size_t b, std::size_t& out,
                        std::size_t max = std::numeric_limits<std::size_t>::max()) noexcept {
    if (a != 0 && b > max / a) return false;
    out = a * b;
    return true;
}

/// The number of values an array-format file holds, which is also the most
/// entries a coordinate file can have without duplicates: m*n for general,
/// n(n+1)/2 for symmetric and hermitian, n(n-1)/2 for skew-symmetric (whose
/// diagonal is zero and not stored). Throws if it overflows `max`.
inline std::size_t mm_value_count(const mm_header& h, std::size_t nrows, std::size_t ncols,
                                  std::size_t max, const char* who) {
    using S = mm_header::symmetry_t;
    std::size_t count = 0;
    bool ok = true;
    if (h.symmetry == S::general) {
        ok = checked_mul(nrows, ncols, count, max);
    } else {
        // n(n+1)/2 or n(n-1)/2, halving whichever factor is even first
        const std::size_t n = nrows;
        const std::size_t a = (h.symmetry == S::skew_symmetric) ? (n == 0 ? 0 : n - 1) : n;
        const std::size_t b = (h.symmetry == S::skew_symmetric) ? n : n + 1;
        if (h.symmetry != S::skew_symmetric && n == std::numeric_limits<std::size_t>::max())
            ok = false;
        else
            ok = (a % 2 == 0) ? checked_mul(a / 2, b, count, max) : checked_mul(a, b / 2, count, max);
    }
    if (!ok)
        throw std::runtime_error(std::string(who) + ": matrix dimensions overflow: "
                                 + std::to_string(nrows) + " x " + std::to_string(ncols));
    return count;
}

/// Checks that Value can hold the file's field.
template <typename Value>
void check_mm_field(const mm_header& h, const char* who) {
    if (h.field == mm_header::field_t::complex && !is_complex_v<Value>)
        throw std::runtime_error(std::string(who) + ": a 'complex' file needs a std::complex "
                                 "value type");
}

/// The next non-blank line, or false at EOF.
inline bool next_data_line(mm_reader& in, std::string& line) {
    while (in.next_line(line))
        if (!is_blank(line)) return true;
    return false;
}

/// Reads one value (two numbers for a complex field) from the cursor.
template <typename Value>
bool read_mm_value(token_cursor& cur, const mm_header& h, Value& v) {
    std::string_view tok;
    if constexpr (is_complex_v<Value>) {
        using R = typename Value::value_type;
        R re{}, im{};
        if (!cur.next(tok) || !parse_value(tok, re)) return false;
        if (h.field == mm_header::field_t::complex)
            if (!cur.next(tok) || !parse_value(tok, im)) return false;
        v = Value(re, im);
        return true;
    } else {
        return cur.next(tok) && parse_value(tok, v);
    }
}

/// The mirrored value at (j, i) for a stored (i, j), i != j.
template <typename Value>
Value mm_mirror(const mm_header& h, const Value& v) {
    using S = mm_header::symmetry_t;
    if (h.symmetry == S::skew_symmetric) return -v;
    if constexpr (is_complex_v<Value>)
        if (h.symmetry == S::hermitian) return std::conj(v);
    return v;
}

/// Everything after the banner up to the data: comment lines, then the size
/// line, checked in full.
struct mm_sizes {
    std::size_t nrows = 0, ncols = 0, nnz = 0;   // nnz: entries (coordinate) or values (array)
};

inline mm_sizes read_mm_sizes(mm_reader& in, const mm_header& h, std::size_t max_index,
                              const std::string& filename, const char* who) {
    std::string line;
    bool have_dimensions = false;
    while (in.next_line(line)) {
        if (is_blank(line) || line[0] == '%') continue;
        have_dimensions = true;
        break;
    }
    if (!have_dimensions)
        throw std::runtime_error(std::string(who) + ": missing dimension line: " + filename);

    mm_sizes z;
    token_cursor cur(line);
    std::string_view tok;
    // A dimension stays below max_index, so the CRS row pointers' nrows + 1
    // cannot wrap: "18446744073709551615 0 0" holds no entries and passes
    // every other check.
    const bool ok = cur.next(tok) && parse_index(tok, z.nrows, max_index - 1) &&
                    cur.next(tok) && parse_index(tok, z.ncols, max_index - 1) &&
                    (!h.coordinate || (cur.next(tok) && parse_index(tok, z.nnz, max_index))) &&
                    cur.done();
    if (!ok)
        throw std::runtime_error(std::string(who) + ": malformed dimension line (expected "
                                 + (h.coordinate ? "rows cols nnz" : "rows cols")
                                 + " as non-negative integers): '" + line + "'");
    if (h.mirrored() && z.nrows != z.ncols)
        throw std::runtime_error(std::string(who) + ": a symmetric, skew-symmetric or "
                                 "hermitian matrix must be square");

    const std::size_t count = mm_value_count(h, z.nrows, z.ncols, max_index, who);
    if (h.coordinate) {
        if (z.nnz > count)
            throw std::runtime_error(std::string(who) + ": header nnz " + std::to_string(z.nnz)
                                     + " exceeds the " + std::to_string(count)
                                     + " entries a " + std::to_string(z.nrows) + " x "
                                     + std::to_string(z.ncols) + " matrix can hold");
    } else {
        z.nnz = count;
    }
    return z;
}

/// After the data: only blank and comment lines may follow.
inline void check_mm_trailer(mm_reader& in, std::size_t expected, const char* who) {
    std::string line;
    while (in.next_line(line)) {
        if (is_blank(line) || line[0] == '%') continue;
        throw std::runtime_error(std::string(who) + ": more data than the header's "
                                 + std::to_string(expected) + " entries: '" + line + "'");
    }
}

/// Visits the array-format values in file order (column-major; for a
/// symmetric, skew-symmetric or hermitian file, the lower triangle by columns,
/// without the diagonal when skew), as f(i, j, v).
template <typename Value, typename F>
void read_mm_array(mm_reader& in, const mm_header& h, const mm_sizes& z, const char* who, F&& f) {
    using S = mm_header::symmetry_t;
    std::string line;
    for (std::size_t j = 0; j < z.ncols; ++j) {
        std::size_t i0 = 0;
        if (h.symmetry == S::skew_symmetric) i0 = j + 1;
        else if (h.mirrored())               i0 = j;
        for (std::size_t i = i0; i < z.nrows; ++i) {
            if (!next_data_line(in, line))
                throw std::runtime_error(std::string(who) + ": unexpected EOF in array data "
                                         "(expected " + std::to_string(z.nnz) + " values)");
            token_cursor cur(line);
            Value v{};
            if (!read_mm_value(cur, h, v) || !cur.done())
                throw std::runtime_error(std::string(who) + ": malformed array value: '"
                                         + line + "'");
            f(i, j, v);
        }
    }
    check_mm_trailer(in, z.nnz, who);
}

/// Visits the coordinate entries as f(i, j, v), 0-based, after checking each
/// line and index. Mirroring is left to the caller.
template <typename Value, typename F>
void read_mm_coordinate(mm_reader& in, const mm_header& h, const mm_sizes& z, const char* who,
                        F&& f) {
    std::string line;
    for (std::size_t k = 0; k < z.nnz; ++k) {
        if (!next_data_line(in, line))
            throw std::runtime_error(std::string(who) + ": unexpected EOF in entries (expected "
                                     + std::to_string(z.nnz) + ")");
        token_cursor cur(line);
        std::string_view tok;
        std::size_t r1 = 0, c1 = 0;
        Value v{};
        if (!(cur.next(tok) && parse_index(tok, r1) && cur.next(tok) && parse_index(tok, c1) &&
              read_mm_value(cur, h, v) && cur.done()))
            throw std::runtime_error(std::string(who) + ": malformed coordinate entry: '"
                                     + line + "'");
        if (r1 == 0 || r1 > z.nrows || c1 == 0 || c1 > z.ncols)
            throw std::runtime_error(std::string(who) + ": coordinate index out of range: '"
                                     + line + "'");
        if (h.symmetry == mm_header::symmetry_t::skew_symmetric && r1 == c1)
            throw std::runtime_error(std::string(who) + ": a skew-symmetric file stores no "
                                     "diagonal entry: '" + line + "'");
        f(r1 - 1, c1 - 1, v);
    }
    check_mm_trailer(in, z.nnz, who);
}

} // namespace detail

/// Read a Matrix Market file into a compressed2D, from either format.
///
/// Supported: coordinate and array; real, integer and complex fields (complex
/// needs a std::complex Value); general, symmetric, skew-symmetric and
/// hermitian (complex only), mirrored into the full matrix. Not supported:
/// pattern. Malformed input throws std::runtime_error: a bad banner, a negative
/// or overflowing size, a header nnz larger than the matrix can hold, a
/// malformed or out-of-range entry, too few or too many entries (#843).
/// Duplicate coordinates are summed.
template <typename Value = double, typename Parameters = mat::parameters<>>
mat::compressed2D<Value, Parameters> mm_read(const std::string& filename) {
    constexpr const char* who = "mm_read";
    detail::mm_reader in(filename);

    std::string line;
    if (!in.next_line(line))                       // banner line
        throw std::runtime_error("mm_read: empty file: " + filename);
    const detail::mm_header h = detail::parse_mm_banner(line);
    detail::check_mm_field<Value>(h, who);

    using size_type = typename Parameters::size_type;
    const std::size_t max_index = static_cast<std::size_t>(std::numeric_limits<size_type>::max());
    const detail::mm_sizes z = detail::read_mm_sizes(in, h, max_index, filename, who);
    const size_type nrows = static_cast<size_type>(z.nrows);
    const size_type ncols = static_cast<size_type>(z.ncols);

    if (h.coordinate) {
        // Direct CRS assembly: gather triplets into one buffer, sort in place,
        // then merge duplicates into CRS. This matches coordinate2D::compress()
        // output exactly but avoids its extra full-size copy -- the transient
        // peak-memory win for very large matrices. The buffer is sized from
        // the header nnz, capped by what the file's size can hold, so a header
        // that overstates nnz does not reserve memory the data cannot fill.
        using triplet = std::tuple<size_type, size_type, Value>;
        std::vector<triplet> ents;
        const std::size_t stored = std::min(z.nnz, in.max_entries());
        ents.reserve(h.mirrored() && stored <= ents.max_size() / 2 ? 2 * stored : stored);

        detail::read_mm_coordinate<Value>(in, h, z, who,
            [&](std::size_t r, std::size_t c, const Value& v) {
                ents.emplace_back(size_type(r), size_type(c), v);
                if (h.mirrored() && r != c)
                    ents.emplace_back(size_type(c), size_type(r), detail::mm_mirror(h, v));
            });

        std::sort(ents.begin(), ents.end(),
            [](const triplet& a, const triplet& b) {
                if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
                return std::get<1>(a) < std::get<1>(b);
            });

        std::vector<size_type> starts(std::size_t(nrows) + 1, size_type(0));
        std::vector<size_type> indices;
        std::vector<Value>     data;
        indices.reserve(ents.size());
        data.reserve(ents.size());
        for (std::size_t k = 0; k < ents.size(); ) {
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
        // Array format into a sparse result: only the nonzeros are stored, so
        // nothing is reserved up front (the dense count could be huge).
        mat::coordinate2D<Value, Parameters> coo(nrows, ncols);
        const Value zero = math::zero<Value>();
        detail::read_mm_array<Value>(in, h, z, who,
            [&](std::size_t i, std::size_t j, const Value& v) {
                if (v == zero) return;
                coo.insert(size_type(i), size_type(j), v);
                if (i != j && h.mirrored())
                    coo.insert(size_type(j), size_type(i), detail::mm_mirror(h, v));
            });
        return coo.compress();
    }
}

/// Read a Matrix Market file into a dense2D, from either format, with the same
/// support and checks as mm_read. Reads ".gz" transparently (zlib).
template <typename Value = double>
mat::dense2D<Value> mm_read_dense(const std::string& filename) {
    constexpr const char* who = "mm_read_dense";
    detail::mm_reader in(filename);

    std::string line;
    if (!in.next_line(line))
        throw std::runtime_error("mm_read_dense: empty file: " + filename);
    const detail::mm_header h = detail::parse_mm_banner(line);
    detail::check_mm_field<Value>(h, who);

    const detail::mm_sizes z = detail::read_mm_sizes(
        in, h, std::numeric_limits<std::size_t>::max(), filename, who);
    std::size_t cells = 0;
    if (!detail::checked_mul(z.nrows, z.ncols, cells))
        throw std::runtime_error("mm_read_dense: matrix dimensions overflow");

    mat::dense2D<Value> A(z.nrows, z.ncols);
    for (std::size_t i = 0; i < z.nrows; ++i)
        for (std::size_t j = 0; j < z.ncols; ++j)
            A(i, j) = math::zero<Value>();

    if (h.coordinate) {
        detail::read_mm_coordinate<Value>(in, h, z, who,
            [&](std::size_t r, std::size_t c, const Value& v) {
                // Duplicate coordinates sum, as in mm_read and SciPy; A starts at zero.
                A(r, c) += v;
                if (h.mirrored() && r != c) A(c, r) += detail::mm_mirror(h, v);
            });
    } else {
        detail::read_mm_array<Value>(in, h, z, who,
            [&](std::size_t i, std::size_t j, const Value& v) {
                A(i, j) = v;
                if (h.mirrored() && i != j) A(j, i) = detail::mm_mirror(h, v);
            });
    }
    return A;
}

namespace detail {

/// The banner field for a value type: complex for std::complex, integer for
/// integral types, real otherwise.
template <typename Value>
constexpr const char* mm_field_name() {
    if constexpr (is_complex_v<Value>) return "complex";
    else if constexpr (std::is_integral_v<Value>) return "integer";
    else return "real";
}

/// Enough digits for the value to read back exactly.
template <typename Value>
void set_mm_precision(std::ostream& out) {
    constexpr int digits = std::numeric_limits<magnitude_t<Value>>::max_digits10;
    if constexpr (digits > 0) out.precision(digits);
}

/// One value: a complex one as "re im".
template <typename Value>
void write_mm_value(std::ostream& out, const Value& v) {
    if constexpr (is_complex_v<Value>) out << v.real() << ' ' << v.imag();
    else out << v;
}

} // namespace detail

/// Write a matrix in Matrix Market array format (column-major), with the
/// field taken from the value type: real, integer or complex.
template <typename Matrix>
void mm_write(const std::string& filename, const Matrix& A,
              const std::string& comment = "") {
    using Value = std::remove_cvref_t<typename Matrix::value_type>;
    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("mm_write: cannot open file: " + filename);

    out << "%%MatrixMarket matrix array " << detail::mm_field_name<Value>() << " general\n";
    if (!comment.empty())
        out << "% " << comment << "\n";
    out << A.num_rows() << " " << A.num_cols() << "\n";

    detail::set_mm_precision<Value>(out);
    for (std::size_t j = 0; j < A.num_cols(); ++j)
        for (std::size_t i = 0; i < A.num_rows(); ++i) {
            detail::write_mm_value(out, static_cast<Value>(A(i, j)));
            out << "\n";
        }
    if (!out)
        throw std::runtime_error("mm_write: write failed: " + filename);
}

/// Write a sparse matrix in Matrix Market coordinate format, with the field
/// taken from the value type: real, integer or complex.
template <typename Value, typename Parameters>
void mm_write_sparse(const std::string& filename,
                     const mat::compressed2D<Value, Parameters>& A,
                     const std::string& comment = "") {
    std::ofstream out(filename);
    if (!out.is_open())
        throw std::runtime_error("mm_write_sparse: cannot open file: " + filename);

    out << "%%MatrixMarket matrix coordinate " << detail::mm_field_name<Value>()
        << " general\n";
    if (!comment.empty())
        out << "% " << comment << "\n";
    out << A.num_rows() << " " << A.num_cols() << " " << A.nnz() << "\n";

    const auto& starts = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data = A.ref_data();

    detail::set_mm_precision<Value>(out);
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k) {
            out << (i + 1) << " " << (indices[k] + 1) << " ";
            detail::write_mm_value(out, data[k]);
            out << "\n";
        }
    if (!out)
        throw std::runtime_error("mm_write_sparse: write failed: " + filename);
}

} // namespace mtl::io
