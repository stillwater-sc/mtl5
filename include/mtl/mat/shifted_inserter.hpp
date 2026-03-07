#pragma once
// MTL5 — Shifted inserter: wraps any inserter and shifts row/col indices.
// Use case: FEM assembly where local element matrices are inserted at
// global offsets.
#include <cstddef>
#include <cassert>

namespace mtl::mat {

/// Offset decorator for sparse matrix inserters.
/// Wraps a base inserter and shifts row/col indices by configurable offsets.
template <typename BaseInserter>
class shifted_inserter {
    using size_type = std::size_t;

    // ── Proxy chain: ins[row][col] << value ─────────────────────────────
    struct col_proxy {
        BaseInserter& base_;
        size_type row_;
        size_type col_;

        template <typename T>
        col_proxy& operator<<(const T& val) {
            base_[row_][col_] << val;
            return *this;
        }
    };

    struct row_proxy {
        BaseInserter& base_;
        size_type row_;
        size_type col_offset_;

        col_proxy operator[](size_type col) {
            return col_proxy{base_, row_, col + col_offset_};
        }
    };

public:
    /// Construct shifted inserter owning a base inserter.
    /// @param matrix     The sparse matrix to insert into
    /// @param slot_size  Slots per row for the base inserter
    /// @param row_offset Offset added to row indices
    /// @param col_offset Offset added to column indices
    template <typename Matrix>
    shifted_inserter(Matrix& matrix, size_type slot_size,
                     size_type row_offset = 0, size_type col_offset = 0)
        : base_(matrix, slot_size),
          row_offset_(row_offset), col_offset_(col_offset) {}

    row_proxy operator[](size_type row) {
        return row_proxy{base_, row + row_offset_, col_offset_};
    }

    // ── Offset control ──────────────────────────────────────────────────

    void set_row_offset(size_type off) { row_offset_ = off; }
    void set_col_offset(size_type off) { col_offset_ = off; }
    size_type get_row_offset() const { return row_offset_; }
    size_type get_col_offset() const { return col_offset_; }

private:
    BaseInserter base_;   // owns the base inserter; RAII finalization in its destructor
    size_type row_offset_;
    size_type col_offset_;
};

} // namespace mtl::mat

namespace mtl { using mat::shifted_inserter; }
