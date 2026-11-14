#pragma once
// MTL5 -- RAII sparse matrix inserter with updater functors
// Slot-based insertion with overflow to std::map, finalized in destructor.
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mtl/detail/exceptions.hpp>
#include <mtl/mat/compressed2D.hpp>

namespace mtl::mat {

// -- Updater functors ----------------------------------------------------

/// Overwrite: a = b (default updater)
template <typename T>
struct update_store {
    void operator()(T& a, const T& b) const { a = b; }
};

/// Accumulate: a += b
template <typename T>
struct update_plus {
    void operator()(T& a, const T& b) const { a += b; }
};

// -- Inserter ------------------------------------------------------------

/// RAII inserter for compressed2D. Allocates flat working arrays with
/// `slot_size` slots per row. Overflow entries go into a std::map.
/// Destructor merges everything into the final sorted CRS arrays.
///
/// The inserter AMENDS the matrix, as MTL4's does (#825): the entries already
/// in `m` are loaded into the slots at construction, so inserting at an
/// existing position applies the updater to the stored value (`update_store`
/// replaces it, `update_plus` adds to it) and every other entry is kept. To
/// refill a matrix from scratch, empty it first (`m.make_empty()` or
/// `m.change_dim(r, c)`). It used to rebuild the matrix from the inserter's own
/// entries only, so a second inserter silently discarded everything before it.
///
/// Indices are checked in every build (#825): `ins[r]` with r >= num_rows() and
/// `ins[r][c]` with c >= num_cols() throw std::out_of_range, via
/// `detail::throw_or_abort`. The check has to happen at insertion -- the row
/// indexes the slot arrays straight away, and finalize() runs in the
/// destructor, which must not throw -- so it is one comparison per index, not
/// a pass at the end.
template <typename Value, typename Parameters = parameters<>,
          typename Updater = update_store<Value>>
class compressed2D_inserter {
    using matrix_type = compressed2D<Value, Parameters>;
    using size_type   = typename matrix_type::size_type;

    // -- Proxy chain: inserter[row][col] << value ------------------------

    struct col_proxy {
        compressed2D_inserter& ins_;
        size_type row_;
        size_type col_;

        col_proxy& operator<<(const Value& val) {
            ins_.do_insert(row_, col_, val);
            return *this;
        }
    };

    struct row_proxy {
        compressed2D_inserter& ins_;
        size_type row_;

        col_proxy operator[](size_type col) {
            if (col >= ins_.ncols_)
                ::mtl::detail::throw_or_abort(
                    std::out_of_range("compressed2D_inserter: column index out of range"));
            return col_proxy{ins_, row_, col};
        }
    };

public:
    /// Construct inserter for matrix `m` with `slot_size` slots per row.
    explicit compressed2D_inserter(matrix_type& m, size_type slot_size = 5)
        : mat_(m), nrows_(m.num_rows()), ncols_(m.num_cols()), slot_size_(slot_size),
          slots_(nrows_ * slot_size),
          col_idx_(nrows_ * slot_size, size_type(-1)),
          count_(nrows_, size_type(0)),
          updater_{}
    {
        load_existing();
    }

    ~compressed2D_inserter() { finalize(); }

    // Non-copyable, non-movable
    compressed2D_inserter(const compressed2D_inserter&) = delete;
    compressed2D_inserter& operator=(const compressed2D_inserter&) = delete;

    row_proxy operator[](size_type row) {
        if (row >= nrows_)
            ::mtl::detail::throw_or_abort(
                std::out_of_range("compressed2D_inserter: row index out of range"));
        return row_proxy{*this, row};
    }

private:
    /// Put the matrix's current entries into the slots, the ones past a row's
    /// `slot_size` into the overflow map. CRS rows are sorted and duplicate-free,
    /// so the map keys arrive in increasing order and each emplace is O(1) at
    /// the end hint.
    void load_existing() {
        const auto& starts  = mat_.ref_major();
        const auto& indices = mat_.ref_minor();
        const auto& data    = mat_.ref_data();
        if (data.empty() || starts.size() != nrows_ + 1) return;
        for (size_type r = 0; r < nrows_; ++r) {
            const size_type base = r * slot_size_;
            for (size_type k = starts[r]; k < starts[r + 1]; ++k) {
                if (count_[r] < slot_size_) {
                    col_idx_[base + count_[r]] = indices[k];
                    slots_[base + count_[r]]   = data[k];
                    ++count_[r];
                } else {
                    overflow_.emplace_hint(overflow_.end(), std::make_pair(r, indices[k]), data[k]);
                }
            }
        }
    }

    void do_insert(size_type r, size_type c, const Value& val) {
        // Search existing slots for this row
        size_type base = r * slot_size_;
        for (size_type s = 0; s < count_[r]; ++s) {
            if (col_idx_[base + s] == c) {
                updater_(slots_[base + s], val);
                return;
            }
        }
        // Check the overflow map -- only once a row has spilled into it.
        auto key = std::make_pair(r, c);
        if (!overflow_.empty()) {
            auto it = overflow_.find(key);
            if (it != overflow_.end()) {
                updater_(it->second, val);
                return;
            }
        }
        // Insert new entry
        if (count_[r] < slot_size_) {
            size_type pos = base + count_[r];
            col_idx_[pos] = c;
            slots_[pos] = val;
            ++count_[r];
        } else {
            overflow_.emplace(key, val);
        }
    }

    void finalize() {
        auto& starts  = mat_.ref_major();
        auto& indices = mat_.ref_minor();
        auto& data    = mat_.ref_data();

        // Count total nnz
        size_type total_nnz = 0;
        for (size_type r = 0; r < nrows_; ++r)
            total_nnz += count_[r];
        total_nnz += overflow_.size();

        indices.resize(total_nnz);
        data.resize(total_nnz);
        starts.resize(nrows_ + 1);

        // Build per-row sorted entries, merge slots + overflow. The overflow map
        // is ordered by (row, col), so one iterator walks it alongside the rows,
        // and each row reserves only its own entries (#825: it used to reserve
        // the whole map's size for every row, and reallocate the vector per row).
        std::vector<std::pair<size_type, Value>> entries;
        auto ov = overflow_.begin();
        size_type pos = 0;
        for (size_type r = 0; r < nrows_; ++r) {
            starts[r] = pos;

            auto ov_end = ov;
            while (ov_end != overflow_.end() && ov_end->first.first == r) ++ov_end;

            entries.clear();
            entries.reserve(count_[r] + static_cast<size_type>(std::distance(ov, ov_end)));

            size_type base = r * slot_size_;
            for (size_type s = 0; s < count_[r]; ++s)
                entries.emplace_back(col_idx_[base + s], slots_[base + s]);
            for (; ov != ov_end; ++ov)
                entries.emplace_back(ov->first.second, ov->second);

            // Sort by column
            std::sort(entries.begin(), entries.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            for (const auto& [col, val] : entries) {
                indices[pos] = col;
                data[pos] = val;
                ++pos;
            }
        }
        starts[nrows_] = pos;
    }

    matrix_type& mat_;
    size_type nrows_;
    size_type ncols_;
    size_type slot_size_;
    std::vector<Value>     slots_;     // flat: nrows * slot_size
    std::vector<size_type> col_idx_;   // flat: nrows * slot_size
    std::vector<size_type> count_;     // entries per row in slots, <= slot_size
    std::map<std::pair<size_type, size_type>, Value> overflow_;
    Updater updater_;
};

/// Convenience alias: inserter<MatrixType, Updater>
template <typename Matrix, typename Updater = update_store<typename Matrix::value_type>>
using inserter = compressed2D_inserter<
    typename Matrix::value_type,
    typename Matrix::param_type,
    Updater>;

} // namespace mtl::mat

// -- Convenience alias --------------------------------------------------
namespace mtl { using mat::inserter; }
