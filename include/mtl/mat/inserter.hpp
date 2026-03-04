#pragma once
// MTL5 — RAII sparse matrix inserter with updater functors
// Slot-based insertion with overflow to std::map, finalized in destructor.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <map>
#include <vector>

#include <mtl/mat/compressed2D.hpp>

namespace mtl::mat {

// ── Updater functors ────────────────────────────────────────────────────

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

// ── Inserter ────────────────────────────────────────────────────────────

/// RAII inserter for compressed2D. Allocates flat working arrays with
/// `slot_size` slots per row. Overflow entries go into a std::map.
/// Destructor merges everything into the final sorted CRS arrays.
template <typename Value, typename Parameters = parameters<>,
          typename Updater = update_store<Value>>
class compressed2D_inserter {
    using matrix_type = compressed2D<Value, Parameters>;
    using size_type   = typename matrix_type::size_type;

    // ── Proxy chain: inserter[row][col] << value ────────────────────────

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
            return col_proxy{ins_, row_, col};
        }
    };

public:
    /// Construct inserter for matrix `m` with `slot_size` slots per row.
    explicit compressed2D_inserter(matrix_type& m, size_type slot_size = 5)
        : mat_(m), nrows_(m.num_rows()), slot_size_(slot_size),
          slots_(nrows_ * slot_size),
          col_idx_(nrows_ * slot_size, size_type(-1)),
          count_(nrows_, size_type(0)),
          updater_{}
    {}

    ~compressed2D_inserter() { finalize(); }

    // Non-copyable, non-movable
    compressed2D_inserter(const compressed2D_inserter&) = delete;
    compressed2D_inserter& operator=(const compressed2D_inserter&) = delete;

    row_proxy operator[](size_type row) {
        assert(row < nrows_);
        return row_proxy{*this, row};
    }

private:
    void do_insert(size_type r, size_type c, const Value& val) {
        // Search existing slots for this row
        size_type base = r * slot_size_;
        for (size_type s = 0; s < count_[r] && s < slot_size_; ++s) {
            if (col_idx_[base + s] == c) {
                updater_(slots_[base + s], val);
                return;
            }
        }
        // Check overflow map
        auto key = std::make_pair(r, c);
        auto it = overflow_.find(key);
        if (it != overflow_.end()) {
            updater_(it->second, val);
            return;
        }
        // Insert new entry
        if (count_[r] < slot_size_) {
            size_type pos = base + count_[r];
            col_idx_[pos] = c;
            slots_[pos] = val;
            ++count_[r];
        } else {
            overflow_[key] = val;
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

        // Build per-row sorted entries, merge slots + overflow
        size_type pos = 0;
        for (size_type r = 0; r < nrows_; ++r) {
            starts[r] = pos;

            // Collect entries for this row
            std::vector<std::pair<size_type, Value>> entries;
            entries.reserve(count_[r] + overflow_.size());

            size_type base = r * slot_size_;
            for (size_type s = 0; s < count_[r] && s < slot_size_; ++s) {
                entries.emplace_back(col_idx_[base + s], slots_[base + s]);
            }

            // Add overflow entries for this row
            auto lo = overflow_.lower_bound(std::make_pair(r, size_type(0)));
            auto hi = overflow_.lower_bound(std::make_pair(r + 1, size_type(0)));
            for (auto it = lo; it != hi; ++it) {
                entries.emplace_back(it->first.second, it->second);
            }

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
    size_type slot_size_;
    std::vector<Value>     slots_;     // flat: nrows * slot_size
    std::vector<size_type> col_idx_;   // flat: nrows * slot_size
    std::vector<size_type> count_;     // entries per row in slots
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

// ── Convenience alias ──────────────────────────────────────────────────
namespace mtl { using mat::inserter; }
