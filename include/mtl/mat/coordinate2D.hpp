#pragma once
// MTL5 -- COO (coordinate) sparse matrix format
// Stores triplets (row, col, value). Supports sort() and compress() to CRS.
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <mtl/detail/exceptions.hpp>

#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::mat {

/// COO sparse matrix: stores (row, col, value) triplets.
/// Supports unordered insertion, sorting, and compression to compressed2D.
template <typename Value, typename Parameters = parameters<>>
class coordinate2D {
public:
    using value_type      = Value;
    using size_type       = typename Parameters::size_type;
    using const_reference = const Value&;
    using reference       = Value&;
    using triplet_type    = std::tuple<size_type, size_type, Value>;

    /// Default: empty 0x0
    coordinate2D() : nrows_(0), ncols_(0) {}

    /// Empty nrows x ncols matrix
    coordinate2D(size_type nrows, size_type ncols)
        : nrows_(nrows), ncols_(ncols) {}

    // -- Insertion --------------------------------------------------------

    /// Insert a single entry (accumulates with existing entries on compress).
    /// The indices are not checked here: compress() checks every entry, in
    /// every build, once per matrix rather than once per insert (#825).
    void insert(size_type r, size_type c, Value v) {
        entries_.emplace_back(r, c, v);
        sorted_ = false;
    }

    /// Reserve space for nnz entries
    void reserve(size_type nnz) { entries_.reserve(nnz); }

    // -- Element access (linear scan, mainly for testing) -----------------

    value_type operator()(size_type r, size_type c) const {
        Value sum = math::zero<Value>();
        for (const auto& [er, ec, ev] : entries_)
            if (er == r && ec == c) sum += ev;
        return sum;
    }

    // -- Size / shape ----------------------------------------------------

    size_type num_rows() const { return nrows_; }
    size_type num_cols() const { return ncols_; }
    size_type size()     const { return nrows_ * ncols_; }
    size_type nnz()      const { return entries_.size(); }

    /// Raw access to the (row, col, value) triplets (unsorted unless sort()ed).
    const std::vector<triplet_type>& ref_entries() const { return entries_; }

    // -- Sorting ----------------------------------------------------------

    /// Sort entries by (row, col) in-place. Stable: duplicates of one position
    /// keep their insertion order, so compress() sums them in that order.
    void sort() {
        std::stable_sort(entries_.begin(), entries_.end(), by_position);
        sorted_ = true;
    }

    bool is_sorted() const { return sorted_; }

    // -- Compression to CRS -----------------------------------------------

    /// Convert to compressed2D, accumulating duplicate entries.
    ///
    /// Duplicates are summed in insertion order (#825): the sort is stable, so
    /// the result is reproducible for rounding-sensitive and non-commutative
    /// values. It used to be std::sort, which put duplicates in an arbitrary
    /// order. After sort(), with no insert since, the entries are used as they
    /// are, with no copy and no sort.
    ///
    /// Throws std::out_of_range (via detail::throw_or_abort) if an entry's row
    /// or column is outside the matrix, in every build: an O(nnz) pass, once
    /// per matrix (#825). Unchecked, a bad row was a heap write out of bounds
    /// below, at starts[r + 1].
    compressed2D<Value, Parameters> compress() const {
        for (const auto& [r, c, v] : entries_)
            if (r >= nrows_ || c >= ncols_)
                ::mtl::detail::throw_or_abort(
                    std::out_of_range("coordinate2D::compress: entry index out of range"));

        std::vector<triplet_type> copy;
        if (!sorted_) {
            copy = entries_;
            std::stable_sort(copy.begin(), copy.end(), by_position);
        }
        const std::vector<triplet_type>& sorted_entries = sorted_ ? entries_ : copy;

        // Merge duplicates and build CRS arrays
        std::vector<size_type> starts(nrows_ + 1, size_type(0));
        std::vector<size_type> indices;
        std::vector<Value>     data;

        for (size_type k = 0; k < sorted_entries.size(); ) {
            auto [r, c, v] = sorted_entries[k];
            Value acc = v;
            ++k;
            // Accumulate duplicates
            while (k < sorted_entries.size() &&
                   std::get<0>(sorted_entries[k]) == r &&
                   std::get<1>(sorted_entries[k]) == c) {
                acc += std::get<2>(sorted_entries[k]);
                ++k;
            }
            indices.push_back(c);
            data.push_back(acc);
            starts[r + 1]++;
        }

        // Cumulative sum for row pointers
        for (size_type i = 0; i < nrows_; ++i)
            starts[i + 1] += starts[i];

        size_type total_nnz = data.size();
        return compressed2D<Value, Parameters>(
            nrows_, ncols_, total_nnz,
            starts.data(), indices.data(), data.data());
    }

    // -- Raw access -------------------------------------------------------

    const std::vector<triplet_type>& entries() const { return entries_; }

private:
    static bool by_position(const triplet_type& a, const triplet_type& b) {
        if (std::get<0>(a) != std::get<0>(b))
            return std::get<0>(a) < std::get<0>(b);
        return std::get<1>(a) < std::get<1>(b);
    }

    size_type nrows_, ncols_;
    std::vector<triplet_type> entries_;
    bool sorted_ = true;
};

} // namespace mtl::mat

namespace mtl::traits {
template <typename Value, typename Parameters>
struct category<mat::coordinate2D<Value, Parameters>> { using type = tag::sparse; };
} // namespace mtl::traits

namespace mtl::ashape {
template <typename Value, typename Parameters>
struct ashape<::mtl::mat::coordinate2D<Value, Parameters>> { using type = mat<Value>; };
} // namespace mtl::ashape

namespace mtl { using mat::coordinate2D; }
