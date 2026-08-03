#pragma once
// MTL5 -- ELLPACK sparse matrix format
// Fixed-width per-ROW storage, and now enforced -- see the static_assert below
// (#355). Good for GPU and matrices with uniform row lengths.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <mtl/tag/orientation.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::mat {

/// ELLPACK sparse matrix: each row stores up to max_width non-zeros.
/// Two flat arrays of size nrows * max_width: indices and values.
/// Unused slots have index == size_type(-1).
template <typename Value, typename Parameters = parameters<>>
class ell_matrix {
public:
    using param_type      = Parameters;   // as compressed2D exposes it
    using value_type      = Value;
    using size_type       = typename Parameters::size_type;
    using const_reference = const Value&;
    using reference       = Value&;

    static constexpr size_type invalid = size_type(-1);

    // ELL here is row-padded, unconditionally: both arrays are nrows*width and
    // every access is indices_[r * width_ + k]. Nothing in this container has
    // ever read Parameters::orientation, so a col_major instantiation was
    // accepted silently while being byte-for-byte a row-major ELL -- the same
    // inert-tag defect as compressed2D (#355).
    //
    // The severity here is lower, and worth stating precisely rather than by
    // analogy: the ONLY fill path is the compressed2D constructor below, which
    // for col_major parameters is now itself a compile error, so no col_major
    // ELL can currently be populated and none can be returning wrong answers.
    // What it can do today is be constructed, empty, while claiming a layout it
    // does not implement -- and the moment an inserter or a setter is added,
    // the hole reopens silently. Close it now, at the point of misuse, rather
    // than leaving callers to hit a confusing diagnostic about compressed2D.
    static_assert(std::is_same_v<typename Parameters::orientation, tag::row_major>,
        "ell_matrix is row-padded ELLPACK storage only; a col_major instantiation "
        "would be byte-identical to the row-major layout and silently misdescribe "
        "it (#355). Use the default parameters, and mtl::trans for a transposed "
        "sparse product.");

    /// Default: empty
    ell_matrix() : nrows_(0), ncols_(0), width_(0) {}

    /// Empty nrows x ncols with given max row width
    ell_matrix(size_type nrows, size_type ncols, size_type max_width)
        : nrows_(nrows), ncols_(ncols), width_(max_width),
          indices_(nrows * max_width, invalid),
          data_(nrows * max_width, math::zero<Value>()) {}

    /// Construct from compressed2D (auto-detect max row width)
    explicit ell_matrix(const compressed2D<Value, Parameters>& crs)
        : nrows_(crs.num_rows()), ncols_(crs.num_cols()) {
        const auto& starts = crs.ref_major();
        const auto& col_idx = crs.ref_minor();
        const auto& vals = crs.ref_data();

        // Find max row width
        width_ = 0;
        for (size_type i = 0; i < nrows_; ++i)
            width_ = std::max(width_, starts[i + 1] - starts[i]);

        indices_.assign(nrows_ * width_, invalid);
        data_.assign(nrows_ * width_, math::zero<Value>());

        // Fill from CRS
        for (size_type i = 0; i < nrows_; ++i) {
            size_type slot = 0;
            for (size_type k = starts[i]; k < starts[i + 1]; ++k, ++slot) {
                indices_[i * width_ + slot] = col_idx[k];
                data_[i * width_ + slot] = vals[k];
            }
        }
    }

    // -- Element access --------------------------------------------------

    value_type operator()(size_type r, size_type c) const {
        assert(r < nrows_ && c < ncols_);
        for (size_type k = 0; k < width_; ++k) {
            size_type idx = indices_[r * width_ + k];
            if (idx == invalid) break;
            if (idx == c) return data_[r * width_ + k];
        }
        return math::zero<Value>();
    }

    // -- Size / shape ----------------------------------------------------

    size_type num_rows()  const { return nrows_; }
    size_type num_cols()  const { return ncols_; }
    size_type size()      const { return nrows_ * ncols_; }
    size_type max_width() const { return width_; }

    // -- Raw access ------------------------------------------------------

    const std::vector<size_type>& ref_indices() const { return indices_; }
    const std::vector<Value>&     ref_data()    const { return data_; }

private:
    size_type nrows_, ncols_, width_;
    std::vector<size_type> indices_;  // nrows * width
    std::vector<Value>     data_;     // nrows * width
};

} // namespace mtl::mat

namespace mtl::traits {
template <typename Value, typename Parameters>
struct category<mat::ell_matrix<Value, Parameters>> { using type = tag::sparse; };
} // namespace mtl::traits

namespace mtl::ashape {
template <typename Value, typename Parameters>
struct ashape<::mtl::mat::ell_matrix<Value, Parameters>> { using type = mat<Value>; };
} // namespace mtl::ashape

namespace mtl { using mat::ell_matrix; }
