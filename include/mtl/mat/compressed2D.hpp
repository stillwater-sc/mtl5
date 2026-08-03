#pragma once
// MTL5 -- CRS (Compressed Row Storage) sparse matrix
// Row-major ONLY, and now enforced -- see the static_assert below (#355).
// Three-array storage: data_, indices_, starts_.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <mtl/config.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::mat {

/// CRS sparse matrix: three arrays -- data (values), indices (column indices),
/// starts (row pointers of length nrows+1).
template <typename Value, typename Parameters = parameters<>>
class compressed2D {
public:
    using param_type      = Parameters;
    using value_type      = Value;
    using size_type       = typename Parameters::size_type;
    using const_reference = const Value&;
    using reference       = Value&;

    // compressed2D is CSR, unconditionally: `starts_` is indexed by ROW
    // everywhere -- operator(), the array constructor's own assert, the
    // inserter and every mult kernel. Nothing in this container has ever read
    // Parameters::orientation.
    //
    // A col_major instantiation was therefore accepted silently and was
    // byte-for-byte a CSR matrix. That is worse than unsupported: a caller who
    // populated it from genuine CSC arrays got the TRANSPOSE while the
    // constructor reported success -- mult returned A^T*x, and A(0,2) returned
    // the value belonging at A(2,0), with no diagnostic anywhere (#355).
    //
    // Reject it at compile time instead. This costs existing callers nothing:
    // no col_major compressed2D can have been correct, so none can exist.
    static_assert(std::is_same_v<typename Parameters::orientation, tag::row_major>,
        "compressed2D is CSR (row-major) storage only; a col_major instantiation "
        "would be byte-identical to CSR and would silently read CSC input as its "
        "transpose (#355). Convert CSC to CSR at the boundary, and use mtl::trans "
        "for a transposed sparse product.");

    // -- Constructors ----------------------------------------------------

    /// Default: empty 0x0 matrix
    compressed2D() : nrows_(0), ncols_(0), starts_(1, size_type(0)) {}

    /// Construct empty nrows x ncols matrix
    compressed2D(size_type nrows, size_type ncols)
        : nrows_(nrows), ncols_(ncols), starts_(nrows + 1, size_type(0)) {}

    /// Construct from raw CSR arrays (copies data)
    compressed2D(size_type nrows, size_type ncols, size_type nnz_count,
                 const size_type* starts, const size_type* indices, const Value* data)
        : nrows_(nrows), ncols_(ncols),
          starts_(starts, starts + nrows + 1),
          indices_(indices, indices + nnz_count),
          data_(data, data + nnz_count)
    {
        assert(starts_[nrows_] == nnz_count);
    }

    // -- Element access --------------------------------------------------

    /// Read-only access: binary search within row r for column c.
    /// Returns zero if element is absent.
    value_type operator()(size_type r, size_type c) const {
        assert(r < nrows_ && c < ncols_);
        auto begin = indices_.begin() + starts_[r];
        auto end   = indices_.begin() + starts_[r + 1];
        auto it = std::lower_bound(begin, end, c);
        if (it != end && *it == c) {
            return data_[it - indices_.begin()];
        }
        return math::zero<value_type>();
    }

    // -- Size / shape ----------------------------------------------------

    size_type num_rows() const { return nrows_; }
    size_type num_cols() const { return ncols_; }
    size_type size()     const { return nrows_ * ncols_; }
    size_type nnz()      const { return data_.size(); }

    // -- Raw CRS access --------------------------------------------------

    const std::vector<size_type>& ref_major() const { return starts_; }
    std::vector<size_type>&       ref_major()       { return starts_; }

    const std::vector<size_type>& ref_minor() const { return indices_; }
    std::vector<size_type>&       ref_minor()       { return indices_; }

    const std::vector<Value>&     ref_data()  const { return data_; }
    std::vector<Value>&           ref_data()        { return data_; }

    // -- Mutation --------------------------------------------------------

    void change_dim(size_type r, size_type c) {
        nrows_ = r;
        ncols_ = c;
        starts_.assign(r + 1, size_type(0));
        indices_.clear();
        data_.clear();
    }

    void make_empty() {
        starts_.assign(nrows_ + 1, size_type(0));
        indices_.clear();
        data_.clear();
    }

private:
    size_type              nrows_;
    size_type              ncols_;
    std::vector<size_type> starts_;   // length nrows_+1
    std::vector<size_type> indices_;  // column indices
    std::vector<Value>     data_;     // values
};

} // namespace mtl::mat

// -- Traits specializations ---------------------------------------------

namespace mtl::traits {

template <typename Value, typename Parameters>
struct category<mat::compressed2D<Value, Parameters>> {
    using type = tag::sparse;
};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename Value, typename Parameters>
struct ashape<::mtl::mat::compressed2D<Value, Parameters>> {
    using type = mat<Value>;
};

} // namespace mtl::ashape

// -- Convenience alias --------------------------------------------------
namespace mtl { using mat::compressed2D; }
