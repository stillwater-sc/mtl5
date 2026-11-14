#pragma once
// MTL5 -- Dense 2D matrix (replaces MTL4 dense2D)
// Composition-based design with contiguous_memory_block, C++20 concepts
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <mtl/config.hpp>
#include <mtl/detail/exceptions.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/tag/storage.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/traits/is_expression.hpp>
#include <mtl/detail/contiguous_memory_block.hpp>
#include <mtl/detail/ewise.hpp>
#include <mtl/mat/expr/expr_alias.hpp>
#include <mtl/detail/index.hpp>
#include <mtl/mat/dimension.hpp>
#include <mtl/mat/parameter.hpp>

namespace mtl::mat {

/// Dense row/column-major matrix with configurable dimensions and storage.
template <typename Value, typename Parameters = parameters<>>
class dense2D {
public:
    // -- Parameter extraction --------------------------------------------
    using param_type   = Parameters;
    using orientation  = typename Parameters::orientation;
    using index_type   = typename Parameters::index_type;
    using dim_type     = typename Parameters::dimensions_type;
    using size_type    = typename Parameters::size_type;

    // Derive static size for memory block
    static constexpr bool is_fixed = dim_type::is_fixed;
    static constexpr std::size_t static_size = []() constexpr {
        if constexpr (is_fixed) return dim_type::rows * dim_type::cols;
        else return std::size_t{0};
    }();

    using actual_storage = typename Parameters::storage;
    using memory_type = detail::contiguous_memory_block<Value, actual_storage, static_size>;

    /// A read-only view (tag::read_only storage) over a caller's `const Value*`:
    /// only const access, so writing through it does not compile. It is still a
    /// dense2D<Value, Parameters>, so every operation that takes
    /// `const dense2D<V, P>&` accepts it. Copying it gives another view of the
    /// same buffer; it cannot be assigned to or resized.
    static constexpr bool read_only = std::is_same_v<actual_storage, tag::read_only>;

    // -- Standard type aliases -------------------------------------------
    using value_type      = Value;
    using reference       = std::conditional_t<read_only, const Value&, Value&>;
    using const_reference = const Value&;
    using pointer         = std::conditional_t<read_only, const Value*, Value*>;
    using const_pointer   = const Value*;

    // -- Offset computation ----------------------------------------------
private:
    static constexpr size_type compute_offset(size_type r, size_type c, size_type ldim) {
        if constexpr (std::is_same_v<orientation, tag::row_major>)
            return r * ldim + c;
        else
            return c * ldim + r;
    }

    void set_ldim() {
        if constexpr (std::is_same_v<orientation, tag::row_major>)
            ldim_ = num_cols();
        else
            ldim_ = num_rows();
    }

public:
    // -- Default constructor ---------------------------------------------
    dense2D() : mem_{}, dims_{}, ldim_{0} {
        if constexpr (is_fixed) set_ldim();
    }

    // -- (rows, cols) constructor ----------------------------------------
    dense2D(size_type rows, size_type cols) requires (!read_only)
        : mem_(rows * cols), dims_{}, ldim_{0}
    {
        if constexpr (!is_fixed) {
            dims_.set_dimensions(rows, cols);
        } else {
            assert(rows == dim_type::rows && cols == dim_type::cols
                   && "Size must match fixed dimensions");
        }
        set_ldim();
    }

    // -- Dimension object constructor ------------------------------------
    explicit dense2D(const dim_type& d) requires (!is_fixed && !read_only)
        : mem_(d.num_rows() * d.num_cols()), dims_(d), ldim_{0}
    {
        set_ldim();
    }

    // -- External pointer constructor ------------------------------------
    dense2D(size_type rows, size_type cols, Value* ptr) requires (!read_only)
        : mem_(ptr, rows * cols, false), dims_{}, ldim_{0}
    {
        if constexpr (!is_fixed) {
            dims_.set_dimensions(rows, cols);
        }
        set_ldim();
    }

    // -- Read-only view constructor --------------------------------------
    // Over a const buffer (or a mutable one, viewed read-only), rows x cols in
    // this orientation with no padding. The pointer is stored without const
    // only because the memory block is shared with the mutable matrices; no
    // member of a read-only matrix writes through it.
    dense2D(size_type rows, size_type cols, const Value* ptr) requires read_only
        : mem_(const_cast<Value*>(ptr), rows * cols, true), dims_{}, ldim_{0}
    {
        if constexpr (!is_fixed) {
            dims_.set_dimensions(rows, cols);
        } else {
            assert(rows == dim_type::rows && cols == dim_type::cols
                   && "Size must match fixed dimensions");
        }
        set_ldim();
    }

    // -- Initializer list of initializer lists ---------------------------
    // Every row must have the first row's length (#824): a longer one wrote
    // past the buffer, a shorter one left zeros.
    dense2D(std::initializer_list<std::initializer_list<Value>> il) requires (!read_only)
        : dense2D(il.size(), il.size() > 0 ? il.begin()->size() : 0)
    {
        for (const auto& row : il) {
            if (row.size() != num_cols())
                ::mtl::detail::throw_or_abort(
                    std::invalid_argument("dense2D: initializer-list rows differ in length"));
        }
        size_type r = 0;
        for (const auto& row : il) {
            size_type c = 0;
            for (const auto& val : row) {
                mem_[compute_offset(r, c, ldim_)] = val;
                ++c;
            }
            ++r;
        }
    }

    // -- Copy / Move -----------------------------------------------------
    dense2D(const dense2D&) requires (!read_only) = default;
    dense2D& operator=(const dense2D&) requires (!read_only) = default;
    ~dense2D() = default;

    /// An owning matrix from a read-only view: copies the elements, so the
    /// copy can be changed. The view may differ in orientation and index base.
    template <typename P2>
        requires (!read_only && !std::same_as<P2, Parameters>
                  && dense2D<Value, P2>::read_only)
    dense2D(const dense2D<Value, P2>& view)
        : dense2D(static_cast<size_type>(view.num_rows()), static_cast<size_type>(view.num_cols()))
    {
        using view_orientation = typename P2::orientation;
        const auto* src = view.data();
        const std::size_t ld = view.get_ldim();
        for (size_type r = 0; r < num_rows(); ++r)
            for (size_type c = 0; c < num_cols(); ++c) {
                const std::size_t off = std::is_same_v<view_orientation, tag::row_major>
                    ? std::size_t(r) * ld + std::size_t(c) : std::size_t(c) * ld + std::size_t(r);
                mem_[compute_offset(r, c, ldim_)] = src[off];
            }
    }

    /// Copying a read-only view gives another view of the same buffer.
    dense2D(const dense2D& other) requires read_only
        : mem_(const_cast<Value*>(other.data()), other.mem_.size(), true),
          dims_(other.dims_), ldim_(other.ldim_) {}
    dense2D& operator=(const dense2D&) requires read_only = delete;

    dense2D(dense2D&& other) noexcept(std::is_nothrow_move_constructible_v<memory_type>)
        : mem_(std::move(other.mem_)), dims_(other.dims_), ldim_(other.ldim_)
    {
        // A fixed-size source keeps its shape (and, on the heap, gets fresh
        // storage), so it keeps its leading dimension too.
        if constexpr (!is_fixed) {
            other.dims_.set_dimensions(0, 0);
            other.ldim_ = 0;
        }
    }

    /// A matrix over an external buffer with as many elements keeps its
    /// buffer and receives the elements, in the source's shape; the source
    /// keeps its own (#829).
    dense2D& operator=(dense2D&&) requires read_only = delete;
    dense2D& operator=(dense2D&& other) noexcept requires (!read_only) {
        if (this != &other) {
            const bool in_place = mem_.assigns_in_place(other.mem_.size());
            mem_ = std::move(other.mem_);
            dims_ = other.dims_;
            ldim_ = other.ldim_;
            if (!in_place) {
                if constexpr (!is_fixed) {
                    other.dims_.set_dimensions(0, 0);
                    other.ldim_ = 0;
                }
            }
        }
        return *this;
    }

    // -- Expression template construction / assignment -------------------
    template <typename Expr>
        requires (!read_only && Matrix<Expr> && traits::is_expression_v<Expr>
                  && std::convertible_to<typename Expr::value_type, Value>)
    dense2D(const Expr& expr)
        : dense2D(static_cast<size_type>(expr.num_rows()),
                   static_cast<size_type>(expr.num_cols()))
    {
        // Element-wise sweep over the FLATTENED (r,c) space: each element is
        // independent, so the contiguous element chunking is bit-identical to the
        // serial nested loop, and a wide/short expression splits just like a tall
        // one (serial at MTL5_NUM_THREADS=1). See mtl/detail/ewise.hpp.
        detail::parallel_ewise_2d(num_rows(), num_cols(), 1,
                                  [&](std::size_t r, std::size_t c) {
            (*this)(r, c) = static_cast<Value>(expr(r, c));
        });
    }

    template <typename Expr>
        requires (!read_only && Matrix<Expr> && traits::is_expression_v<Expr>
                  && std::convertible_to<typename Expr::value_type, Value>)
    dense2D& operator=(const Expr& expr) {
        // A matrix over an external buffer is never re-pointed by assignment:
        // an expression with another element count throws. change_dim on it,
        // an explicit resize, does detach it (#829).
        if (!owns_memory()
            && static_cast<size_type>(expr.num_rows()) * static_cast<size_type>(expr.num_cols()) != size())
            ::mtl::detail::throw_or_abort(
                std::domain_error("dense2D: cannot assign a matrix of another size to an external-pointer matrix"));
        // An expression that reads this matrix other than at (r, c) -- through
        // trans(), another view or a lazy product -- is evaluated into a
        // temporary first (#824, mat/expr/expr_alias.hpp).
        if (reads_this_elsewhere(expr)) {
            const dense2D tmp(expr);
            change_dim(tmp.num_rows(), tmp.num_cols());
            assign_elements(tmp);
            return *this;
        }
        change_dim(static_cast<size_type>(expr.num_rows()),
                   static_cast<size_type>(expr.num_cols()));
        detail::parallel_ewise_2d(num_rows(), num_cols(), 1,
                                  [&](std::size_t r, std::size_t c) {
            (*this)(r, c) = static_cast<Value>(expr(r, c));
        });
        return *this;
    }

    template <typename Expr>
        requires (!read_only && Matrix<Expr> && traits::is_expression_v<Expr>
                  && std::convertible_to<typename Expr::value_type, Value>)
    dense2D& operator+=(const Expr& expr) {
        assert(num_rows() == expr.num_rows() && num_cols() == expr.num_cols());
        if (reads_this_elsewhere(expr)) {   // as in operator=(expr)
            const dense2D tmp(expr);
            detail::parallel_ewise_2d(num_rows(), num_cols(), 1,
                                      [&](std::size_t r, std::size_t c) {
                (*this)(r, c) += tmp(r, c);
            });
            return *this;
        }
        detail::parallel_ewise_2d(num_rows(), num_cols(), 1,
                                  [&](std::size_t r, std::size_t c) {
            (*this)(r, c) += static_cast<Value>(expr(r, c));
        });
        return *this;
    }

    template <typename Expr>
        requires (!read_only && Matrix<Expr> && traits::is_expression_v<Expr>
                  && std::convertible_to<typename Expr::value_type, Value>)
    dense2D& operator-=(const Expr& expr) {
        assert(num_rows() == expr.num_rows() && num_cols() == expr.num_cols());
        if (reads_this_elsewhere(expr)) {   // as in operator=(expr)
            const dense2D tmp(expr);
            detail::parallel_ewise_2d(num_rows(), num_cols(), 1,
                                      [&](std::size_t r, std::size_t c) {
                (*this)(r, c) -= tmp(r, c);
            });
            return *this;
        }
        detail::parallel_ewise_2d(num_rows(), num_cols(), 1,
                                  [&](std::size_t r, std::size_t c) {
            (*this)(r, c) -= static_cast<Value>(expr(r, c));
        });
        return *this;
    }

private:
    /// Whether expr reads this matrix anywhere but at the element it writes.
    template <typename Expr>
    bool reads_this_elsewhere(const Expr& expr) const {
        const auto* lo = reinterpret_cast<const unsigned char*>(mem_.data());
        const detail::alias_target t{this, lo, lo + size() * sizeof(Value)};
        return detail::alias_reads_target_elsewhere(expr, t);
    }

    /// Copy src's elements into this matrix, of the same shape, in place:
    /// the storage is kept, so an external-pointer matrix keeps writing its
    /// caller's buffer.
    void assign_elements(const dense2D& src) {
        detail::parallel_ewise_2d(num_rows(), num_cols(), 1,
                                  [&](std::size_t r, std::size_t c) {
            (*this)(r, c) = src(r, c);
        });
    }

public:
    // -- Element access --------------------------------------------------
    Value& operator()(size_type r, size_type c) requires (!read_only) {
        auto ri = index_type::to_internal(r);
        auto ci = index_type::to_internal(c);
        if constexpr (bounds_checking) {
            if (ri >= num_rows() || ci >= num_cols())
                ::mtl::detail::throw_or_abort(std::out_of_range("dense2D: index out of range"));
        }
        return mem_[compute_offset(ri, ci, ldim_)];
    }

    const_reference operator()(size_type r, size_type c) const {
        auto ri = index_type::to_internal(r);
        auto ci = index_type::to_internal(c);
        if constexpr (bounds_checking) {
            if (ri >= num_rows() || ci >= num_cols())
                ::mtl::detail::throw_or_abort(std::out_of_range("dense2D: index out of range"));
        }
        return mem_[compute_offset(ri, ci, ldim_)];
    }

    // -- Size / shape ----------------------------------------------------
    size_type num_rows() const {
        if constexpr (is_fixed) return dim_type::rows;
        else return dims_.num_rows();
    }

    size_type num_cols() const {
        if constexpr (is_fixed) return dim_type::cols;
        else return dims_.num_cols();
    }

    size_type size() const { return num_rows() * num_cols(); }

    size_type get_ldim() const { return ldim_; }

    // -- Data access -----------------------------------------------------
    /// Whether this matrix owns its elements: false for one made over an
    /// external pointer, whose elements belong to the caller.
    bool owns_memory() const noexcept {
        return mem_.category() == ::mtl::detail::memory_category::own;
    }

    Value*        data()        requires (!read_only) { return mem_.data(); }
    const_pointer data() const { return mem_.data(); }

    Value*        begin()       requires (!read_only) { return data(); }
    const_pointer begin() const { return data(); }
    Value*        end()         requires (!read_only) { return data() + size(); }
    const_pointer end()   const { return data() + size(); }

    // -- Dimension modification ------------------------------------------
    void change_dim(size_type r, size_type c) requires (!read_only) {
        if constexpr (is_fixed) {
            assert(r == dim_type::rows && c == dim_type::cols
                   && "Cannot change fixed dimensions");
        } else {
            auto new_sz = r * c;
            if (new_sz != size()) {
                mem_.realloc(new_sz);
            }
            dims_.set_dimensions(r, c);
            set_ldim();
        }
    }

    void checked_change_dim(size_type r, size_type c) requires (!read_only) {
        if constexpr (is_fixed) {
            if (r != dim_type::rows || c != dim_type::cols)
                ::mtl::detail::throw_or_abort(std::domain_error("dense2D: cannot resize fixed-size matrix"));
        } else {
            change_dim(r, c);
        }
    }

    // -- Compound assignment operators -----------------------------------
    dense2D& operator+=(const dense2D& other) requires (!read_only) {
        assert(num_rows() == other.num_rows() && num_cols() == other.num_cols());
        for (size_type i = 0; i < size(); ++i) mem_[i] += other.mem_[i];
        return *this;
    }

    dense2D& operator-=(const dense2D& other) requires (!read_only) {
        assert(num_rows() == other.num_rows() && num_cols() == other.num_cols());
        for (size_type i = 0; i < size(); ++i) mem_[i] -= other.mem_[i];
        return *this;
    }

    template <typename S> requires (!read_only && Scalar<S>)
    dense2D& operator*=(const S& alpha) {
        for (size_type i = 0; i < size(); ++i) mem_[i] *= alpha;
        return *this;
    }

    template <typename S> requires (!read_only && Field<S>)
    dense2D& operator/=(const S& alpha) {
        for (size_type i = 0; i < size(); ++i) mem_[i] /= alpha;
        return *this;
    }

    // -- Swap ------------------------------------------------------------
    void swap(dense2D& other) noexcept requires (!read_only) {
        mem_.swap(other.mem_);
        if constexpr (!is_fixed) {
            auto tr = dims_.num_rows(), tc = dims_.num_cols();
            dims_.set_dimensions(other.dims_.num_rows(), other.dims_.num_cols());
            other.dims_.set_dimensions(tr, tc);
        }
        using std::swap;
        swap(ldim_, other.ldim_);
    }

private:
    memory_type mem_;
    [[no_unique_address]] dim_type dims_;
    size_type ldim_;
};

// -- Read-only views --------------------------------------------------------

/// A read-only dense matrix over memory the caller owns (tag::read_only):
/// const access only, no allocation, no copy of the elements. The buffer
/// holds rows x cols elements in Orientation order with no padding.
template <typename Value, typename Orientation = tag::row_major>
using const_matrix_view = dense2D<Value,
    parameters<Orientation, detail::c_index, non_fixed::dimensions, tag::read_only>>;

/// A read-only row-major view of the rows x cols elements at ptr. For a
/// column-major buffer, use const_matrix_view<Value, tag::col_major> directly.
/// The buffer must outlive the view.
template <typename Value>
const_matrix_view<Value> const_view(std::size_t rows, std::size_t cols, const Value* ptr) {
    return const_matrix_view<Value>(rows, cols, ptr);
}

/// A read-only view of A's elements, in A's orientation and index base: hand
/// a matrix to code that must not change it, without copying it. A must
/// outlive the view.
template <typename Value, typename Parameters>
auto const_view(const dense2D<Value, Parameters>& A) {
    using view_params = parameters<typename Parameters::orientation, typename Parameters::index_type,
                                   non_fixed::dimensions, tag::read_only, typename Parameters::size_type>;
    return dense2D<Value, view_params>(
        static_cast<typename view_params::size_type>(A.num_rows()),
        static_cast<typename view_params::size_type>(A.num_cols()), A.data());
}

/// A view of a temporary would dangle at the end of the full-expression.
template <typename Value, typename Parameters>
void const_view(const dense2D<Value, Parameters>&&) = delete;

// -- Free functions in mtl::mat ------------------------------------------

template <typename Value, typename Parameters>
auto num_rows(const dense2D<Value, Parameters>& m) { return m.num_rows(); }

template <typename Value, typename Parameters>
auto num_cols(const dense2D<Value, Parameters>& m) { return m.num_cols(); }

template <typename Value, typename Parameters>
auto size(const dense2D<Value, Parameters>& m) { return m.size(); }

} // namespace mtl::mat

// -- Traits specializations ---------------------------------------------

namespace mtl::traits {

template <typename Value, typename Parameters>
struct category<mat::dense2D<Value, Parameters>> {
    using type = tag::dense;
};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename Value, typename Parameters>
struct ashape<::mtl::mat::dense2D<Value, Parameters>> {
    using type = mat<Value>;
};

} // namespace mtl::ashape

// -- Convenience alias --------------------------------------------------
namespace mtl { using mat::dense2D; }
