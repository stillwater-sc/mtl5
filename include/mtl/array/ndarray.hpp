#pragma once
// MTL5 -- N-dimensional array (ndarray) core type
//
// ndarray<T, N, Order> provides a NumPy-compatible multi-dimensional array:
//   - Static rank N, runtime shape and strides
//   - Owning (heap-allocated) and non-owning (view) modes
//   - C-order (row-major, default) or F-order (column-major)
//   - Element access via operator()(i, j, k, ...)
//   - Slicing returns views (no copies)
//
// Follows MTL5 patterns: composition with contiguous_memory_block,
// [[no_unique_address]], C++20 concepts, bounds checking in debug builds.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <mtl/config.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/traits/is_expression.hpp>
#include <mtl/detail/contiguous_memory_block.hpp>
#include <mtl/detail/exceptions.hpp>
#include <mtl/array/shape.hpp>

namespace mtl::array {

namespace detail_nd {

/// Calls f(off) for every element of an array of shape `sh`, in Order's index
/// order -- c_order: the last index fastest; f_order: the first -- where off[k]
/// is the element's offset under the strides st[k]. One walk serves several
/// arrays of one shape (a copy, a broadcast, an axis reduction). The innermost
/// dimension is a plain loop; the outer ones advance an odometer, so there is
/// no index array to build or check per element (#856).
template <typename Order, std::size_t N, std::size_t K, typename F>
void walk_offsets(const shape<N>& sh, const std::array<std::array<std::size_t, N>, K>& st,
                  F&& f) {
    std::array<std::size_t, K> off{};
    if constexpr (N == 0) {
        f(off);   // rank 0: one element, at offset 0
    } else {
        if (sh.total_size() == 0) return;
        constexpr bool c = !std::is_same_v<Order, f_order>;
        constexpr auto dim = [](std::size_t level) { return c ? level : N - 1 - level; };
        constexpr std::size_t in = dim(N - 1);   // the fastest-varying dimension
        const std::size_t n_in = sh[in];
        std::array<std::size_t, N> idx{};
        for (;;) {
            std::array<std::size_t, K> o = off;
            for (std::size_t i = 0; i < n_in; ++i) {
                f(std::as_const(o));
                for (std::size_t q = 0; q < K; ++q) o[q] += st[q][in];
            }
            std::size_t level = N - 1;   // advance the outer dimensions
            for (;;) {
                if (level == 0) return;
                --level;
                const std::size_t d = dim(level);
                if (++idx[d] < sh[d]) {
                    for (std::size_t q = 0; q < K; ++q) off[q] += st[q][d];
                    break;
                }
                for (std::size_t q = 0; q < K; ++q) off[q] -= (sh[d] - 1) * st[q][d];
                idx[d] = 0;
            }
        }
    }
}

} // namespace detail_nd

/// N-dimensional array with static rank and runtime shape.
///
/// Its logical order -- the order of iterate_indices, for_each_element, reduce
/// and flatten -- is its own Order: c_order arrays walk the last index fastest,
/// f_order arrays the first, whatever the strides of a view say. reshape reads
/// in the same order (#359), so flatten(a) and a.reshape({a.size()}) agree.
///
/// @tparam Value  element type (must be default-constructible)
/// @tparam N      number of dimensions (compile-time rank)
/// @tparam Order  memory layout: c_order (default) or f_order
template <typename Value, std::size_t N, typename Order = c_order>
class ndarray {
public:
    // -- Type aliases --------------------------------------------------------
    using value_type      = Value;
    using reference       = Value&;
    using const_reference = const Value&;
    using pointer         = Value*;
    using const_pointer   = const Value*;
    using size_type       = std::size_t;
    using shape_type      = ::mtl::array::shape<N>;
    using strides_type    = std::array<size_type, N>;

    static constexpr size_type rank = N;

private:
    using memory_type = detail::contiguous_memory_block<Value, tag::on_heap, 0>;

    shape_type   shape_;
    strides_type strides_{};
    memory_type  mem_;

    constexpr void init_strides() {
        strides_ = compute_strides<Order>(shape_);
    }

public:
    // -- Constructors --------------------------------------------------------

    /// Default: empty array with zero extents.
    ndarray() = default;

    /// Construct from shape, zero-initialized.
    explicit ndarray(shape_type sh)
        : mem_(sh.total_size()), shape_(sh) {
        init_strides();
    }

    /// Construct from initializer list of extents.
    ndarray(std::initializer_list<size_type> extents)
        : shape_(extents), mem_(shape_.total_size()) {
        init_strides();
    }

    /// Construct from shape + fill value.
    ndarray(shape_type sh, const Value& val)
        : mem_(sh.total_size()), shape_(sh) {
        init_strides();
        std::fill_n(mem_.data(), mem_.size(), val);
    }

    /// Non-owning view: wraps external memory with given shape and strides.
    ndarray(Value* ptr, shape_type sh, strides_type strides)
        : mem_(ptr, sh.total_size(), /*is_view=*/true),
          shape_(sh), strides_(strides) {}

    /// Non-owning view: wraps external memory with contiguous layout.
    ndarray(Value* ptr, shape_type sh)
        : mem_(ptr, sh.total_size(), /*is_view=*/true),
          shape_(sh) {
        init_strides();
    }

    // -- Copy and move -------------------------------------------------------
    //
    // A copy owns a contiguous array in this Order, filled element by element,
    // so copying a strided view (a slice, a transpose) reads its elements, not
    // the raw memory from its base pointer.
    //
    // Assignment to a view writes the viewed elements and keeps the view's
    // memory, shape and strides (#829). The source must have the same shape:
    // copy assignment throws std::domain_error otherwise; move assignment is
    // noexcept, so it aborts, in every build (#856) -- it used to assert, and
    // under NDEBUG take over the source's storage, detaching the view.
    //
    // Assignment to an owning array replaces its contents. A view source is
    // copied, so the array stays owning: own = a.transpose() or
    // own = std::move(view) used to hand over the view, leaving own aliasing
    // another array's buffer (#856). Only an owning source's storage is taken.

    ndarray(const ndarray& other) : shape_(other.shape_), mem_(other.size()) {
        init_strides();
        copy_elements_from(other);
    }

    /// The source is left empty: no storage, a zero shape.
    ndarray(ndarray&& other) noexcept
        : shape_(other.shape_), strides_(other.strides_), mem_(std::move(other.mem_)) {
        other.shape_   = shape_type{};
        other.strides_ = strides_type{};
    }

    ndarray& operator=(const ndarray& other) {
        if (this == &other) return *this;
        if (is_view()) {
            if (shape_ != other.shape_)
                ::mtl::detail::throw_or_abort(
                    std::domain_error("ndarray: cannot assign an array of another shape to a view"));
            assign_to_view(other);
            return *this;
        }
        ndarray tmp(other);
        *this = std::move(tmp);
        return *this;
    }

    ndarray& operator=(ndarray&& other) noexcept {
        if (this == &other) return *this;
        if (is_view()) {
            if (shape_ != other.shape_)
                ::mtl::detail::contract_abort(
                    "ndarray: move-assigning an array of another shape to a view");
            assign_to_view(other);
            return *this;
        }
        if (other.is_view()) {
            // An owning array stays owning: copy the view's elements. This also
            // covers a view into our own storage (a = a.transpose()), which
            // would dangle once that storage is released. In a noexcept
            // function, a copy that fails to allocate terminates.
            ndarray owned(other);
            return *this = std::move(owned);
        }
        // Take over other's storage, shape and strides.
        memory_type released;
        mem_.swap(released);
        mem_.swap(other.mem_);
        shape_   = other.shape_;
        strides_ = other.strides_;
        other.shape_   = shape_type{};
        other.strides_ = strides_type{};
        return *this;
    }

    // -- Shape and size ------------------------------------------------------

    constexpr const shape_type&   get_shape()   const { return shape_; }
    constexpr const strides_type& get_strides() const { return strides_; }

    constexpr size_type extent(size_type dim) const { return shape_[dim]; }

    constexpr size_type size() const { return shape_.total_size(); }
    constexpr bool      empty() const { return size() == 0; }

    /// Check if this array owns its memory.
    bool is_view() const {
        return mem_.category() == detail::memory_category::view ||
               mem_.category() == detail::memory_category::external;
    }

    /// True when the elements occupy a dense block of `size()` values, in
    /// EITHER C or F order.
    ///
    /// This is the right question for anything order-INdependent -- BLAS
    /// dispatch, fill, a commutative reduction -- and the WRONG one for
    /// anything that walks memory as though it were logical order. A transpose
    /// of a C-order array is F-contiguous, so this returns true while
    /// memory order and logical order disagree (#359). Use
    /// is_c_contiguous() / is_f_contiguous() when the distinction matters.
    bool is_contiguous() const {
        return ::mtl::array::is_contiguous(shape_, strides_);
    }

    /// True when memory is contiguous in C (row-major) order.
    bool is_c_contiguous() const {
        return ::mtl::array::is_contiguous_c(shape_, strides_);
    }

    /// True when memory is contiguous in F (column-major) order.
    /// For N <= 1 this coincides with is_c_contiguous().
    bool is_f_contiguous() const {
        return ::mtl::array::is_contiguous_f(shape_, strides_);
    }

    /// True when memory order matches THIS array's logical iteration order,
    /// i.e. the order its own `Order` parameter implies. This is the predicate
    /// every order-dependent operation wants, and the one whose absence caused
    /// reshape and flatten to return elements in memory order (#359).
    bool is_contiguous_in_own_order() const {
        if constexpr (std::is_same_v<Order, c_order>) {
            return is_c_contiguous();
        } else {
            return is_f_contiguous();
        }
    }

    // -- Element access ------------------------------------------------------

    /// Multi-index access: a(i, j, k, ...)
    template <typename... Indices>
        requires (sizeof...(Indices) == N)
    reference operator()(Indices... indices) {
        if constexpr (bounds_checking) {
            check_bounds({static_cast<size_type>(indices)...});
        }
        const size_type off = offset_from<N>(strides_, indices...);
        return mem_[off];
    }

    template <typename... Indices>
        requires (sizeof...(Indices) == N)
    const_reference operator()(Indices... indices) const {
        if constexpr (bounds_checking) {
            check_bounds({static_cast<size_type>(indices)...});
        }
        const size_type off = offset_from<N>(strides_, indices...);
        return mem_[off];
    }

    /// Array-index access: a[{i, j, k}]
    reference operator[](const std::array<size_type, N>& idx) {
        if constexpr (bounds_checking) check_bounds(idx);
        return mem_[compute_offset(idx, strides_)];
    }

    const_reference operator[](const std::array<size_type, N>& idx) const {
        if constexpr (bounds_checking) check_bounds(idx);
        return mem_[compute_offset(idx, strides_)];
    }

    /// Flat (linear) index access - only valid for contiguous arrays.
    reference flat(size_type i) {
        assert(i < size());
        return mem_[i];
    }

    const_reference flat(size_type i) const {
        assert(i < size());
        return mem_[i];
    }

    // -- Data access ---------------------------------------------------------

    pointer       data()       { return mem_.data(); }
    const_pointer data() const { return mem_.data(); }

    pointer       begin()       { return mem_.begin(); }
    const_pointer begin() const { return mem_.begin(); }
    pointer       end()         { return mem_.end(); }
    const_pointer end()   const { return mem_.end(); }

    // -- Fill ----------------------------------------------------------------

    void fill(const Value& val) {
        if (is_contiguous()) {
            std::fill_n(mem_.data(), size(), val);
        } else {
            Value* p = mem_.data();
            detail_nd::walk_offsets<Order>(shape_, std::array<strides_type, 1>{strides_},
                                           [&](const auto& o) { p[o[0]] = val; });
        }
    }

    // -- View creation -------------------------------------------------------

    /// Create a view (non-owning) of this array with the same shape and strides.
    ndarray view() {
        return ndarray(mem_.data(), shape_, strides_);
    }

    /// Reshape: a view over the same elements, read in this array's Order.
    /// Throws std::invalid_argument if the size differs (#856: it was an
    /// assert, so under NDEBUG a 3-element array reshaped to 100 elements gave
    /// a view past its buffer), and std::runtime_error if the array is not
    /// contiguous in its own order.
    template <std::size_t M>
    ndarray<Value, M, Order> reshape(::mtl::array::shape<M> new_shape) {
        check_reshape(new_shape.total_size());
        return ndarray<Value, M, Order>(mem_.data(), new_shape);
    }

    /// Reshape of a const array: a read-only view (#856: it used to return a
    /// mutable view through a const_cast).
    template <std::size_t M>
    ndarray<const Value, M, Order> reshape(::mtl::array::shape<M> new_shape) const {
        check_reshape(new_shape.total_size());
        return ndarray<const Value, M, Order>(mem_.data(), new_shape);
    }

    /// Transpose: a view with reversed shape and strides.
    ndarray<Value, N, Order> transpose() {
        return ndarray<Value, N, Order>(mem_.data(), reversed(shape_), reversed(strides_));
    }

    /// Transpose of a const array: a read-only view (#856).
    ndarray<const Value, N, Order> transpose() const {
        return ndarray<const Value, N, Order>(mem_.data(), reversed(shape_), reversed(strides_));
    }

    // -- Scalar compound assignment ------------------------------------------

    ndarray& operator+=(const Value& s) {
        for_each_element([&](Value& v) { v += s; });
        return *this;
    }

    ndarray& operator-=(const Value& s) {
        for_each_element([&](Value& v) { v -= s; });
        return *this;
    }

    ndarray& operator*=(const Value& s) {
        for_each_element([&](Value& v) { v *= s; });
        return *this;
    }

    ndarray& operator/=(const Value& s) {
        for_each_element([&](Value& v) { v /= s; });
        return *this;
    }

    // -- Utility -------------------------------------------------------------

    /// Iterate over all valid multi-indices, calling f(idx) for each, in this
    /// array's Order (the last index fastest for c_order, the first for f_order).
    template <typename F>
    void iterate_indices(F&& f) const {
        std::array<size_type, N> idx{};
        iterate_impl(f, idx, 0);
    }

    /// Apply f to every element, in this array's Order. Memory that is
    /// contiguous in that order is walked directly; anything else -- a slice,
    /// a transposed view -- by strides, the innermost dimension a plain loop.
    template <typename F>
    void for_each_element(F&& f) {
        if (is_contiguous_in_own_order()) {
            Value* p = mem_.data();
            for (size_type i = 0; i < size(); ++i) f(p[i]);
        } else {
            Value* p = mem_.data();
            detail_nd::walk_offsets<Order>(shape_, std::array<strides_type, 1>{strides_},
                                           [&](const auto& o) { f(p[o[0]]); });
        }
    }

    template <typename F>
    void for_each_element(F&& f) const {
        if (is_contiguous_in_own_order()) {
            const Value* p = mem_.data();
            for (size_type i = 0; i < size(); ++i) f(p[i]);
        } else {
            const Value* p = mem_.data();
            detail_nd::walk_offsets<Order>(shape_, std::array<strides_type, 1>{strides_},
                                           [&](const auto& o) { f(p[o[0]]); });
        }
    }

private:
    /// The bytes from this array's first element to its last, in memory
    /// order; empty for an array with no elements.
    std::pair<const unsigned char*, const unsigned char*> byte_span() const {
        if (size() == 0 || mem_.data() == nullptr) return {nullptr, nullptr};
        size_type last = 0;
        for (size_type i = 0; i < N; ++i) last += (shape_[i] - 1) * strides_[i];
        const auto* lo = reinterpret_cast<const unsigned char*>(mem_.data());
        return {lo, lo + (last + 1) * sizeof(Value)};
    }

    /// Whether the two arrays may share an element. Their memory spans are
    /// compared first. When the spans overlap and the two have one shape and
    /// one set of strides -- two columns of a matrix, a(all, 0) and a(all, 1)
    /// -- the answer is exact (#856: two interleaved columns used to count as
    /// overlapping, so assigning one to the other always copied through a
    /// temporary). Otherwise it is conservative: an overlap of spans counts.
    bool shares_memory_with(const ndarray& other) const {
        const auto [a0, a1] = byte_span();
        const auto [b0, b1] = other.byte_span();
        if (a0 == nullptr || b0 == nullptr) return false;
        const std::less<const unsigned char*> lt;
        if (!(lt(a0, b1) && lt(b0, a1))) return false;
        if (shape_ != other.shape_ || strides_ != other.strides_) return true;
        // One layout, displaced by d elements: they share an element iff
        // d = sum k_i * stride_i with |k_i| < extent_i. When each stride is at
        // least the span of the smaller ones (true of every view of a
        // contiguous array), taking the largest stride first decides it.
        const auto bytes = (a0 < b0) ? b0 - a0 : a0 - b0;
        if (bytes % sizeof(Value) != 0) return true;
        std::size_t d = static_cast<std::size_t>(bytes) / sizeof(Value);
        std::array<size_type, N> dims{};
        for (size_type i = 0; i < N; ++i) dims[i] = i;
        std::sort(dims.begin(), dims.end(),
                  [&](size_type x, size_type y) { return strides_[x] > strides_[y]; });
        size_type below = 1;   // the span of the dimensions with smaller strides, plus one
        for (size_type k = N; k-- > 0;) {
            const size_type i = dims[k];
            if (shape_[i] > 1 && strides_[i] < below) return true;   // self-overlapping layout
            below += (shape_[i] - 1) * strides_[i];
        }
        for (size_type k = 0; k < N; ++k) {
            const size_type i = dims[k];
            if (strides_[i] == 0) continue;
            const size_type step = std::min<size_type>(d / strides_[i], shape_[i] - 1);
            d -= step * strides_[i];
        }
        return d == 0;
    }

    /// Write other's elements into this view, of the same shape. A source
    /// that shares memory with this view (v = v.transpose(), or v = a for a
    /// view v of a) is read in full first. noexcept callers: a copy that fails
    /// to allocate terminates.
    void assign_to_view(const ndarray& other) {
        if (shares_memory_with(other)) {
            const ndarray tmp(other);
            copy_elements_from(tmp);
        } else {
            copy_elements_from(other);
        }
    }

    /// Element-wise copy from an array of the same shape, in either layout.
    void copy_elements_from(const ndarray& other) {
        if (is_contiguous_in_own_order() && other.is_contiguous_in_own_order()) {
            std::copy_n(other.mem_.data(), size(), mem_.data());
        } else {
            Value* dst = mem_.data();
            const Value* src = other.mem_.data();
            detail_nd::walk_offsets<Order>(shape_,
                                           std::array<strides_type, 2>{strides_, other.strides_},
                                           [&](const auto& o) { dst[o[0]] = src[o[1]]; });
        }
    }

    template <typename A>
    static A reversed(const A& a) {
        A r{};
        for (size_type i = 0; i < N; ++i) r[i] = a[N - 1 - i];
        return r;
    }

    void check_reshape(size_type new_size) const {
        if (new_size != size())
            ::mtl::detail::throw_or_abort(std::invalid_argument(
                "ndarray::reshape: the new shape holds " + std::to_string(new_size)
                + " elements, the array " + std::to_string(size())));
        // Must be contiguous in THIS array's own order, not merely in one of
        // the two. The result is built with `Order` strides over the same
        // memory, so an F-contiguous c_order array (any transpose of a C-order
        // array) would be re-read in the wrong order and silently return
        // elements in memory order rather than logical order (#359).
        if (!is_contiguous_in_own_order())
            ::mtl::detail::throw_or_abort(
                std::runtime_error("Cannot reshape non-contiguous array without copy"));
    }

    void check_bounds(const std::array<size_type, N>& idx) const {
        for (size_type i = 0; i < N; ++i) {
            if (idx[i] >= shape_[i]) {
                throw std::out_of_range("ndarray: index out of bounds");
            }
        }
    }

    /// Level `level` of the walk: dimension `level` for c_order, N-1-level for
    /// f_order, so the last level -- the fastest -- is the Order's fastest index.
    template <typename F>
    void iterate_impl(F& f, std::array<size_type, N>& idx, size_type level) const {
        if (level == N) {
            f(std::as_const(idx));
            return;
        }
        const size_type dim = std::is_same_v<Order, f_order> ? N - 1 - level : level;
        for (size_type i = 0; i < shape_[dim]; ++i) {
            idx[dim] = i;
            iterate_impl(f, idx, level + 1);
        }
    }
};

// -- Deduction guides --------------------------------------------------------

template <typename Value, std::size_t N>
ndarray(Value*, shape<N>) -> ndarray<Value, N, c_order>;

template <typename Value, std::size_t N>
ndarray(Value*, shape<N>, std::array<std::size_t, N>) -> ndarray<Value, N, c_order>;

} // namespace mtl::array

// -- Trait specializations ---------------------------------------------------

namespace mtl::traits {

template <typename V, std::size_t N, typename O>
struct category<::mtl::array::ndarray<V, N, O>> {
    using type = tag::dense;
};

template <typename V, std::size_t N, typename O>
struct is_expression<::mtl::array::ndarray<V, N, O>> : std::false_type {};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename V, std::size_t N, typename O>
struct ashape<::mtl::array::ndarray<V, N, O>> {
    using type = nonscal;
};

} // namespace mtl::ashape
