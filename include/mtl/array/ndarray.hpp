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
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include <mtl/config.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/traits/is_expression.hpp>
#include <mtl/detail/contiguous_memory_block.hpp>
#include <mtl/array/shape.hpp>

namespace mtl::array {

/// N-dimensional array with static rank and runtime shape.
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

    // Copy and move: default (contiguous_memory_block handles ownership)
    ndarray(const ndarray&) = default;
    ndarray(ndarray&&) noexcept = default;
    ndarray& operator=(const ndarray&) = default;
    ndarray& operator=(ndarray&&) noexcept = default;

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

    /// Check if memory is contiguous (enables BLAS dispatch, reshape, etc.)
    bool is_contiguous() const {
        return ::mtl::array::is_contiguous(shape_, strides_);
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

    /// Flat (linear) index access — only valid for contiguous arrays.
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
            iterate_indices([&](const std::array<size_type, N>& idx) {
                (*this)[idx] = val;
            });
        }
    }

    // -- View creation -------------------------------------------------------

    /// Create a view (non-owning) of this array with the same shape and strides.
    ndarray view() {
        return ndarray(mem_.data(), shape_, strides_);
    }

    /// Reshape: returns a view if contiguous, otherwise throws.
    template <std::size_t M>
    ndarray<Value, M, Order> reshape(::mtl::array::shape<M> new_shape) const {
        assert(new_shape.total_size() == size() && "Reshape must preserve total size");
        if (!is_contiguous()) {
            throw std::runtime_error("Cannot reshape non-contiguous array without copy");
        }
        return ndarray<Value, M, Order>(
            const_cast<Value*>(mem_.data()), new_shape);
    }

    /// Transpose: returns a view with reversed shape and strides.
    ndarray<Value, N, Order> transpose() const {
        shape_type   new_shape;
        strides_type new_strides;
        for (size_type i = 0; i < N; ++i) {
            new_shape[i]   = shape_[N - 1 - i];
            new_strides[i] = strides_[N - 1 - i];
        }
        return ndarray<Value, N, Order>(
            const_cast<Value*>(mem_.data()), new_shape, new_strides);
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

    /// Iterate over all valid multi-indices, calling f(idx) for each.
    template <typename F>
    void iterate_indices(F&& f) const {
        std::array<size_type, N> idx{};
        iterate_impl(f, idx, 0);
    }

    /// Apply f to every element (respects strides for non-contiguous arrays).
    template <typename F>
    void for_each_element(F&& f) {
        if (is_contiguous()) {
            for (size_type i = 0; i < size(); ++i) {
                f(mem_[i]);
            }
        } else {
            iterate_indices([&](const std::array<size_type, N>& idx) {
                f((*this)[idx]);
            });
        }
    }

    template <typename F>
    void for_each_element(F&& f) const {
        if (is_contiguous()) {
            for (size_type i = 0; i < size(); ++i) {
                f(mem_[i]);
            }
        } else {
            iterate_indices([&](const std::array<size_type, N>& idx) {
                f((*this)[idx]);
            });
        }
    }

private:
    void check_bounds(const std::array<size_type, N>& idx) const {
        for (size_type i = 0; i < N; ++i) {
            if (idx[i] >= shape_[i]) {
                throw std::out_of_range("ndarray: index out of bounds");
            }
        }
    }

    template <typename F>
    void iterate_impl(F& f, std::array<size_type, N>& idx, size_type dim) const {
        if (dim == N) {
            f(idx);
            return;
        }
        for (size_type i = 0; i < shape_[dim]; ++i) {
            idx[dim] = i;
            iterate_impl(f, idx, dim + 1);
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
