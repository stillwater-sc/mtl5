#pragma once
// MTL5 — Forward declarations for all key types

#include <cstddef>

namespace mtl {

// ── Tags ────────────────────────────────────────────────────────────────
namespace tag {
    struct row_major;
    struct col_major;
    struct dense;
    struct sparse;
    struct on_stack;
    struct on_heap;
} // namespace tag

namespace detail {
    struct c_index;
    struct f_index;
} // namespace detail

// ── Matrix types ────────────────────────────────────────────────────────
namespace mat {
    template <typename Orientation, typename Index, typename Dimensions,
              typename Storage, typename SizeType>
    struct parameters;

    namespace non_fixed { struct dimensions; }
    template <std::size_t R, std::size_t C> struct fixed_dimensions;

    template <typename Value, typename Parameters> class dense2D;
    template <typename Value, typename Parameters> class compressed2D;
    template <typename Value, typename Parameters> class coordinate2D;
    template <typename Value, typename Parameters> class ell_matrix;
    template <typename Value>                      class identity2D;
} // namespace mat

// ── Vector types ────────────────────────────────────────────────────────
namespace vec {
    template <typename Orientation, typename Dimensions,
              typename Storage, typename SizeType>
    struct parameters;

    namespace non_fixed { struct dimension; }
    template <std::size_t N> struct fixed_dimension;

    template <typename Value, typename Parameters> class dense_vector;
    template <typename Value, typename Parameters> class sparse_vector;
} // namespace vec

// ── Math ────────────────────────────────────────────────────────────────
namespace math {
    template <typename T> struct add;
    template <typename T> struct mult;
    template <typename Op, typename T> struct identity_t;

    template <typename T> constexpr T zero();
    template <typename T> constexpr T one();
} // namespace math

// ── ITL ─────────────────────────────────────────────────────────────────
namespace itl {
    template <typename Real> class basic_iteration;
    template <typename Real> class cyclic_iteration;
    template <typename Real> class noisy_iteration;
} // namespace itl

} // namespace mtl
