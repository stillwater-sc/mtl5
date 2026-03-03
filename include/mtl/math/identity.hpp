#pragma once
// MTL5 — Algebraic identity elements (replaces boost/numeric/linear_algebra/identity.hpp)
// Key change from MTL4: constexpr, no reference parameter needed for type deduction
#include <mtl/math/operations.hpp>
#include <type_traits>
#include <limits>

namespace mtl::math {

// ── Primary template ────────────────────────────────────────────────────

/// identity_t<Op, T> — returns the identity element of operation Op for type T
template <typename Op, typename T>
struct identity_t;

// ── Additive identity (zero) ────────────────────────────────────────────

template <typename T>
struct identity_t<add<T>, T> {
    static constexpr T value = T{0};
    constexpr T operator()() const { return value; }
};

// ── Multiplicative identity (one) ───────────────────────────────────────

template <typename T>
struct identity_t<mult<T>, T> {
    static constexpr T value = T{1};
    constexpr T operator()() const { return value; }
};

// ── Max identity (lowest value) ─────────────────────────────────────────

template <typename T>
struct identity_t<max_op<T>, T> {
    static constexpr T value = std::numeric_limits<T>::lowest();
    constexpr T operator()() const { return value; }
};

// ── Min identity (max value) ────────────────────────────────────────────

template <typename T>
struct identity_t<min_op<T>, T> {
    static constexpr T value = std::numeric_limits<T>::max();
    constexpr T operator()() const { return value; }
};

// ── Convenience functions ───────────────────────────────────────────────

/// Returns the additive identity (zero) for type T
template <typename T>
constexpr T zero() { return identity_t<add<T>, T>{}(); }

/// Returns the multiplicative identity (one) for type T
template <typename T>
constexpr T one() { return identity_t<mult<T>, T>{}(); }

} // namespace mtl::math
