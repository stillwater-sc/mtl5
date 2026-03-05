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
    constexpr T operator()() const { return T{0}; }
};

// ── Multiplicative identity (one) ───────────────────────────────────────

template <typename T>
struct identity_t<mult<T>, T> {
    constexpr T operator()() const { return T{1}; }
};

// ── Max identity (lowest value) ─────────────────────────────────────────

template <typename T>
struct identity_t<max_op<T>, T> {
    constexpr T operator()() const { return std::numeric_limits<T>::lowest(); }
};

// ── Min identity (max value) ────────────────────────────────────────────

template <typename T>
struct identity_t<min_op<T>, T> {
    constexpr T operator()() const { return std::numeric_limits<T>::max(); }
};

// ── Convenience functions ───────────────────────────────────────────────

/// Returns the additive identity (zero) for type T
template <typename T>
constexpr T zero() { return identity_t<add<T>, T>{}(); }

/// Returns the multiplicative identity (one) for type T
template <typename T>
constexpr T one() { return identity_t<mult<T>, T>{}(); }

} // namespace mtl::math
