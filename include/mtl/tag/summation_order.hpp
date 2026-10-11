#pragma once
// MTL5 -- Summation-order tags for factorizations (#556)
//
// A factorization entry has the form  a - (p_0 + p_1 + ... + p_{k-1}),  where
// the p_i are products of already-computed factor entries. Two orderings are in
// common use, and they round differently:
//
//   subtract_sum           s = 0; s += p_i ...;  result = a - s
//   subtract_term_by_term  s = a; s -= p_i ...;  result = s
//
// Both are backward stable and neither is more accurate across the board (see
// docs/numerical/summation-order-in-cholesky.md). They exist side by side so a
// caller whose results are pinned to one form can reproduce it bit for bit.
namespace mtl::tag {

/// Sum the products from zero, then subtract the sum once (the default).
struct subtract_sum {};

/// Start from the matrix entry and subtract each product in turn.
struct subtract_term_by_term {};

} // namespace mtl::tag
