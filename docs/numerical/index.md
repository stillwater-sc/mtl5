# Numerical Considerations

Two implementations of the same algorithm, each correct and backward stable, can
still return different bits. Floating-point addition isn't associative, so the
order in which a routine forms its sums is part of its result.

Usually that doesn't matter: both answers are within rounding of the exact one,
and nobody compares them bit for bit. It starts to matter when results are
**pinned**. Some examples:

- a port that has to reproduce the code it replaces exactly, before anything
  else is allowed to change;
- a filter whose accept/reject decisions flip when a residual moves in the last
  bit;
- a regression suite that stores reference outputs and compares them with `==`.

This section documents where MTL5 makes an ordering choice that a caller might
need to control, what each choice computes, and how to select it.

## Pages

| Page | What it covers |
|---|---|
| [Summation order in Cholesky](summation-order-in-cholesky.md) | `cholesky_factor`'s default *sum, then subtract* form, the opt-in *term by term* form (`tag::subtract_term_by_term`), what each costs in accuracy, and how to use the opt-in to reproduce a hand-written factorization bit for bit |
| [Householder reflector forms](householder-reflector-forms.md) | The default LAPACK-form reflector and the opt-in textbook form (`tag::textbook_reflector`) for `householder`, `apply_householder_left` and `qr_factor`: how they differ in rounding and in the sign of R, how zero and NaN columns are skipped, and how to reproduce a hand-written textbook QR bit for bit |

## Related

- [Measuring solver accuracy](../algorithms/measuring-solver-accuracy.md): how
  to measure backward error.
- [Mixed-precision kernels](../algorithms/mixed-precision-kernels.md): choosing
  the precision of an accumulation, as opposed to its order.
- [Loop order and sparsity in mtl::mult](../performance/ikj-gemm-and-crs-dense.md):
  a reordering that keeps every element's summation order, and where FP
  contraction makes "bit-identical" depend on compiler flags.
