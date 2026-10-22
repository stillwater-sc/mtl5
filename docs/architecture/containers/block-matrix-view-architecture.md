# block_matrix_view: architecture of the block-quantized matrix view

How `mtl::mat::block_matrix_view` is put together, and why it is shaped the way
it is. ML weights, and other compressed numerical data, are increasingly stored
**block-quantized**: groups of 16–256 values share a scale, and the elements are
4–8-bit integers or minifloats. Before this type a consumer had two choices:

- dequantize into [`dense2D<float>`](dense2d-architecture.md), at 4–8× the
  memory and bandwidth, which gives up the reason the data was quantized; or
- write its own view and kernels outside MTL5, and give up `Matrix` interop,
  `mult` dispatch, `convert` and expression syntax.

`block_matrix_view` is the format-agnostic middle (#629): a `Matrix` whose
storage *is* the encoded bytes, parameterized by a **`BlockScheme`** that knows
how to decode them. Its design follows the prototype in
[branes-ai/synapse](https://github.com/branes-ai/synapse), a GGML-compatible
runtime built on MTL5.

The implementation is in
[`include/mtl/mat/block_matrix_view.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/block_matrix_view.hpp),
the concept in
[`include/mtl/concepts/block_scheme.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/concepts/block_scheme.hpp),
and the in-tree OCP MX schemes in
[`include/mtl/mat/mx_schemes.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/mx_schemes.hpp).

---

## The split: MTL5 defines the concept, libraries supply schemes

```cpp
template <typename S>
concept BlockScheme = requires(const std::byte* src, float* dst) {
    { S::block_size }  -> std::convertible_to<std::size_t>;   // elements per block
    { S::block_bytes } -> std::convertible_to<std::size_t>;   // bytes per block
    S::decode(src, dst);                                      // reference decode
};
```

A scheme may also provide `S::decode_element(src, j)`, which decodes one
element without decoding its block; element access uses it when it exists.

Schemes work on **raw bytes**. Block storage usually comes from a file or a
memory map and carries no C++ objects, so a scheme reads the bytes the way the
format specifies them, with `memcpy` for multi-byte fields, never through a
`reinterpret_cast` to a struct.

**Which schemes live in MTL5.** The OCP Microscaling (MX) formats are a
vendor-neutral specification, so two of them ship in-tree as `mtl::mx::mxint8`
and `mtl::mx::mxfp4`. GGML's formats (Q4_0, Q8_0, Q4_K, Q6_K, ...) stay with
their consumer: their correctness is defined by a llama.cpp version, not by a
specification, and synapse verifies them against llama.cpp's own encoder.

## The view

```text
block_matrix_view<S>
   rows_, cols_, blocks_per_row_ = cols / S::block_size
   data_ : const std::byte*        rows * blocks_per_row blocks, row after row
```

- **Non-owning.** The caller keeps the bytes alive for the view's lifetime. A
  memory-mapped weight file is the typical owner.
- **Row-major, blocked along rows.** This is GGML's layout for weights, and the
  one `y = W x` reads in order. Column-blocked storage is out of scope; an
  orientation parameter could be added later without breaking anything.
- **Read-only.** Writing one element of a block format is a read-modify-write of
  its whole block, and usually a change of the shared scale. That is an
  *encode*, not an assignment, so `operator()` returns a decoded `float` by value.
- **Checked at construction.** `cols` must be a multiple of `block_size`, and
  the span must hold exactly `rows * (cols / block_size)` blocks. Anything else
  throws `std::invalid_argument`, because a view that silently reads past its
  data is worse than no view.

The view exposes the encoded bytes as well as the values: `block(r, b)` and
`row(r)` are spans into the storage, for kernels that work on the blocks
directly, and `decode_row(r, dst)` decodes a whole row.

**Category `dense`, and no `data()`.** Every element is stored, so the category
tag is `tag::dense` and the view satisfies `DenseMatrix`. It deliberately has no
`data()` returning `const float*`: every BLAS and native-SIMD dispatch in MTL5
is gated on one, so no accelerated dense kernel can mistake encoded bytes for a
float array.

## Operations

| call | path |
|---|---|
| `A(r, c)` | decode one element (or its block) |
| `convert(A, D)`, `convert<T>(A)` | decode a row at a time into a dense matrix |
| `mult(A, x, y)` | a scheme's fused kernel, or decode a row and dot it |
| `mult(A, x, y, scratch)` | the same, with the fused kernel's working memory from the caller |
| `A * x` | `mult` into a new `dense_vector` |

Each of these is an overload more specialized than the generic `Matrix` version.
Generic code still works on the view, through element access, but the generic
mat-vec would decode one block per *element*; the overload decodes each row once.

**The decode-then-dot path** decodes row `r` into a float buffer and dots it with
`x`. For a contiguous `float` `x` that dot is `simd::reduce_dot<float>`; for
other element types it is the generic fused loop. An explicit `Accumulator`
(`mult<double>(A, x, y)`) is honored. Rows are partitioned across the thread
pool like the dense mat-vec, each band with its own decode buffer, and because
each `y(r)` comes from row `r` alone the result does not depend on the thread
count (`tests/unit/mat/test_block_matrix_view_mt.cpp`).

**Fused kernels plug in per scheme**, through a customization point:

```cpp
template <> struct mtl::block_kernels<my_scheme> {
    // y = A x. scratch holds at least scratch_bytes(A.num_cols()) bytes,
    // aligned to scratch_alignment.
    static void gemv(const mtl::mat::block_matrix_view<my_scheme>& A,
                     const float* x, float* y, std::span<std::byte> scratch);

    // Optional: working memory per call (default 0) and its alignment
    // (default alignof(std::max_align_t), a power of two).
    static std::size_t scratch_bytes(std::size_t cols);
    static constexpr std::size_t scratch_alignment = 64;
};
```

`mult` calls it for contiguous `float` input and output and no explicit
`Accumulator`, and takes the decode path otherwise. A typical fused kernel
quantizes `x` per block and runs
[`simd::reduce_dot_block_scaled`](../../algorithms/mixed-precision-kernels.md)
on the integer blocks. The kernel owns its accuracy contract; the decode path is
the reference it is measured against.

**The scratch is the caller's memory.** A fused kernel needs somewhere to put
`x` once it is quantized, and a runtime that sizes its buffers when it loads a
model should not have the mat-vec allocate behind its back. So the hook takes
the scratch as an argument:

- `mult(A, x, y, scratch)` passes it through. It must hold
  `block_gemv_scratch_bytes<S>(cols)` bytes aligned to
  `block_gemv_scratch_alignment<S>`, or `mult` throws `std::invalid_argument`
  before the kernel runs. Off the fused path it is unused, so an empty span is
  fine.
- `mult(A, x, y)` and `A * x` allocate it for the one call, when the kernel
  declares any.

The kernel is called once per `mult`, with the whole matrix, so the scratch is
built from `x` once and read by every row. A kernel that threads over rows does
so itself, and shares the scratch read-only.

synapse's GGML schemes (Q4_0, Q8_0, Q4_K, Q6_K) were checked against this hook
before 5.20.0. Their fused gemv takes caller-owned, aligned scratch for the
quantized activations, and it plugs in unchanged.

## The OCP MX schemes

The OCP Microscaling Formats specification (v1.0) defines a block as 32 elements
sharing one **E8M0** scale *X*, each element *Pᵢ* decoding to *X · Pᵢ*. E8M0 is
a bare biased exponent: *X* = 2^(e−127) for e in [0, 254], and e = 0xFF is NaN,
which makes every element of its block NaN.

| scheme | element | bytes per block | element value |
|---|---|---|---|
| `mx::mxint8` | INT8, two's complement | 1 + 32 = 33 | *q* / 64 · *X* |
| `mx::mxfp4` | FP4 E2M1: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} | 1 + 16 = 17 | E2M1(*c*) · *X* |

**The byte layout is ours to state**, because the specification defines the
encodings but not how a block is laid out in memory. Both schemes put the scale
byte first. `mxfp4` follows GGML's `block_mxfp4` (llama.cpp's `GGML_TYPE_MXFP4`,
the format of gpt-oss GGUF files), so such data can be viewed in place: for
j < 16, element j is the low nibble of byte 1 + j and element j + 16 its high
nibble. GGML has no MXINT8 type, so `mxint8` is the scale followed by the 32
elements in order.

**Range.** Decoding is to `float`, which covers E8M0's range at the bottom
(2^−127 is a float subnormal) but not quite at the top. Under the largest scale,
2^127, any element of magnitude 2 or more exceeds `FLT_MAX` and decodes to
infinity: the E2M1 codes 2, 3, 4 and 6, and INT8's −128, which is exactly −2.

MTL5 stays Universal-free, so the schemes decode by hand rather than through
Universal's `mxfloat`. They are small enough to check exhaustively, and the
tests do: every E2M1 code in both nibble positions, E8M0 at its extremes and
NaN, and INT8's implicit 2^−6.

## Not here yet

- **The encode direction**: quantizing a float vector or matrix into a block
  format, e.g. activations to MXINT8 for a fused integer gemv. It needs a
  rounding contract per scheme (how the shared scale is chosen, how elements
  round) and is a follow-up.
- **Fused kernels for the MX schemes.** The customization point is in place;
  the in-tree schemes use the decode path.
- **An owning variant**, and column-blocked or writable block matrices.

## File map

| file | contents |
|---|---|
| `include/mtl/concepts/block_scheme.hpp` | `BlockScheme`, `ElementDecodableScheme` |
| `include/mtl/mat/block_matrix_view.hpp` | the view, `block_kernels` and its scratch queries, `mult`, `convert`, `operator*` |
| `include/mtl/mat/mx_schemes.hpp` | `mx::mxint8`, `mx::mxfp4`, `e8m0_to_float`, `e2m1_to_float` |
| `tests/unit/mat/test_block_matrix_view.cpp` | the encodings, the view, convert, mult, the fused-kernel hook and its scratch |
| `tests/unit/mat/test_block_matrix_view_mt.cpp` | threaded mult: bit-identity and that the rows split |
