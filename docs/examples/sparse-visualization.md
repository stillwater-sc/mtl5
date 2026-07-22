# Visualizing Sparse Matrices (spy)

MATLAB's `spy` is one of the fastest ways to understand a sparse matrix: it draws
the non-zero *pattern* so bandedness, block structure, arrowheads, and — after a
factorization — **fill-in** become visible at a glance. MTL5 provides the same,
emitting **PNG** files through a dependency-free writer (no libpng, no zlib —
see `include/mtl/io/png.hpp`).

## API

All three live in `namespace mtl::io` (`#include <mtl/io/spy.hpp>`) and take any
matrix type — `compressed2D` (CRS), `coordinate2D` (COO), `ell_matrix` (ELLPACK),
or `dense2D`:

```cpp
#include <mtl/io/spy.hpp>

mtl::io::spy(A, "pattern.png");                 // binary non-zero pattern
mtl::io::spy_magnitude(A, "magnitude.png",      // colored by |a_ij|
                       {.max_pixels = 512, .log_scale = true});
mtl::io::spy_density(A, "density.png");         // colored by non-zeros per cell
```

`spy_options` controls the longest image edge (`max_pixels`, default 1024) and
whether magnitude/density coloring is linear or `log10` (`log_scale`). A matrix
larger than `max_pixels` is **down-sampled** by binning its non-zeros into pixel
cells, so an N×N matrix with N ≫ image size still renders a faithful structure at
a fixed resolution.

## Example: a 2D Laplacian and its LU fill-in

A 24×24-grid 5-point Laplacian (576 unknowns) has the classic banded pattern —
the main diagonal plus the four stencil neighbours:

![spy of a 2D Laplacian](img/spy-laplacian.png)

Coloring the same matrix by **non-zero density** (down-sampled to 64×64) shows
where the bands concentrate:

![density spy of the Laplacian](img/spy-laplacian-density.png)

Now factor it with dense LU and spy the factor. The band **fills in** — the
2,784 structural non-zeros of the original become 27,118 in `L\U`, exactly the
fill-in that fill-reducing orderings (RCM/AMD/COLAMD in `mtl::sparse::ordering`)
exist to reduce:

![spy of the LU factor showing fill-in](img/spy-laplacian-lu.png)

## Uses

- **Debug a sparse assembly** — confirm the pattern matches the intended stencil
  or connectivity.
- **See fill-in** — spy an `L`/`U`/Cholesky factor to judge an ordering's effect
  (spy the matrix before and after `permute`).
- **Inspect reordering** — visualize a matrix and its RCM/AMD-reordered form
  side by side.
- **Heatmaps generally** — the underlying `write_png_gray` / `write_png_rgb`
  (in `mtl/io/png.hpp`) render any 8-bit raster, not just spy plots.

## How the PNG is written

The encoder is written from first principles: the PNG container (signature +
`IHDR`/`IDAT`/`IEND` chunks, each CRC-32-checked) with an `IDAT` that is a zlib
stream built from DEFLATE **stored** (uncompressed) blocks plus an Adler-32
checksum. No compression library is required — just correct framing and the two
checksums — so `spy` works in any build with zero external dependencies.
