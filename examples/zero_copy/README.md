# Zero-Copy Overlay

An MTL5 vector or matrix laid over a buffer that someone else owns is a conduit to that buffer: MTL5 algorithms read and write it in place, with no copy in or out. Use it when the data is already in memory you share: a solver's callback pointers, a frame from a driver, an array from NumPy or a C library.

| Example | Description |
|---------|-------------|
| `ode_rhs_callback.cpp` | A C-style RK4 integrator calls `rhs(t, const double* y, double* dydt, n, user)`. The callback wraps `y` with `vec::const_view` and `dydt` with `dense_vector(n, dydt)` and computes `dydt = L y` in place. Checks the heat-equation solution, that no buffer was copied, and times copy-in/copy-out against it. |
| `shared_frame_pipeline.cpp` | A producer thread fills double-buffered frames in a shared region. The processing thread corrects each frame in place, with a `dense2D` over the whole frame and `ndarray` slices over its tiles, and hands the slot back. Checks every frame and that it was never copied. |
| `zero_copy_rules.cpp` | The contract, one checked demonstration per rule: write-through, size, resize, copy construction, read-only views, aliasing, layout, lifetime and synchronization. |

Each program exits non-zero if a check fails, and CTest runs all three. The walkthrough is [docs/examples/zero-copy-overlay.md](../../docs/examples/zero-copy-overlay.md).
