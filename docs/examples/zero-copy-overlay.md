# Zero-Copy Overlay: MTL5 Types over Your Buffers

Sometimes the data an algorithm needs already sits in memory that something else owns. A solver library hands your callback raw pointers into its state. A camera driver or a DMA engine fills frames in a shared region. NumPy, a C library or a GPU staging buffer owns an array you want to work on. Copying that data into an MTL5 vector and copying the result back costs time and memory on every call, and in a pipeline it means the next stage no longer sees the same memory.

MTL5 lets you lay a vector or matrix over the buffer instead. The MTL5 object becomes a **conduit**: an algorithm written with MTL5 operations reads and writes the buffer in place, and whoever shares the buffer sees the result without any hand-off of data.

```cpp
vec::dense_vector<double> v(n, ptr);            // over a buffer you can write
mat::dense2D<double>      A(rows, cols, ptr);    // row-major, rows x cols, no padding

const auto y = vec::const_view(n, cptr);         // over a const buffer: read-only
const auto B = mat::const_view(rows, cols, cptr);
```

Three programs in `examples/zero_copy/` show the pattern and check every claim they make. CTest runs them, so a regression fails the build.

## The callback: `ode_rhs_callback.cpp`

An ODE integrator owns the state `y` and its stage buffers, and calls the user's right-hand side with raw pointers. CVODE, GSL's `odeiv2` and most Fortran solvers have this shape; the example writes a minimal RK4 with this interface so it has no dependency:

```cpp
using rhs_fn = void (*)(double t, const double* y, double* dydt, std::size_t n, void* user);
```

The right-hand side of the semi-discrete heat equation, `dy/dt = L y` with `L` the sparse 1D Laplacian, wraps both pointers and multiplies straight into the integrator's buffer:

```cpp
void heat_rhs(double, const double* y, double* dydt, std::size_t n, void* user) {
    auto& p = *static_cast<heat_problem*>(user);
    const auto Y = vec::const_view(n, y);      // a view of the integrator's y
    vec::dense_vector<double> dY(n, dydt);     // a vector over its output buffer
    mult(p.L, Y, dY);                          // dydt = L y, written in place
}
```

`Y` is a read-only view: `y` arrives as `const double*`, and `Y(0) = 0.0` does not compile. `mult(A, x, y)` writes its result into `y`; `dY = p.L * Y` would also write `dydt`, but through a temporary vector.

The program integrates 8000 right-hand-side calls and checks:
- the result against the exact solution of the semi-discrete system (relative error 3e-16);
- that every call used the integrator's `y` and `dydt` in place;
- that a copy-in/copy-out version of the same function gives bit-identical results.

It then times one call at n = 2,000,000 both ways. The copying version moves 32 MB more per call and allocates twice. In a Release GCC build on an i7-12700K, one zero-copy call took 4.5 ms and one copying call 14.5 ms, the same in three runs: 3.2 times as long, because a sparse matrix-vector product with three entries per row does little more work than the copies themselves. The ratio depends on the machine and on how much work the right-hand side does; a Debug build hides most of it.

## The shared buffer: `shared_frame_pipeline.cpp`

A producer thread stands in for a sensor driver. It fills frames in a region with two slots, double-buffered, and hands each slot to a processing thread. The processing stage corrects each frame in place and hands the slot back. No frame is ever copied.

```cpp
mat::dense2D<float> F(H, W, frame);          // the slot, as a matrix
F = gain * (F - dark);                       // dark-frame and gain correction

array::ndarray<float, 2> nd(frame, {H, W});
auto tile = array::slice(nd, array::range(r0, r0 + T), array::range(c0, c0 + T));
tile -= tile_mean;                           // background flattening per tile
```

The whole frame is contiguous, so `dense2D` describes it. A tile's rows are `W` elements apart, so it needs a strided view: an `ndarray` slice. Both write the slot.

Who may touch a slot is the application's business: a mutex and a condition variable guard a per-slot state, and each thread works on a slot only while it owns it. MTL5 adds no locking. Without the hand-off, the producer would overwrite a frame while it is being corrected, and the result would be a mix of two frames.

The program checks every processed frame against values computed independently from the synthetic scene, and that every view stayed on its slot.

## The rules: `zero_copy_rules.cpp`

A tour of the contract, one checked demonstration per rule.

| Rule | What it means |
|---|---|
| **Assignment writes through** | `v = w`, `v = f()`, `v = a + b` and `v *= s` write the buffer, and `v` stays non-owning. |
| **The buffer fixes the size** | Copy or expression assignment of another size throws `std::domain_error`. A move of another size asserts, because a move can't throw. |
| **An explicit resize detaches** | `change_dim` to another size gives the object its own storage; writes stop reaching the buffer, and `owns_memory()` becomes true. |
| **Copy construction copies** | `auto w = v;` is an owning deep copy. Keep the conduit with a reference, or a second wrapper over the same pointer. Pass conduits by reference. |
| **Read-only buffers get read-only views** | `vec::const_view` and `mat::const_view` take a `const T*`. A write through them does not compile, and every operation that reads a `dense_vector` or `dense2D` accepts them. `dense_vector<double> w(view)` makes an owning copy. |
| **Overlap and aliasing** | Two wrappers over overlapping parts of one buffer copy safely. An expression that reads the target elsewhere, such as `E = E + trans(E)`, goes through a temporary and still lands in the buffer. MTL5 cannot see another thread writing the buffer during an expression: that is a data race. |
| **Layout** | `dense2D(rows, cols, ptr)` assumes rows (or columns) are packed with no padding. Over a padded buffer, such as an image with a row pitch, it reads the padding. Use an `ndarray` with the real strides instead. |
| **Lifetime** | MTL5 never frees memory it did not allocate. The buffer must outlive every object over it. |
| **Synchronization** | Yours. Hand a buffer between threads as the pipeline does. |

## Running them

```bash
cmake -B build -DMTL5_BUILD_EXAMPLES=ON
cmake --build build --target example_ode_rhs_callback example_shared_frame_pipeline example_zero_copy_rules -j4
ctest --test-dir build -R example_ --output-on-failure
```

## Related

- [dense2D architecture](../architecture/containers/dense2d-architecture.md), Decision 3: the ownership modes, the assignment rule and the read-only views.
- `include/mtl/detail/contiguous_memory_block.hpp`: the memory block that implements them.
