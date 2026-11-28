## ryzen-9-8945hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 79.2 GFLOP/s, f32 126.6 GFLOP/s |
| ... hardware, AVX-512F | f64 79.2, f32 126.6 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 36.6, update 52.0, triad 37.6 GB/s (ceiling 52.0) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 98% | 0 |
| blas2 | 10 | 75% | 88% | 0 |
| blas3 | 10 | 64% | 89% | 0 |
| factor | 12 | 10% | 23% | 0 |
| eigen | 8 | 14% | 43% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 21% | 56% | 0 |
| direct | 19 | 1% | 16% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 12% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 25% | 35% | 0 |
| mg | 1 | 28% | 28% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.761 ms | 1.43 | 11.41 | 22% | ok |
| blas1 | dot | f32 | n=8388608 | 12.285 ms | 1.37 | 5.46 | 11% | ok |
| blas1 | axpy | f64 | n=8388608 | 14.959 ms | 1.12 | 13.46 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 14.923 ms | 1.12 | 6.75 | 13% | ok |
| blas1 | scale | f64 | n=8388608 | 2.628 ms | 3.19 | 51.06 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 12.261 ms | 1.37 | 5.47 | 11% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.398 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.407 ms | 1.31 | 10.47 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 103.113 ms | 1.79 | 1.30 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.421 ms | 9.81 | 39.25 | 76% | ok |
| blas2 | gemv | f32 | n=4096 | 1.553 ms | 21.61 | 43.24 | 83% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.988 ms | 8.41 | 33.67 | 65% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.460 ms | 22.97 | 45.97 | 88% | ok |
| blas2 | symv | f64 | n=4096 | 1.912 ms | 17.55 | 35.14 | 68% | ok |
| blas2 | trmv | f64 | n=4096 | 1.770 ms | 9.48 | 37.95 | 73% | ok |
| blas2 | trmv | f32 | n=4096 | 0.931 ms | 18.02 | 36.08 | 69% | ok |
| blas2 | trsv | f64 | n=4096 | 1.713 ms | 9.79 | 39.21 | 75% | ok |
| blas2 | trsv | f32 | n=4096 | 1.028 ms | 16.33 | 32.69 | 63% | ok |
| blas2 | ger | f64 | n=4096 | 6.624 ms | 5.07 | 40.53 | 78% | ok |
| blas3 | gemm | f64 | 512x512x512 | 4.565 ms | 58.81 | 1.38 | 74% | ok |
| blas3 | gemm | f32 | 512x512x512 | 2.377 ms | 112.94 | 1.32 | 89% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 3.426 ms | 39.17 | 1.84 | 49% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.747 ms | 56.55 | 1.33 | 71% | ok |
| blas3 | syrk | f64 | 512x512x512 | 3.712 ms | 36.15 | 1.13 | 46% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 6.124 ms | 43.83 | 1.03 | 55% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.361 ms | 56.86 | 2.22 | 72% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.496 ms | 38.39 | 1.50 | 48% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.644 ms | 81.65 | 1.59 | 64% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.159 ms | 0.48 | 3.81 | 7% | ok |
| factor | lu | f64 | 512x512 | 6.334 ms | 14.13 | 0.66 | 18% | ok |
| factor | lu | f32 | 512x512 | 3.050 ms | 29.33 | 0.69 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 5.475 ms | 8.17 | 0.77 | 10% | ok |
| factor | cholesky | f32 | 512x512 | 2.109 ms | 21.21 | 0.99 | 17% | ok |
| factor | ldlt | f64 | 512x512 | 14.775 ms | 3.03 | 0.28 | 4% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.860 ms | 3.48 | 0.33 | 4% | ok |
| factor | qr | f64 | 512x512 | 12.952 ms | 13.82 | 0.32 | 17% | ok |
| factor | lq | f64 | 512x512 | 81.757 ms | 2.19 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 942.771 ms | 0.47 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.966 ms | 0.21 | 0.84 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 77.179 ms | 3.48 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.986 ms | 12.81 | 0.60 | 16% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 2.140 ms | 10.45 | 0.49 | 13% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 164.449 ms | 0.92 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 647.753 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 21.247 ms | 7.90 | 0.05 | 10% | ok |
| eigen | schur | f64 | 256x256 | 27.460 ms | 15.27 | 0.04 | 19% | ok |
| eigen | svd | f64 | 256x256 | 10.958 ms | 33.68 | 0.10 | 43% | ok |
| eigen | expm | f64 | 256x256 | 19.764 ms | 12.73 | 0.05 | 16% | ok |
| eigen | sylvester | f64 | 256x256 | 91.195 ms | 11.04 | 0.02 | 14% | ok |
| elementwise | exp | f64 | n=8388608 | 33.660 ms | 0.25 | 3.99 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 53.038 ms | 0.16 | 2.53 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 47.970 ms | 0.17 | 2.80 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.467 ms | 0.32 | 5.07 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 42.572 ms | 0.20 | 3.15 | 6% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.929 ms | 2.82 | 29.30 | 56% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.974 ms | 1.33 | 13.80 | 27% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.898 ms | 5.30 | 11.15 | 21% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 23.341 ms | 0.14 | 0.49 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.924 ms | 1.04 | 8.65 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.203 ms | 0.48 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.861 ms | 0.29 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.873 ms | 0.56 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.048 ms | 0.34 | 0.01 | 0% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 70.868 ms | 0.32 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.491 ms | 0.73 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.542 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.664 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.967 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 74.483 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.238 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.634 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.639 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.658 ms | 0.63 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.367 ms | 2.06 | 8.23 | 16%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.092 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.247 ms | 1.54 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.368 ms | 2.05 | 8.21 | 16%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.068 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.729 ms | 0.39 | 0.01 | 0% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.288 ms | 0.59 | 2.35 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.874 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.601 ms | 0.38 | 0.01 | 0% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.270 ms | 0.60 | 2.38 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.980 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 333.647 ms | 0.92 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.208 ms | 1.53 | 6.10 | 12%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.413 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 85.252 ms | 3.62 | 0.01 | 5% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.466 ms | 1.37 | 5.46 | 11%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.590 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1105.680 ms | 0.56 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.117 ms | 0.30 | 1.21 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.570 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 625.843 ms | 0.99 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.094 ms | 0.30 | 1.21 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.475 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 74.264 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 39.862 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.173 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.415 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.349 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 67.956 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.322 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.094 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.001 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 31.122 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 32.244 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.950 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.056 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.909 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 27.589 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.235 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.159 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.654 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.630 ms | 1.61 | 17.99 | 35% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.522 ms | 1.04 | 11.63 | 22% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.269 ms | 1.15 | 12.92 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.769 ms | 1.95 | 14.70 | 28% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.467 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.250 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 64.889 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.836 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.279 ms | 1.37 | 5.47 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 37.255 ms | 7.21 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.833 ms | 9.15 | 9.72 | 19% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.035 ms | 4.16 | 16.63 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.244 ms | 5.17 | 10.34 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.123 ms | 7.90 | 7.90 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.176 ms | 26.38 | 0.31 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.302 ms | 7.01 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 34.594 ms | 7.76 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 34.595 ms | 7.76 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.482 ms | 7.78 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

