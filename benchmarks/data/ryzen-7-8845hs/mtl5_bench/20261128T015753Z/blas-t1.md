## ryzen-7-8845hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 80.1 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.1, f32 160.1 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 43.1, update 53.2, triad 38.8 GB/s (ceiling 53.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 110% | 0 |
| blas2 | 10 | 78% | 90% | 0 |
| blas3 | 10 | 70% | 92% | 0 |
| factor | 12 | 13% | 23% | 0 |
| eigen | 8 | 17% | 53% | 0 |
| elementwise | 5 | 7% | 12% | 0 |
| sparse | 5 | 26% | 60% | 0 |
| direct | 19 | 1% | 20% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 32% | 34% | 0 |
| mg | 1 | 30% | 30% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 13% | 23% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 9.138 ms | 1.84 | 14.69 | 28% | ok |
| blas1 | dot | f32 | n=8388608 | 9.852 ms | 1.70 | 6.81 | 13% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.029 ms | 1.39 | 16.74 | 31% | ok |
| blas1 | axpy | f32 | n=8388608 | 12.009 ms | 1.40 | 8.38 | 16% | ok |
| blas1 | scale | f64 | n=8388608 | 2.292 ms | 3.66 | 58.55 | 110% | ok |
| blas1 | two_norm | f64 | n=8388608 | 9.838 ms | 1.71 | 6.82 | 13% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.063 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.071 ms | 1.65 | 13.23 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 83.257 ms | 2.22 | 1.61 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.221 ms | 10.42 | 41.68 | 78% | ok |
| blas2 | gemv | f32 | n=4096 | 1.409 ms | 23.82 | 47.66 | 90% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.367 ms | 9.97 | 39.89 | 75% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.496 ms | 22.43 | 44.87 | 84% | ok |
| blas2 | symv | f64 | n=4096 | 1.818 ms | 18.45 | 36.95 | 69% | ok |
| blas2 | trmv | f64 | n=4096 | 1.592 ms | 10.54 | 42.21 | 79% | ok |
| blas2 | trmv | f32 | n=4096 | 0.884 ms | 18.99 | 38.01 | 71% | ok |
| blas2 | trsv | f64 | n=4096 | 1.717 ms | 9.77 | 39.12 | 74% | ok |
| blas2 | trsv | f32 | n=4096 | 0.866 ms | 19.38 | 38.79 | 73% | ok |
| blas2 | ger | f64 | n=4096 | 6.365 ms | 5.27 | 42.19 | 79% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.646 ms | 73.62 | 1.73 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.899 ms | 141.36 | 1.66 | 88% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.737 ms | 49.03 | 2.30 | 61% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.795 ms | 70.73 | 1.66 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.934 ms | 45.75 | 1.43 | 57% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.756 ms | 56.44 | 1.32 | 70% | ok |
| blas3 | trmm | f64 | 512x512x512 | 1.897 ms | 70.74 | 2.76 | 88% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.847 ms | 47.15 | 1.84 | 59% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.347 ms | 99.66 | 1.95 | 62% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.382 ms | 0.57 | 4.53 | 9% | ok |
| factor | lu | f64 | 512x512 | 4.977 ms | 17.98 | 0.84 | 22% | ok |
| factor | lu | f32 | 512x512 | 2.436 ms | 36.72 | 0.86 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 4.346 ms | 10.29 | 0.97 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 1.676 ms | 26.70 | 1.25 | 17% | ok |
| factor | ldlt | f64 | 512x512 | 11.748 ms | 3.81 | 0.36 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.441 ms | 4.28 | 0.40 | 5% | ok |
| factor | qr | f64 | 512x512 | 10.185 ms | 17.57 | 0.41 | 22% | ok |
| factor | lq | f64 | 512x512 | 64.665 ms | 2.77 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 737.984 ms | 0.61 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.912 ms | 0.27 | 1.07 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 60.173 ms | 4.46 | 0.07 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.532 ms | 16.17 | 0.76 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.696 ms | 13.19 | 0.62 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 129.234 ms | 1.17 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 525.892 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.932 ms | 9.91 | 0.06 | 12% | ok |
| eigen | schur | f64 | 256x256 | 22.235 ms | 18.86 | 0.05 | 24% | ok |
| eigen | svd | f64 | 256x256 | 8.719 ms | 42.33 | 0.12 | 53% | ok |
| eigen | expm | f64 | 256x256 | 15.714 ms | 16.02 | 0.07 | 20% | ok |
| eigen | sylvester | f64 | 256x256 | 72.498 ms | 13.89 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 27.797 ms | 0.30 | 4.83 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 43.232 ms | 0.19 | 3.10 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 39.164 ms | 0.21 | 3.43 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.388 ms | 0.39 | 6.28 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 34.552 ms | 0.24 | 3.88 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.852 ms | 3.07 | 31.98 | 60% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.481 ms | 1.77 | 18.39 | 35% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.434 ms | 6.51 | 13.68 | 26% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 17.923 ms | 0.18 | 0.64 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.701 ms | 1.18 | 9.79 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.401 ms | 0.58 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.080 ms | 0.36 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.543 ms | 0.72 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.208 ms | 0.41 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.425 ms | 0.39 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 26.356 ms | 0.85 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 37.657 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.707 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.673 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.210 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.227 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.347 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.950 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.287 ms | 0.78 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.278 ms | 2.72 | 10.90 | 20%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.602 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.017 ms | 1.86 | 0.12 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.291 ms | 2.59 | 10.38 | 20%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.391 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.512 ms | 0.47 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.113 ms | 0.68 | 2.72 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.854 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.289 ms | 0.45 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.066 ms | 0.71 | 2.84 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.593 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 255.657 ms | 1.21 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.984 ms | 1.70 | 6.79 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.819 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.567 ms | 4.44 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.024 ms | 1.66 | 6.66 | 13%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.185 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 899.491 ms | 0.69 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.408 ms | 0.32 | 1.29 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.534 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 517.755 ms | 1.19 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.468 ms | 0.32 | 1.29 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 38.966 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 57.677 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.662 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.588 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.523 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.578 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 53.064 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.148 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.689 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 25.912 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 24.460 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.308 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 32.904 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.462 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 22.322 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.811 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.773 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.337 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.612 ms | 1.62 | 18.20 | 34% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.735 ms | 1.51 | 16.90 | 32% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.869 ms | 0.91 | 10.22 | 19% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.448 ms | 2.08 | 15.76 | 30% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.482 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.999 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.952 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.811 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 9.796 ms | 1.71 | 6.85 | 13% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.502 ms | 9.10 | 0.14 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.482 ms | 11.32 | 12.03 | 23% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.201 ms | 5.24 | 20.96 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.865 ms | 5.86 | 11.71 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.389 ms | 12.08 | 12.08 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.392 ms | 12.05 | 12.05 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.387 ms | 32.00 | 0.38 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.396 ms | 8.83 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 27.390 ms | 9.80 | 0.06 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 27.400 ms | 9.80 | 0.06 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.371 ms | 9.81 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

