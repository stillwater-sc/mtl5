## ryzen-9-8945hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.8 GFLOP/s, f32 126.6 GFLOP/s |
| ... hardware, AVX-512F | f64 78.8, f32 126.6 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 36.6, update 51.5, triad 37.8 GB/s (ceiling 51.5) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 99% | 0 |
| blas2 | 10 | 75% | 90% | 0 |
| blas3 | 10 | 64% | 88% | 0 |
| factor | 12 | 10% | 22% | 0 |
| eigen | 8 | 14% | 43% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 21% | 59% | 0 |
| direct | 19 | 1% | 7% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 12% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 23% | 34% | 0 |
| mg | 1 | 28% | 28% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.775 ms | 1.42 | 11.40 | 22% | ok |
| blas1 | dot | f32 | n=8388608 | 12.240 ms | 1.37 | 5.48 | 11% | ok |
| blas1 | axpy | f64 | n=8388608 | 14.965 ms | 1.12 | 13.45 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 15.067 ms | 1.11 | 6.68 | 13% | ok |
| blas1 | scale | f64 | n=8388608 | 2.642 ms | 3.18 | 50.81 | 99% | ok |
| blas1 | two_norm | f64 | n=8388608 | 12.264 ms | 1.37 | 5.47 | 11% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.399 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.410 ms | 1.31 | 10.47 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.396 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 101.404 ms | 1.82 | 1.32 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.278 ms | 10.24 | 40.97 | 79% | ok |
| blas2 | gemv | f32 | n=4096 | 1.633 ms | 20.55 | 41.11 | 80% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.940 ms | 8.52 | 34.08 | 66% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.443 ms | 23.26 | 46.54 | 90% | ok |
| blas2 | symv | f64 | n=4096 | 1.985 ms | 16.90 | 33.83 | 66% | ok |
| blas2 | trmv | f64 | n=4096 | 1.747 ms | 9.60 | 38.45 | 75% | ok |
| blas2 | trmv | f32 | n=4096 | 0.928 ms | 18.08 | 36.19 | 70% | ok |
| blas2 | trsv | f64 | n=4096 | 1.753 ms | 9.57 | 38.33 | 74% | ok |
| blas2 | trsv | f32 | n=4096 | 0.946 ms | 17.73 | 35.49 | 69% | ok |
| blas2 | ger | f64 | n=4096 | 6.640 ms | 5.05 | 40.44 | 78% | ok |
| blas3 | gemm | f64 | 512x512x512 | 4.660 ms | 57.60 | 1.35 | 73% | ok |
| blas3 | gemm | f32 | 512x512x512 | 2.412 ms | 111.27 | 1.30 | 88% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 3.566 ms | 37.64 | 1.76 | 48% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.860 ms | 55.23 | 1.29 | 70% | ok |
| blas3 | syrk | f64 | 512x512x512 | 3.883 ms | 34.57 | 1.08 | 44% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 6.237 ms | 43.04 | 1.01 | 55% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.409 ms | 55.72 | 2.18 | 71% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.542 ms | 37.89 | 1.48 | 48% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.646 ms | 81.52 | 1.59 | 64% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.276 ms | 0.52 | 4.13 | 8% | ok |
| factor | lu | f64 | 512x512 | 6.321 ms | 14.16 | 0.66 | 18% | ok |
| factor | lu | f32 | 512x512 | 3.156 ms | 28.35 | 0.66 | 22% | ok |
| factor | cholesky | f64 | 512x512 | 5.456 ms | 8.20 | 0.77 | 10% | ok |
| factor | cholesky | f32 | 512x512 | 2.266 ms | 19.74 | 0.93 | 16% | ok |
| factor | ldlt | f64 | 512x512 | 14.818 ms | 3.02 | 0.28 | 4% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.431 ms | 3.60 | 0.34 | 5% | ok |
| factor | qr | f64 | 512x512 | 12.936 ms | 13.83 | 0.32 | 18% | ok |
| factor | lq | f64 | 512x512 | 80.965 ms | 2.21 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 941.376 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.896 ms | 0.21 | 0.86 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 76.172 ms | 3.52 | 0.06 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.957 ms | 12.86 | 0.60 | 16% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 2.205 ms | 10.14 | 0.48 | 13% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 162.599 ms | 0.93 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 630.339 ms | 0.24 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 21.223 ms | 7.91 | 0.05 | 10% | ok |
| eigen | schur | f64 | 256x256 | 27.656 ms | 15.17 | 0.04 | 19% | ok |
| eigen | svd | f64 | 256x256 | 10.896 ms | 33.87 | 0.10 | 43% | ok |
| eigen | expm | f64 | 256x256 | 19.609 ms | 12.83 | 0.05 | 16% | ok |
| eigen | sylvester | f64 | 256x256 | 92.602 ms | 10.87 | 0.02 | 14% | ok |
| elementwise | exp | f64 | n=8388608 | 32.560 ms | 0.26 | 4.12 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 52.011 ms | 0.16 | 2.58 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 47.070 ms | 0.18 | 2.85 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 25.593 ms | 0.33 | 5.24 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 41.695 ms | 0.20 | 3.22 | 6% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.894 ms | 2.93 | 30.47 | 59% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.545 ms | 1.69 | 17.63 | 34% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.952 ms | 5.27 | 11.07 | 21% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.446 ms | 0.17 | 0.59 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.937 ms | 1.04 | 8.59 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 24.259 ms | 0.46 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 40.765 ms | 0.27 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 20.218 ms | 0.55 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.444 ms | 0.34 | 0.01 | 0% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 70.356 ms | 0.32 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.014 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 43.238 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.577 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.086 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.477 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.242 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.824 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.598 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.523 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.959 ms | 0.79 | 3.15 | 6%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.055 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.010 ms | 1.60 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.896 ms | 0.84 | 3.37 | 7%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.178 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.803 ms | 0.39 | 0.01 | 0% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.608 ms | 0.47 | 1.88 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.014 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 58.973 ms | 0.38 | 0.01 | 0% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.381 ms | 0.55 | 2.19 | 4%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.975 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 338.853 ms | 0.91 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.182 ms | 1.54 | 6.18 | 12%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.204 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 85.718 ms | 3.60 | 0.01 | 5% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.266 ms | 1.49 | 5.95 | 12%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.856 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1120.360 ms | 0.55 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.214 ms | 0.30 | 1.20 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.740 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 639.654 ms | 0.96 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.274 ms | 0.30 | 1.20 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.498 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 74.664 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.050 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.178 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.530 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.454 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 68.623 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.784 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.119 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.252 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 31.447 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 32.544 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.784 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.054 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.925 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 27.869 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.222 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.137 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.664 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.691 ms | 1.55 | 17.34 | 34% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.437 ms | 1.07 | 12.04 | 23% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.847 ms | 0.92 | 10.30 | 20% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 20.130 ms | 1.91 | 14.44 | 28% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 43.177 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 43.899 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 73.262 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.826 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.256 ms | 1.37 | 5.48 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 37.760 ms | 7.11 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.835 ms | 9.14 | 9.71 | 19% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.051 ms | 4.14 | 16.56 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.246 ms | 5.17 | 10.34 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.123 ms | 7.90 | 7.90 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.124 ms | 7.90 | 7.90 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.240 ms | 26.21 | 0.31 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.355 ms | 7.00 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 34.581 ms | 7.76 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 34.815 ms | 7.71 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.571 ms | 7.76 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

