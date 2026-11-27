## i7-12700k -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.5 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.0, update 38.3, triad 32.5 GB/s (ceiling 38.3) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 98% | 0 |
| blas2 | 10 | 79% | 95% | 0 |
| blas3 | 10 | 17% | 18% | 0 |
| factor | 12 | 8% | 19% | 0 |
| eigen | 8 | 6% | 13% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 50% | 95% | 0 |
| direct | 19 | 1% | 31% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 24% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 63% | 78% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 36% | 86% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.612 ms | 2.99 | 23.92 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.236 ms | 3.96 | 15.84 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.483 ms | 2.59 | 31.05 | 81% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.723 ms | 3.55 | 21.31 | 56% | ok |
| blas1 | scale | f64 | n=8388608 | 3.578 ms | 2.34 | 37.51 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.940 ms | 3.40 | 13.58 | 35% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.044 ms | 2.07 | 16.60 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.047 ms | 1.19 | 9.52 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.993 ms | 2.10 | 16.81 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 77.522 ms | 2.38 | 1.73 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.678 ms | 9.12 | 36.51 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 2.729 ms | 12.30 | 24.60 | 64% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.783 ms | 7.02 | 28.08 | 73% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.221 ms | 15.10 | 30.22 | 79% | ok |
| blas2 | symv | f64 | n=4096 | 3.182 ms | 10.54 | 21.11 | 55% | ok |
| blas2 | trmv | f64 | n=4096 | 2.032 ms | 8.26 | 33.06 | 86% | ok |
| blas2 | trmv | f32 | n=4096 | 1.425 ms | 11.77 | 23.56 | 61% | ok |
| blas2 | trsv | f64 | n=4096 | 1.979 ms | 8.48 | 33.94 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 1.397 ms | 12.01 | 24.05 | 63% | ok |
| blas2 | ger | f64 | n=4096 | 7.688 ms | 4.36 | 34.93 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 19.871 ms | 13.51 | 0.32 | 17% | ok |
| blas3 | gemm | f32 | 512x512x512 | 9.996 ms | 26.86 | 0.31 | 17% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 12.529 ms | 10.71 | 0.50 | 14% | ok |
| blas3 | symm | f64 | 512x512x512 | 19.745 ms | 13.60 | 0.32 | 17% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.600 ms | 9.87 | 0.31 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 24.471 ms | 10.97 | 0.26 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.551 ms | 14.05 | 0.55 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.715 ms | 13.82 | 0.54 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.215 ms | 25.74 | 0.50 | 16% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.143 ms | 0.40 | 3.23 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.624 ms | 7.70 | 0.36 | 10% | ok |
| factor | lu | f32 | 512x512 | 4.489 ms | 19.93 | 0.47 | 13% | ok |
| factor | cholesky | f64 | 512x512 | 7.328 ms | 6.11 | 0.57 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.596 ms | 6.78 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.475 ms | 5.99 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 4.834 ms | 9.26 | 0.87 | 12% | ok |
| factor | qr | f64 | 512x512 | 234.563 ms | 0.76 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 25.838 ms | 6.93 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 275.555 ms | 1.62 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.642 ms | 0.64 | 2.55 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 54.086 ms | 4.96 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.955 ms | 15.02 | 0.70 | 19% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.739 ms | 1.26 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 58.463 ms | 2.58 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 749.050 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 33.215 ms | 5.05 | 0.03 | 6% | ok |
| eigen | schur | f64 | 256x256 | 81.636 ms | 5.14 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 111.975 ms | 3.30 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 25.712 ms | 9.79 | 0.04 | 13% | ok |
| eigen | sylvester | f64 | 256x256 | 190.794 ms | 5.28 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.389 ms | 0.21 | 3.41 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.913 ms | 0.18 | 2.92 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.135 ms | 0.14 | 2.23 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.066 ms | 0.29 | 4.62 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 204.049 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.744 ms | 3.52 | 36.61 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.916 ms | 2.86 | 29.73 | 78% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.634 ms | 9.04 | 19.00 | 50% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 5.263 ms | 0.62 | 2.19 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.329 ms | 1.51 | 12.52 | 33%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.536 ms | 0.77 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.967 ms | 0.45 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.404 ms | 1.33 | 0.09 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.919 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.095 ms | 0.51 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.624 ms | 1.35 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.933 ms | 0.08 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.363 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.980 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.482 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.185 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.215 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.091 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.804 ms | 0.87 | 0.06 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.253 ms | 2.98 | 11.94 | 31%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.059 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.073 ms | 2.75 | 0.18 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.251 ms | 3.02 | 12.07 | 31%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.104 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.872 ms | 0.62 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.630 ms | 1.20 | 4.80 | 13%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.453 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.310 ms | 0.58 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.639 ms | 1.18 | 4.73 | 12%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.920 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 205.875 ms | 1.50 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.456 ms | 2.31 | 9.26 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.634 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 59.916 ms | 5.15 | 0.01 | 7% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.457 ms | 2.31 | 9.25 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.887 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 581.940 ms | 1.06 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.537 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.254 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 453.938 ms | 1.36 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.538 ms | 0.45 | 1.79 | 5%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.318 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 48.204 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.958 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.574 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.146 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.139 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.617 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.223 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.332 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.182 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.720 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.282 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 19.195 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.042 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.561 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 11.520 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.087 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.017 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.703 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.986 ms | 2.65 | 29.74 | 78% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.221 ms | 2.14 | 24.03 | 63% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.564 ms | 1.67 | 18.75 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.335 ms | 2.88 | 21.80 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.787 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 30.699 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.389 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.023 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.927 ms | 3.41 | 13.62 | 36% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.987 ms | 6.89 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.543 ms | 30.90 | 32.83 | 86%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.314 ms | 7.25 | 29.00 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.999 ms | 16.80 | 33.59 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.723 ms | 9.74 | 9.74 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.724 ms | 9.73 | 9.73 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.543 ms | 15.30 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.819 ms | 12.89 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.232 ms | 26.23 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.229 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.805 ms | 10.01 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

