## ryzen-9-8945hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 63.2 GFLOP/s, f32 126.6 GFLOP/s |
| ... hardware, AVX-512F | f64 63.2, f32 126.6 GFLOP/s |
| ... this build, SSE2 | f64 40.7, f32 81.3 GFLOP/s |
| bandwidth | read 45.9, update 51.7, triad 37.9 GB/s (ceiling 51.7) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 105% | 128% | 0 |
| blas2 | 10 | 64% | 77% | 0 |
| blas3 | 10 | 17% | 31% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 16% | 53% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 30% | 31% | 0 |
| mg | 1 | 29% | 29% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 60% | 60% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.506 ms | 6.70 | 53.57 | 104% | ok |
| blas1 | dot | f32 | n=8388608 | 1.238 ms | 13.55 | 54.20 | 105% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.691 ms | 4.55 | 54.54 | 105% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.734 ms | 9.68 | 58.06 | 112% | ok |
| blas1 | scale | f64 | n=8388608 | 2.025 ms | 4.14 | 66.28 | 128% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.182 ms | 14.20 | 56.79 | 110% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.444 ms | 1.30 | 10.41 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.456 ms | 1.30 | 10.40 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.441 ms | 1.30 | 10.42 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 106.259 ms | 1.74 | 1.26 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.401 ms | 9.87 | 39.48 | 76% | ok |
| blas2 | gemv | f32 | n=4096 | 1.681 ms | 19.96 | 39.94 | 77% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 22.715 ms | 1.48 | 5.91 | 11% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 7.382 ms | 4.55 | 9.10 | 18% | ok |
| blas2 | symv | f64 | n=4096 | 10.190 ms | 3.29 | 6.59 | 13% | ok |
| blas2 | trmv | f64 | n=4096 | 2.028 ms | 8.27 | 33.12 | 64% | ok |
| blas2 | trmv | f32 | n=4096 | 1.012 ms | 16.57 | 33.17 | 64% | ok |
| blas2 | trsv | f64 | n=4096 | 1.971 ms | 8.51 | 34.08 | 66% | ok |
| blas2 | trsv | f32 | n=4096 | 1.013 ms | 16.56 | 33.15 | 64% | ok |
| blas2 | ger | f64 | n=4096 | 16.385 ms | 2.05 | 16.39 | 32% | ok |
| blas3 | gemm | f64 | 512x512x512 | 13.883 ms | 19.34 | 0.45 | 31% | ok |
| blas3 | gemm | f32 | 512x512x512 | 6.958 ms | 38.58 | 0.45 | 30% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 9.319 ms | 14.40 | 0.68 | 23% | ok |
| blas3 | symm | f64 | 512x512x512 | 15.136 ms | 17.73 | 0.42 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.781 ms | 8.00 | 0.25 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 29.373 ms | 9.14 | 0.21 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 12.214 ms | 10.99 | 0.43 | 17% | ok |
| blas3 | trsm | f64 | 512x512x512 | 12.407 ms | 10.82 | 0.42 | 17% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.238 ms | 16.29 | 0.32 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.342 ms | 0.47 | 3.74 | 7% | ok |
| factor | lu | f64 | 512x512 | 40.283 ms | 2.22 | 0.10 | 4% | ok |
| factor | lu | f32 | 512x512 | 39.171 ms | 2.28 | 0.05 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.374 ms | 3.11 | 0.29 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 13.806 ms | 3.24 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.509 ms | 3.08 | 0.29 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 208.838 ms | 0.21 | 0.02 | 0% | ok |
| factor | qr | f64 | 512x512 | 818.881 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 82.060 ms | 2.18 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 942.390 ms | 0.47 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.962 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 111.863 ms | 2.40 | 0.04 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 42.353 ms | 2.11 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 48.246 ms | 0.46 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 167.565 ms | 0.90 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 650.720 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 75.665 ms | 2.22 | 0.01 | 4% | ok |
| eigen | schur | f64 | 256x256 | 146.790 ms | 2.86 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 152.497 ms | 2.42 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 33.081 ms | 7.61 | 0.03 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 332.851 ms | 3.02 | 0.00 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 33.815 ms | 0.25 | 3.97 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 60.647 ms | 0.14 | 2.21 | 4% | ok |
| elementwise | sin | f64 | n=8388608 | 51.013 ms | 0.16 | 2.63 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.721 ms | 0.31 | 5.02 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 39.385 ms | 0.21 | 3.41 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.998 ms | 2.62 | 27.28 | 53% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 4.006 ms | 0.65 | 6.80 | 13% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.782 ms | 5.38 | 11.31 | 22% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.242 ms | 0.15 | 0.54 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.956 ms | 1.03 | 8.51 | 16%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 125.450 ms | 0.09 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 37.056 ms | 0.30 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.893 ms | 0.56 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 66.158 ms | 0.34 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 91.351 ms | 0.24 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 31.941 ms | 0.70 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 105.233 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.598 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.009 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.498 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.995 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.366 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 39.183 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 34.905 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.124 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.753 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 19.896 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 39.593 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 36.804 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.005 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 24.632 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 23.311 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 24.859 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 28.186 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.048 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.926 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.771 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.326 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.227 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.806 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.871 ms | 1.40 | 15.68 | 30% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.837 ms | 1.43 | 15.97 | 31% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.226 ms | 1.18 | 13.17 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.519 ms | 1.97 | 14.89 | 29% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.853 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 58.734 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 72.987 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.121 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 2.174 ms | 7.72 | 30.87 | 60% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 19.282 ms | 13.92 | 0.22 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.578 ms | 29.01 | 30.82 | 60% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.428 ms | 11.75 | 47.00 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.494 ms | 33.99 | 67.98 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.430 ms | 38.99 | 38.99 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.389 ms | 43.18 | 43.18 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.942 ms | 14.17 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.051 ms | 12.17 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.725 ms | 11.81 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.586 ms | 58.53 | 0.34 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.558 ms | 58.89 | 0.35 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

