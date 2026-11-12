## ryzen-9-8945hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 65.3 GFLOP/s, f32 125.9 GFLOP/s |
| ... hardware, AVX-512F | f64 65.3, f32 125.9 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 36.7, update 52.1, triad 37.9 GB/s (ceiling 52.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 98% | 0 |
| blas2 | 10 | 69% | 88% | 0 |
| blas3 | 10 | 67% | 90% | 0 |
| factor | 12 | 12% | 23% | 0 |
| eigen | 8 | 17% | 52% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 22% | 56% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 27% | 32% | 0 |
| mg | 1 | 29% | 29% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 14.421 ms | 1.16 | 9.31 | 18% | ok |
| blas1 | dot | f32 | n=8388608 | 12.500 ms | 1.34 | 5.37 | 10% | ok |
| blas1 | axpy | f64 | n=8388608 | 15.001 ms | 1.12 | 13.42 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 12.800 ms | 1.31 | 7.86 | 15% | ok |
| blas1 | scale | f64 | n=8388608 | 2.643 ms | 3.17 | 50.77 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 13.389 ms | 1.25 | 5.01 | 10% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.393 ms | 1.31 | 10.50 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.419 ms | 1.31 | 10.46 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.50 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 111.621 ms | 1.65 | 1.20 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.712 ms | 9.04 | 36.18 | 69% | ok |
| blas2 | gemv | f32 | n=4096 | 1.474 ms | 22.77 | 45.56 | 88% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.564 ms | 9.41 | 37.67 | 72% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.560 ms | 21.51 | 43.04 | 83% | ok |
| blas2 | symv | f64 | n=4096 | 2.079 ms | 16.14 | 32.31 | 62% | ok |
| blas2 | trmv | f64 | n=4096 | 1.869 ms | 8.98 | 35.95 | 69% | ok |
| blas2 | trmv | f32 | n=4096 | 0.950 ms | 17.65 | 35.34 | 68% | ok |
| blas2 | trsv | f64 | n=4096 | 1.910 ms | 8.78 | 35.17 | 68% | ok |
| blas2 | trsv | f32 | n=4096 | 0.942 ms | 17.81 | 35.65 | 68% | ok |
| blas2 | ger | f64 | n=4096 | 16.284 ms | 2.06 | 16.49 | 32% | ok |
| blas3 | gemm | f64 | 512x512x512 | 4.577 ms | 58.65 | 1.37 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 2.377 ms | 112.93 | 1.32 | 90% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 3.421 ms | 39.24 | 1.84 | 60% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.759 ms | 56.41 | 1.32 | 86% | ok |
| blas3 | syrk | f64 | 512x512x512 | 3.713 ms | 36.15 | 1.13 | 55% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 6.137 ms | 43.74 | 1.03 | 67% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.363 ms | 56.80 | 2.22 | 87% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.503 ms | 38.32 | 1.50 | 59% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.638 ms | 81.94 | 1.60 | 65% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.075 ms | 0.48 | 3.83 | 7% | ok |
| factor | lu | f64 | 512x512 | 6.379 ms | 14.03 | 0.66 | 21% | ok |
| factor | lu | f32 | 512x512 | 3.067 ms | 29.18 | 0.68 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 5.494 ms | 8.14 | 0.76 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 2.107 ms | 21.24 | 1.00 | 17% | ok |
| factor | ldlt | f64 | 512x512 | 14.838 ms | 3.02 | 0.28 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 204.407 ms | 0.22 | 0.02 | 0% | ok |
| factor | qr | f64 | 512x512 | 12.976 ms | 13.79 | 0.32 | 21% | ok |
| factor | lq | f64 | 512x512 | 82.283 ms | 2.17 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 942.426 ms | 0.47 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.943 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 76.845 ms | 3.49 | 0.05 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.013 ms | 12.76 | 0.60 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 2.110 ms | 10.60 | 0.50 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 164.794 ms | 0.92 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 644.865 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 21.049 ms | 7.97 | 0.05 | 12% | ok |
| eigen | schur | f64 | 256x256 | 27.429 ms | 15.29 | 0.04 | 23% | ok |
| eigen | svd | f64 | 256x256 | 10.890 ms | 33.89 | 0.10 | 52% | ok |
| eigen | expm | f64 | 256x256 | 20.222 ms | 12.44 | 0.05 | 19% | ok |
| eigen | sylvester | f64 | 256x256 | 91.736 ms | 10.97 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 33.658 ms | 0.25 | 3.99 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 54.591 ms | 0.15 | 2.46 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 48.203 ms | 0.17 | 2.78 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.540 ms | 0.32 | 5.06 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 42.861 ms | 0.20 | 3.13 | 6% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.938 ms | 2.79 | 29.02 | 56% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.930 ms | 1.36 | 14.11 | 27% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.784 ms | 5.38 | 11.31 | 22% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.528 ms | 0.15 | 0.53 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.953 ms | 1.03 | 8.52 | 16%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 130.309 ms | 0.09 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.935 ms | 0.29 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 22.716 ms | 0.49 | 0.03 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 66.643 ms | 0.34 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 91.394 ms | 0.24 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.953 ms | 0.72 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 102.434 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.602 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.959 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.501 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.630 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 51.336 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.327 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.844 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.042 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 30.578 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.951 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 70.681 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 44.039 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.640 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 31.292 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 30.084 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 33.481 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.022 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.045 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.930 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.703 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.276 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.171 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.745 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.772 ms | 1.48 | 16.55 | 32% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.058 ms | 1.27 | 14.25 | 27% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.228 ms | 1.17 | 13.16 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.582 ms | 1.96 | 14.84 | 29% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.726 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.567 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 69.577 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.961 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 14.386 ms | 1.17 | 4.66 | 9% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 37.456 ms | 7.17 | 0.11 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.835 ms | 9.14 | 9.71 | 19% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.281 ms | 3.92 | 15.67 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.235 ms | 5.19 | 10.37 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.72 | 9.72 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.728 ms | 9.71 | 9.71 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.216 ms | 26.28 | 0.31 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.424 ms | 6.99 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 38.719 ms | 6.93 | 0.04 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 38.830 ms | 6.91 | 0.04 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.534 ms | 7.77 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

