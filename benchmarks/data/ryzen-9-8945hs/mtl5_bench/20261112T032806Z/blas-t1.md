## ryzen-9-8945hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 63.3 GFLOP/s, f32 126.6 GFLOP/s |
| ... hardware, AVX-512F | f64 63.3, f32 126.6 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 37.7, update 51.7, triad 37.5 GB/s (ceiling 51.7) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 18% | 98% | 0 |
| blas2 | 10 | 76% | 83% | 0 |
| blas3 | 10 | 71% | 93% | 0 |
| factor | 12 | 13% | 23% | 0 |
| eigen | 8 | 17% | 53% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 22% | 58% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 31% | 32% | 0 |
| mg | 1 | 29% | 29% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 14.419 ms | 1.16 | 9.31 | 18% | ok |
| blas1 | dot | f32 | n=8388608 | 12.263 ms | 1.37 | 5.47 | 11% | ok |
| blas1 | axpy | f64 | n=8388608 | 14.961 ms | 1.12 | 13.46 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 12.802 ms | 1.31 | 7.86 | 15% | ok |
| blas1 | scale | f64 | n=8388608 | 2.645 ms | 3.17 | 50.75 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 15.421 ms | 1.09 | 4.35 | 8% | ok |
| blas1 | one_norm | f64 | n=8388608 | 7.357 ms | 1.14 | 9.12 | 18% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.411 ms | 1.31 | 10.47 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.396 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 111.774 ms | 1.65 | 1.20 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.276 ms | 10.24 | 40.99 | 79% | ok |
| blas2 | gemv | f32 | n=4096 | 1.651 ms | 20.32 | 40.66 | 79% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.593 ms | 9.34 | 37.38 | 72% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.556 ms | 21.57 | 43.16 | 83% | ok |
| blas2 | symv | f64 | n=4096 | 2.032 ms | 16.51 | 33.06 | 64% | ok |
| blas2 | trmv | f64 | n=4096 | 1.707 ms | 9.83 | 39.36 | 76% | ok |
| blas2 | trmv | f32 | n=4096 | 0.968 ms | 17.33 | 34.69 | 67% | ok |
| blas2 | trsv | f64 | n=4096 | 1.693 ms | 9.91 | 39.68 | 77% | ok |
| blas2 | trsv | f32 | n=4096 | 0.921 ms | 18.21 | 36.45 | 70% | ok |
| blas2 | ger | f64 | n=4096 | 16.270 ms | 2.06 | 16.50 | 32% | ok |
| blas3 | gemm | f64 | 512x512x512 | 4.563 ms | 58.82 | 1.38 | 93% | ok |
| blas3 | gemm | f32 | 512x512x512 | 2.374 ms | 113.07 | 1.33 | 89% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 3.425 ms | 39.18 | 1.84 | 62% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.749 ms | 56.53 | 1.32 | 89% | ok |
| blas3 | syrk | f64 | 512x512x512 | 3.710 ms | 36.17 | 1.13 | 57% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 5.998 ms | 44.75 | 1.05 | 71% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.361 ms | 56.85 | 2.22 | 90% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.504 ms | 38.31 | 1.50 | 60% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.637 ms | 82.01 | 1.60 | 65% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.129 ms | 0.48 | 3.82 | 7% | ok |
| factor | lu | f64 | 512x512 | 6.393 ms | 14.00 | 0.66 | 22% | ok |
| factor | lu | f32 | 512x512 | 3.063 ms | 29.21 | 0.68 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 5.484 ms | 8.16 | 0.76 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 2.114 ms | 21.16 | 0.99 | 17% | ok |
| factor | ldlt | f64 | 512x512 | 14.811 ms | 3.02 | 0.28 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 211.072 ms | 0.21 | 0.02 | 0% | ok |
| factor | qr | f64 | 512x512 | 12.960 ms | 13.81 | 0.32 | 22% | ok |
| factor | lq | f64 | 512x512 | 81.665 ms | 2.19 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 935.107 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.952 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 76.588 ms | 3.50 | 0.05 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.977 ms | 12.82 | 0.60 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 2.121 ms | 10.55 | 0.49 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 160.066 ms | 0.94 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 650.132 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 20.908 ms | 8.02 | 0.05 | 13% | ok |
| eigen | schur | f64 | 256x256 | 27.263 ms | 15.38 | 0.04 | 24% | ok |
| eigen | svd | f64 | 256x256 | 10.946 ms | 33.72 | 0.10 | 53% | ok |
| eigen | expm | f64 | 256x256 | 20.395 ms | 12.34 | 0.05 | 19% | ok |
| eigen | sylvester | f64 | 256x256 | 91.010 ms | 11.06 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 33.461 ms | 0.25 | 4.01 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 53.847 ms | 0.16 | 2.49 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 47.973 ms | 0.17 | 2.80 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.544 ms | 0.32 | 5.06 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 42.476 ms | 0.20 | 3.16 | 6% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.914 ms | 2.86 | 29.80 | 58% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.433 ms | 1.83 | 19.01 | 37% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.879 ms | 5.32 | 11.18 | 22% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 22.341 ms | 0.15 | 0.51 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.955 ms | 1.03 | 8.51 | 16%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 129.966 ms | 0.09 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 37.994 ms | 0.29 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 22.885 ms | 0.49 | 0.03 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 67.095 ms | 0.33 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 91.993 ms | 0.24 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 31.939 ms | 0.70 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 101.803 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.551 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.957 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.504 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.597 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 50.950 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 57.576 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 39.396 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.799 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.238 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.672 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 68.911 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.678 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.558 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 30.784 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 29.718 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 33.378 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.427 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.046 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.938 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 27.106 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.287 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.503 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.031 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.765 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.756 ms | 1.49 | 16.70 | 32% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.838 ms | 1.42 | 15.96 | 31% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.219 ms | 1.18 | 13.22 | 26% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.322 ms | 1.99 | 15.04 | 29% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.615 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.327 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 69.312 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.928 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 14.628 ms | 1.15 | 4.59 | 9% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 37.348 ms | 7.19 | 0.11 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.836 ms | 9.14 | 9.71 | 19% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.277 ms | 3.92 | 15.69 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.235 ms | 5.19 | 10.37 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.238 ms | 26.22 | 0.31 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.570 ms | 6.96 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 38.705 ms | 6.94 | 0.04 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 38.695 ms | 6.94 | 0.04 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.459 ms | 7.79 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

