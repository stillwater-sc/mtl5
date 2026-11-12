## ryzen-7-8845hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 80.0 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.0, f32 160.1 GFLOP/s |
| ... this build, SSE2 | f64 40.0, f32 80.1 GFLOP/s |
| bandwidth | read 47.5, update 53.4, triad 38.9 GB/s (ceiling 53.4) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 105% | 132% | 0 |
| blas2 | 10 | 63% | 76% | 0 |
| blas3 | 10 | 17% | 31% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 8% | 11% | 0 |
| sparse | 5 | 18% | 59% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 34% | 37% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 71% | 71% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.403 ms | 6.98 | 55.86 | 105% | ok |
| blas1 | dot | f32 | n=8388608 | 1.195 ms | 14.04 | 56.16 | 105% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.505 ms | 4.79 | 57.44 | 108% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.804 ms | 9.30 | 55.79 | 105% | ok |
| blas1 | scale | f64 | n=8388608 | 1.901 ms | 4.41 | 70.61 | 132% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.160 ms | 14.47 | 57.86 | 108% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.063 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.071 ms | 1.65 | 13.23 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 85.060 ms | 2.17 | 1.58 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.312 ms | 10.13 | 40.54 | 76% | ok |
| blas2 | gemv | f32 | n=4096 | 1.766 ms | 19.00 | 38.01 | 71% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 20.625 ms | 1.63 | 6.51 | 12% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 9.826 ms | 3.41 | 6.83 | 13% | ok |
| blas2 | symv | f64 | n=4096 | 8.482 ms | 3.96 | 7.92 | 15% | ok |
| blas2 | trmv | f64 | n=4096 | 1.820 ms | 9.22 | 36.90 | 69% | ok |
| blas2 | trmv | f32 | n=4096 | 0.994 ms | 16.87 | 33.78 | 63% | ok |
| blas2 | trsv | f64 | n=4096 | 1.794 ms | 9.35 | 37.43 | 70% | ok |
| blas2 | trsv | f32 | n=4096 | 1.079 ms | 15.55 | 31.14 | 58% | ok |
| blas2 | ger | f64 | n=4096 | 13.411 ms | 2.50 | 20.02 | 38% | ok |
| blas3 | gemm | f64 | 512x512x512 | 10.969 ms | 24.47 | 0.57 | 31% | ok |
| blas3 | gemm | f32 | 512x512x512 | 5.536 ms | 48.49 | 0.57 | 30% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 7.405 ms | 18.13 | 0.85 | 23% | ok |
| blas3 | symm | f64 | 512x512x512 | 12.086 ms | 22.21 | 0.52 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.389 ms | 10.02 | 0.31 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 22.961 ms | 11.69 | 0.27 | 15% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.600 ms | 13.98 | 0.55 | 17% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.771 ms | 13.74 | 0.54 | 17% | ok |
| blas3 | trsm | f32 | 512x512x512 | 6.516 ms | 20.60 | 0.40 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.524 ms | 0.56 | 4.46 | 8% | ok |
| factor | lu | f64 | 512x512 | 32.712 ms | 2.74 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 31.529 ms | 2.84 | 0.07 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.391 ms | 3.93 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 11.014 ms | 4.06 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.462 ms | 3.90 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 165.982 ms | 0.27 | 0.03 | 0% | ok |
| factor | qr | f64 | 512x512 | 647.066 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 64.907 ms | 2.76 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 742.786 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.890 ms | 0.27 | 1.08 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 89.194 ms | 3.01 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 34.374 ms | 2.60 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 38.142 ms | 0.59 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 128.810 ms | 1.17 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 499.047 ms | 0.30 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 58.495 ms | 2.87 | 0.02 | 4% | ok |
| eigen | schur | f64 | 256x256 | 114.551 ms | 3.66 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 117.887 ms | 3.13 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 26.188 ms | 9.61 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 261.501 ms | 3.85 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 28.096 ms | 0.30 | 4.78 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 42.180 ms | 0.20 | 3.18 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 39.954 ms | 0.21 | 3.36 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.972 ms | 0.38 | 6.11 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 31.964 ms | 0.26 | 4.20 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.858 ms | 3.05 | 31.73 | 59% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 5.473 ms | 0.48 | 4.98 | 9% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.099 ms | 6.87 | 14.44 | 27% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.363 ms | 0.20 | 0.70 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.761 ms | 1.14 | 9.45 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 102.554 ms | 0.11 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 29.615 ms | 0.38 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.753 ms | 0.71 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.005 ms | 0.41 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 72.574 ms | 0.31 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.700 ms | 0.87 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 86.297 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.608 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.544 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.413 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.058 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.666 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 31.042 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 28.572 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.474 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.880 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.687 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 31.050 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.304 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.324 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 19.311 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 18.383 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 19.705 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 22.013 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.356 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 25.996 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.801 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.725 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.648 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.616 ms | 1.62 | 18.15 | 34% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.505 ms | 1.74 | 19.49 | 37% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.784 ms | 1.47 | 16.44 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.865 ms | 2.42 | 18.32 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 31.901 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 46.498 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 54.119 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.437 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.769 ms | 9.48 | 37.93 | 71% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 15.267 ms | 17.58 | 0.27 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.473 ms | 35.46 | 37.68 | 71% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.222 ms | 13.73 | 54.90 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.459 ms | 36.53 | 73.06 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.346 ms | 48.46 | 48.46 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.317 ms | 53.01 | 53.01 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 14.989 ms | 17.91 | 0.21 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 17.356 ms | 15.47 | 0.12 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 17.746 ms | 15.13 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.662 ms | 73.30 | 0.43 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.648 ms | 73.58 | 0.43 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

