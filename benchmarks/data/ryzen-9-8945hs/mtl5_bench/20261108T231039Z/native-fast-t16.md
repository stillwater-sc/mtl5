## AMD Ryzen 9 8945HS w/ Radeon 780M Graphics -- native-fast, 16 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36247 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `685e2711d5ff` |
| FMA peak (ceiling) | f64 604.8 GFLOP/s, f32 1213.5 GFLOP/s |
| ... hardware, AVX-512F | f64 604.8, f32 1213.5 GFLOP/s |
| ... this build, SSE2 | f64 299.8, f32 599.3 GFLOP/s |
| bandwidth | read 52.0, update 52.9, triad 39.8 GB/s (ceiling 52.9) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 106% | 120% | 0 | 6% |
| blas2 | 6 | 67% | 108% | 0 | 6% |
| blas3 | 10 | 4% | 15% | 0 | 10% |
| factor | 12 | 0% | 7% | 0 | 8% |
| eigen | 8 | 0% | 2% | 0 | 6% |
| elementwise | 5 | 7% | 10% | 0 | 5% |
| sparse | 5 | 37% | 159% | 0 | 9% |
| direct | 7 | 0% | 0% | 0 | 6% |
| ordering | 4 | n/a | n/a | 0 | 6% |
| krylov | 13 | n/a | n/a | 0 | 6% |
| pc | 7 | n/a | n/a | 0 | 6% |
| smoother | 3 | 27% | 31% | 0 | 5% |
| mg | 1 | 31% | 31% | 0 | 6% |
| sparse_eig | 4 | n/a | n/a | 0 | 6% |
| mixed | 3 | 64% | 65% | 0 | 6% |
| integer | 9 | n/a | n/a | 0 | 17% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.387 ms | 7.03 | 56.22 | 106% | ok | 1.1x (7%) |
| blas1 | dot | f32 | n=8388608 | 1.064 ms | 15.76 | 63.05 | 119% | ok | 1.1x (7%) |
| blas1 | axpy | f64 | n=8388608 | 3.669 ms | 4.57 | 54.87 | 104% | ok | 1.0x (6%) |
| blas1 | axpy | f32 | n=8388608 | 1.718 ms | 9.76 | 58.58 | 111% | ok | 1.0x (6%) |
| blas1 | scale | f64 | n=8388608 | 2.225 ms | 3.77 | 60.33 | 114% | ok | 1.0x (6%) |
| blas1 | two_norm | f64 | n=8388608 | 1.062 ms | 15.80 | 63.21 | 120% | ok | 1.1x (7%) |
| blas1 | one_norm | f64 | n=8388608 | 5.008 ms | 1.68 | 13.40 | 25% | ok | 1.0x (6%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.234 ms | 1.60 | 12.82 | 24% | ok | 1.0x (6%) |
| blas1 | sum | f64 | n=8388608 | 4.999 ms | 1.68 | 13.42 | 25% | ok | 1.0x (6%) |
| blas1 | dot_k3 | f64 | n=8388608 | 87.385 ms | 2.11 | 1.54 | 3% | ok | 0.9x (6%) |
| blas2 | gemv | f64 | n=4096 | 2.419 ms | 13.87 | 55.51 | 105% | ok | 1.4x (9%) |
| blas2 | gemv | f32 | n=4096 | 1.174 ms | 28.59 | 57.21 | 108% | ok | 1.3x (8%) |
| blas2 | symv | f64 | n=4096 | 10.979 ms | 3.06 | 6.12 | 12% | ok | 0.8x (5%) |
| blas2 | trmv | f64 | n=4096 | 1.907 ms | 8.80 | 35.23 | 67% | ok | 1.0x (6%) |
| blas2 | trsv | f64 | n=4096 | 1.960 ms | 8.56 | 34.28 | 65% | ok | 1.0x (6%) |
| blas2 | ger | f64 | n=4096 | 14.160 ms | 2.37 | 18.96 | 36% | ok | 1.0x (6%) |
| blas3 | gemm | f64 | 512x512x512 | 2.960 ms | 90.68 | 2.13 | 15% | ok | 4.2x (26%) |
| blas3 | gemm | f32 | 512x512x512 | 1.809 ms | 148.38 | 1.74 | 12% | ok | 3.3x (20%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.240 ms | 25.62 | 1.20 | 4% | ok | 1.5x (10%) |
| blas3 | symm | f64 | 512x512x512 | 3.670 ms | 73.14 | 1.71 | 12% | ok | 3.6x (23%) |
| blas3 | syrk | f64 | 512x512x512 | 22.893 ms | 5.86 | 0.18 | 1% | ok | 1.1x (7%) |
| blas3 | syr2k | f64 | 512x512x512 | 32.505 ms | 8.26 | 0.19 | 1% | ok | 2.3x (14%) |
| blas3 | trmm | f64 | 512x512x512 | 7.636 ms | 17.58 | 0.69 | 3% | ok | 1.4x (9%) |
| blas3 | trsm | f64 | 512x512x512 | 7.226 ms | 18.57 | 0.73 | 3% | ok | 1.7x (10%) |
| blas3 | trsm | f32 | 512x512x512 | 5.617 ms | 23.90 | 0.47 | 2% | ok | 1.3x (8%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.500 ms | 0.51 | 4.04 | 8% | ok | 0.9x (5%) |
| factor | lu | f64 | 512x512 | 27.398 ms | 3.27 | 0.15 | 1% | ok | 1.3x (8%) |
| factor | lu | f32 | 512x512 | 26.398 ms | 3.39 | 0.08 | 0% | ok | 1.3x (8%) |
| factor | cholesky | f64 | 512x512 | 12.307 ms | 3.64 | 0.34 | 1%\* | ok | 1.1x (7%) |
| factor | cholesky | f32 | 512x512 | 11.908 ms | 3.76 | 0.18 | 0%\* | ok | 0.9x (6%) |
| factor | ldlt | f64 | 512x512 | 12.439 ms | 3.60 | 0.34 | 1%\* | ok | 1.3x (8%) |
| factor | ldlt_bk | f64 | 512x512 | 177.705 ms | 0.25 | 0.02 | 0%\* | ok | 1.0x (6%) |
| factor | qr | f64 | 512x512 | 435.826 ms | 0.41 | 0.01 | 0% | ok | 2.0x (12%) |
| factor | lq | f64 | 512x512 | 48.609 ms | 3.68 | 0.09 | 1% | ok | 1.3x (8%) |
| factor | hessenberg | f64 | 512x512 | 517.322 ms | 0.86 | 0.01 | 0% | ok | 1.5x (10%) |
| factor | householder | f64 | 512x512 | 1.179 ms | 0.89 | 3.56 | 7%\* | ok | 3.9x (24%) |
| factor | inv | f64 | 512x512 | 115.532 ms | 2.32 | 0.04 | 0% | ok | 1.0x (6%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 31.650 ms | 2.83 | 0.13 | 0% | ok | 1.1x (7%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 38.367 ms | 0.58 | 0.03 | 0% | ok | 1.1x (7%) |
| eigen | eigen_symmetric | f64 | 256x256 | 130.422 ms | 1.16 | 0.01 | 0% | ok | 1.0x (6%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 538.470 ms | 0.28 | 0.00 | 0% | ok | 0.9x (6%) |
| eigen | eigenvalue | f64 | 256x256 | 58.864 ms | 2.85 | 0.02 | 0% | ok | 1.0x (6%) |
| eigen | schur | f64 | 256x256 | 120.082 ms | 3.49 | 0.01 | 1% | ok | 1.0x (6%) |
| eigen | svd | f64 | 256x256 | 124.708 ms | 2.96 | 0.01 | 0% | ok | 0.9x (6%) |
| eigen | expm | f64 | 256x256 | 24.788 ms | 10.15 | 0.04 | 2% | ok | 1.2x (7%) |
| eigen | sylvester | f64 | 256x256 | 277.213 ms | 3.63 | 0.01 | 1% | ok | 1.0x (6%) |
| elementwise | exp | f64 | n=8388608 | 36.584 ms | 0.23 | 3.67 | 7% | ok | 0.8x (5%) |
| elementwise | log | f64 | n=8388608 | 45.084 ms | 0.19 | 2.98 | 6% | ok | 0.9x (6%) |
| elementwise | sin | f64 | n=8388608 | 45.019 ms | 0.19 | 2.98 | 6% | ok | 0.9x (5%) |
| elementwise | sqrt | f64 | n=8388608 | 26.082 ms | 0.32 | 5.15 | 10% | ok | 0.8x (5%) |
| elementwise | erf | f64 | n=8388608 | 37.959 ms | 0.22 | 3.54 | 7% | ok | 0.8x (5%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.323 ms | 8.10 | 84.23 | 159% ! | ok | 2.8x (17%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.398 ms | 1.87 | 19.48 | 37% | ok | 1.4x (9%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.537 ms | 16.51 | 34.71 | 66% | ok | 2.4x (15%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.242 ms | 0.27 | 0.94 | 2%\* | ok | 1.0x (6%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.091 ms | 0.96 | 7.96 | 15%\* | ok | 1.0x (6%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 111.857 ms | 0.10 | 0.01 | 0% | ok | 0.9x (6%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.459 ms | 0.37 | 0.02 | 0% | ok | 1.0x (6%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 16.430 ms | 0.68 | 0.04 | 0% | ok | 1.0x (6%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 58.357 ms | 0.38 | 0.01 | 0% | ok | 0.9x (6%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 78.016 ms | 0.29 | 0.01 | 0% | ok | 0.9x (6%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 28.265 ms | 0.79 | 0.03 | 0% | ok | 0.9x (6%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 292.591 ms | 0.01 | 0.00 | 0%\* | ok | 1.0x (6%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.913 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.886 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.427 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 19.194 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 36.706 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 42.137 ms | 0.00 | 0.00 | n/a | ok | 0.8x (5%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 29.757 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 14.705 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 19.121 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.236 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 35.837 ms | 0.00 | 0.00 | n/a | ok | 0.9x (5%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.840 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.699 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 25.590 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 26.631 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.973 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.418 ms | 0.00 | 0.00 | n/a | ok | 0.9x (5%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.047 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.626 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ilut | f64 | 128^2 grid | 26.354 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ic_0 | f64 | 128^2 grid | 2.019 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ildl | f64 | 128^2 grid | 1.878 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok | 1.0x (7%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.710 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.080 ms | 1.26 | 14.10 | 27% | ok | 0.8x (5%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.765 ms | 1.48 | 16.62 | 31% | ok | 0.9x (6%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.186 ms | 1.20 | 13.41 | 25% | ok | 0.9x (5%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 17.712 ms | 2.17 | 16.41 | 31% | ok | 1.0x (6%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 36.531 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 50.737 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 93.215 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.839 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| mixed | dot | f32>f64 | n=8388608 | 1.956 ms | 8.58 | 34.30 | 65% | ok | 1.0x (6%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.725 ms | 56.81 | 0.89 | 9% | ok | 3.5x (22%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.529 ms | 31.73 | 33.72 | 64% | ok | 1.0x (6%) |
| integer | dot | i32 | n=8388608, |v| <= 15 | 1.509 ms | 11.12 | 44.47 | n/a | ok | 1.0x (6%) |
| integer | dot | i16>i32 | n=8388608, |v| <= 15 | 0.582 ms | 28.81 | 57.61 | n/a | ok | 1.0x (6%) |
| integer | dot | i8>i32 | n=8388608, |v| <= 15 | 0.401 ms | 41.88 | 41.88 | n/a | ok | 1.0x (6%) |
| integer | dot | u8i8>i32 | n=8388608, |v| <= 15 | 0.364 ms | 46.05 | 46.05 | n/a | ok | 0.9x (6%) |
| integer | gemm | i32 | 512^3, |v| <= 2047 | 4.619 ms | 58.11 | 0.68 | n/a | ok | 3.3x (21%) |
| integer | gemm | i16>i32 | 512^3, |v| <= 2047 | 5.194 ms | 51.68 | 0.40 | n/a | ok | 3.4x (21%) |
| integer | gemm | i8>i32 | 512^3, |v| <= 127 | 5.171 ms | 51.92 | 0.30 | n/a | ok | 3.4x (21%) |
| integer | gemm_quad | i8>i32 | 512^3, |v| <= 127 | 1.322 ms | 203.11 | 1.19 | n/a | ok | 2.8x (17%) |
| integer | gemm_quad | u8i8>i32 | 512^3, |v| <= 127 | 1.271 ms | 211.20 | 1.24 | n/a | ok | 2.7x (17%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

