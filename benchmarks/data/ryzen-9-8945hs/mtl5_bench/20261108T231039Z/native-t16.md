## AMD Ryzen 9 8945HS w/ Radeon 780M Graphics -- native, 16 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36247 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `685e2711d5ff` |
| FMA peak (ceiling) | f64 602.3 GFLOP/s, f32 1196.6 GFLOP/s |
| ... hardware, AVX-512F | f64 602.3, f32 1196.6 GFLOP/s |
| ... this build, scalar | f64 20.2, f32 19.8 GFLOP/s |
| bandwidth | read 53.2, update 55.0, triad 41.0 GB/s (ceiling 55.0) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 98% | 0 | 17% |
| blas2 | 6 | 34% | 93% | 0 | 6% |
| blas3 | 10 | 2% | 12% | 0 | 7% |
| factor | 12 | 0% | 2% | 0 | 8% |
| eigen | 8 | 1% | 2% | 0 | 6% |
| elementwise | 5 | 8% | 12% | 0 | 6% |
| sparse | 5 | 37% | 130% | 0 | 6% |
| direct | 7 | 0% | 0% | 0 | 6% |
| ordering | 4 | n/a | n/a | 0 | 6% |
| krylov | 13 | n/a | n/a | 0 | 7% |
| pc | 7 | n/a | n/a | 0 | 6% |
| smoother | 3 | 35% | 36% | 0 | 7% |
| mg | 1 | 31% | 31% | 0 | 7% |
| sparse_eig | 4 | n/a | n/a | 0 | 7% |
| mixed | 3 | 12% | 25% | 0 | 6% |
| integer | 9 | n/a | n/a | 0 | 20% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.531 ms | 6.63 | 53.04 | 96% | ok | 3.7x (23%) |
| blas1 | dot | f32 | n=8388608 | 1.649 ms | 10.18 | 40.71 | 74% | ok | 6.3x (40%) |
| blas1 | axpy | f64 | n=8388608 | 3.911 ms | 4.29 | 51.48 | 94% | ok | 2.7x (17%) |
| blas1 | axpy | f32 | n=8388608 | 2.227 ms | 7.53 | 45.20 | 82% | ok | 4.6x (29%) |
| blas1 | scale | f64 | n=8388608 | 2.487 ms | 3.37 | 53.98 | 98% | ok | 0.9x (6%) |
| blas1 | two_norm | f64 | n=8388608 | 1.282 ms | 13.08 | 52.34 | 95% | ok | 7.7x (48%) |
| blas1 | one_norm | f64 | n=8388608 | 4.965 ms | 1.69 | 13.52 | 25% | ok | 1.0x (6%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.158 ms | 1.63 | 13.01 | 24% | ok | 1.0x (6%) |
| blas1 | sum | f64 | n=8388608 | 4.971 ms | 1.69 | 13.50 | 25% | ok | 1.0x (6%) |
| blas1 | dot_k3 | f64 | n=8388608 | 85.340 ms | 2.16 | 1.57 | 3% | ok | 0.9x (6%) |
| blas2 | gemv | f64 | n=4096 | 2.639 ms | 12.71 | 50.88 | 93% | ok | 4.0x (25%) |
| blas2 | gemv | f32 | n=4096 | 1.540 ms | 21.79 | 43.61 | 79% | ok | 6.5x (40%) |
| blas2 | symv | f64 | n=4096 | 8.666 ms | 3.87 | 7.75 | 14% | ok | 1.0x (6%) |
| blas2 | trmv | f64 | n=4096 | 5.853 ms | 2.87 | 11.48 | 21% | ok | 1.0x (6%) |
| blas2 | trsv | f64 | n=4096 | 5.861 ms | 2.86 | 11.46 | 21% | ok | 1.0x (6%) |
| blas2 | ger | f64 | n=4096 | 14.179 ms | 2.37 | 18.94 | 34% | ok | 1.0x (6%) |
| blas3 | gemm | f64 | 512x512x512 | 3.586 ms | 74.86 | 1.75 | 12% | ok | 4.7x (30%) |
| blas3 | gemm | f32 | 512x512x512 | 1.969 ms | 136.37 | 1.60 | 11% | ok | 4.5x (28%) |
| blas3 | gemmt | f64 | 512x512x512 | 12.409 ms | 10.82 | 0.51 | 2% | ok | 1.1x (7%) |
| blas3 | symm | f64 | 512x512x512 | 22.864 ms | 11.74 | 0.28 | 2% | ok | 1.1x (7%) |
| blas3 | syrk | f64 | 512x512x512 | 39.358 ms | 3.41 | 0.11 | 1% | ok | 1.1x (7%) |
| blas3 | syr2k | f64 | 512x512x512 | 75.624 ms | 3.55 | 0.08 | 1% | ok | 1.0x (6%) |
| blas3 | trmm | f64 | 512x512x512 | 26.578 ms | 5.05 | 0.20 | 1% | ok | 1.0x (6%) |
| blas3 | trsm | f64 | 512x512x512 | 31.219 ms | 4.30 | 0.17 | 1% | ok | 1.0x (6%) |
| blas3 | trsm | f32 | 512x512x512 | 29.604 ms | 4.53 | 0.09 | 0% | ok | 1.0x (6%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.991 ms | 0.59 | 4.72 | 9% | ok | 1.0x (6%) |
| factor | lu | f64 | 512x512 | 26.537 ms | 3.37 | 0.16 | 1% | ok | 1.3x (8%) |
| factor | lu | f32 | 512x512 | 24.738 ms | 3.62 | 0.08 | 0% | ok | 1.3x (8%) |
| factor | cholesky | f64 | 512x512 | 17.935 ms | 2.49 | 0.23 | 0%\* | ok | 0.8x (5%) |
| factor | cholesky | f32 | 512x512 | 11.277 ms | 3.97 | 0.19 | 0%\* | ok | 1.0x (6%) |
| factor | ldlt | f64 | 512x512 | 18.289 ms | 2.45 | 0.23 | 0%\* | ok | 0.6x (4%) |
| factor | ldlt_bk | f64 | 512x512 | 170.746 ms | 0.26 | 0.02 | 0%\* | ok | 1.0x (6%) |
| factor | qr | f64 | 512x512 | 473.817 ms | 0.38 | 0.01 | 0% | ok | 1.7x (11%) |
| factor | lq | f64 | 512x512 | 46.030 ms | 3.89 | 0.09 | 1% | ok | 1.4x (9%) |
| factor | hessenberg | f64 | 512x512 | 624.079 ms | 0.72 | 0.01 | 0% | ok | 1.5x (9%) |
| factor | householder | f64 | 512x512 | 3.351 ms | 0.31 | 1.25 | 2%\* | ok | 1.2x (7%) |
| factor | inv | f64 | 512x512 | 148.570 ms | 1.81 | 0.03 | 0% | ok | 1.0x (6%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 28.886 ms | 3.10 | 0.15 | 1% | ok | 1.2x (8%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 43.858 ms | 0.51 | 0.02 | 0% | ok | 0.9x (5%) |
| eigen | eigen_symmetric | f64 | 256x256 | 133.385 ms | 1.13 | 0.01 | 0% | ok | 1.0x (6%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 474.234 ms | 0.32 | 0.00 | 0% | ok | 1.0x (7%) |
| eigen | eigenvalue | f64 | 256x256 | 72.463 ms | 2.32 | 0.01 | 0% | ok | 0.9x (6%) |
| eigen | schur | f64 | 256x256 | 129.447 ms | 3.24 | 0.01 | 1% | ok | 1.0x (6%) |
| eigen | svd | f64 | 256x256 | 118.899 ms | 3.10 | 0.01 | 1% | ok | 1.0x (6%) |
| eigen | expm | f64 | 256x256 | 25.738 ms | 9.78 | 0.04 | 2% | ok | 1.4x (9%) |
| eigen | sylvester | f64 | 256x256 | 305.155 ms | 3.30 | 0.01 | 1% | ok | 1.0x (6%) |
| elementwise | exp | f64 | n=8388608 | 27.454 ms | 0.31 | 4.89 | 9% | ok | 1.0x (6%) |
| elementwise | log | f64 | n=8388608 | 42.511 ms | 0.20 | 3.16 | 6% | ok | 1.0x (6%) |
| elementwise | sin | f64 | n=8388608 | 38.620 ms | 0.22 | 3.48 | 6% | ok | 1.0x (6%) |
| elementwise | sqrt | f64 | n=8388608 | 20.344 ms | 0.41 | 6.60 | 12% | ok | 1.0x (6%) |
| elementwise | erf | f64 | n=8388608 | 31.714 ms | 0.26 | 4.23 | 8% | ok | 1.0x (6%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.380 ms | 6.89 | 71.72 | 130% | ok | 2.5x (16%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.355 ms | 1.93 | 20.09 | 37% | ok | 1.0x (6%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.257 ms | 18.55 | 39.01 | 71% | ok | 2.8x (17%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.916 ms | 0.25 | 0.89 | 2%\* | ok | 1.0x (6%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.951 ms | 0.68 | 5.64 | 10%\* | ok | 0.8x (5%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 99.854 ms | 0.11 | 0.01 | 0% | ok | 1.0x (6%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 32.900 ms | 0.34 | 0.02 | 0% | ok | 1.0x (6%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 20.229 ms | 0.55 | 0.04 | 0% | ok | 1.1x (7%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 55.482 ms | 0.40 | 0.01 | 0% | ok | 1.0x (7%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 75.041 ms | 0.30 | 0.01 | 0% | ok | 1.0x (6%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 27.339 ms | 0.82 | 0.03 | 0% | ok | 1.0x (6%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 264.092 ms | 0.01 | 0.00 | 0%\* | ok | 1.0x (7%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.210 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.860 ms | 0.00 | 0.00 | n/a | ok | 0.8x (5%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.459 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 20.208 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.442 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 46.531 ms | 0.00 | 0.00 | n/a | ok | 1.2x (8%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 30.886 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.868 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.895 ms | 0.00 | 0.00 | n/a | ok | 1.0x (7%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.890 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 52.375 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.076 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.439 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 32.171 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 34.510 ms | 0.00 | 0.00 | n/a | ok | 1.0x (7%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.643 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.504 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.047 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.622 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ilut | f64 | 128^2 grid | 24.963 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| pc | ic_0 | f64 | 128^2 grid | 2.070 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ildl | f64 | 128^2 grid | 1.970 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ssor | f64 | 128^2 grid | 0.028 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.831 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.500 ms | 1.75 | 19.55 | 36% | ok | 1.1x (7%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.505 ms | 1.74 | 19.48 | 35% | ok | 1.0x (6%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.798 ms | 1.46 | 16.31 | 30% | ok | 1.4x (8%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 17.098 ms | 2.25 | 17.00 | 31% | ok | 1.1x (7%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 35.952 ms | 0.00 | 0.00 | n/a | ok | 1.3x (8%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 50.493 ms | 0.00 | 0.00 | n/a | ok | 1.2x (7%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 88.719 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.754 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| mixed | dot | f32>f64 | n=8388608 | 9.810 ms | 1.71 | 6.84 | 12% | ok | 1.0x (6%) |
| mixed | gemm | f32>f64 | 512x512x512 | 6.503 ms | 41.28 | 0.64 | 7% | ok | 4.5x (28%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.290 ms | 13.00 | 13.82 | 25% | ok | 1.0x (6%) |
| integer | dot | i32 | n=8388608, |v| <= 15 | 3.214 ms | 5.22 | 20.88 | n/a | ok | 1.0x (6%) |
| integer | dot | i16>i32 | n=8388608, |v| <= 15 | 2.615 ms | 6.42 | 12.83 | n/a | ok | 1.0x (6%) |
| integer | dot | i8>i32 | n=8388608, |v| <= 15 | 1.370 ms | 12.25 | 12.25 | n/a | ok | 1.0x (6%) |
| integer | dot | u8i8>i32 | n=8388608, |v| <= 15 | 1.734 ms | 9.68 | 9.68 | n/a | ok | 1.0x (6%) |
| integer | gemm | i32 | 512^3, |v| <= 2047 | 1.420 ms | 189.00 | 2.21 | n/a | ok | 6.5x (40%) |
| integer | gemm | i16>i32 | 512^3, |v| <= 2047 | 5.879 ms | 45.66 | 0.36 | n/a | ok | 6.5x (41%) |
| integer | gemm | i8>i32 | 512^3, |v| <= 127 | 6.590 ms | 40.73 | 0.24 | n/a | ok | 4.7x (29%) |
| integer | gemm_quad | i8>i32 | 512^3, |v| <= 127 | 6.035 ms | 44.48 | 0.26 | n/a | ok | 5.1x (32%) |
| integer | gemm_quad | u8i8>i32 | 512^3, |v| <= 127 | 8.431 ms | 31.84 | 0.19 | n/a | ok | 3.3x (20%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

