# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy power scheme copy of Balanced with processor state 100-100% and EPP 0, boost aggressive; AC, 2026-11-12T03:39:40Z.

## ryzen-7-8845hs -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 79.9 GFLOP/s, f32 159.3 GFLOP/s |
| ... hardware, AVX-512F | f64 79.9, f32 159.3 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 43.1, update 53.2, triad 38.9 GB/s (ceiling 53.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 110% | 0 |
| blas2 | 10 | 53% | 80% | 0 |
| blas3 | 10 | 15% | 21% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 11% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 26% | 60% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 34% | 35% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 14% | 24% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 9.804 ms | 1.71 | 13.69 | 26% | ok |
| blas1 | dot | f32 | n=8388608 | 9.699 ms | 1.73 | 6.92 | 13% | ok |
| blas1 | axpy | f64 | n=8388608 | 11.852 ms | 1.42 | 16.99 | 32% | ok |
| blas1 | axpy | f32 | n=8388608 | 11.804 ms | 1.42 | 8.53 | 16% | ok |
| blas1 | scale | f64 | n=8388608 | 2.297 ms | 3.65 | 58.42 | 110% | ok |
| blas1 | two_norm | f64 | n=8388608 | 9.280 ms | 1.81 | 7.23 | 14% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.075 ms | 1.65 | 13.22 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.063 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 88.197 ms | 2.09 | 1.52 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.174 ms | 10.57 | 42.30 | 80% | ok |
| blas2 | gemv | f32 | n=4096 | 2.379 ms | 14.11 | 28.22 | 53% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.062 ms | 4.75 | 19.02 | 36% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.215 ms | 15.15 | 30.32 | 57% | ok |
| blas2 | symv | f64 | n=4096 | 8.589 ms | 3.91 | 7.82 | 15% | ok |
| blas2 | trmv | f64 | n=4096 | 1.811 ms | 9.26 | 37.09 | 70% | ok |
| blas2 | trmv | f32 | n=4096 | 1.508 ms | 11.13 | 22.28 | 42% | ok |
| blas2 | trsv | f64 | n=4096 | 1.790 ms | 9.37 | 37.54 | 71% | ok |
| blas2 | trsv | f32 | n=4096 | 1.564 ms | 10.73 | 21.48 | 40% | ok |
| blas2 | ger | f64 | n=4096 | 13.479 ms | 2.49 | 19.92 | 37% | ok |
| blas3 | gemm | f64 | 512x512x512 | 15.726 ms | 17.07 | 0.40 | 21% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.255 ms | 32.52 | 0.38 | 20% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.034 ms | 12.16 | 0.57 | 15% | ok |
| blas3 | symm | f64 | 512x512x512 | 15.954 ms | 16.83 | 0.39 | 21% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.480 ms | 9.96 | 0.31 | 12% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 22.836 ms | 11.76 | 0.28 | 15% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.617 ms | 5.24 | 0.20 | 7% | ok |
| blas3 | trsm | f64 | 512x512x512 | 28.121 ms | 4.77 | 0.19 | 6% | ok |
| blas3 | trsm | f32 | 512x512x512 | 27.018 ms | 4.97 | 0.10 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.193 ms | 0.58 | 4.62 | 9% | ok |
| factor | lu | f64 | 512x512 | 32.361 ms | 2.76 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 31.115 ms | 2.88 | 0.07 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.379 ms | 3.93 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 11.036 ms | 4.05 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.464 ms | 3.90 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 162.628 ms | 0.28 | 0.03 | 0% | ok |
| factor | qr | f64 | 512x512 | 645.984 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 64.987 ms | 2.75 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 741.265 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.880 ms | 0.27 | 1.08 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 89.804 ms | 2.99 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 34.068 ms | 2.63 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.480 ms | 0.60 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 125.038 ms | 1.21 | 0.01 | 2% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 507.004 ms | 0.30 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 58.866 ms | 2.85 | 0.02 | 4% | ok |
| eigen | schur | f64 | 256x256 | 115.454 ms | 3.63 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 118.009 ms | 3.13 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 29.955 ms | 8.40 | 0.04 | 11% | ok |
| eigen | sylvester | f64 | 256x256 | 264.371 ms | 3.81 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 32.711 ms | 0.26 | 4.10 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 42.275 ms | 0.20 | 3.17 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 39.091 ms | 0.21 | 3.43 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.923 ms | 0.38 | 6.12 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 31.902 ms | 0.26 | 4.21 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.852 ms | 3.07 | 31.98 | 60% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.258 ms | 2.08 | 21.64 | 41% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.324 ms | 6.62 | 13.92 | 26% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.678 ms | 0.20 | 0.69 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.170 ms | 0.92 | 7.67 | 14%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 106.776 ms | 0.10 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.083 ms | 0.36 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 17.583 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.119 ms | 0.41 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 71.682 ms | 0.31 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 26.848 ms | 0.83 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 75.885 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.611 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.483 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.414 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.017 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 39.164 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 43.702 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 30.540 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.637 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.655 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.615 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 50.818 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 33.763 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.759 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.684 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.838 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.251 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 31.233 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.043 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.333 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 21.896 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.794 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.710 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.651 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.618 ms | 1.62 | 18.12 | 34% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.554 ms | 1.68 | 18.87 | 35% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.785 ms | 1.47 | 16.43 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 16.048 ms | 2.40 | 18.11 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.324 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 46.556 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 56.593 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.729 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 9.288 ms | 1.81 | 7.23 | 14% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.084 ms | 9.23 | 0.14 | 12% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.371 ms | 12.24 | 13.01 | 24% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.122 ms | 5.37 | 21.50 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.877 ms | 5.83 | 11.66 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.374 ms | 12.21 | 12.21 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.375 ms | 12.20 | 12.20 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.044 ms | 33.37 | 0.39 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.301 ms | 8.86 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 30.602 ms | 8.77 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 30.587 ms | 8.78 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.303 ms | 9.83 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-7-8845hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 630.6 GFLOP/s, f32 1260.3 GFLOP/s |
| ... hardware, AVX-512F | f64 630.6, f32 1260.3 GFLOP/s |
| ... this build, scalar | f64 14.4, f32 14.3 GFLOP/s |
| bandwidth | read 53.1, update 53.4, triad 41.1 GB/s (ceiling 53.4) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 120% | 0 | 42% |
| blas2 | 10 | 71% | 109% | 0 | 13% |
| blas3 | 10 | 2% | 17% | 0 | 12% |
| factor | 12 | 1% | 8% | 0 | 17% |
| eigen | 8 | 0% | 2% | 0 | 12% |
| elementwise | 5 | 8% | 11% | 0 | 12% |
| sparse | 5 | 39% | 256% | 0 | 16% |
| direct | 7 | 0% | 0% | 0 | 12% |
| ordering | 4 | n/a | n/a | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 30% | 33% | 0 | 11% |
| mg | 1 | 35% | 35% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 13% | 24% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 81% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.315 ms | 7.25 | 57.97 | 109% | ok | 4.2x (53%) |
| blas1 | dot | f32 | n=8388608 | 1.349 ms | 12.43 | 49.74 | 93% | ok | 7.2x (90%) |
| blas1 | axpy | f64 | n=8388608 | 3.521 ms | 4.76 | 57.17 | 107% | ok | 3.4x (42%) |
| blas1 | axpy | f32 | n=8388608 | 1.836 ms | 9.14 | 54.82 | 103% | ok | 6.4x (80%) |
| blas1 | scale | f64 | n=8388608 | 2.088 ms | 4.02 | 64.28 | 120% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=8388608 | 1.294 ms | 12.97 | 51.87 | 97% | ok | 7.2x (90%) |
| blas1 | one_norm | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.074 ms | 1.65 | 13.23 | 25% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 87.533 ms | 2.11 | 1.53 | 3% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.362 ms | 14.20 | 56.84 | 106% | ok | 1.3x (17%) |
| blas2 | gemv | f32 | n=4096 | 1.149 ms | 29.20 | 58.42 | 109% | ok | 2.1x (26%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.307 ms | 14.54 | 58.20 | 109% | ok | 3.1x (38%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.169 ms | 28.70 | 57.42 | 107% | ok | 1.9x (24%) |
| blas2 | symv | f64 | n=4096 | 8.528 ms | 3.93 | 7.88 | 15% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.779 ms | 9.43 | 37.77 | 71% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.540 ms | 10.90 | 21.81 | 41% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.764 ms | 9.51 | 38.08 | 71% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.563 ms | 10.73 | 21.48 | 40% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 13.551 ms | 2.48 | 19.81 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.487 ms | 107.94 | 2.53 | 17% | ok | 6.3x (79%) |
| blas3 | gemm | f32 | 512x512x512 | 1.230 ms | 218.26 | 2.56 | 17% | ok | 6.7x (84%) |
| blas3 | gemmt | f64 | 512x512x512 | 11.297 ms | 11.88 | 0.56 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 16.042 ms | 16.73 | 0.39 | 3% | ok | 1.0x (12%) |
| blas3 | syrk | f64 | 512x512x512 | 13.714 ms | 9.79 | 0.31 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 23.268 ms | 11.54 | 0.27 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 25.172 ms | 5.33 | 0.21 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f64 | 512x512x512 | 27.071 ms | 4.96 | 0.19 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 28.089 ms | 4.78 | 0.09 | 0% | ok | 1.0x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.585 ms | 0.55 | 4.43 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 26.602 ms | 3.36 | 0.16 | 1% | ok | 1.2x (15%) |
| factor | lu | f32 | 512x512 | 21.944 ms | 4.08 | 0.10 | 0% | ok | 1.4x (18%) |
| factor | cholesky | f64 | 512x512 | 11.426 ms | 3.92 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.930 ms | 4.09 | 0.19 | 0%\* | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 11.464 ms | 3.90 | 0.37 | 1%\* | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 169.655 ms | 0.26 | 0.02 | 0%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 407.152 ms | 0.44 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 44.795 ms | 4.00 | 0.09 | 1% | ok | 1.5x (18%) |
| factor | hessenberg | f64 | 512x512 | 479.837 ms | 0.93 | 0.01 | 0% | ok | 1.5x (19%) |
| factor | householder | f64 | 512x512 | 1.019 ms | 1.03 | 4.12 | 8%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 78.886 ms | 3.40 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 24.968 ms | 3.58 | 0.17 | 1% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.700 ms | 0.59 | 0.03 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 129.571 ms | 1.17 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 513.566 ms | 0.29 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 59.034 ms | 2.84 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 113.511 ms | 3.70 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 118.902 ms | 3.10 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 19.576 ms | 12.86 | 0.05 | 2% | ok | 1.5x (19%) |
| eigen | sylvester | f64 | 256x256 | 260.615 ms | 3.86 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 31.514 ms | 0.27 | 4.26 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 42.606 ms | 0.20 | 3.15 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 39.222 ms | 0.21 | 3.42 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.911 ms | 0.38 | 6.13 | 11% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 32.216 ms | 0.26 | 4.17 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.199 ms | 13.15 | 136.84 | 256% ! | ok | 4.3x (53%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.293 ms | 2.02 | 21.06 | 39% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.268 ms | 18.47 | 38.82 | 73% | ok | 2.8x (35%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.669 ms | 0.20 | 0.69 | 1%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.682 ms | 1.19 | 9.90 | 19%\* | ok | 1.3x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 104.859 ms | 0.11 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 32.011 ms | 0.35 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 17.339 ms | 0.65 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.218 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 71.693 ms | 0.31 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.643 ms | 0.87 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 76.585 ms | 0.02 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.663 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.553 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.422 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.061 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 39.194 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 43.698 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 30.565 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.630 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.635 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.566 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 51.080 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 33.962 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.814 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.682 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.865 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.315 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 31.071 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.372 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 22.739 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.810 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.754 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.652 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.645 ms | 1.59 | 17.83 | 33% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.843 ms | 1.42 | 15.91 | 30% | ok | 0.8x (11%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.032 ms | 1.29 | 14.43 | 27% | ok | 0.9x (11%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.472 ms | 2.49 | 18.79 | 35% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.123 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 46.676 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 54.638 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.790 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 9.326 ms | 1.80 | 7.20 | 13% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.006 ms | 67.02 | 1.05 | 11% | ok | 7.3x (91%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.386 ms | 12.11 | 12.87 | 24% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.121 ms | 5.38 | 21.51 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.881 ms | 5.82 | 11.65 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.381 ms | 12.15 | 12.15 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.381 ms | 12.15 | 12.15 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.248 ms | 215.06 | 2.52 | n/a | ok | 6.4x (81%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.153 ms | 64.64 | 0.51 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.176 ms | 64.28 | 0.38 | n/a | ok | 7.3x (92%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.178 ms | 64.26 | 0.38 | n/a | ok | 7.3x (92%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.775 ms | 71.10 | 0.42 | n/a | ok | 7.2x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

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

## ryzen-7-8845hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 626.5 GFLOP/s, f32 1254.5 GFLOP/s |
| ... hardware, AVX-512F | f64 626.5, f32 1254.5 GFLOP/s |
| ... this build, SSE2 | f64 313.7, f32 627.5 GFLOP/s |
| bandwidth | read 53.0, update 53.4, triad 41.0 GB/s (ceiling 53.4) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 111% | 126% | 0 | 13% |
| blas2 | 10 | 67% | 111% | 0 | 13% |
| blas3 | 10 | 6% | 19% | 0 | 21% |
| factor | 12 | 1% | 8% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 8% | 11% | 0 | 12% |
| sparse | 5 | 40% | 250% | 0 | 34% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 32% | 34% | 0 | 12% |
| mg | 1 | 35% | 35% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 70% | 71% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 68% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.259 ms | 7.43 | 59.41 | 111% | ok | 1.1x (13%) |
| blas1 | dot | f32 | n=8388608 | 1.000 ms | 16.77 | 67.10 | 126% | ok | 1.2x (15%) |
| blas1 | axpy | f64 | n=8388608 | 3.486 ms | 4.81 | 57.75 | 108% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=8388608 | 1.667 ms | 10.07 | 60.40 | 113% | ok | 1.1x (14%) |
| blas1 | scale | f64 | n=8388608 | 2.116 ms | 3.96 | 63.42 | 119% | ok | 0.9x (11%) |
| blas1 | two_norm | f64 | n=8388608 | 1.054 ms | 15.92 | 63.69 | 119% | ok | 1.1x (14%) |
| blas1 | one_norm | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.076 ms | 1.65 | 13.22 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.063 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 91.399 ms | 2.02 | 1.47 | 3% | ok | 0.9x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.339 ms | 14.35 | 57.42 | 107% | ok | 1.4x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.136 ms | 29.53 | 59.10 | 111% | ok | 1.6x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.678 ms | 3.47 | 13.88 | 26% | ok | 2.1x (27%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.733 ms | 7.09 | 14.19 | 27% | ok | 2.1x (26%) |
| blas2 | symv | f64 | n=4096 | 8.715 ms | 3.85 | 7.71 | 14% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.888 ms | 8.89 | 35.59 | 67% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 0.940 ms | 17.86 | 35.75 | 67% | ok | 1.1x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.839 ms | 9.12 | 36.53 | 68% | ok | 1.0x (12%) |
| blas2 | trsv | f32 | n=4096 | 1.076 ms | 15.60 | 31.23 | 58% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 13.438 ms | 2.50 | 19.98 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.248 ms | 119.43 | 2.80 | 19% | ok | 4.9x (61%) |
| blas3 | gemm | f32 | 512x512x512 | 1.206 ms | 222.56 | 2.61 | 18% | ok | 4.6x (57%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.498 ms | 38.37 | 1.80 | 6% | ok | 2.1x (26%) |
| blas3 | symm | f64 | 512x512x512 | 3.481 ms | 77.11 | 1.81 | 12% | ok | 3.5x (43%) |
| blas3 | syrk | f64 | 512x512x512 | 13.822 ms | 9.71 | 0.30 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 24.502 ms | 10.96 | 0.26 | 2% | ok | 0.9x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 5.673 ms | 23.66 | 0.92 | 4% | ok | 1.7x (21%) |
| blas3 | trsm | f64 | 512x512x512 | 6.040 ms | 22.22 | 0.87 | 4% | ok | 1.6x (20%) |
| blas3 | trsm | f32 | 512x512x512 | 4.533 ms | 29.61 | 0.58 | 2% | ok | 1.4x (18%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.619 ms | 0.55 | 4.42 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 23.230 ms | 3.85 | 0.18 | 1% | ok | 1.4x (18%) |
| factor | lu | f32 | 512x512 | 22.367 ms | 4.00 | 0.09 | 0% | ok | 1.4x (18%) |
| factor | cholesky | f64 | 512x512 | 11.388 ms | 3.93 | 0.37 | 1%\* | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 10.905 ms | 4.10 | 0.19 | 0%\* | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 11.454 ms | 3.91 | 0.37 | 1%\* | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 166.835 ms | 0.27 | 0.03 | 0%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 396.951 ms | 0.45 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 44.755 ms | 4.00 | 0.09 | 1% | ok | 1.5x (18%) |
| factor | hessenberg | f64 | 512x512 | 467.536 ms | 0.96 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.026 ms | 1.02 | 4.09 | 8%\* | ok | 3.8x (47%) |
| factor | inv | f64 | 512x512 | 78.505 ms | 3.42 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 24.907 ms | 3.59 | 0.17 | 1% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.219 ms | 0.57 | 0.03 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 129.294 ms | 1.17 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 504.947 ms | 0.30 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 60.029 ms | 2.79 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 111.431 ms | 3.76 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 117.845 ms | 3.13 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 19.518 ms | 12.89 | 0.05 | 2% | ok | 1.3x (17%) |
| eigen | sylvester | f64 | 256x256 | 258.496 ms | 3.89 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 28.558 ms | 0.29 | 4.70 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 43.320 ms | 0.19 | 3.10 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 40.207 ms | 0.21 | 3.34 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 22.524 ms | 0.37 | 5.96 | 11% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 32.893 ms | 0.26 | 4.08 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.204 ms | 12.86 | 133.74 | 250% ! | ok | 4.2x (53%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.271 ms | 2.06 | 21.42 | 40% | ok | 4.3x (54%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.271 ms | 18.44 | 38.76 | 73% | ok | 2.7x (34%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.131 ms | 0.20 | 0.71 | 1%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.709 ms | 1.17 | 9.74 | 18%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 98.366 ms | 0.11 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 28.695 ms | 0.39 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.257 ms | 0.73 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 55.586 ms | 0.40 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 71.695 ms | 0.31 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 26.142 ms | 0.86 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 83.696 ms | 0.02 | 0.00 | 0%\* | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.720 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.580 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.421 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.706 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 31.364 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 27.588 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.428 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.881 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.729 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 30.895 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.306 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 19.305 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 18.364 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 19.536 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 22.079 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.411 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 23.668 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ic_0 | f64 | 128^2 grid | 1.826 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.791 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.658 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.635 ms | 1.60 | 17.93 | 34% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.700 ms | 1.54 | 17.25 | 32% | ok | 0.9x (11%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.779 ms | 1.47 | 16.49 | 31% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.404 ms | 2.50 | 18.87 | 35% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.136 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 46.781 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 54.484 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.437 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.778 ms | 9.43 | 37.73 | 71% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 2.570 ms | 104.46 | 1.63 | 17% | ok | 5.9x (74%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.479 ms | 35.05 | 37.24 | 70% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.204 ms | 13.93 | 55.72 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.473 ms | 35.50 | 71.00 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.348 ms | 48.25 | 48.25 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.318 ms | 52.76 | 52.76 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.376 ms | 112.98 | 1.32 | n/a | ok | 6.3x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.538 ms | 105.78 | 0.83 | n/a | ok | 6.8x (85%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 2.561 ms | 104.80 | 0.61 | n/a | ok | 6.9x (87%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.670 ms | 400.71 | 2.35 | n/a | ok | 5.5x (68%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.670 ms | 400.77 | 2.35 | n/a | ok | 5.4x (68%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## ryzen-7-8845hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 80.0 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.0, f32 160.1 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 43.2, update 53.5, triad 38.8 GB/s (ceiling 53.5) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 110% | 0 |
| blas2 | 10 | 76% | 90% | 0 |
| blas3 | 10 | 71% | 92% | 0 |
| factor | 12 | 13% | 23% | 0 |
| eigen | 8 | 17% | 53% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 25% | 61% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 34% | 37% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 23% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.471 ms | 1.46 | 11.70 | 22% | ok |
| blas1 | dot | f32 | n=8388608 | 9.772 ms | 1.72 | 6.87 | 13% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.040 ms | 1.39 | 16.72 | 31% | ok |
| blas1 | axpy | f32 | n=8388608 | 10.255 ms | 1.64 | 9.82 | 18% | ok |
| blas1 | scale | f64 | n=8388608 | 2.283 ms | 3.67 | 58.78 | 110% | ok |
| blas1 | two_norm | f64 | n=8388608 | 10.584 ms | 1.59 | 6.34 | 12% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.070 ms | 1.65 | 13.24 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 89.225 ms | 2.07 | 1.50 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.217 ms | 10.43 | 41.74 | 78% | ok |
| blas2 | gemv | f32 | n=4096 | 1.573 ms | 21.33 | 42.68 | 80% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.210 ms | 10.45 | 41.83 | 78% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.388 ms | 24.17 | 48.37 | 90% | ok |
| blas2 | symv | f64 | n=4096 | 1.813 ms | 18.51 | 37.05 | 69% | ok |
| blas2 | trmv | f64 | n=4096 | 1.646 ms | 10.19 | 40.82 | 76% | ok |
| blas2 | trmv | f32 | n=4096 | 0.863 ms | 19.44 | 38.91 | 73% | ok |
| blas2 | trsv | f64 | n=4096 | 1.780 ms | 9.42 | 37.73 | 71% | ok |
| blas2 | trsv | f32 | n=4096 | 0.925 ms | 18.13 | 36.30 | 68% | ok |
| blas2 | ger | f64 | n=4096 | 13.593 ms | 2.47 | 19.75 | 37% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.630 ms | 73.96 | 1.73 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.887 ms | 142.27 | 1.67 | 89% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.730 ms | 49.16 | 2.30 | 61% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.777 ms | 71.08 | 1.67 | 89% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.925 ms | 45.89 | 1.43 | 57% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.740 ms | 56.63 | 1.33 | 71% | ok |
| blas3 | trmm | f64 | 512x512x512 | 1.888 ms | 71.07 | 2.78 | 89% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.809 ms | 47.77 | 1.87 | 60% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.328 ms | 101.09 | 1.97 | 63% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.594 ms | 0.55 | 4.43 | 8% | ok |
| factor | lu | f64 | 512x512 | 4.986 ms | 17.94 | 0.84 | 22% | ok |
| factor | lu | f32 | 512x512 | 2.423 ms | 36.94 | 0.87 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 4.366 ms | 10.25 | 0.96 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 1.678 ms | 26.66 | 1.25 | 17% | ok |
| factor | ldlt | f64 | 512x512 | 11.813 ms | 3.79 | 0.36 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 163.637 ms | 0.27 | 0.03 | 0% | ok |
| factor | qr | f64 | 512x512 | 10.232 ms | 17.49 | 0.41 | 22% | ok |
| factor | lq | f64 | 512x512 | 64.712 ms | 2.77 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 741.036 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.894 ms | 0.27 | 1.08 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 60.728 ms | 4.42 | 0.07 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.533 ms | 16.17 | 0.76 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.670 ms | 13.40 | 0.63 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 129.184 ms | 1.17 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 505.111 ms | 0.30 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.748 ms | 10.02 | 0.06 | 13% | ok |
| eigen | schur | f64 | 256x256 | 21.953 ms | 19.11 | 0.05 | 24% | ok |
| eigen | svd | f64 | 256x256 | 8.660 ms | 42.62 | 0.12 | 53% | ok |
| eigen | expm | f64 | 256x256 | 16.114 ms | 15.62 | 0.07 | 20% | ok |
| eigen | sylvester | f64 | 256x256 | 73.897 ms | 13.62 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 28.026 ms | 0.30 | 4.79 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 43.694 ms | 0.19 | 3.07 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 39.203 ms | 0.21 | 3.42 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.901 ms | 0.38 | 6.13 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 35.048 ms | 0.24 | 3.83 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.838 ms | 3.12 | 32.51 | 61% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.260 ms | 2.08 | 21.61 | 40% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.533 ms | 6.41 | 13.48 | 25% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.213 ms | 0.20 | 0.71 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.770 ms | 1.13 | 9.41 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 102.856 ms | 0.11 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.253 ms | 0.37 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 17.319 ms | 0.65 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.818 ms | 0.42 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 72.251 ms | 0.31 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.912 ms | 0.86 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 80.136 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.639 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.563 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.417 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 13.929 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.256 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 45.978 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.183 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.229 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.340 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.973 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 54.827 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.678 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.181 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 24.296 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 23.479 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 26.430 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 32.682 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.414 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 24.125 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.824 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.741 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.316 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.597 ms | 1.64 | 18.36 | 34% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.461 ms | 1.79 | 20.07 | 37% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.783 ms | 1.47 | 16.45 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.889 ms | 2.42 | 18.29 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.001 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.685 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 53.435 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.908 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 11.377 ms | 1.47 | 5.90 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.622 ms | 9.06 | 0.14 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.469 ms | 11.42 | 12.13 | 23% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.214 ms | 5.22 | 20.88 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.880 ms | 5.83 | 11.65 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.387 ms | 12.09 | 12.09 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.387 ms | 12.10 | 12.10 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.328 ms | 32.23 | 0.38 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.305 ms | 8.86 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 30.582 ms | 8.78 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 30.607 ms | 8.77 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.278 ms | 9.84 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-7-8845hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 626.9 GFLOP/s, f32 1253.3 GFLOP/s |
| ... hardware, AVX-512F | f64 626.9, f32 1253.3 GFLOP/s |
| ... this build, scalar | f64 12.2, f32 14.4 GFLOP/s |
| bandwidth | read 53.3, update 53.3, triad 41.3 GB/s (ceiling 53.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 93% | 114% | 0 | 43% |
| blas2 | 10 | 100% | 107% | 0 | 16% |
| blas3 | 10 | 40% | 71% | 0 | 63% |
| factor | 12 | 2% | 8% | 0 | 11% |
| eigen | 8 | 2% | 5% | 0 | 11% |
| elementwise | 5 | 7% | 11% | 0 | 12% |
| sparse | 5 | 40% | 252% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 12% |
| ordering | 4 | n/a | n/a | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 34% | 37% | 0 | 12% |
| mg | 1 | 35% | 35% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 11% | 22% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.317 ms | 7.24 | 57.91 | 109% | ok | 4.9x (62%) |
| blas1 | dot | f32 | n=8388608 | 1.357 ms | 12.36 | 49.44 | 93% | ok | 7.2x (90%) |
| blas1 | axpy | f64 | n=8388608 | 3.534 ms | 4.75 | 56.97 | 107% | ok | 3.4x (43%) |
| blas1 | axpy | f32 | n=8388608 | 1.997 ms | 8.40 | 50.42 | 95% | ok | 5.1x (64%) |
| blas1 | scale | f64 | n=8388608 | 2.203 ms | 3.81 | 60.92 | 114% | ok | 1.0x (13%) |
| blas1 | two_norm | f64 | n=8388608 | 1.461 ms | 11.48 | 45.92 | 86% | ok | 7.2x (91%) |
| blas1 | one_norm | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.076 ms | 1.65 | 13.22 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 87.462 ms | 2.11 | 1.53 | 3% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.576 ms | 13.03 | 52.13 | 98% | ok | 1.2x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.174 ms | 28.58 | 57.18 | 107% | ok | 1.3x (17%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.667 ms | 12.58 | 50.36 | 94% | ok | 1.2x (15%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.257 ms | 26.70 | 53.42 | 100% | ok | 1.1x (14%) |
| blas2 | symv | f64 | n=4096 | 1.243 ms | 27.00 | 54.06 | 101% | ok | 1.5x (18%) |
| blas2 | trmv | f64 | n=4096 | 1.228 ms | 13.66 | 54.68 | 103% | ok | 1.3x (17%) |
| blas2 | trmv | f32 | n=4096 | 0.590 ms | 28.42 | 56.90 | 107% | ok | 1.5x (18%) |
| blas2 | trsv | f64 | n=4096 | 1.773 ms | 9.46 | 37.88 | 71% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.880 ms | 19.06 | 38.16 | 72% | ok | 1.1x (13%) |
| blas2 | ger | f64 | n=4096 | 13.716 ms | 2.45 | 19.58 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 0.606 ms | 442.82 | 10.38 | 71% | ok | 6.0x (75%) |
| blas3 | gemm | f32 | 512x512x512 | 0.337 ms | 797.02 | 9.34 | 64% | ok | 5.6x (70%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.358 ms | 98.83 | 4.63 | 16% | ok | 2.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 0.609 ms | 440.56 | 10.33 | 70% | ok | 6.2x (77%) |
| blas3 | syrk | f64 | 512x512x512 | 1.694 ms | 79.25 | 2.48 | 13% | ok | 1.7x (22%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.385 ms | 112.53 | 2.64 | 18% | ok | 2.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 0.389 ms | 345.30 | 13.49 | 55% | ok | 4.9x (61%) |
| blas3 | trsm | f64 | 512x512x512 | 0.534 ms | 251.20 | 9.81 | 40% | ok | 5.3x (66%) |
| blas3 | trsm | f32 | 512x512x512 | 0.265 ms | 507.25 | 9.91 | 40% | ok | 5.0x (63%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.413 ms | 0.56 | 4.51 | 8% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 9.206 ms | 9.72 | 0.46 | 2% | ok | 0.5x (7%) |
| factor | lu | f32 | 512x512 | 4.327 ms | 20.68 | 0.48 | 2% | ok | 0.6x (7%) |
| factor | cholesky | f64 | 512x512 | 6.006 ms | 7.45 | 0.70 | 1%\* | ok | 0.7x (9%) |
| factor | cholesky | f32 | 512x512 | 2.333 ms | 19.18 | 0.90 | 2%\* | ok | 0.7x (9%) |
| factor | ldlt | f64 | 512x512 | 11.812 ms | 3.79 | 0.36 | 1%\* | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 165.676 ms | 0.27 | 0.03 | 0%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 11.927 ms | 15.00 | 0.35 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 45.876 ms | 3.90 | 0.09 | 1% | ok | 1.4x (18%) |
| factor | hessenberg | f64 | 512x512 | 474.331 ms | 0.94 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.018 ms | 1.03 | 4.12 | 8%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 66.451 ms | 4.04 | 0.06 | 1% | ok | 0.9x (11%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.328 ms | 12.21 | 0.57 | 2% | ok | 0.8x (9%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 3.598 ms | 6.22 | 0.29 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 128.680 ms | 1.17 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 503.438 ms | 0.30 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 21.106 ms | 7.95 | 0.05 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 26.400 ms | 15.89 | 0.04 | 3% | ok | 0.8x (10%) |
| eigen | svd | f64 | 256x256 | 12.154 ms | 30.37 | 0.09 | 5% | ok | 0.7x (9%) |
| eigen | expm | f64 | 256x256 | 18.149 ms | 13.87 | 0.06 | 2% | ok | 0.9x (11%) |
| eigen | sylvester | f64 | 256x256 | 81.732 ms | 12.32 | 0.02 | 2% | ok | 0.9x (11%) |
| elementwise | exp | f64 | n=8388608 | 28.661 ms | 0.29 | 4.68 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 44.373 ms | 0.19 | 3.02 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 40.050 ms | 0.21 | 3.35 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.928 ms | 0.38 | 6.12 | 11% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 35.602 ms | 0.24 | 3.77 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.202 ms | 12.93 | 134.47 | 252% ! | ok | 4.1x (52%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.283 ms | 2.04 | 21.22 | 40% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.281 ms | 18.36 | 38.60 | 72% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.841 ms | 0.19 | 0.68 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.706 ms | 1.18 | 9.76 | 18%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 101.796 ms | 0.11 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.910 ms | 0.36 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 18.414 ms | 0.61 | 0.04 | 0% | ok | 0.9x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.705 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 73.417 ms | 0.30 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 26.165 ms | 0.86 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 80.932 ms | 0.02 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.731 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.635 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.428 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.197 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.191 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 45.928 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.257 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.233 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.332 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.938 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.251 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.581 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.216 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 24.296 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 23.598 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 26.977 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 32.847 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.415 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 25.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.841 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.748 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.786 ms | 0.00 | 0.00 | n/a | ok | 0.7x (9%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.607 ms | 1.63 | 18.25 | 34% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.491 ms | 1.76 | 19.67 | 37% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.763 ms | 1.48 | 16.64 | 31% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.463 ms | 2.49 | 18.80 | 35% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 33.321 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.852 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 54.581 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.975 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 11.439 ms | 1.47 | 5.87 | 11% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.097 ms | 65.52 | 1.02 | 10% | ok | 7.2x (90%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.491 ms | 11.25 | 11.95 | 22% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.175 ms | 5.28 | 21.13 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.891 ms | 5.80 | 11.60 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.402 ms | 11.97 | 11.97 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.402 ms | 11.96 | 11.96 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.200 ms | 223.73 | 2.62 | n/a | ok | 6.9x (87%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.205 ms | 63.84 | 0.50 | n/a | ok | 7.2x (90%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.234 ms | 63.40 | 0.37 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.240 ms | 63.30 | 0.37 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.841 ms | 69.88 | 0.41 | n/a | ok | 7.1x (89%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

