# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,6, CPUs 0,1,2,3,4,5, clock policy nvpmodel 25W (mode 1), jetson_clocks; schedutil 1344-1344 MHz on cpus 0,1,2,3,4,5, 2026-11-11T19:06:10Z.

## jetson-orin -- native, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, scalar | f64 5.4, f32 5.4 GFLOP/s |
| bandwidth | read 11.5, update 16.2, triad 14.8 GB/s (ceiling 16.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 72% | 131% | 0 |
| blas2 | 10 | 33% | 124% | 0 |
| blas3 | 10 | 33% | 33% | 0 |
| factor | 12 | 13% | 27% | 0 |
| eigen | 8 | 16% | 23% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 31% | 73% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 60% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.539 ms | 1.45 | 11.63 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 6.005 ms | 2.79 | 11.18 | 69% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.258 ms | 1.37 | 16.42 | 101% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.964 ms | 2.81 | 16.88 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.336 ms | 1.32 | 21.18 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.605 ms | 2.99 | 11.97 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.750 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.367 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 21.711 ms | 1.55 | 6.18 | 38% | ok |
| blas2 | gemv | f32 | n=4096 | 12.416 ms | 2.70 | 5.41 | 33% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 16.004 ms | 2.10 | 8.39 | 52% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.275 ms | 5.35 | 10.70 | 66% | ok |
| blas2 | symv | f64 | n=4096 | 21.330 ms | 1.57 | 3.15 | 19% | ok |
| blas2 | trmv | f64 | n=4096 | 13.093 ms | 1.28 | 5.13 | 32% | ok |
| blas2 | trmv | f32 | n=4096 | 26.561 ms | 0.63 | 1.26 | 8% | ok |
| blas2 | trsv | f64 | n=4096 | 13.112 ms | 1.28 | 5.12 | 32% | ok |
| blas2 | trsv | f32 | n=4096 | 27.035 ms | 0.62 | 1.24 | 8% | ok |
| blas2 | ger | f64 | n=4096 | 13.406 ms | 2.50 | 20.03 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 75.215 ms | 3.57 | 0.08 | 33% | ok |
| blas3 | gemm | f32 | 512x512x512 | 38.166 ms | 7.03 | 0.08 | 33% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 49.261 ms | 2.72 | 0.13 | 25% | ok |
| blas3 | symm | f64 | 512x512x512 | 75.813 ms | 3.54 | 0.08 | 33% | ok |
| blas3 | syrk | f64 | 512x512x512 | 51.808 ms | 2.59 | 0.08 | 24% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 93.019 ms | 2.89 | 0.07 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 37.963 ms | 3.54 | 0.14 | 33% | ok |
| blas3 | trsm | f64 | 512x512x512 | 38.344 ms | 3.50 | 0.14 | 33% | ok |
| blas3 | trsm | f32 | 512x512x512 | 19.279 ms | 6.96 | 0.14 | 32% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.564 ms | 0.56 | 4.44 | 27% | ok |
| factor | lu | f64 | 512x512 | 36.144 ms | 2.48 | 0.12 | 23% | ok |
| factor | lu | f32 | 512x512 | 20.469 ms | 4.37 | 0.10 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 31.825 ms | 1.41 | 0.13 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 30.100 ms | 1.49 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 34.450 ms | 1.30 | 0.12 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 85.787 ms | 0.52 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 334.500 ms | 0.53 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 91.967 ms | 1.95 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 473.116 ms | 0.95 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.013 ms | 0.52 | 2.08 | 13% | ok |
| factor | inv | f64 | 512x512 | 192.447 ms | 1.39 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 30.696 ms | 2.91 | 0.14 | 27% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.380 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 144.863 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 947.450 ms | 0.16 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 98.408 ms | 1.70 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 214.235 ms | 1.96 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.627 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 102.700 ms | 2.45 | 0.01 | 23% | ok |
| eigen | sylvester | f64 | 256x256 | 536.759 ms | 1.88 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 75.790 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.000 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.948 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.073 ms | 0.15 | 2.44 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.536 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.298 ms | 1.14 | 11.85 | 73% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.063 ms | 0.85 | 8.89 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.596 ms | 2.38 | 5.00 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.012 ms | 0.16 | 0.57 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.815 ms | 0.17 | 1.41 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 278.019 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 99.276 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.576 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 129.032 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 184.741 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 62.381 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 290.265 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.118 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.450 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.134 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.791 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.085 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.114 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.906 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.588 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.776 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.796 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.785 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.108 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.066 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.847 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.707 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.636 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.819 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.167 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.829 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.473 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.510 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.250 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.235 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.991 ms | 0.87 | 9.80 | 60% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.735 ms | 0.70 | 7.85 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 7.751 ms | 0.34 | 3.78 | 23% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.854 ms | 0.86 | 6.48 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 351.669 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 410.617 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 201.091 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.148 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.501 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.423 ms | 3.56 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.566 ms | 3.67 | 3.90 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.731 ms | 2.93 | 11.71 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.729 ms | 6.15 | 12.30 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.643 ms | 2.97 | 2.97 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 37.858 ms | 7.09 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.098 ms | 6.53 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.863 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.874 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.131 ms | 4.46 | 0.03 | n/a | ok |

## jetson-orin -- native, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.6 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.6 GFLOP/s |
| ... this build, scalar | f64 32.2, f32 32.1 GFLOP/s |
| bandwidth | read 31.2, update 34.2, triad 32.7 GB/s (ceiling 34.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 93% | 138% | 0 | 40% |
| blas2 | 10 | 43% | 59% | 0 | 18% |
| blas3 | 10 | 6% | 19% | 0 | 17% |
| factor | 12 | 2% | 23% | 0 | 21% |
| eigen | 8 | 3% | 8% | 0 | 17% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 26% | 98% | 0 | 18% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 29% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 16% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.225 ms | 3.97 | 31.77 | 93% | ok | 2.7x (46%) |
| blas1 | dot | f32 | n=8388608 | 2.123 ms | 7.90 | 31.61 | 93% | ok | 2.8x (47%) |
| blas1 | axpy | f64 | n=8388608 | 4.928 ms | 3.40 | 40.85 | 120% | ok | 2.5x (41%) |
| blas1 | axpy | f32 | n=8388608 | 2.460 ms | 6.82 | 40.92 | 120% | ok | 2.4x (40%) |
| blas1 | scale | f64 | n=8388608 | 2.842 ms | 2.95 | 47.22 | 138% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.042 ms | 8.21 | 32.86 | 96% | ok | 2.7x (46%) |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.750 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 192.670 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 6.720 ms | 4.99 | 19.98 | 58% | ok | 3.2x (54%) |
| blas2 | gemv | f32 | n=4096 | 4.572 ms | 7.34 | 14.69 | 43% | ok | 2.7x (45%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.814 ms | 4.92 | 19.71 | 58% | ok | 2.3x (39%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.418 ms | 7.60 | 15.20 | 44% | ok | 1.4x (24%) |
| blas2 | symv | f64 | n=4096 | 21.300 ms | 1.58 | 3.15 | 9% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 12.384 ms | 1.35 | 5.42 | 16% | ok | 1.1x (18%) |
| blas2 | trmv | f32 | n=4096 | 26.602 ms | 0.63 | 1.26 | 4% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 12.352 ms | 1.36 | 5.44 | 16% | ok | 1.1x (18%) |
| blas2 | trsv | f32 | n=4096 | 27.011 ms | 0.62 | 1.24 | 4% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.386 ms | 2.51 | 20.06 | 59% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 21.511 ms | 12.48 | 0.29 | 19% | ok | 3.5x (58%) |
| blas3 | gemm | f32 | 512x512x512 | 12.534 ms | 21.42 | 0.25 | 17% | ok | 3.0x (51%) |
| blas3 | gemmt | f64 | 512x512x512 | 48.401 ms | 2.77 | 0.13 | 4% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 75.648 ms | 3.55 | 0.08 | 6% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 51.121 ms | 2.63 | 0.08 | 4% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 92.848 ms | 2.89 | 0.07 | 4% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 37.946 ms | 3.54 | 0.14 | 6% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 37.885 ms | 3.54 | 0.14 | 6% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 19.232 ms | 6.98 | 0.14 | 5% | ok | 1.0x (17%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.271 ms | 0.57 | 4.58 | 13% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 26.334 ms | 3.40 | 0.16 | 5% | ok | 1.4x (23%) |
| factor | lu | f32 | 512x512 | 16.065 ms | 5.57 | 0.13 | 4% | ok | 1.3x (21%) |
| factor | cholesky | f64 | 512x512 | 31.719 ms | 1.41 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.064 ms | 1.49 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 34.421 ms | 1.30 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 83.966 ms | 0.53 | 0.05 | 1% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 207.576 ms | 0.86 | 0.02 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 62.030 ms | 2.88 | 0.07 | 4% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 328.244 ms | 1.36 | 0.01 | 2% | ok | 1.4x (24%) |
| factor | householder | f64 | 512x512 | 0.545 ms | 1.93 | 7.70 | 23% | ok | 3.7x (62%) |
| factor | inv | f64 | 512x512 | 182.168 ms | 1.47 | 0.02 | 2% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 26.828 ms | 3.34 | 0.16 | 5% | ok | 1.1x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.956 ms | 0.37 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.409 ms | 1.05 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 943.243 ms | 0.16 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 98.212 ms | 1.71 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 213.956 ms | 1.96 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 358.137 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 49.477 ms | 5.09 | 0.02 | 8% | ok | 2.1x (35%) |
| eigen | sylvester | f64 | 256x256 | 501.273 ms | 2.01 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 75.684 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 107.914 ms | 0.08 | 1.24 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 178.207 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.978 ms | 0.15 | 2.44 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.559 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.815 ms | 3.21 | 33.39 | 98% | ok | 2.8x (47%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.041 ms | 0.86 | 8.95 | 26% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.871 ms | 7.13 | 15.00 | 44% | ok | 3.0x (50%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.596 ms | 0.16 | 0.56 | 2% | ok | 1.0x (16%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.127 ms | 0.18 | 1.50 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 277.400 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 99.100 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.440 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 130.002 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 184.649 ms | 0.12 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.853 ms | 0.36 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 284.997 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.290 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.131 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.511 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.716 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.725 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.555 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.723 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.495 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.453 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.137 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.823 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.689 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.497 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.433 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 69.021 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.838 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 45.108 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.471 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.236 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.231 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.964 ms | 0.88 | 9.89 | 29% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.741 ms | 0.70 | 7.84 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 7.664 ms | 0.34 | 3.83 | 11% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.363 ms | 0.91 | 6.86 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 352.401 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 412.038 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 190.563 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.881 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.499 ms | 1.34 | 5.37 | 16% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.939 ms | 20.75 | 0.32 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.571 ms | 3.67 | 3.90 | 11% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.588 ms | 3.00 | 12.01 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.692 ms | 6.23 | 12.47 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.693 ms | 2.95 | 2.95 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.005 ms | 29.81 | 0.35 | n/a | ok | 4.2x (70%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.087 ms | 37.88 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.485 ms | 48.94 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.486 ms | 48.93 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.322 ms | 26.01 | 0.15 | n/a | ok | 5.8x (97%) |

## jetson-orin -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, NEON | f64 10.7, f32 21.5 GFLOP/s |
| bandwidth | read 11.5, update 16.2, triad 14.8 GB/s (ceiling 16.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 73% | 131% | 0 |
| blas2 | 10 | 48% | 124% | 0 |
| blas3 | 10 | 50% | 61% | 0 |
| factor | 12 | 13% | 27% | 0 |
| eigen | 8 | 16% | 33% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 31% | 74% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 56% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 66% | 119% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.566 ms | 1.45 | 11.60 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 5.672 ms | 2.96 | 11.83 | 73% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.227 ms | 1.37 | 16.47 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.979 ms | 2.81 | 16.84 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.337 ms | 1.32 | 21.18 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.598 ms | 3.00 | 11.99 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.503 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.296 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 13.922 ms | 2.41 | 9.65 | 60% | ok |
| blas2 | gemv | f32 | n=4096 | 6.269 ms | 5.35 | 10.71 | 66% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 50.052 ms | 0.67 | 2.68 | 17% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 24.652 ms | 1.36 | 2.72 | 17% | ok |
| blas2 | symv | f64 | n=4096 | 22.082 ms | 1.52 | 3.04 | 19% | ok |
| blas2 | trmv | f64 | n=4096 | 9.049 ms | 1.85 | 7.42 | 46% | ok |
| blas2 | trmv | f32 | n=4096 | 4.330 ms | 3.87 | 7.76 | 48% | ok |
| blas2 | trsv | f64 | n=4096 | 9.282 ms | 1.81 | 7.24 | 45% | ok |
| blas2 | trsv | f32 | n=4096 | 4.264 ms | 3.93 | 7.88 | 49% | ok |
| blas2 | ger | f64 | n=4096 | 13.401 ms | 2.50 | 20.04 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 41.288 ms | 6.50 | 0.15 | 61% | ok |
| blas3 | gemm | f32 | 512x512x512 | 20.477 ms | 13.11 | 0.15 | 61% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 25.834 ms | 5.20 | 0.24 | 48% | ok |
| blas3 | symm | f64 | 512x512x512 | 42.560 ms | 6.31 | 0.15 | 59% | ok |
| blas3 | syrk | f64 | 512x512x512 | 49.944 ms | 2.69 | 0.08 | 25% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 92.722 ms | 2.90 | 0.07 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 24.957 ms | 5.38 | 0.21 | 50% | ok |
| blas3 | trsm | f64 | 512x512x512 | 25.768 ms | 5.21 | 0.20 | 49% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.595 ms | 10.66 | 0.21 | 50% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.633 ms | 0.55 | 4.41 | 27% | ok |
| factor | lu | f64 | 512x512 | 35.522 ms | 2.52 | 0.12 | 23% | ok |
| factor | lu | f32 | 512x512 | 20.248 ms | 4.42 | 0.10 | 21% | ok |
| factor | cholesky | f64 | 512x512 | 33.959 ms | 1.32 | 0.12 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 30.034 ms | 1.49 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 32.294 ms | 1.39 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 85.848 ms | 0.52 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 333.709 ms | 0.54 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 92.005 ms | 1.95 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 470.798 ms | 0.95 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 1.995 ms | 0.53 | 2.10 | 13% | ok |
| factor | inv | f64 | 512x512 | 191.363 ms | 1.40 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 30.445 ms | 2.94 | 0.14 | 27% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.220 ms | 0.37 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 144.446 ms | 1.05 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 904.419 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 98.464 ms | 1.70 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 215.345 ms | 1.95 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.885 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 72.043 ms | 3.49 | 0.01 | 33% | ok |
| eigen | sylvester | f64 | 256x256 | 516.424 ms | 1.95 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 76.034 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.148 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.607 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.112 ms | 0.15 | 2.44 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.586 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.283 ms | 1.15 | 11.93 | 74% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.075 ms | 0.85 | 8.85 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.316 ms | 2.42 | 5.08 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.833 ms | 0.16 | 0.55 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.825 ms | 0.17 | 1.41 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 277.967 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 99.500 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.503 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 134.874 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 191.193 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.861 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 291.572 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.258 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.988 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.135 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.428 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.151 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.007 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.989 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.487 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.744 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.650 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.320 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.076 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.951 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.831 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.635 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.323 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.075 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.737 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 44.908 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.501 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.227 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.089 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.230 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.231 ms | 0.81 | 9.08 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.740 ms | 0.70 | 7.84 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.515 ms | 0.58 | 6.50 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 45.031 ms | 0.85 | 6.45 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 353.745 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 419.285 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 197.613 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.815 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 6.263 ms | 2.68 | 10.72 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 64.044 ms | 4.19 | 0.07 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.310 ms | 12.81 | 13.61 | 119% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.604 ms | 2.99 | 11.97 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.684 ms | 6.25 | 12.50 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.208 ms | 13.89 | 13.89 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.348 ms | 7.15 | 7.15 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 26.016 ms | 10.32 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 27.808 ms | 9.65 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 33.397 ms | 8.04 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 8.563 ms | 31.35 | 0.18 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 23.385 ms | 11.48 | 0.07 | n/a | ok |

## jetson-orin -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.6 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.6 GFLOP/s |
| ... this build, NEON | f64 64.3, f32 128.6 GFLOP/s |
| bandwidth | read 29.6, update 34.5, triad 32.8 GB/s (ceiling 34.5) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 94% | 137% | 0 | 40% |
| blas2 | 10 | 23% | 88% | 0 | 17% |
| blas3 | 10 | 19% | 51% | 0 | 38% |
| factor | 12 | 2% | 22% | 0 | 20% |
| eigen | 8 | 3% | 8% | 0 | 17% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 25% | 91% | 0 | 18% |
| direct | 7 | 0% | 0% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 27% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 34% | 40% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 80% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.254 ms | 3.94 | 31.55 | 92% | ok | 2.7x (45%) |
| blas1 | dot | f32 | n=8388608 | 2.065 ms | 8.13 | 32.50 | 94% | ok | 2.7x (46%) |
| blas1 | axpy | f64 | n=8388608 | 5.015 ms | 3.35 | 40.15 | 116% | ok | 2.4x (41%) |
| blas1 | axpy | f32 | n=8388608 | 2.486 ms | 6.75 | 40.48 | 117% | ok | 2.4x (40%) |
| blas1 | scale | f64 | n=8388608 | 2.842 ms | 2.95 | 47.23 | 137% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.036 ms | 8.24 | 32.97 | 96% | ok | 2.8x (46%) |
| blas1 | one_norm | f64 | n=8388608 | 12.504 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.757 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.504 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 191.694 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.150 ms | 6.52 | 26.07 | 76% | ok | 2.7x (45%) |
| blas2 | gemv | f32 | n=4096 | 2.218 ms | 15.13 | 30.27 | 88% | ok | 2.8x (47%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 11.407 ms | 2.94 | 11.77 | 34% | ok | 4.4x (73%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 9.384 ms | 3.58 | 7.15 | 21% | ok | 2.6x (44%) |
| blas2 | symv | f64 | n=4096 | 21.446 ms | 1.56 | 3.13 | 9% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 8.978 ms | 1.87 | 7.48 | 22% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 4.285 ms | 3.92 | 7.84 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 9.055 ms | 1.85 | 7.42 | 22% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.176 ms | 4.02 | 8.04 | 23% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.362 ms | 2.51 | 20.09 | 58% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 8.212 ms | 32.69 | 0.77 | 51% | ok | 5.0x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 4.110 ms | 65.31 | 0.77 | 51% | ok | 5.0x (83%) |
| blas3 | gemmt | f64 | 512x512x512 | 7.623 ms | 17.61 | 0.83 | 27% | ok | 3.4x (56%) |
| blas3 | symm | f64 | 512x512x512 | 9.158 ms | 29.31 | 0.69 | 46% | ok | 4.6x (77%) |
| blas3 | syrk | f64 | 512x512x512 | 50.996 ms | 2.63 | 0.08 | 4% | ok | 1.0x (16%) |
| blas3 | syr2k | f64 | 512x512x512 | 93.274 ms | 2.88 | 0.07 | 4% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 10.875 ms | 12.34 | 0.48 | 19% | ok | 2.3x (38%) |
| blas3 | trsm | f64 | 512x512x512 | 11.667 ms | 11.50 | 0.45 | 18% | ok | 2.2x (37%) |
| blas3 | trsm | f32 | 512x512x512 | 5.718 ms | 23.47 | 0.46 | 18% | ok | 2.2x (37%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.676 ms | 0.55 | 4.39 | 13% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 26.586 ms | 3.37 | 0.16 | 5% | ok | 1.3x (22%) |
| factor | lu | f32 | 512x512 | 16.603 ms | 5.39 | 0.13 | 4% | ok | 1.2x (20%) |
| factor | cholesky | f64 | 512x512 | 33.987 ms | 1.32 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.179 ms | 1.48 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 32.261 ms | 1.39 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 86.633 ms | 0.52 | 0.05 | 1% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 207.341 ms | 0.86 | 0.02 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 61.112 ms | 2.93 | 0.07 | 5% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 304.859 ms | 1.47 | 0.01 | 2% | ok | 1.5x (26%) |
| factor | householder | f64 | 512x512 | 0.544 ms | 1.93 | 7.71 | 22% | ok | 3.7x (61%) |
| factor | inv | f64 | 512x512 | 182.763 ms | 1.47 | 0.02 | 2% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 26.118 ms | 3.43 | 0.16 | 5% | ok | 1.2x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.996 ms | 0.37 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.305 ms | 1.05 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 900.812 ms | 0.17 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 98.476 ms | 1.70 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 213.700 ms | 1.96 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 358.027 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 50.135 ms | 5.02 | 0.02 | 8% | ok | 1.4x (24%) |
| eigen | sylvester | f64 | 256x256 | 499.925 ms | 2.01 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 75.211 ms | 0.11 | 1.78 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 107.315 ms | 0.08 | 1.25 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 177.101 ms | 0.05 | 0.76 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.810 ms | 0.15 | 2.45 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.133 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.864 ms | 3.03 | 31.51 | 91% | ok | 2.6x (44%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.127 ms | 0.84 | 8.71 | 25% | ok | 1.0x (16%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.613 ms | 7.46 | 15.69 | 46% | ok | 3.1x (51%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.211 ms | 0.16 | 0.57 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.158 ms | 0.18 | 1.49 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 277.428 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 99.051 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.314 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 135.416 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 190.845 ms | 0.12 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 72.865 ms | 0.31 | 0.01 | 0% | ok | 0.8x (14%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 288.246 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.072 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.933 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.173 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.783 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.396 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.779 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.495 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.559 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.397 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.718 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.118 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.658 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.824 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.407 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.156 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.233 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.732 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.856 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.504 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.228 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.254 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.197 ms | 0.82 | 9.17 | 27% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.737 ms | 0.70 | 7.85 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.514 ms | 0.58 | 6.50 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.313 ms | 0.91 | 6.87 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 351.319 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 411.405 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 199.138 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.055 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| mixed | dot | f32>f64 | n=8388608 | 6.259 ms | 2.68 | 10.72 | 31% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.278 ms | 21.86 | 0.34 | 34% | ok | 5.2x (87%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.292 ms | 12.99 | 13.80 | 40% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.633 ms | 2.98 | 11.91 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.648 ms | 6.34 | 12.67 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.199 ms | 14.00 | 14.00 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.348 ms | 7.15 | 7.15 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 5.155 ms | 52.08 | 0.61 | n/a | ok | 5.0x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.316 ms | 50.50 | 0.39 | n/a | ok | 5.2x (87%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 6.473 ms | 41.47 | 0.24 | n/a | ok | 5.2x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.782 ms | 150.65 | 0.88 | n/a | ok | 4.8x (80%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.562 ms | 58.84 | 0.34 | n/a | ok | 5.1x (85%) |

## jetson-orin -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, scalar | f64 5.4, f32 5.4 GFLOP/s |
| bandwidth | read 11.5, update 16.2, triad 14.8 GB/s (ceiling 16.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 71% | 131% | 0 |
| blas2 | 10 | 64% | 123% | 0 |
| blas3 | 10 | 89% | 95% | 0 |
| factor | 12 | 41% | 61% | 0 |
| eigen | 8 | 37% | 80% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 32% | 74% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 60% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.633 ms | 1.44 | 11.54 | 71% | ok |
| blas1 | dot | f32 | n=8388608 | 6.587 ms | 2.55 | 10.19 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.267 ms | 1.37 | 16.41 | 101% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.963 ms | 2.81 | 16.88 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.355 ms | 1.32 | 21.12 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.600 ms | 3.00 | 11.98 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.384 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 12.620 ms | 2.66 | 10.64 | 66% | ok |
| blas2 | gemv | f32 | n=4096 | 5.885 ms | 5.70 | 11.41 | 71% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 12.953 ms | 2.59 | 10.37 | 64% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.832 ms | 5.75 | 11.51 | 71% | ok |
| blas2 | symv | f64 | n=4096 | 15.986 ms | 2.10 | 4.20 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 8.649 ms | 1.94 | 7.77 | 48% | ok |
| blas2 | trmv | f32 | n=4096 | 4.868 ms | 3.45 | 6.90 | 43% | ok |
| blas2 | trsv | f64 | n=4096 | 8.637 ms | 1.94 | 7.78 | 48% | ok |
| blas2 | trsv | f32 | n=4096 | 4.851 ms | 3.46 | 6.92 | 43% | ok |
| blas2 | ger | f64 | n=4096 | 13.491 ms | 2.49 | 19.90 | 123% | ok |
| blas3 | gemm | f64 | 512x512x512 | 26.731 ms | 10.04 | 0.24 | 94% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.122 ms | 20.46 | 0.24 | 95% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.622 ms | 8.07 | 0.38 | 75% | ok |
| blas3 | symm | f64 | 512x512x512 | 26.694 ms | 10.06 | 0.24 | 94% | ok |
| blas3 | syrk | f64 | 512x512x512 | 14.433 ms | 9.30 | 0.29 | 87% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 28.179 ms | 9.53 | 0.22 | 89% | ok |
| blas3 | trmm | f64 | 512x512x512 | 13.634 ms | 9.84 | 0.38 | 92% | ok |
| blas3 | trsm | f64 | 512x512x512 | 14.984 ms | 8.96 | 0.35 | 83% | ok |
| blas3 | trsm | f32 | 512x512x512 | 7.581 ms | 17.70 | 0.35 | 83% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.800 ms | 0.54 | 4.33 | 27% | ok |
| factor | lu | f64 | 512x512 | 13.587 ms | 6.59 | 0.31 | 61% | ok |
| factor | lu | f32 | 512x512 | 8.464 ms | 10.57 | 0.25 | 49% | ok |
| factor | cholesky | f64 | 512x512 | 7.870 ms | 5.68 | 0.53 | 53% | ok |
| factor | cholesky | f32 | 512x512 | 5.051 ms | 8.86 | 0.42 | 41% | ok |
| factor | ldlt | f64 | 512x512 | 32.350 ms | 1.38 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 85.727 ms | 0.52 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 28.432 ms | 6.29 | 0.15 | 59% | ok |
| factor | lq | f64 | 512x512 | 93.379 ms | 1.92 | 0.04 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 479.353 ms | 0.93 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.000 ms | 0.53 | 2.10 | 13% | ok |
| factor | inv | f64 | 512x512 | 173.614 ms | 1.55 | 0.02 | 14% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 17.699 ms | 5.06 | 0.24 | 47% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 10.351 ms | 2.16 | 0.10 | 20% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.168 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 899.228 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 62.978 ms | 2.66 | 0.02 | 25% | ok |
| eigen | schur | f64 | 256x256 | 90.345 ms | 4.64 | 0.01 | 43% | ok |
| eigen | svd | f64 | 256x256 | 43.220 ms | 8.54 | 0.02 | 80% | ok |
| eigen | expm | f64 | 256x256 | 57.335 ms | 4.39 | 0.02 | 41% | ok |
| eigen | sylvester | f64 | 256x256 | 253.673 ms | 3.97 | 0.01 | 37% | ok |
| elementwise | exp | f64 | n=8388608 | 75.836 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.342 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.313 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.075 ms | 0.15 | 2.44 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.498 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.279 ms | 1.15 | 11.95 | 74% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.085 ms | 0.85 | 8.83 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.136 ms | 2.44 | 5.14 | 32% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.222 ms | 0.16 | 0.57 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.891 ms | 0.17 | 1.40 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 280.496 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 101.423 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.396 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 130.286 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 189.415 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.944 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 291.222 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.182 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.421 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.138 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 67.378 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.576 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.448 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.008 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.451 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.700 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.607 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.655 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.107 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.799 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.822 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.664 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.629 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.485 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.683 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.083 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.494 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.229 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.939 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.003 ms | 0.87 | 9.77 | 60% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.735 ms | 0.70 | 7.85 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.502 ms | 0.58 | 6.51 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.941 ms | 0.86 | 6.47 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 351.388 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 356.687 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 194.035 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.238 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.499 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.646 ms | 3.55 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.557 ms | 3.68 | 3.91 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.600 ms | 3.00 | 11.98 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.687 ms | 6.24 | 12.49 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.642 ms | 2.97 | 2.97 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.757 ms | 9.55 | 9.55 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 38.114 ms | 7.04 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.110 ms | 6.53 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.870 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.862 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.142 ms | 4.46 | 0.03 | n/a | ok |

## jetson-orin -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 64.2 GFLOP/s, f32 128.4 GFLOP/s |
| ... hardware, NEON | f64 64.2, f32 128.4 GFLOP/s |
| ... this build, scalar | f64 32.1, f32 32.1 GFLOP/s |
| bandwidth | read 27.4, update 34.5, triad 33.0 GB/s (ceiling 34.5) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 92% | 140% | 0 | 42% |
| blas2 | 10 | 60% | 86% | 0 | 43% |
| blas3 | 10 | 72% | 88% | 0 | 86% |
| factor | 12 | 11% | 25% | 0 | 27% |
| eigen | 8 | 7% | 19% | 0 | 20% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 26% | 96% | 0 | 18% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 28% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 16% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.222 ms | 3.97 | 31.79 | 92% | ok | 2.8x (46%) |
| blas1 | dot | f32 | n=8388608 | 2.229 ms | 7.53 | 30.10 | 87% | ok | 3.0x (49%) |
| blas1 | axpy | f64 | n=8388608 | 4.922 ms | 3.41 | 40.90 | 119% | ok | 2.5x (42%) |
| blas1 | axpy | f32 | n=8388608 | 2.378 ms | 7.06 | 42.34 | 123% | ok | 2.5x (42%) |
| blas1 | scale | f64 | n=8388608 | 2.770 ms | 3.03 | 48.46 | 140% | ok | 2.3x (38%) |
| blas1 | two_norm | f64 | n=8388608 | 2.034 ms | 8.25 | 33.00 | 96% | ok | 2.8x (46%) |
| blas1 | one_norm | f64 | n=8388608 | 12.499 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.749 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.498 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 192.154 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 4.620 ms | 7.26 | 29.07 | 84% | ok | 2.7x (46%) |
| blas2 | gemv | f32 | n=4096 | 2.272 ms | 14.77 | 29.55 | 86% | ok | 2.6x (43%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.504 ms | 5.16 | 20.65 | 60% | ok | 2.0x (33%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.761 ms | 8.92 | 17.85 | 52% | ok | 1.6x (26%) |
| blas2 | symv | f64 | n=4096 | 5.256 ms | 6.38 | 12.78 | 37% | ok | 3.0x (51%) |
| blas2 | trmv | f64 | n=4096 | 2.927 ms | 5.73 | 22.95 | 66% | ok | 3.0x (49%) |
| blas2 | trmv | f32 | n=4096 | 1.442 ms | 11.63 | 23.29 | 67% | ok | 3.4x (56%) |
| blas2 | trsv | f64 | n=4096 | 8.547 ms | 1.96 | 7.86 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.868 ms | 3.45 | 6.90 | 20% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.411 ms | 2.50 | 20.02 | 58% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 4.817 ms | 55.72 | 1.31 | 87% | ok | 5.5x (92%) |
| blas3 | gemm | f32 | 512x512x512 | 2.470 ms | 108.66 | 1.27 | 85% | ok | 5.3x (89%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.086 ms | 32.85 | 1.54 | 51% | ok | 4.1x (68%) |
| blas3 | symm | f64 | 512x512x512 | 4.755 ms | 56.45 | 1.32 | 88% | ok | 5.6x (94%) |
| blas3 | syrk | f64 | 512x512x512 | 3.592 ms | 37.36 | 1.17 | 58% | ok | 4.0x (67%) |
| blas3 | syr2k | f64 | 512x512x512 | 6.209 ms | 43.23 | 1.01 | 67% | ok | 4.5x (76%) |
| blas3 | trmm | f64 | 512x512x512 | 2.573 ms | 52.17 | 2.04 | 81% | ok | 5.3x (88%) |
| blas3 | trsm | f64 | 512x512x512 | 2.899 ms | 46.31 | 1.81 | 72% | ok | 5.2x (86%) |
| blas3 | trsm | f32 | 512x512x512 | 1.569 ms | 85.57 | 1.67 | 67% | ok | 4.8x (81%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.246 ms | 0.57 | 4.59 | 13% | ok | 1.1x (18%) |
| factor | lu | f64 | 512x512 | 5.679 ms | 15.76 | 0.74 | 25% | ok | 2.4x (40%) |
| factor | lu | f32 | 512x512 | 4.542 ms | 19.70 | 0.46 | 15% | ok | 1.9x (31%) |
| factor | cholesky | f64 | 512x512 | 4.181 ms | 10.70 | 1.00 | 17% | ok | 1.9x (31%) |
| factor | cholesky | f32 | 512x512 | 3.146 ms | 14.22 | 0.67 | 11% | ok | 1.6x (27%) |
| factor | ldlt | f64 | 512x512 | 32.224 ms | 1.39 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 83.669 ms | 0.53 | 0.05 | 1% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 13.457 ms | 13.30 | 0.31 | 21% | ok | 2.1x (35%) |
| factor | lq | f64 | 512x512 | 61.635 ms | 2.90 | 0.07 | 5% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 321.671 ms | 1.39 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.544 ms | 1.93 | 7.71 | 22% | ok | 3.7x (61%) |
| factor | inv | f64 | 512x512 | 165.769 ms | 1.62 | 0.03 | 3% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 14.755 ms | 6.06 | 0.28 | 9% | ok | 1.2x (20%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 6.960 ms | 3.21 | 0.15 | 5% | ok | 1.5x (25%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.951 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 923.716 ms | 0.16 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 56.087 ms | 2.99 | 0.02 | 5% | ok | 1.1x (19%) |
| eigen | schur | f64 | 256x256 | 73.843 ms | 5.68 | 0.01 | 9% | ok | 1.2x (20%) |
| eigen | svd | f64 | 256x256 | 29.696 ms | 12.43 | 0.04 | 19% | ok | 1.5x (24%) |
| eigen | expm | f64 | 256x256 | 37.525 ms | 6.71 | 0.03 | 10% | ok | 1.5x (25%) |
| eigen | sylvester | f64 | 256x256 | 210.412 ms | 4.78 | 0.01 | 7% | ok | 1.2x (20%) |
| elementwise | exp | f64 | n=8388608 | 75.696 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 108.161 ms | 0.08 | 1.24 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 178.387 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.973 ms | 0.15 | 2.44 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.267 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.819 ms | 3.19 | 33.23 | 96% | ok | 2.8x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.062 ms | 0.85 | 8.89 | 26% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.800 ms | 7.22 | 15.18 | 44% | ok | 3.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.569 ms | 0.16 | 0.56 | 2% | ok | 1.0x (16%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.168 ms | 0.18 | 1.49 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 280.166 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 101.441 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.359 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 131.332 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 190.234 ms | 0.12 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.993 ms | 0.36 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 287.318 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.184 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.341 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.128 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.800 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.510 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.132 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.895 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.363 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.695 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.514 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.319 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.112 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.829 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.729 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.610 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.773 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.763 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.680 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.929 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.498 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.230 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 4.078 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.005 ms | 0.87 | 9.76 | 28% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.737 ms | 0.70 | 7.85 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.520 ms | 0.58 | 6.49 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.106 ms | 0.91 | 6.90 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 349.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 354.940 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 192.255 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.039 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.501 ms | 1.34 | 5.37 | 16% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.966 ms | 20.70 | 0.32 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 5.603 ms | 2.99 | 3.18 | 9% | ok | 0.8x (14%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.592 ms | 3.00 | 12.00 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.678 ms | 6.27 | 12.53 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.643 ms | 2.97 | 2.97 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.065 ms | 29.61 | 0.35 | n/a | ok | 4.2x (70%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.079 ms | 37.92 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.485 ms | 48.94 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.478 ms | 49.01 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.299 ms | 26.07 | 0.15 | n/a | ok | 5.8x (97%) |

