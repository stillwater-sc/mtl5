# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy governor performance on the P-cores, turbo on; performance 800-4900 MHz on cpus 0,2,8,10,12,14; performance 800-5000 MHz on cpus 4,6, 2026-11-11T19:00:41Z.

## i7-12700k -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.4 GFLOP/s |
| bandwidth | read 33.0, update 38.5, triad 32.5 GB/s (ceiling 38.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 97% | 0 |
| blas2 | 10 | 78% | 93% | 0 |
| blas3 | 10 | 17% | 18% | 0 |
| factor | 12 | 8% | 18% | 0 |
| eigen | 8 | 6% | 12% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 95% | 0 |
| direct | 7 | 1% | 2% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 77% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 85% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.857 ms | 2.86 | 22.91 | 59% | ok |
| blas1 | dot | f32 | n=8388608 | 4.218 ms | 3.98 | 15.91 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.351 ms | 2.64 | 31.70 | 82% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.725 ms | 3.55 | 21.30 | 55% | ok |
| blas1 | scale | f64 | n=8388608 | 3.576 ms | 2.35 | 37.54 | 97% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.888 ms | 3.43 | 13.73 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.037 ms | 2.08 | 16.62 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.042 ms | 1.19 | 9.53 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.995 ms | 2.10 | 16.80 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 77.567 ms | 2.38 | 1.73 | 4% | ok |
| blas2 | gemv | f64 | n=4096 | 3.756 ms | 8.93 | 35.75 | 93% | ok |
| blas2 | gemv | f32 | n=4096 | 2.728 ms | 12.30 | 24.62 | 64% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.811 ms | 6.97 | 27.91 | 72% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.248 ms | 14.93 | 29.87 | 78% | ok |
| blas2 | symv | f64 | n=4096 | 6.593 ms | 5.09 | 10.19 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 2.034 ms | 8.25 | 33.02 | 86% | ok |
| blas2 | trmv | f32 | n=4096 | 1.453 ms | 11.55 | 23.12 | 60% | ok |
| blas2 | trsv | f64 | n=4096 | 2.038 ms | 8.23 | 32.96 | 86% | ok |
| blas2 | trsv | f32 | n=4096 | 1.414 ms | 11.87 | 23.75 | 62% | ok |
| blas2 | ger | f64 | n=4096 | 7.717 ms | 4.35 | 34.79 | 90% | ok |
| blas3 | gemm | f64 | 512x512x512 | 20.198 ms | 13.29 | 0.31 | 17% | ok |
| blas3 | gemm | f32 | 512x512x512 | 10.618 ms | 25.28 | 0.30 | 16% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 10.773 ms | 12.46 | 0.58 | 16% | ok |
| blas3 | symm | f64 | 512x512x512 | 19.298 ms | 13.91 | 0.33 | 18% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.043 ms | 10.29 | 0.32 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 23.572 ms | 11.39 | 0.27 | 15% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.743 ms | 13.78 | 0.54 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.708 ms | 13.83 | 0.54 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.142 ms | 26.10 | 0.51 | 17% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.214 ms | 0.40 | 3.21 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.625 ms | 7.70 | 0.36 | 10% | ok |
| factor | lu | f32 | 512x512 | 4.233 ms | 21.14 | 0.50 | 13% | ok |
| factor | cholesky | f64 | 512x512 | 7.345 ms | 6.09 | 0.57 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.611 ms | 6.77 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.458 ms | 6.00 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 47.970 ms | 0.93 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 227.701 ms | 0.79 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 25.974 ms | 6.89 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 273.695 ms | 1.63 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.638 ms | 0.64 | 2.56 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 54.709 ms | 4.91 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.243 ms | 14.33 | 0.67 | 18% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 18.156 ms | 1.23 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 59.144 ms | 2.55 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 743.077 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 32.983 ms | 5.09 | 0.03 | 6% | ok |
| eigen | schur | f64 | 256x256 | 81.936 ms | 5.12 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 110.166 ms | 3.35 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 26.373 ms | 9.54 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 186.214 ms | 5.41 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.618 ms | 0.21 | 3.39 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 46.018 ms | 0.18 | 2.92 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.309 ms | 0.14 | 2.23 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.098 ms | 0.29 | 4.61 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 203.386 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.746 ms | 3.51 | 36.50 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.943 ms | 2.77 | 28.87 | 75% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.649 ms | 9.01 | 18.94 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.874 ms | 0.67 | 2.36 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.278 ms | 1.57 | 13.03 | 34%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.448 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.587 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.901 ms | 1.26 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.158 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 54.223 ms | 0.41 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.808 ms | 1.33 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 101.709 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.364 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.987 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.230 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.082 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.258 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 33.126 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.161 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.540 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.050 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.074 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.765 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.335 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.321 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.229 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.456 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.854 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.322 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.555 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.376 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.069 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.005 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.692 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.992 ms | 2.64 | 29.56 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.221 ms | 2.14 | 24.02 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.564 ms | 1.67 | 18.75 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.339 ms | 2.88 | 21.79 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.970 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 44.412 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 39.759 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.079 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 5.136 ms | 3.27 | 13.07 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 39.013 ms | 6.88 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.545 ms | 30.80 | 32.72 | 85%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.305 ms | 7.28 | 29.12 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.009 ms | 16.62 | 33.25 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.724 ms | 9.73 | 9.73 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.73 | 9.73 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.466 ms | 15.37 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.813 ms | 12.90 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.230 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.230 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.812 ms | 10.01 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- native, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 600.6 GFLOP/s, f32 1201.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.6, f32 1201.5 GFLOP/s |
| ... this build, scalar | f64 9.9, f32 10.7 GFLOP/s |
| bandwidth | read 83.5, update 78.6, triad 76.6 GB/s (ceiling 83.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 84% | 123% | 0 | 28% |
| blas2 | 10 | 42% | 138% | 0 | 13% |
| blas3 | 10 | 2% | 16% | 0 | 13% |
| factor | 12 | 1% | 12% | 0 | 13% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 282% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 36% | 0 | 13% |
| mg | 1 | 25% | 25% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.475 ms | 11.37 | 90.97 | 109% | ok | 4.0x (50%) |
| blas1 | dot | f32 | n=8388608 | 0.653 ms | 25.68 | 102.72 | 123% | ok | 6.5x (81%) |
| blas1 | axpy | f64 | n=8388608 | 2.834 ms | 5.92 | 71.04 | 85% | ok | 2.2x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.453 ms | 11.55 | 69.29 | 83% | ok | 3.3x (41%) |
| blas1 | scale | f64 | n=8388608 | 1.910 ms | 4.39 | 70.28 | 84% | ok | 1.9x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.742 ms | 22.62 | 90.47 | 108% | ok | 6.6x (82%) |
| blas1 | one_norm | f64 | n=8388608 | 3.977 ms | 2.11 | 16.87 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.932 ms | 1.21 | 9.68 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.980 ms | 2.11 | 16.86 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.714 ms | 2.44 | 1.77 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.435 ms | 23.39 | 93.59 | 112% | ok | 2.6x (33%) |
| blas2 | gemv | f32 | n=4096 | 0.582 ms | 57.69 | 115.44 | 138% | ok | 4.7x (59%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.560 ms | 21.51 | 86.06 | 103% | ok | 3.1x (39%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.866 ms | 38.73 | 77.51 | 93% | ok | 2.6x (32%) |
| blas2 | symv | f64 | n=4096 | 6.506 ms | 5.16 | 10.32 | 12% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.022 ms | 8.30 | 33.22 | 40% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.433 ms | 11.71 | 23.45 | 28% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 2.019 ms | 8.31 | 33.27 | 40% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.394 ms | 12.04 | 24.09 | 29% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.645 ms | 4.39 | 35.12 | 42% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.796 ms | 96.02 | 2.25 | 16% | ok | 7.2x (90%) |
| blas3 | gemm | f32 | 512x512x512 | 1.553 ms | 172.80 | 2.03 | 14% | ok | 6.8x (85%) |
| blas3 | gemmt | f64 | 512x512x512 | 10.229 ms | 13.12 | 0.62 | 2% | ok | 1.1x (13%) |
| blas3 | symm | f64 | 512x512x512 | 19.274 ms | 13.93 | 0.33 | 2% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 12.753 ms | 10.52 | 0.33 | 2% | ok | 1.0x (13%) |
| blas3 | syr2k | f64 | 512x512x512 | 20.393 ms | 13.16 | 0.31 | 2% | ok | 1.2x (14%) |
| blas3 | trmm | f64 | 512x512x512 | 10.655 ms | 12.60 | 0.49 | 2% | ok | 0.9x (11%) |
| blas3 | trsm | f64 | 512x512x512 | 10.717 ms | 12.52 | 0.49 | 2% | ok | 0.9x (11%) |
| blas3 | trsm | f32 | 512x512x512 | 5.508 ms | 24.37 | 0.48 | 2% | ok | 0.9x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.892 ms | 0.41 | 3.29 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 7.486 ms | 11.95 | 0.56 | 2% | ok | 1.6x (19%) |
| factor | lu | f32 | 512x512 | 4.133 ms | 21.65 | 0.51 | 2% | ok | 1.0x (13%) |
| factor | cholesky | f64 | 512x512 | 7.098 ms | 6.30 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.453 ms | 6.93 | 0.32 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.189 ms | 6.22 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 45.056 ms | 0.99 | 0.09 | 0% | ok | 1.1x (13%) |
| factor | qr | f64 | 512x512 | 128.974 ms | 1.39 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 17.508 ms | 10.22 | 0.24 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 165.489 ms | 2.70 | 0.03 | 0% | ok | 1.7x (21%) |
| factor | householder | f64 | 512x512 | 0.421 ms | 2.50 | 9.97 | 12%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 51.398 ms | 5.22 | 0.08 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.092 ms | 14.69 | 0.69 | 2% | ok | 1.0x (13%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.763 ms | 1.26 | 0.06 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.948 ms | 2.61 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 736.881 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 32.627 ms | 5.14 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 80.316 ms | 5.22 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 106.789 ms | 3.46 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 11.613 ms | 21.67 | 0.09 | 4% | ok | 2.3x (28%) |
| eigen | sylvester | f64 | 256x256 | 173.171 ms | 5.81 | 0.01 | 1% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.840 ms | 0.22 | 3.46 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.429 ms | 0.18 | 2.95 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.971 ms | 0.14 | 2.28 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.520 ms | 0.29 | 4.71 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 200.401 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.116 ms | 22.60 | 235.16 | 282% ! | ok | 6.4x (81%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.904 ms | 2.90 | 30.14 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.411 ms | 29.68 | 62.41 | 75% | ok | 3.3x (41%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.817 ms | 0.68 | 2.39 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.387 ms | 1.45 | 12.01 | 14%\* | ok | 0.9x (12%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 90.206 ms | 0.12 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.387 ms | 0.44 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.009 ms | 1.24 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.926 ms | 0.56 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 52.716 ms | 0.42 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 21.140 ms | 1.06 | 0.03 | 0% | ok | 0.8x (10%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 90.302 ms | 0.02 | 0.00 | 0% | ok | 1.1x (14%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.383 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.849 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.226 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.831 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.473 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.177 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.466 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.218 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.611 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.713 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.332 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.570 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.908 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.870 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.153 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.591 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 20.939 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.515 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 12.056 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.048 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.985 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.020 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.675 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.972 ms | 2.69 | 30.17 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.199 ms | 2.18 | 24.45 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.536 ms | 1.70 | 19.09 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.745 ms | 2.80 | 21.15 | 25% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.658 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 43.775 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 34.846 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.921 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.805 ms | 3.49 | 13.97 | 17% | ok | 1.1x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.211 ms | 51.52 | 0.80 | 9% | ok | 7.5x (94%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.533 ms | 31.46 | 33.43 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.299 ms | 7.30 | 29.20 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.997 ms | 16.82 | 33.65 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.686 ms | 9.95 | 9.95 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.686 ms | 9.95 | 9.95 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.497 ms | 107.49 | 1.26 | n/a | ok | 7.0x (87%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.802 ms | 95.80 | 0.75 | n/a | ok | 7.4x (93%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.94 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.357 ms | 197.89 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.532 ms | 76.01 | 0.45 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## i7-12700k -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, AVX2 | f64 78.3, f32 156.6 GFLOP/s |
| bandwidth | read 37.8, update 41.5, triad 31.5 GB/s (ceiling 41.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 87% | 103% | 0 |
| blas2 | 10 | 91% | 96% | 0 |
| blas3 | 10 | 65% | 91% | 0 |
| factor | 12 | 8% | 21% | 0 |
| eigen | 8 | 7% | 27% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 52% | 88% | 0 |
| direct | 7 | 1% | 2% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 54% | 72% | 0 |
| mg | 1 | 49% | 49% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 85% | 132% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 3.697 ms | 4.54 | 36.30 | 87% | ok |
| blas1 | dot | f32 | n=8388608 | 1.704 ms | 9.85 | 39.39 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.786 ms | 3.51 | 42.06 | 101% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.357 ms | 7.12 | 42.71 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 3.459 ms | 2.42 | 38.80 | 93% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.212 ms | 7.59 | 30.34 | 73% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.022 ms | 2.09 | 16.69 | 40% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.036 ms | 1.19 | 9.54 | 23% | ok |
| blas1 | sum | f64 | n=8388608 | 3.964 ms | 2.12 | 16.93 | 41% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 71.043 ms | 2.60 | 1.89 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.419 ms | 9.81 | 39.27 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 1.685 ms | 19.92 | 39.85 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.430 ms | 3.56 | 14.24 | 34% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.014 ms | 8.36 | 16.73 | 40% | ok |
| blas2 | symv | f64 | n=4096 | 6.205 ms | 5.41 | 10.83 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 1.796 ms | 9.34 | 37.41 | 90% | ok |
| blas2 | trmv | f32 | n=4096 | 0.864 ms | 19.43 | 38.89 | 94% | ok |
| blas2 | trsv | f64 | n=4096 | 1.818 ms | 9.23 | 36.95 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 0.889 ms | 18.88 | 37.79 | 91% | ok |
| blas2 | ger | f64 | n=4096 | 7.139 ms | 4.70 | 37.61 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.787 ms | 70.87 | 1.66 | 91% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.896 ms | 141.60 | 1.66 | 90% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.626 ms | 51.11 | 2.40 | 65% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.040 ms | 66.44 | 1.56 | 85% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.819 ms | 13.67 | 0.43 | 17% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 20.122 ms | 13.34 | 0.31 | 17% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.705 ms | 49.62 | 1.94 | 63% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.748 ms | 48.84 | 1.91 | 62% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.283 ms | 104.59 | 2.04 | 67% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.320 ms | 0.40 | 3.19 | 8% | ok |
| factor | lu | f64 | 512x512 | 10.475 ms | 8.54 | 0.40 | 11% | ok |
| factor | lu | f32 | 512x512 | 3.391 ms | 26.38 | 0.62 | 17% | ok |
| factor | cholesky | f64 | 512x512 | 7.083 ms | 6.32 | 0.59 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.545 ms | 6.84 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.201 ms | 6.21 | 0.58 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 46.075 ms | 0.97 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 232.890 ms | 0.77 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 24.261 ms | 7.38 | 0.17 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 267.725 ms | 1.67 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.661 ms | 0.63 | 2.52 | 6%\* | ok |
| factor | inv | f64 | 512x512 | 55.093 ms | 4.87 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.458 ms | 16.39 | 0.77 | 21% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.564 ms | 1.35 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 56.664 ms | 2.66 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 742.662 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 30.493 ms | 5.50 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 79.022 ms | 5.31 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 89.919 ms | 4.10 | 0.01 | 5% | ok |
| eigen | expm | f64 | 256x256 | 11.746 ms | 21.42 | 0.09 | 27% | ok |
| eigen | sylvester | f64 | 256x256 | 175.459 ms | 5.74 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.502 ms | 0.21 | 3.40 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 47.053 ms | 0.18 | 2.85 | 7% | ok |
| elementwise | sin | f64 | n=8388608 | 60.051 ms | 0.14 | 2.24 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.134 ms | 0.29 | 4.61 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 203.371 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.743 ms | 3.52 | 36.67 | 88% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.926 ms | 2.83 | 29.42 | 71% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.056 ms | 10.32 | 21.71 | 52% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.828 ms | 0.67 | 2.38 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.277 ms | 1.57 | 13.03 | 31%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 72.881 ms | 0.15 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.404 ms | 0.41 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.064 ms | 1.23 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.598 ms | 0.55 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.829 ms | 0.39 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.033 ms | 1.31 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.783 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.410 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.034 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.227 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.264 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.443 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.591 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 21.367 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.491 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.766 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.680 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.212 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.882 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.509 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.686 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.976 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.074 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.336 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.576 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.142 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.012 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.698 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.983 ms | 2.66 | 29.83 | 72% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.312 ms | 1.99 | 22.35 | 54% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.511 ms | 1.73 | 19.41 | 47% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.388 ms | 2.67 | 20.20 | 49% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 28.542 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 39.810 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 39.924 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.597 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.896 ms | 8.85 | 35.40 | 85% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 5.999 ms | 44.75 | 0.70 | 57% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.324 ms | 51.73 | 54.96 | 132%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.921 ms | 8.73 | 34.94 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.767 ms | 21.88 | 43.75 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.292 ms | 57.54 | 57.54 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.279 ms | 60.03 | 60.03 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.958 ms | 67.83 | 0.79 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.623 ms | 47.74 | 0.37 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.682 ms | 47.25 | 0.28 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 2.115 ms | 126.92 | 0.74 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 1.904 ms | 141.01 | 0.83 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 600.8 GFLOP/s, f32 1201.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.8, f32 1201.5 GFLOP/s |
| ... this build, AVX2 | f64 600.8, f32 1201.6 GFLOP/s |
| bandwidth | read 82.0, update 78.5, triad 76.5 GB/s (ceiling 82.0) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 88% | 146% | 0 | 22% |
| blas2 | 10 | 49% | 150% | 0 | 13% |
| blas3 | 10 | 11% | 62% | 0 | 16% |
| factor | 12 | 1% | 12% | 0 | 13% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 302% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 28% | 37% | 0 | 13% |
| mg | 1 | 24% | 24% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 49% | 71% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 79% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.431 ms | 11.72 | 93.79 | 114% | ok | 2.6x (32%) |
| blas1 | dot | f32 | n=8388608 | 0.561 ms | 29.88 | 119.54 | 146% | ok | 3.0x (38%) |
| blas1 | axpy | f64 | n=8388608 | 2.781 ms | 6.03 | 72.40 | 88% | ok | 1.7x (22%) |
| blas1 | axpy | f32 | n=8388608 | 1.360 ms | 12.33 | 74.00 | 90% | ok | 1.7x (22%) |
| blas1 | scale | f64 | n=8388608 | 1.893 ms | 4.43 | 70.91 | 86% | ok | 1.8x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.572 ms | 29.35 | 117.39 | 143% | ok | 3.9x (48%) |
| blas1 | one_norm | f64 | n=8388608 | 3.991 ms | 2.10 | 16.81 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.940 ms | 1.21 | 9.67 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.932 ms | 2.13 | 17.07 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 70.036 ms | 2.64 | 1.92 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.438 ms | 23.34 | 93.41 | 114% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.546 ms | 61.42 | 122.90 | 150% | ok | 3.1x (39%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.164 ms | 15.50 | 62.05 | 76% | ok | 4.4x (54%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.919 ms | 36.51 | 73.05 | 89% | ok | 4.4x (55%) |
| blas2 | symv | f64 | n=4096 | 6.176 ms | 5.43 | 10.88 | 13% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.758 ms | 9.54 | 38.21 | 47% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.843 ms | 19.89 | 39.82 | 49% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.797 ms | 9.34 | 37.39 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.862 ms | 19.46 | 38.95 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.105 ms | 4.72 | 37.79 | 46% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 1.167 ms | 230.07 | 5.39 | 38% | ok | 3.2x (41%) |
| blas3 | gemm | f32 | 512x512x512 | 0.360 ms | 745.23 | 8.73 | 62% | ok | 5.3x (66%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.250 ms | 107.34 | 5.03 | 18% | ok | 2.1x (26%) |
| blas3 | symm | f64 | 512x512x512 | 1.142 ms | 235.00 | 5.51 | 39% | ok | 3.5x (44%) |
| blas3 | syrk | f64 | 512x512x512 | 10.572 ms | 12.70 | 0.40 | 2% | ok | 0.9x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 16.443 ms | 16.33 | 0.38 | 3% | ok | 1.2x (15%) |
| blas3 | trmm | f64 | 512x512x512 | 2.102 ms | 63.87 | 2.49 | 11% | ok | 1.3x (16%) |
| blas3 | trsm | f64 | 512x512x512 | 2.121 ms | 63.29 | 2.47 | 11% | ok | 1.3x (16%) |
| blas3 | trsm | f32 | 512x512x512 | 0.974 ms | 137.85 | 2.69 | 11% | ok | 1.3x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.966 ms | 0.41 | 3.28 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.150 ms | 14.55 | 0.68 | 2% | ok | 1.7x (21%) |
| factor | lu | f32 | 512x512 | 3.445 ms | 25.98 | 0.61 | 2% | ok | 1.0x (12%) |
| factor | cholesky | f64 | 512x512 | 6.899 ms | 6.48 | 0.61 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.424 ms | 6.96 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.054 ms | 6.34 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 44.085 ms | 1.01 | 0.10 | 0% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 126.257 ms | 1.42 | 0.03 | 0% | ok | 1.8x (23%) |
| factor | lq | f64 | 512x512 | 15.027 ms | 11.91 | 0.28 | 2% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 162.508 ms | 2.75 | 0.03 | 0% | ok | 1.6x (21%) |
| factor | householder | f64 | 512x512 | 0.426 ms | 2.47 | 9.86 | 12%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 51.962 ms | 5.17 | 0.08 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.504 ms | 16.26 | 0.76 | 3% | ok | 1.0x (12%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.806 ms | 1.33 | 0.06 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 56.196 ms | 2.69 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 738.239 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 31.162 ms | 5.38 | 0.03 | 1% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 80.053 ms | 5.24 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 92.743 ms | 3.98 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 10.262 ms | 24.52 | 0.10 | 4% | ok | 1.1x (14%) |
| eigen | sylvester | f64 | 256x256 | 172.552 ms | 5.83 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.867 ms | 0.22 | 3.45 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 46.005 ms | 0.18 | 2.92 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.765 ms | 0.14 | 2.28 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.583 ms | 0.29 | 4.70 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 201.350 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.110 ms | 23.82 | 247.77 | 302% ! | ok | 6.8x (84%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.921 ms | 2.84 | 29.58 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.377 ms | 30.41 | 63.94 | 78% | ok | 2.9x (37%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.681 ms | 0.70 | 2.46 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.387 ms | 1.45 | 12.00 | 15%\* | ok | 0.9x (12%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 71.904 ms | 0.16 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 26.928 ms | 0.42 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.181 ms | 1.22 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.650 ms | 0.56 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 55.654 ms | 0.40 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.889 ms | 1.32 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 88.733 ms | 0.02 | 0.00 | 0% | ok | 1.1x (14%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.386 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.949 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.224 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.958 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 27.998 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.963 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.389 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.621 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.523 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.074 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.532 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.381 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.489 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.795 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.828 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.022 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.519 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.891 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.060 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.002 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.696 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.967 ms | 2.71 | 30.33 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.288 ms | 2.03 | 22.77 | 28% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.485 ms | 1.76 | 19.75 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.804 ms | 2.60 | 19.63 | 24% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.840 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 39.314 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 38.633 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.499 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.902 ms | 8.82 | 35.29 | 43% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 0.913 ms | 294.11 | 4.60 | 49% | ok | 6.6x (82%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.306 ms | 54.85 | 58.28 | 71%\* | ok | 1.1x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.910 ms | 8.78 | 35.13 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.768 ms | 21.85 | 43.69 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.266 ms | 63.13 | 63.13 | n/a | ok | 1.1x (14%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.255 ms | 65.77 | 65.77 | n/a | ok | 1.1x (14%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 0.625 ms | 429.61 | 5.03 | n/a | ok | 6.3x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 0.819 ms | 327.83 | 2.56 | n/a | ok | 6.9x (86%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 0.803 ms | 334.34 | 1.96 | n/a | ok | 7.1x (88%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.333 ms | 806.11 | 4.72 | n/a | ok | 6.4x (79%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.302 ms | 888.58 | 5.21 | n/a | ok | 6.3x (79%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## i7-12700k -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.1, update 38.5, triad 32.5 GB/s (ceiling 38.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 98% | 0 |
| blas2 | 10 | 97% | 106% | 0 |
| blas3 | 10 | 81% | 94% | 0 |
| factor | 12 | 25% | 40% | 0 |
| eigen | 8 | 24% | 71% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 95% | 0 |
| direct | 7 | 1% | 2% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 76% | 0 |
| mg | 1 | 56% | 56% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 35% | 83% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.616 ms | 2.99 | 23.90 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.224 ms | 3.97 | 15.89 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.438 ms | 2.61 | 31.27 | 81% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.926 ms | 3.41 | 20.44 | 53% | ok |
| blas1 | scale | f64 | n=8388608 | 3.573 ms | 2.35 | 37.57 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.878 ms | 3.44 | 13.76 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.023 ms | 2.09 | 16.68 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.036 ms | 1.19 | 9.54 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.994 ms | 2.10 | 16.80 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.947 ms | 2.40 | 1.74 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.508 ms | 9.56 | 38.28 | 99% | ok |
| blas2 | gemv | f32 | n=4096 | 1.642 ms | 20.43 | 40.89 | 106% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.628 ms | 9.25 | 37.01 | 96% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.638 ms | 20.48 | 40.98 | 106% | ok |
| blas2 | symv | f64 | n=4096 | 1.914 ms | 17.53 | 35.09 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 1.856 ms | 9.04 | 36.20 | 94% | ok |
| blas2 | trmv | f32 | n=4096 | 0.943 ms | 17.79 | 35.62 | 93% | ok |
| blas2 | trsv | f64 | n=4096 | 1.795 ms | 9.35 | 37.43 | 97% | ok |
| blas2 | trsv | f32 | n=4096 | 0.876 ms | 19.15 | 38.34 | 100% | ok |
| blas2 | ger | f64 | n=4096 | 7.697 ms | 4.36 | 34.89 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.709 ms | 72.38 | 1.70 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.832 ms | 146.52 | 1.72 | 94% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.573 ms | 52.16 | 2.45 | 67% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.718 ms | 72.21 | 1.69 | 92% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.546 ms | 52.72 | 1.65 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.716 ms | 56.92 | 1.33 | 73% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.119 ms | 63.35 | 2.47 | 81% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.143 ms | 62.63 | 2.45 | 80% | ok |
| blas3 | trsm | f32 | 512x512x512 | 0.962 ms | 139.55 | 2.73 | 89% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.272 ms | 0.40 | 3.20 | 8% | ok |
| factor | lu | f64 | 512x512 | 3.248 ms | 27.55 | 1.29 | 35% | ok |
| factor | lu | f32 | 512x512 | 1.885 ms | 47.48 | 1.11 | 30% | ok |
| factor | cholesky | f64 | 512x512 | 2.051 ms | 21.81 | 2.04 | 28% | ok |
| factor | cholesky | f32 | 512x512 | 1.121 ms | 39.92 | 1.87 | 25% | ok |
| factor | ldlt | f64 | 512x512 | 7.444 ms | 6.01 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 46.117 ms | 0.97 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 5.752 ms | 31.11 | 0.73 | 40% | ok |
| factor | lq | f64 | 512x512 | 25.999 ms | 6.88 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 270.923 ms | 1.65 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.638 ms | 0.64 | 2.56 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 46.193 ms | 5.81 | 0.09 | 7% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 3.785 ms | 23.64 | 1.11 | 30% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.710 ms | 13.08 | 0.61 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 59.238 ms | 2.55 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 746.680 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.216 ms | 10.35 | 0.06 | 13% | ok |
| eigen | schur | f64 | 256x256 | 21.003 ms | 19.97 | 0.05 | 26% | ok |
| eigen | svd | f64 | 256x256 | 6.658 ms | 55.44 | 0.16 | 71% | ok |
| eigen | expm | f64 | 256x256 | 11.271 ms | 22.33 | 0.09 | 29% | ok |
| eigen | sylvester | f64 | 256x256 | 53.545 ms | 18.80 | 0.03 | 24% | ok |
| elementwise | exp | f64 | n=8388608 | 39.558 ms | 0.21 | 3.39 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.787 ms | 0.18 | 2.93 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.319 ms | 0.14 | 2.23 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.137 ms | 0.29 | 4.61 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 202.087 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.747 ms | 3.51 | 36.47 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.927 ms | 2.82 | 29.36 | 76% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.665 ms | 8.98 | 18.88 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.850 ms | 0.67 | 2.37 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.267 ms | 1.58 | 13.14 | 34%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.327 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.635 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.915 ms | 1.25 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 41.045 ms | 0.55 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.887 ms | 0.42 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.927 ms | 1.32 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.990 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.353 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.980 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.220 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.231 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.228 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.851 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.036 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.522 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.031 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.069 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.802 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.404 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.317 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.225 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.500 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 18.011 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.394 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.546 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.395 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.065 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 0.997 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.118 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.002 ms | 2.61 | 29.26 | 76% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.222 ms | 2.14 | 23.99 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.565 ms | 1.67 | 18.74 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.440 ms | 2.86 | 21.63 | 56% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 31.798 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.267 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 38.901 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.014 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.916 ms | 3.41 | 13.65 | 35% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 39.048 ms | 6.87 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.560 ms | 29.97 | 31.84 | 83%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.300 ms | 7.29 | 29.18 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.014 ms | 16.54 | 33.08 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.72 | 9.72 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.73 | 9.73 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.476 ms | 15.36 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.817 ms | 12.89 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.235 ms | 26.23 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.235 ms | 26.23 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.851 ms | 10.00 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 600.4 GFLOP/s, f32 1200.9 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.4, f32 1200.9 GFLOP/s |
| ... this build, scalar | f64 10.1, f32 10.0 GFLOP/s |
| bandwidth | read 82.1, update 77.5, triad 75.7 GB/s (ceiling 82.1) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 86% | 125% | 0 | 28% |
| blas2 | 10 | 119% | 197% | 0 | 31% |
| blas3 | 10 | 66% | 85% | 0 | 78% |
| factor | 12 | 5% | 12% | 0 | 17% |
| eigen | 8 | 3% | 10% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 294% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 36% | 0 | 13% |
| mg | 1 | 25% | 25% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 39% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 90% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.463 ms | 11.46 | 91.72 | 112% | ok | 3.8x (48%) |
| blas1 | dot | f32 | n=8388608 | 0.655 ms | 25.60 | 102.40 | 125% | ok | 6.4x (81%) |
| blas1 | axpy | f64 | n=8388608 | 2.831 ms | 5.93 | 71.12 | 87% | ok | 2.3x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.426 ms | 11.76 | 70.57 | 86% | ok | 3.5x (43%) |
| blas1 | scale | f64 | n=8388608 | 1.910 ms | 4.39 | 70.26 | 86% | ok | 1.9x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.751 ms | 22.35 | 89.38 | 109% | ok | 6.5x (81%) |
| blas1 | one_norm | f64 | n=8388608 | 4.025 ms | 2.08 | 16.67 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.961 ms | 1.21 | 9.64 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.976 ms | 2.11 | 16.88 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 76.839 ms | 2.40 | 1.75 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.441 ms | 23.28 | 93.16 | 113% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.557 ms | 60.21 | 120.49 | 147% | ok | 2.9x (37%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.448 ms | 23.17 | 92.71 | 113% | ok | 2.5x (31%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.690 ms | 48.65 | 97.35 | 119% | ok | 2.4x (30%) |
| blas2 | symv | f64 | n=4096 | 0.626 ms | 53.58 | 107.27 | 131% | ok | 3.1x (38%) |
| blas2 | trmv | f64 | n=4096 | 0.605 ms | 27.74 | 111.08 | 135% | ok | 3.1x (38%) |
| blas2 | trmv | f32 | n=4096 | 0.207 ms | 80.89 | 161.93 | 197% ! | ok | 4.5x (57%) |
| blas2 | trsv | f64 | n=4096 | 1.761 ms | 9.53 | 38.14 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.856 ms | 19.61 | 39.26 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.657 ms | 4.38 | 35.07 | 43% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.539 ms | 497.80 | 11.67 | 83% | ok | 6.9x (86%) |
| blas3 | gemm | f32 | 512x512x512 | 0.264 ms | 1017.39 | 11.92 | 85% | ok | 6.9x (87%) |
| blas3 | gemmt | f64 | 512x512x512 | 0.706 ms | 190.21 | 8.92 | 32% | ok | 3.6x (46%) |
| blas3 | symm | f64 | 512x512x512 | 0.540 ms | 497.44 | 11.66 | 83% | ok | 6.9x (86%) |
| blas3 | syrk | f64 | 512x512x512 | 0.749 ms | 179.08 | 5.60 | 30% | ok | 3.4x (42%) |
| blas3 | syr2k | f64 | 512x512x512 | 1.123 ms | 239.07 | 5.60 | 40% | ok | 4.2x (53%) |
| blas3 | trmm | f64 | 512x512x512 | 0.341 ms | 393.78 | 15.38 | 66% | ok | 6.2x (78%) |
| blas3 | trsm | f64 | 512x512x512 | 0.338 ms | 396.51 | 15.49 | 66% | ok | 6.3x (79%) |
| blas3 | trsm | f32 | 512x512x512 | 0.160 ms | 838.72 | 16.38 | 70% | ok | 6.0x (75%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.021 ms | 0.41 | 3.26 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 2.126 ms | 42.09 | 1.97 | 7% | ok | 1.5x (19%) |
| factor | lu | f32 | 512x512 | 1.252 ms | 71.47 | 1.68 | 6% | ok | 1.5x (19%) |
| factor | cholesky | f64 | 512x512 | 1.533 ms | 29.19 | 2.74 | 5% | ok | 1.3x (17%) |
| factor | cholesky | f32 | 512x512 | 0.943 ms | 47.43 | 2.22 | 4% | ok | 1.2x (15%) |
| factor | ldlt | f64 | 512x512 | 7.270 ms | 6.15 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 43.304 ms | 1.03 | 0.10 | 0% | ok | 1.1x (13%) |
| factor | qr | f64 | 512x512 | 4.146 ms | 43.16 | 1.01 | 7% | ok | 1.4x (17%) |
| factor | lq | f64 | 512x512 | 17.482 ms | 10.24 | 0.24 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 164.842 ms | 2.71 | 0.03 | 0% | ok | 1.6x (21%) |
| factor | householder | f64 | 512x512 | 0.431 ms | 2.44 | 9.73 | 12%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 46.394 ms | 5.79 | 0.09 | 1% | ok | 1.0x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 2.759 ms | 32.43 | 1.52 | 5% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.641 ms | 13.63 | 0.64 | 2% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.936 ms | 2.61 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 739.098 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 17.426 ms | 9.63 | 0.06 | 2% | ok | 0.9x (12%) |
| eigen | schur | f64 | 256x256 | 21.686 ms | 19.34 | 0.05 | 3% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 6.151 ms | 60.01 | 0.17 | 10% | ok | 1.1x (14%) |
| eigen | expm | f64 | 256x256 | 8.753 ms | 28.75 | 0.12 | 5% | ok | 1.3x (16%) |
| eigen | sylvester | f64 | 256x256 | 55.162 ms | 18.25 | 0.03 | 3% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 39.075 ms | 0.21 | 3.43 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 46.105 ms | 0.18 | 2.91 | 4% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 59.795 ms | 0.14 | 2.24 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.700 ms | 0.29 | 4.68 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 202.511 ms | 0.04 | 0.66 | 1% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.23 | 241.72 | 294% ! | ok | 6.6x (83%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.912 ms | 2.87 | 29.86 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.418 ms | 29.54 | 62.10 | 76% | ok | 3.3x (41%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.820 ms | 0.68 | 2.39 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.378 ms | 1.46 | 12.08 | 15%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 90.181 ms | 0.12 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.378 ms | 0.44 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.963 ms | 1.25 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.411 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 52.932 ms | 0.42 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.012 ms | 1.32 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 89.053 ms | 0.02 | 0.00 | 0% | ok | 1.1x (14%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.344 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.918 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.215 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.919 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.712 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.214 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.606 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.252 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.797 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.770 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.226 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.897 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.993 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.932 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.658 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.538 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 12.528 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.054 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.125 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.993 ms | 2.64 | 29.54 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.209 ms | 2.17 | 24.26 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.595 ms | 1.64 | 18.39 | 22% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.204 ms | 2.71 | 20.46 | 25% | ok | 0.9x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.596 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.574 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 39.274 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.096 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 4.916 ms | 3.41 | 13.65 | 17% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 8.815 ms | 30.45 | 0.48 | 5% | ok | 4.4x (55%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.562 ms | 29.84 | 31.71 | 39%\* | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.323 ms | 7.22 | 28.89 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.004 ms | 16.71 | 33.42 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.684 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.686 ms | 9.95 | 9.95 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.434 ms | 110.29 | 1.29 | n/a | ok | 7.2x (90%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.827 ms | 94.96 | 0.74 | n/a | ok | 7.4x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.357 ms | 197.84 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.358 ms | 197.65 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.538 ms | 75.87 | 0.44 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

