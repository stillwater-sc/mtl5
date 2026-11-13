## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 67.8 GFLOP/s, f32 135.0 GFLOP/s |
| ... hardware, NEON | f64 67.7, f32 135.0 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.8, f32 134.9 GFLOP/s |
| bandwidth | read 23.3, update 25.1, triad 23.0 GB/s (ceiling 25.1) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 96% | 104% | 0 | 27% |
| blas2 | 10 | 55% | 88% | 0 | 26% |
| blas3 | 10 | 27% | 56% | 0 | 60% |
| factor | 12 | 3% | 37% | 0 | 36% |
| eigen | 8 | 4% | 13% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 27% |
| sparse | 5 | 56% | 90% | 0 | 28% |
| direct | 19 | 0% | 6% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 44% | 59% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 38% | 73% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 98% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.554 ms | 3.02 | 24.16 | 96% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.792 ms | 6.01 | 24.04 | 96% | ok | 1.0x (26%) |
| blas1 | axpy | f64 | n=8388608 | 8.038 ms | 2.09 | 25.05 | 100% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.990 ms | 4.20 | 25.23 | 100% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 5.151 ms | 1.63 | 26.06 | 104% | ok | 1.0x (26%) |
| blas1 | two_norm | f64 | n=8388608 | 2.627 ms | 6.39 | 25.55 | 102% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.401 ms | 1.13 | 9.07 | 36% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.974 ms | 0.53 | 4.20 | 17% | ok | 1.1x (27%) |
| blas1 | sum | f64 | n=8388608 | 7.465 ms | 1.12 | 8.99 | 36% | ok | 1.1x (28%) |
| blas1 | dot_k3 | f64 | n=8388608 | 177.963 ms | 1.04 | 0.75 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.738 ms | 4.98 | 19.93 | 79% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 3.039 ms | 11.04 | 22.09 | 88% | ok | 1.3x (33%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.681 ms | 4.37 | 17.48 | 70% | ok | 1.0x (25%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.932 ms | 6.80 | 13.61 | 54% | ok | 0.8x (20%) |
| blas2 | symv | f64 | n=4096 | 15.838 ms | 2.12 | 4.24 | 17% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.840 ms | 3.47 | 13.88 | 55% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 2.579 ms | 6.51 | 13.02 | 52% | ok | 1.0x (25%) |
| blas2 | trsv | f64 | n=4096 | 4.845 ms | 3.46 | 13.86 | 55% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.551 ms | 6.58 | 13.17 | 52% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 15.026 ms | 2.23 | 17.87 | 71% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.429 ms | 36.14 | 0.85 | 53% | ok | 3.8x (95%) |
| blas3 | gemm | f32 | 512x512x512 | 3.575 ms | 75.09 | 0.88 | 56% | ok | 3.9x (97%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.869 ms | 22.87 | 1.07 | 34% | ok | 3.1x (76%) |
| blas3 | symm | f64 | 512x512x512 | 8.116 ms | 33.08 | 0.78 | 49% | ok | 3.6x (90%) |
| blas3 | syrk | f64 | 512x512x512 | 36.927 ms | 3.63 | 0.11 | 5% | ok | 1.1x (28%) |
| blas3 | syr2k | f64 | 512x512x512 | 77.699 ms | 3.45 | 0.08 | 5% | ok | 1.1x (27%) |
| blas3 | trmm | f64 | 512x512x512 | 7.731 ms | 17.36 | 0.68 | 26% | ok | 2.3x (58%) |
| blas3 | trsm | f64 | 512x512x512 | 7.934 ms | 16.92 | 0.66 | 25% | ok | 2.3x (57%) |
| blas3 | trsm | f32 | 512x512x512 | 3.631 ms | 36.97 | 0.72 | 27% | ok | 2.4x (60%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.301 ms | 0.18 | 1.45 | 6% | ok | 1.0x (26%) |
| factor | lu | f64 | 512x512 | 17.247 ms | 5.19 | 0.24 | 8% | ok | 1.4x (36%) |
| factor | lu | f32 | 512x512 | 8.933 ms | 10.02 | 0.23 | 7% | ok | 1.6x (40%) |
| factor | cholesky | f64 | 512x512 | 19.926 ms | 2.25 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.611 ms | 2.54 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.281 ms | 2.21 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.877 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 147.322 ms | 1.21 | 0.03 | 2% | ok | 1.7x (43%) |
| factor | lq | f64 | 512x512 | 36.695 ms | 4.88 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 199.764 ms | 2.24 | 0.02 | 3% | ok | 1.8x (44%) |
| factor | householder | f64 | 512x512 | 0.450 ms | 2.33 | 9.32 | 37% | ok | 3.5x (87%) |
| factor | inv | f64 | 512x512 | 126.765 ms | 2.12 | 0.03 | 3% | ok | 1.2x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.187 ms | 7.34 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.778 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.901 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 526.387 ms | 0.29 | 0.00 | 0% | ok | 1.1x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 60.205 ms | 2.79 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.676 ms | 3.11 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.826 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 29.163 ms | 8.63 | 0.04 | 13% | ok | 1.6x (41%) |
| eigen | sylvester | f64 | 256x256 | 312.718 ms | 3.22 | 0.01 | 5% | ok | 1.1x (29%) |
| elementwise | exp | f64 | n=8388608 | 80.963 ms | 0.10 | 1.66 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 99.473 ms | 0.08 | 1.35 | 5% | ok | 1.1x (27%) |
| elementwise | sin | f64 | n=8388608 | 135.627 ms | 0.06 | 0.99 | 4% | ok | 1.1x (27%) |
| elementwise | sqrt | f64 | n=8388608 | 65.678 ms | 0.13 | 2.04 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 106.526 ms | 0.08 | 1.26 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.200 ms | 2.18 | 22.70 | 90% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.920 ms | 1.36 | 14.18 | 56% | ok | 1.1x (28%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.090 ms | 6.88 | 14.46 | 58% | ok | 2.0x (50%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.100 ms | 0.27 | 0.95 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.640 ms | 0.23 | 1.93 | 8% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 168.079 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 55.332 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.449 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 94.008 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 129.625 ms | 0.17 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.952 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 189.308 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.698 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.354 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.629 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.602 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.330 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 160.509 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.999 ms | 0.38 | 1.51 | 6% | ok | 0.9x (24%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.683 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.358 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.947 ms | 0.39 | 1.55 | 6% | ok | 0.9x (22%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.977 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.556 ms | 0.28 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.594 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.020 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 108.201 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.533 ms | 0.14 | 0.55 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.286 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 15064.515 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.676 ms | 0.44 | 1.76 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 42.308 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 185.502 ms | 1.66 | 0.00 | 2% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.971 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.557 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1589.092 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.003 ms | 0.12 | 0.46 | 2% | ok | 1.0x (25%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.319 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2346.841 ms | 0.26 | 0.00 | 0% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.998 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.813 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 95.793 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.077 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 25.980 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.281 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.604 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.069 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.335 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.248 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.611 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.156 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.336 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.897 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.559 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.436 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.874 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.742 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.600 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.975 ms | 1.33 | 14.85 | 59% | ok | 1.1x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.067 ms | 0.64 | 7.21 | 29% | ok | 1.1x (29%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.681 ms | 0.98 | 10.94 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 31.839 ms | 1.21 | 9.13 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 208.066 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 240.396 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 134.712 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.946 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.676 ms | 4.56 | 18.26 | 73% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 10.528 ms | 25.50 | 0.40 | 38% | ok | 3.9x (98%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.368 ms | 7.08 | 7.53 | 30% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.786 ms | 6.02 | 24.09 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.374 ms | 12.21 | 24.41 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok | 1.1x (27%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 3.810 ms | 4.40 | 4.40 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.482 ms | 59.89 | 0.70 | n/a | ok | 3.9x (98%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.251 ms | 63.14 | 0.49 | n/a | ok | 3.9x (98%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.194 ms | 64.01 | 0.38 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.959 ms | 45.04 | 0.26 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 12.721 ms | 21.10 | 0.12 | n/a | ok | 4.0x (100%) |

