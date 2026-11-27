## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 134.9 GFLOP/s |
| ... hardware, NEON | f64 67.5, f32 134.9 GFLOP/s |
| ... this build, scalar | f64 33.8, f32 33.8 GFLOP/s |
| bandwidth | read 23.0, update 24.7, triad 22.4 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 115% | 0 | 28% |
| blas2 | 10 | 62% | 88% | 0 | 27% |
| blas3 | 10 | 9% | 17% | 0 | 28% |
| factor | 12 | 6% | 37% | 0 | 35% |
| eigen | 8 | 4% | 10% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 57% | 92% | 0 | 28% |
| direct | 19 | 1% | 6% | 0 | 27% |
| ordering | 5 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 60% | 0 | 27% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 32% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 53% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.544 ms | 3.03 | 24.21 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.807 ms | 5.98 | 23.91 | 97% | ok | 1.6x (41%) |
| blas1 | axpy | f64 | n=8388608 | 8.188 ms | 2.05 | 24.59 | 100% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.948 ms | 4.25 | 25.50 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.742 ms | 1.77 | 28.30 | 115% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.603 ms | 6.45 | 25.78 | 104% | ok | 1.2x (31%) |
| blas1 | one_norm | f64 | n=8388608 | 7.322 ms | 1.15 | 9.17 | 37% | ok | 1.2x (29%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.606 ms | 0.54 | 4.30 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.779 ms | 1.08 | 8.63 | 35% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 182.569 ms | 1.01 | 0.74 | 3% | ok | 1.1x (27%) |
| blas2 | gemv | f64 | n=4096 | 6.707 ms | 5.00 | 20.02 | 81% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 3.089 ms | 10.86 | 21.73 | 88% | ok | 1.9x (47%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.940 ms | 4.23 | 16.91 | 68% | ok | 1.2x (31%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.478 ms | 6.13 | 12.26 | 50% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 11.398 ms | 2.94 | 5.89 | 24% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.380 ms | 3.83 | 15.34 | 62% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 3.113 ms | 5.39 | 10.79 | 44% | ok | 1.1x (27%) |
| blas2 | trsv | f64 | n=4096 | 4.416 ms | 3.80 | 15.21 | 62% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.871 ms | 5.84 | 11.70 | 47% | ok | 1.1x (26%) |
| blas2 | ger | f64 | n=4096 | 15.456 ms | 2.17 | 17.37 | 70% | ok | 1.0x (25%) |
| blas3 | gemm | f64 | 512x512x512 | 27.021 ms | 9.93 | 0.23 | 15% | ok | 1.9x (48%) |
| blas3 | gemm | f32 | 512x512x512 | 11.703 ms | 22.94 | 0.27 | 17% | ok | 2.1x (51%) |
| blas3 | gemmt | f64 | 512x512x512 | 37.905 ms | 3.54 | 0.17 | 5% | ok | 1.1x (27%) |
| blas3 | symm | f64 | 512x512x512 | 47.667 ms | 5.63 | 0.13 | 8% | ok | 1.1x (27%) |
| blas3 | syrk | f64 | 512x512x512 | 39.794 ms | 3.37 | 0.11 | 5% | ok | 1.0x (25%) |
| blas3 | syr2k | f64 | 512x512x512 | 82.696 ms | 3.25 | 0.08 | 5% | ok | 1.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 22.868 ms | 5.87 | 0.23 | 9% | ok | 1.1x (28%) |
| blas3 | trsm | f64 | 512x512x512 | 23.135 ms | 5.80 | 0.23 | 9% | ok | 1.1x (28%) |
| blas3 | trsm | f32 | 512x512x512 | 10.656 ms | 12.60 | 0.25 | 9% | ok | 1.1x (28%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.477 ms | 0.19 | 1.49 | 6% | ok | 1.1x (27%) |
| factor | lu | f64 | 512x512 | 17.251 ms | 5.19 | 0.24 | 8% | ok | 1.4x (35%) |
| factor | lu | f32 | 512x512 | 8.843 ms | 10.12 | 0.24 | 8% | ok | 1.5x (37%) |
| factor | cholesky | f64 | 512x512 | 20.002 ms | 2.24 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.588 ms | 2.54 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 19.070 ms | 2.35 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 10.213 ms | 4.38 | 0.41 | 6% | ok | 1.1x (27%) |
| factor | qr | f64 | 512x512 | 146.283 ms | 1.22 | 0.03 | 2% | ok | 1.7x (43%) |
| factor | lq | f64 | 512x512 | 36.398 ms | 4.92 | 0.12 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 199.401 ms | 2.24 | 0.02 | 3% | ok | 1.7x (44%) |
| factor | householder | f64 | 512x512 | 0.463 ms | 2.27 | 9.07 | 37% | ok | 3.4x (84%) |
| factor | inv | f64 | 512x512 | 125.644 ms | 2.14 | 0.03 | 3% | ok | 1.2x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.039 ms | 7.43 | 0.35 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.928 ms | 0.62 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.102 ms | 1.73 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 526.518 ms | 0.29 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.503 ms | 2.77 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.135 ms | 3.13 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.466 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 37.876 ms | 6.64 | 0.03 | 10% | ok | 1.8x (44%) |
| eigen | sylvester | f64 | 256x256 | 319.782 ms | 3.15 | 0.00 | 5% | ok | 1.2x (29%) |
| elementwise | exp | f64 | n=8388608 | 81.588 ms | 0.10 | 1.65 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 100.193 ms | 0.08 | 1.34 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 137.115 ms | 0.06 | 0.98 | 4% | ok | 1.1x (27%) |
| elementwise | sqrt | f64 | n=8388608 | 65.329 ms | 0.13 | 2.05 | 8% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 107.460 ms | 0.08 | 1.25 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.201 ms | 2.18 | 22.67 | 92% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.931 ms | 1.36 | 14.10 | 57% | ok | 1.1x (28%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.282 ms | 6.67 | 14.02 | 57% | ok | 1.9x (48%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.821 ms | 0.28 | 0.97 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.234 ms | 0.22 | 1.80 | 7% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 33.838 ms | 0.33 | 0.02 | 0% | ok | 1.1x (27%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.122 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.182 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 94.465 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 110.589 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.125 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 66.351 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.694 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.200 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 112.539 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.525 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.851 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.044 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 26.280 ms | 0.43 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.230 ms | 0.34 | 1.36 | 5% | ok | 0.8x (20%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.143 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.711 ms | 1.04 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.172 ms | 0.35 | 1.39 | 6% | ok | 0.8x (21%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.130 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 81.209 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.729 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.873 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 93.322 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.689 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 10.763 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 610.937 ms | 0.51 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.926 ms | 0.43 | 1.70 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 24.383 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 166.921 ms | 1.85 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.954 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.853 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1560.277 ms | 0.40 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.877 ms | 0.12 | 0.47 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 14.988 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1107.195 ms | 0.56 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.607 ms | 0.12 | 0.47 | 2% | ok | 1.1x (27%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.454 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.763 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.207 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.274 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.519 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.521 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.531 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.129 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.389 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.466 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.601 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.768 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.665 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.621 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.569 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.876 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.778 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.602 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.978 ms | 1.32 | 14.83 | 60% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.275 ms | 1.15 | 12.89 | 52% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.491 ms | 0.58 | 6.53 | 26% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.100 ms | 1.20 | 9.05 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 211.039 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 212.404 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 132.784 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.900 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.208 ms | 2.33 | 9.31 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.390 ms | 21.67 | 0.34 | 32% | ok | 3.9x (98%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.669 ms | 6.29 | 6.68 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.801 ms | 5.99 | 23.96 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.374 ms | 12.21 | 24.42 | n/a | ok | 1.0x (26%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.262 ms | 5.14 | 5.14 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.054 ms | 15.92 | 15.92 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.655 ms | 23.03 | 0.27 | n/a | ok | 2.1x (53%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.680 ms | 40.18 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.142 ms | 52.20 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.143 ms | 52.19 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.739 ms | 27.56 | 0.16 | n/a | ok | 4.0x (99%) |

