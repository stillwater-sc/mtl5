## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 134.9 GFLOP/s |
| ... hardware, NEON | f64 67.5, f32 134.9 GFLOP/s |
| ... this build, scalar | f64 33.9, f32 33.8 GFLOP/s |
| bandwidth | read 22.9, update 24.8, triad 22.8 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 114% | 0 | 28% |
| blas2 | 10 | 62% | 88% | 0 | 27% |
| blas3 | 10 | 8% | 17% | 0 | 28% |
| factor | 12 | 4% | 38% | 0 | 35% |
| eigen | 8 | 4% | 10% | 0 | 29% |
| elementwise | 5 | 6% | 9% | 0 | 28% |
| sparse | 5 | 55% | 92% | 0 | 28% |
| direct | 19 | 0% | 6% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 28% |
| smoother | 3 | 53% | 59% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 32% | 37% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 56% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.553 ms | 3.02 | 24.17 | 97% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.827 ms | 5.93 | 23.74 | 96% | ok | 1.6x (40%) |
| blas1 | axpy | f64 | n=8388608 | 8.337 ms | 2.01 | 24.15 | 97% | ok | 0.9x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.985 ms | 4.21 | 25.26 | 102% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.739 ms | 1.77 | 28.32 | 114% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.632 ms | 6.37 | 25.50 | 103% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.230 ms | 1.16 | 9.28 | 37% | ok | 1.2x (29%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.421 ms | 0.54 | 4.35 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.286 ms | 1.15 | 9.21 | 37% | ok | 1.2x (29%) |
| blas1 | dot_k3 | f64 | n=8388608 | 177.416 ms | 1.04 | 0.76 | 3% | ok | 1.0x (26%) |
| blas2 | gemv | f64 | n=4096 | 6.647 ms | 5.05 | 20.20 | 81% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 3.090 ms | 10.86 | 21.73 | 88% | ok | 1.8x (46%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.757 ms | 4.33 | 17.31 | 70% | ok | 1.3x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.403 ms | 6.21 | 12.43 | 50% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 15.432 ms | 2.17 | 4.35 | 18% | ok | 1.1x (28%) |
| blas2 | trmv | f64 | n=4096 | 4.398 ms | 3.81 | 15.27 | 62% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 4.449 ms | 3.77 | 7.55 | 30% | ok | 1.1x (27%) |
| blas2 | trsv | f64 | n=4096 | 4.456 ms | 3.77 | 15.08 | 61% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 4.357 ms | 3.85 | 7.71 | 31% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 15.498 ms | 2.17 | 17.33 | 70% | ok | 1.0x (25%) |
| blas3 | gemm | f64 | 512x512x512 | 26.604 ms | 10.09 | 0.24 | 15% | ok | 2.0x (49%) |
| blas3 | gemm | f32 | 512x512x512 | 11.483 ms | 23.38 | 0.27 | 17% | ok | 2.2x (54%) |
| blas3 | gemmt | f64 | 512x512x512 | 38.825 ms | 3.46 | 0.16 | 5% | ok | 1.1x (27%) |
| blas3 | symm | f64 | 512x512x512 | 48.170 ms | 5.57 | 0.13 | 8% | ok | 1.1x (28%) |
| blas3 | syrk | f64 | 512x512x512 | 43.782 ms | 3.07 | 0.10 | 5% | ok | 0.9x (23%) |
| blas3 | syr2k | f64 | 512x512x512 | 83.861 ms | 3.20 | 0.08 | 5% | ok | 1.0x (24%) |
| blas3 | trmm | f64 | 512x512x512 | 23.764 ms | 5.65 | 0.22 | 8% | ok | 1.1x (28%) |
| blas3 | trsm | f64 | 512x512x512 | 23.350 ms | 5.75 | 0.22 | 9% | ok | 1.1x (28%) |
| blas3 | trsm | f32 | 512x512x512 | 11.426 ms | 11.75 | 0.23 | 9% | ok | 1.1x (27%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 23.188 ms | 0.23 | 1.83 | 7% | ok | 1.3x (33%) |
| factor | lu | f64 | 512x512 | 17.838 ms | 5.02 | 0.24 | 7% | ok | 1.4x (35%) |
| factor | lu | f32 | 512x512 | 8.895 ms | 10.06 | 0.24 | 7% | ok | 1.5x (37%) |
| factor | cholesky | f64 | 512x512 | 18.616 ms | 2.40 | 0.23 | 4% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.576 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.629 ms | 2.17 | 0.20 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.673 ms | 0.74 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 147.244 ms | 1.22 | 0.03 | 2% | ok | 1.7x (43%) |
| factor | lq | f64 | 512x512 | 36.240 ms | 4.94 | 0.12 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 202.473 ms | 2.21 | 0.02 | 3% | ok | 1.7x (44%) |
| factor | householder | f64 | 512x512 | 0.447 ms | 2.35 | 9.38 | 38% | ok | 3.5x (88%) |
| factor | inv | f64 | 512x512 | 126.331 ms | 2.12 | 0.03 | 3% | ok | 1.2x (30%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.243 ms | 7.31 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.564 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.955 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 528.038 ms | 0.29 | 0.00 | 0% | ok | 1.1x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 60.003 ms | 2.80 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 133.910 ms | 3.13 | 0.01 | 5% | ok | 1.2x (29%) |
| eigen | svd | f64 | 256x256 | 205.360 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 38.270 ms | 6.58 | 0.03 | 10% | ok | 1.8x (44%) |
| eigen | sylvester | f64 | 256x256 | 317.094 ms | 3.17 | 0.00 | 5% | ok | 1.2x (29%) |
| elementwise | exp | f64 | n=8388608 | 75.182 ms | 0.11 | 1.79 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 96.967 ms | 0.09 | 1.38 | 6% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 136.418 ms | 0.06 | 0.98 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 62.694 ms | 0.13 | 2.14 | 9% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 102.073 ms | 0.08 | 1.31 | 5% | ok | 1.2x (29%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.195 ms | 2.19 | 22.78 | 92% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.998 ms | 1.31 | 13.63 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.106 ms | 6.86 | 14.42 | 58% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.606 ms | 0.28 | 0.99 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.721 ms | 0.23 | 1.91 | 8% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.320 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.614 ms | 0.19 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.422 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 92.192 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 124.235 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.920 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 185.159 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.749 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.504 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.631 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.270 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.329 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 155.572 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.266 ms | 0.33 | 1.33 | 5% | ok | 0.8x (20%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.587 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.323 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.200 ms | 0.34 | 1.37 | 6% | ok | 0.8x (20%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.913 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 77.786 ms | 0.29 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.567 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.993 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 106.968 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.687 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.413 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14709.630 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.721 ms | 0.44 | 1.75 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 41.547 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 181.104 ms | 1.70 | 0.00 | 3% | ok | 1.1x (29%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.976 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.560 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1595.252 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.929 ms | 0.12 | 0.48 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 25.818 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2247.524 ms | 0.27 | 0.00 | 0% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.544 ms | 0.12 | 0.49 | 2% | ok | 1.0x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.853 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.448 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.632 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.199 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.356 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.981 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 57.058 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.824 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.390 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.938 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.474 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.782 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.281 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.471 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilut | f64 | 128^2 grid | 27.442 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.873 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ildl | f64 | 128^2 grid | 2.733 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.594 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.005 ms | 1.31 | 14.63 | 59% | ok | 1.1x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.253 ms | 1.16 | 13.01 | 53% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.673 ms | 0.98 | 10.97 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.299 ms | 1.19 | 9.00 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 209.029 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 242.032 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 134.843 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.886 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.224 ms | 2.32 | 9.29 | 37% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.334 ms | 21.76 | 0.34 | 32% | ok | 3.9x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.634 ms | 6.37 | 6.77 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.812 ms | 5.97 | 23.86 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.373 ms | 12.22 | 24.43 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.274 ms | 5.12 | 5.12 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.063 ms | 15.78 | 15.78 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.479 ms | 23.38 | 0.27 | n/a | ok | 2.2x (56%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.679 ms | 40.19 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.136 ms | 52.27 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.137 ms | 52.26 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.738 ms | 27.57 | 0.16 | n/a | ok | 4.0x (99%) |

