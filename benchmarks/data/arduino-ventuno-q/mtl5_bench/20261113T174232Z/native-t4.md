## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 67.8 GFLOP/s, f32 134.9 GFLOP/s |
| ... hardware, NEON | f64 67.8, f32 134.9 GFLOP/s |
| ... this build, scalar | f64 33.8, f32 33.8 GFLOP/s |
| bandwidth | read 23.0, update 24.7, triad 22.7 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 106% | 0 | 28% |
| blas2 | 10 | 63% | 88% | 0 | 27% |
| blas3 | 10 | 8% | 17% | 0 | 28% |
| factor | 12 | 4% | 39% | 0 | 36% |
| eigen | 8 | 4% | 10% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 56% | 92% | 0 | 28% |
| direct | 19 | 0% | 6% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 27% |
| direct3d | 12 | 2% | 7% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 28% |
| smoother | 3 | 52% | 59% | 0 | 27% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 31% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 51% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.556 ms | 3.02 | 24.16 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.799 ms | 5.99 | 23.97 | 97% | ok | 1.5x (39%) |
| blas1 | axpy | f64 | n=8388608 | 8.233 ms | 2.04 | 24.45 | 99% | ok | 0.9x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.024 ms | 4.17 | 25.01 | 101% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 5.113 ms | 1.64 | 26.25 | 106% | ok | 1.0x (26%) |
| blas1 | two_norm | f64 | n=8388608 | 2.617 ms | 6.41 | 25.65 | 104% | ok | 1.2x (31%) |
| blas1 | one_norm | f64 | n=8388608 | 7.331 ms | 1.14 | 9.15 | 37% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.560 ms | 0.54 | 4.31 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.410 ms | 1.13 | 9.06 | 37% | ok | 1.2x (29%) |
| blas1 | dot_k3 | f64 | n=8388608 | 168.062 ms | 1.10 | 0.80 | 3% | ok | 1.1x (27%) |
| blas2 | gemv | f64 | n=4096 | 6.676 ms | 5.03 | 20.12 | 81% | ok | 1.2x (29%) |
| blas2 | gemv | f32 | n=4096 | 3.081 ms | 10.89 | 21.80 | 88% | ok | 1.9x (46%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.769 ms | 4.32 | 17.28 | 70% | ok | 1.3x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.649 ms | 5.94 | 11.89 | 48% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 15.010 ms | 2.24 | 4.48 | 18% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.374 ms | 3.84 | 15.36 | 62% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 4.369 ms | 3.84 | 7.69 | 31% | ok | 1.1x (28%) |
| blas2 | trsv | f64 | n=4096 | 4.339 ms | 3.87 | 15.48 | 63% | ok | 1.0x (26%) |
| blas2 | trsv | f32 | n=4096 | 4.353 ms | 3.85 | 7.72 | 31% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 15.039 ms | 2.23 | 17.85 | 72% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 27.219 ms | 9.86 | 0.23 | 15% | ok | 1.9x (48%) |
| blas3 | gemm | f32 | 512x512x512 | 11.550 ms | 23.24 | 0.27 | 17% | ok | 2.1x (53%) |
| blas3 | gemmt | f64 | 512x512x512 | 40.583 ms | 3.31 | 0.16 | 5% | ok | 1.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 49.009 ms | 5.48 | 0.13 | 8% | ok | 1.1x (29%) |
| blas3 | syrk | f64 | 512x512x512 | 40.384 ms | 3.32 | 0.10 | 5% | ok | 1.0x (25%) |
| blas3 | syr2k | f64 | 512x512x512 | 82.731 ms | 3.24 | 0.08 | 5% | ok | 1.0x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 24.171 ms | 5.55 | 0.22 | 8% | ok | 1.1x (28%) |
| blas3 | trsm | f64 | 512x512x512 | 23.894 ms | 5.62 | 0.22 | 8% | ok | 1.1x (28%) |
| blas3 | trsm | f32 | 512x512x512 | 11.159 ms | 12.03 | 0.23 | 9% | ok | 1.2x (30%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 25.451 ms | 0.21 | 1.67 | 7% | ok | 1.1x (28%) |
| factor | lu | f64 | 512x512 | 17.312 ms | 5.17 | 0.24 | 8% | ok | 1.4x (36%) |
| factor | lu | f32 | 512x512 | 8.960 ms | 9.99 | 0.23 | 7% | ok | 1.4x (36%) |
| factor | cholesky | f64 | 512x512 | 18.582 ms | 2.41 | 0.23 | 4% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.579 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.274 ms | 2.21 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 61.075 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 148.124 ms | 1.21 | 0.03 | 2% | ok | 1.7x (43%) |
| factor | lq | f64 | 512x512 | 36.670 ms | 4.88 | 0.11 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 202.009 ms | 2.21 | 0.02 | 3% | ok | 1.8x (44%) |
| factor | householder | f64 | 512x512 | 0.439 ms | 2.39 | 9.54 | 39% | ok | 3.6x (90%) |
| factor | inv | f64 | 512x512 | 127.813 ms | 2.10 | 0.03 | 3% | ok | 1.2x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.349 ms | 7.25 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.531 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.874 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 527.507 ms | 0.29 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.093 ms | 2.79 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.219 ms | 3.12 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.403 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 38.219 ms | 6.58 | 0.03 | 10% | ok | 1.8x (44%) |
| eigen | sylvester | f64 | 256x256 | 316.501 ms | 3.18 | 0.00 | 5% | ok | 1.2x (30%) |
| elementwise | exp | f64 | n=8388608 | 81.559 ms | 0.10 | 1.65 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 99.961 ms | 0.08 | 1.34 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 135.132 ms | 0.06 | 0.99 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 66.418 ms | 0.13 | 2.02 | 8% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 107.102 ms | 0.08 | 1.25 | 5% | ok | 1.0x (26%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.199 ms | 2.18 | 22.71 | 92% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.954 ms | 1.34 | 13.94 | 56% | ok | 1.1x (28%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.011 ms | 6.97 | 14.65 | 59% | ok | 2.0x (50%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.574 ms | 0.28 | 0.99 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.601 ms | 0.23 | 1.94 | 8% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.328 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.512 ms | 0.19 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.380 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 92.220 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 124.112 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.899 ms | 0.58 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 185.001 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.755 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.575 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.637 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.749 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.361 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 155.660 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.000 ms | 0.38 | 1.51 | 6% | ok | 0.9x (23%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.653 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.317 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.992 ms | 0.38 | 1.52 | 6% | ok | 0.9x (22%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.998 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 78.203 ms | 0.29 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.554 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.069 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 106.565 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.555 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.485 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14695.201 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.631 ms | 0.44 | 1.77 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 41.857 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 181.443 ms | 1.70 | 0.00 | 3% | ok | 1.1x (29%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.953 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.912 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1582.879 ms | 0.39 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.613 ms | 0.12 | 0.49 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.292 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2286.196 ms | 0.27 | 0.00 | 0% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 26.997 ms | 0.12 | 0.50 | 2% | ok | 1.0x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.391 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 95.684 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.373 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.054 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.195 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.649 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.661 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.614 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.026 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.557 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.336 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.489 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.177 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.104 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.479 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilut | f64 | 128^2 grid | 27.470 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.863 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ildl | f64 | 128^2 grid | 2.746 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.577 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.002 ms | 1.31 | 14.65 | 59% | ok | 1.0x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.288 ms | 1.14 | 12.82 | 52% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.658 ms | 0.98 | 11.03 | 45% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.640 ms | 1.18 | 8.90 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 208.548 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 241.245 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 135.885 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.941 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.210 ms | 2.33 | 9.31 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.845 ms | 20.90 | 0.33 | 31% | ok | 3.8x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.657 ms | 6.31 | 6.71 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.812 ms | 5.97 | 23.87 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.357 ms | 12.36 | 24.72 | n/a | ok | 1.0x (26%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.268 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.037 ms | 16.19 | 16.19 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.831 ms | 22.69 | 0.27 | n/a | ok | 2.0x (51%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.677 ms | 40.20 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.139 ms | 52.24 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.136 ms | 52.26 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.765 ms | 27.49 | 0.16 | n/a | ok | 3.9x (99%) |

