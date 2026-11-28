## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 67.6 GFLOP/s, f32 135.0 GFLOP/s |
| ... hardware, NEON | f64 67.6, f32 135.0 GFLOP/s |
| ... this build, scalar | f64 33.7, f32 33.7 GFLOP/s |
| bandwidth | read 23.1, update 24.6, triad 22.7 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 104% | 0 | 26% |
| blas2 | 10 | 63% | 80% | 0 | 27% |
| blas3 | 10 | 8% | 17% | 0 | 28% |
| factor | 12 | 6% | 38% | 0 | 35% |
| eigen | 8 | 4% | 10% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 56% | 92% | 0 | 27% |
| direct | 19 | 1% | 6% | 0 | 27% |
| ordering | 5 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 62% | 0 | 28% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 31% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 51% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.570 ms | 3.01 | 24.10 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.793 ms | 6.01 | 24.03 | 98% | ok | 1.5x (39%) |
| blas1 | axpy | f64 | n=8388608 | 8.189 ms | 2.05 | 24.58 | 100% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.001 ms | 4.19 | 25.16 | 102% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 5.426 ms | 1.55 | 24.74 | 100% | ok | 1.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 2.624 ms | 6.39 | 25.57 | 104% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.316 ms | 1.15 | 9.17 | 37% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.443 ms | 0.54 | 4.35 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.900 ms | 1.06 | 8.50 | 34% | ok | 1.0x (26%) |
| blas1 | dot_k3 | f64 | n=8388608 | 179.235 ms | 1.03 | 0.75 | 3% | ok | 1.0x (26%) |
| blas2 | gemv | f64 | n=4096 | 7.008 ms | 4.79 | 19.16 | 78% | ok | 1.1x (28%) |
| blas2 | gemv | f32 | n=4096 | 3.407 ms | 9.85 | 19.71 | 80% | ok | 1.7x (42%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.868 ms | 4.26 | 17.07 | 69% | ok | 1.3x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.757 ms | 5.83 | 11.66 | 47% | ok | 0.8x (20%) |
| blas2 | symv | f64 | n=4096 | 11.007 ms | 3.05 | 6.10 | 25% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.347 ms | 3.86 | 15.45 | 63% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 3.038 ms | 5.52 | 11.06 | 45% | ok | 1.1x (27%) |
| blas2 | trsv | f64 | n=4096 | 4.438 ms | 3.78 | 15.14 | 61% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.929 ms | 5.73 | 11.47 | 47% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.816 ms | 2.26 | 18.12 | 74% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 26.844 ms | 10.00 | 0.23 | 15% | ok | 2.0x (49%) |
| blas3 | gemm | f32 | 512x512x512 | 11.584 ms | 23.17 | 0.27 | 17% | ok | 2.1x (52%) |
| blas3 | gemmt | f64 | 512x512x512 | 38.165 ms | 3.52 | 0.16 | 5% | ok | 1.0x (26%) |
| blas3 | symm | f64 | 512x512x512 | 48.034 ms | 5.59 | 0.13 | 8% | ok | 1.2x (29%) |
| blas3 | syrk | f64 | 512x512x512 | 37.120 ms | 3.62 | 0.11 | 5% | ok | 1.1x (28%) |
| blas3 | syr2k | f64 | 512x512x512 | 80.018 ms | 3.35 | 0.08 | 5% | ok | 1.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 23.600 ms | 5.69 | 0.22 | 8% | ok | 1.1x (27%) |
| blas3 | trsm | f64 | 512x512x512 | 24.701 ms | 5.43 | 0.21 | 8% | ok | 1.1x (27%) |
| blas3 | trsm | f32 | 512x512x512 | 10.756 ms | 12.48 | 0.24 | 9% | ok | 1.1x (28%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.315 ms | 0.18 | 1.45 | 6% | ok | 0.9x (21%) |
| factor | lu | f64 | 512x512 | 17.438 ms | 5.13 | 0.24 | 8% | ok | 1.4x (35%) |
| factor | lu | f32 | 512x512 | 9.057 ms | 9.88 | 0.23 | 7% | ok | 1.4x (36%) |
| factor | cholesky | f64 | 512x512 | 18.565 ms | 2.41 | 0.23 | 4% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.621 ms | 2.54 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.319 ms | 2.20 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 10.254 ms | 4.36 | 0.41 | 6% | ok | 1.1x (27%) |
| factor | qr | f64 | 512x512 | 149.322 ms | 1.20 | 0.03 | 2% | ok | 1.7x (42%) |
| factor | lq | f64 | 512x512 | 36.533 ms | 4.90 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 201.343 ms | 2.22 | 0.02 | 3% | ok | 1.7x (43%) |
| factor | householder | f64 | 512x512 | 0.445 ms | 2.36 | 9.42 | 38% | ok | 3.5x (87%) |
| factor | inv | f64 | 512x512 | 129.347 ms | 2.08 | 0.03 | 3% | ok | 1.1x (28%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.349 ms | 7.25 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.842 ms | 0.62 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.157 ms | 1.73 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 532.789 ms | 0.28 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.400 ms | 2.78 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.226 ms | 3.12 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.387 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 38.012 ms | 6.62 | 0.03 | 10% | ok | 1.8x (44%) |
| eigen | sylvester | f64 | 256x256 | 317.491 ms | 3.17 | 0.00 | 5% | ok | 1.2x (29%) |
| elementwise | exp | f64 | n=8388608 | 80.557 ms | 0.10 | 1.67 | 7% | ok | 1.1x (28%) |
| elementwise | log | f64 | n=8388608 | 100.216 ms | 0.08 | 1.34 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 134.889 ms | 0.06 | 1.00 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 65.869 ms | 0.13 | 2.04 | 8% | ok | 1.0x (26%) |
| elementwise | erf | f64 | n=8388608 | 106.680 ms | 0.08 | 1.26 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.207 ms | 2.17 | 22.55 | 92% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.973 ms | 1.33 | 13.80 | 56% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.201 ms | 6.75 | 14.20 | 58% | ok | 2.0x (50%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.735 ms | 0.28 | 0.98 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.309 ms | 0.22 | 1.79 | 7% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 34.171 ms | 0.33 | 0.02 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 54.081 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.278 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 94.582 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 110.984 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.135 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 66.742 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.824 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.418 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 111.425 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.517 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.067 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.019 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 26.840 ms | 0.42 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.925 ms | 0.39 | 1.57 | 6% | ok | 1.0x (24%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.077 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.659 ms | 1.05 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.953 ms | 0.39 | 1.55 | 6% | ok | 0.9x (23%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.991 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 80.134 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.675 ms | 0.13 | 0.53 | 2% | ok | 1.1x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.897 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 93.920 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.708 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 10.626 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 615.140 ms | 0.50 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.979 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 24.813 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 161.732 ms | 1.91 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.925 ms | 0.43 | 1.70 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.596 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1612.240 ms | 0.38 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.523 ms | 0.11 | 0.46 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 14.851 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1143.888 ms | 0.54 | 0.00 | 1% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 30.327 ms | 0.11 | 0.44 | 2% | ok | 1.0x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.380 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 108.438 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.642 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.405 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.495 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.997 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.864 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.446 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.179 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.583 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.677 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.845 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.742 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.104 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.565 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.147 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.883 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.776 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.075 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.592 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.934 ms | 1.35 | 15.16 | 62% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.282 ms | 1.15 | 12.85 | 52% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.692 ms | 0.97 | 10.90 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.293 ms | 1.19 | 9.00 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 212.407 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 211.906 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 134.729 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 12.134 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| mixed | dot | f32>f64 | n=8388608 | 7.260 ms | 2.31 | 9.24 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.814 ms | 20.95 | 0.33 | 31% | ok | 3.8x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.635 ms | 6.37 | 6.77 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.832 ms | 5.92 | 23.69 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.400 ms | 11.98 | 23.97 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.272 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.041 ms | 16.12 | 16.12 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.814 ms | 22.72 | 0.27 | n/a | ok | 2.0x (51%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.679 ms | 40.19 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.139 ms | 52.23 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.139 ms | 52.23 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.741 ms | 27.56 | 0.16 | n/a | ok | 4.0x (99%) |

