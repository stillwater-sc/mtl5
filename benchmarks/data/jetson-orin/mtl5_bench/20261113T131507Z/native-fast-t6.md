## jetson-orin -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.7 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.7 GFLOP/s |
| ... this build, NEON | f64 64.2, f32 128.2 GFLOP/s |
| bandwidth | read 27.3, update 34.8, triad 32.4 GB/s (ceiling 34.8) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 87% | 136% | 0 | 40% |
| blas2 | 10 | 50% | 82% | 0 | 17% |
| blas3 | 10 | 19% | 51% | 0 | 38% |
| factor | 12 | 2% | 22% | 0 | 23% |
| eigen | 8 | 3% | 8% | 0 | 17% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 25% | 93% | 0 | 18% |
| direct | 19 | 0% | 4% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 1% | 4% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 19% | 23% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 34% | 39% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 81% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.565 ms | 3.68 | 29.40 | 85% | ok | 2.5x (42%) |
| blas1 | dot | f32 | n=8388608 | 2.230 ms | 7.52 | 30.09 | 87% | ok | 2.5x (42%) |
| blas1 | axpy | f64 | n=8388608 | 5.038 ms | 3.33 | 39.96 | 115% | ok | 2.4x (40%) |
| blas1 | axpy | f32 | n=8388608 | 2.467 ms | 6.80 | 40.80 | 117% | ok | 2.4x (40%) |
| blas1 | scale | f64 | n=8388608 | 2.838 ms | 2.96 | 47.29 | 136% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.182 ms | 7.69 | 30.75 | 88% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.749 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 190.873 ms | 0.97 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.340 ms | 6.28 | 25.14 | 72% | ok | 2.7x (45%) |
| blas2 | gemv | f32 | n=4096 | 2.346 ms | 14.30 | 28.62 | 82% | ok | 2.7x (45%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.667 ms | 4.38 | 17.51 | 50% | ok | 1.8x (30%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.443 ms | 9.75 | 19.50 | 56% | ok | 1.8x (30%) |
| blas2 | symv | f64 | n=4096 | 21.343 ms | 1.57 | 3.15 | 9% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 9.142 ms | 1.84 | 7.35 | 21% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 4.257 ms | 3.94 | 7.89 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 9.246 ms | 1.81 | 7.27 | 21% | ok | 1.0x (16%) |
| blas2 | trsv | f32 | n=4096 | 4.250 ms | 3.95 | 7.90 | 23% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.403 ms | 2.50 | 20.03 | 58% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 8.182 ms | 32.81 | 0.77 | 51% | ok | 5.1x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 4.092 ms | 65.61 | 0.77 | 51% | ok | 5.0x (83%) |
| blas3 | gemmt | f64 | 512x512x512 | 7.401 ms | 18.14 | 0.85 | 28% | ok | 3.5x (58%) |
| blas3 | symm | f64 | 512x512x512 | 9.194 ms | 29.20 | 0.68 | 45% | ok | 4.6x (77%) |
| blas3 | syrk | f64 | 512x512x512 | 49.054 ms | 2.74 | 0.09 | 4% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 93.353 ms | 2.88 | 0.07 | 4% | ok | 1.0x (16%) |
| blas3 | trmm | f64 | 512x512x512 | 10.861 ms | 12.36 | 0.48 | 19% | ok | 2.3x (38%) |
| blas3 | trsm | f64 | 512x512x512 | 11.602 ms | 11.57 | 0.45 | 18% | ok | 2.2x (37%) |
| blas3 | trsm | f32 | 512x512x512 | 5.642 ms | 23.79 | 0.46 | 18% | ok | 2.2x (37%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.705 ms | 0.55 | 4.38 | 13% | ok | 1.0x (16%) |
| factor | lu | f64 | 512x512 | 24.906 ms | 3.59 | 0.17 | 6% | ok | 1.4x (24%) |
| factor | lu | f32 | 512x512 | 14.418 ms | 6.21 | 0.15 | 5% | ok | 1.4x (23%) |
| factor | cholesky | f64 | 512x512 | 33.889 ms | 1.32 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.092 ms | 1.49 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 34.425 ms | 1.30 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 83.917 ms | 0.53 | 0.05 | 1% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 203.731 ms | 0.88 | 0.02 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 59.739 ms | 3.00 | 0.07 | 5% | ok | 1.6x (26%) |
| factor | hessenberg | f64 | 512x512 | 299.925 ms | 1.49 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.542 ms | 1.94 | 7.74 | 22% | ok | 3.7x (61%) |
| factor | inv | f64 | 512x512 | 181.126 ms | 1.48 | 0.02 | 2% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.555 ms | 4.58 | 0.21 | 7% | ok | 1.3x (22%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.446 ms | 0.36 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 145.085 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 942.397 ms | 0.16 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 98.833 ms | 1.70 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 214.587 ms | 1.95 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 359.105 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 49.596 ms | 5.07 | 0.02 | 8% | ok | 1.5x (24%) |
| eigen | sylvester | f64 | 256x256 | 501.888 ms | 2.01 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 75.460 ms | 0.11 | 1.78 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 107.344 ms | 0.08 | 1.25 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 178.525 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.986 ms | 0.15 | 2.44 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.727 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.841 ms | 3.11 | 32.36 | 93% | ok | 2.7x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.093 ms | 0.85 | 8.80 | 25% | ok | 1.0x (16%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.726 ms | 8.86 | 18.63 | 54% | ok | 3.7x (62%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.611 ms | 0.16 | 0.56 | 2% | ok | 1.1x (18%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.008 ms | 0.18 | 1.51 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 282.218 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 89.435 ms | 0.13 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.293 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 137.821 ms | 0.16 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 194.543 ms | 0.12 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 74.319 ms | 0.30 | 0.01 | 0% | ok | 0.8x (14%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 287.032 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.007 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.095 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.128 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.447 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 8.835 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 270.205 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.142 ms | 0.35 | 1.41 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 18.175 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.531 ms | 0.68 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.087 ms | 0.36 | 1.45 | 4% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.720 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 117.440 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.625 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 14.740 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 167.649 ms | 0.13 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.632 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 31.490 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 25346.402 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.867 ms | 0.34 | 1.37 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 65.027 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 280.349 ms | 1.10 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.180 ms | 0.33 | 1.32 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.261 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1897.301 ms | 0.33 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.891 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 42.494 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 3097.120 ms | 0.20 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.976 ms | 0.09 | 0.36 | 1% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.192 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.449 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.681 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.433 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.588 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.555 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.318 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.919 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.759 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.527 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.453 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.410 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.993 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 45.202 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.474 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.263 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.397 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 7.514 ms | 0.35 | 3.90 | 11% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.744 ms | 0.70 | 7.83 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.518 ms | 0.58 | 6.49 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.124 ms | 0.91 | 6.90 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 358.587 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 419.394 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 194.431 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 6.261 ms | 2.68 | 10.72 | 31% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.238 ms | 21.93 | 0.34 | 34% | ok | 5.2x (87%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.322 ms | 12.69 | 13.49 | 39% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.601 ms | 3.00 | 11.98 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.680 ms | 6.26 | 12.52 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.198 ms | 14.01 | 14.01 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.349 ms | 7.14 | 7.14 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 5.141 ms | 52.22 | 0.61 | n/a | ok | 5.1x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.320 ms | 50.45 | 0.39 | n/a | ok | 5.2x (87%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 6.467 ms | 41.51 | 0.24 | n/a | ok | 5.2x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.767 ms | 151.89 | 0.89 | n/a | ok | 4.9x (81%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.533 ms | 59.22 | 0.35 | n/a | ok | 5.2x (86%) |

