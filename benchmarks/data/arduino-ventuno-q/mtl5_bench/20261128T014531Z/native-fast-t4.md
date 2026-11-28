## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 67.6 GFLOP/s, f32 134.9 GFLOP/s |
| ... hardware, NEON | f64 67.6, f32 134.9 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.5, f32 134.9 GFLOP/s |
| bandwidth | read 22.8, update 24.8, triad 22.8 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 109% | 0 | 27% |
| blas2 | 10 | 57% | 88% | 0 | 26% |
| blas3 | 10 | 37% | 55% | 0 | 77% |
| factor | 12 | 6% | 35% | 0 | 35% |
| eigen | 8 | 4% | 13% | 0 | 28% |
| elementwise | 5 | 5% | 9% | 0 | 27% |
| sparse | 5 | 56% | 91% | 0 | 27% |
| direct | 19 | 1% | 6% | 0 | 27% |
| ordering | 5 | n/a | n/a | 0 | 26% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 51% | 62% | 0 | 27% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 73% | 98% | 0 | 27% |
| integer | 9 | n/a | n/a | 0 | 96% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.569 ms | 3.01 | 24.10 | 97% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.802 ms | 5.99 | 23.95 | 96% | ok | 1.0x (26%) |
| blas1 | axpy | f64 | n=8388608 | 8.181 ms | 2.05 | 24.61 | 99% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.941 ms | 4.26 | 25.54 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.956 ms | 1.69 | 27.08 | 109% | ok | 1.1x (27%) |
| blas1 | two_norm | f64 | n=8388608 | 2.636 ms | 6.36 | 25.46 | 102% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.558 ms | 1.11 | 8.88 | 36% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.719 ms | 0.53 | 4.27 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.897 ms | 1.06 | 8.50 | 34% | ok | 1.1x (26%) |
| blas1 | dot_k3 | f64 | n=8388608 | 182.617 ms | 1.01 | 0.73 | 3% | ok | 1.1x (27%) |
| blas2 | gemv | f64 | n=4096 | 6.603 ms | 5.08 | 20.34 | 82% | ok | 1.3x (32%) |
| blas2 | gemv | f32 | n=4096 | 3.058 ms | 10.97 | 21.95 | 88% | ok | 1.3x (33%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.653 ms | 4.38 | 17.55 | 71% | ok | 1.0x (25%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.792 ms | 7.00 | 14.01 | 56% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 11.017 ms | 3.05 | 6.10 | 25% | ok | 1.2x (29%) |
| blas2 | trmv | f64 | n=4096 | 4.708 ms | 3.56 | 14.27 | 57% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 2.764 ms | 6.07 | 12.15 | 49% | ok | 1.0x (25%) |
| blas2 | trsv | f64 | n=4096 | 4.855 ms | 3.46 | 13.84 | 56% | ok | 1.0x (26%) |
| blas2 | trsv | f32 | n=4096 | 2.640 ms | 6.35 | 12.72 | 51% | ok | 1.1x (26%) |
| blas2 | ger | f64 | n=4096 | 14.516 ms | 2.31 | 18.50 | 74% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.395 ms | 36.30 | 0.85 | 54% | ok | 3.8x (95%) |
| blas3 | gemm | f32 | 512x512x512 | 3.585 ms | 74.88 | 0.88 | 55% | ok | 3.8x (96%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.918 ms | 22.68 | 1.06 | 34% | ok | 3.0x (75%) |
| blas3 | symm | f64 | 512x512x512 | 8.170 ms | 32.86 | 0.77 | 49% | ok | 3.5x (89%) |
| blas3 | syrk | f64 | 512x512x512 | 5.317 ms | 25.24 | 0.79 | 37% | ok | 3.1x (77%) |
| blas3 | syr2k | f64 | 512x512x512 | 10.502 ms | 25.56 | 0.60 | 38% | ok | 3.1x (78%) |
| blas3 | trmm | f64 | 512x512x512 | 7.609 ms | 17.64 | 0.69 | 26% | ok | 2.3x (59%) |
| blas3 | trsm | f64 | 512x512x512 | 7.841 ms | 17.12 | 0.67 | 25% | ok | 2.3x (57%) |
| blas3 | trsm | f32 | 512x512x512 | 3.585 ms | 37.44 | 0.73 | 28% | ok | 2.4x (59%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 22.576 ms | 0.24 | 1.88 | 8% | ok | 1.2x (29%) |
| factor | lu | f64 | 512x512 | 17.931 ms | 4.99 | 0.23 | 7% | ok | 1.4x (34%) |
| factor | lu | f32 | 512x512 | 9.035 ms | 9.90 | 0.23 | 7% | ok | 1.4x (36%) |
| factor | cholesky | f64 | 512x512 | 20.056 ms | 2.23 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 18.771 ms | 2.38 | 0.11 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 19.004 ms | 2.35 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 10.467 ms | 4.27 | 0.40 | 6% | ok | 1.1x (27%) |
| factor | qr | f64 | 512x512 | 148.349 ms | 1.21 | 0.03 | 2% | ok | 1.7x (41%) |
| factor | lq | f64 | 512x512 | 36.818 ms | 4.86 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 203.969 ms | 2.19 | 0.02 | 3% | ok | 1.7x (42%) |
| factor | householder | f64 | 512x512 | 0.482 ms | 2.18 | 8.70 | 35% | ok | 3.1x (78%) |
| factor | inv | f64 | 512x512 | 134.285 ms | 2.00 | 0.03 | 3% | ok | 1.1x (27%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.097 ms | 7.40 | 0.35 | 11% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.600 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.093 ms | 1.73 | 0.01 | 3% | ok | 1.1x (29%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 530.936 ms | 0.28 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.655 ms | 2.77 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 135.534 ms | 3.09 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.929 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 28.610 ms | 8.80 | 0.04 | 13% | ok | 1.7x (41%) |
| eigen | sylvester | f64 | 256x256 | 317.061 ms | 3.17 | 0.00 | 5% | ok | 1.1x (29%) |
| elementwise | exp | f64 | n=8388608 | 74.927 ms | 0.11 | 1.79 | 7% | ok | 1.1x (29%) |
| elementwise | log | f64 | n=8388608 | 100.766 ms | 0.08 | 1.33 | 5% | ok | 1.1x (26%) |
| elementwise | sin | f64 | n=8388608 | 137.336 ms | 0.06 | 0.98 | 4% | ok | 1.1x (27%) |
| elementwise | sqrt | f64 | n=8388608 | 58.829 ms | 0.14 | 2.28 | 9% | ok | 1.2x (31%) |
| elementwise | erf | f64 | n=8388608 | 104.297 ms | 0.08 | 1.29 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.211 ms | 2.16 | 22.49 | 91% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.945 ms | 1.35 | 14.00 | 56% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.232 ms | 6.72 | 14.13 | 57% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.935 ms | 0.27 | 0.96 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.953 ms | 0.22 | 1.86 | 7% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 34.856 ms | 0.32 | 0.02 | 0% | ok | 1.1x (27%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 52.965 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.512 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 95.065 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 112.752 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.611 ms | 0.58 | 0.02 | 1% | ok | 1.1x (27%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 71.022 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.156 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.999 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 113.310 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.522 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.861 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.095 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 27.206 ms | 0.41 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.245 ms | 0.34 | 1.35 | 5% | ok | 0.8x (20%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.152 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.836 ms | 1.03 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.149 ms | 0.35 | 1.41 | 6% | ok | 0.8x (20%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.028 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 80.411 ms | 0.28 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.580 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.021 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 95.279 ms | 0.23 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.578 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 10.933 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 616.709 ms | 0.50 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.968 ms | 0.42 | 1.69 | 7% | ok | 0.9x (23%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 25.278 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 169.083 ms | 1.83 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.666 ms | 0.44 | 1.76 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.691 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1583.355 ms | 0.39 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.702 ms | 0.12 | 0.47 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 15.143 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1137.605 ms | 0.54 | 0.00 | 1% | ok | 1.1x (26%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.087 ms | 0.12 | 0.46 | 2% | ok | 1.0x (25%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.985 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 107.797 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.528 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.088 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.390 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.489 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.127 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.360 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.404 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.656 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.523 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.608 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.363 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.622 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 28.181 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.895 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.751 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.604 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.901 ms | 1.38 | 15.43 | 62% | ok | 1.1x (28%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.315 ms | 1.13 | 12.67 | 51% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.711 ms | 0.97 | 10.82 | 44% | ok | 1.1x (27%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.529 ms | 1.18 | 8.94 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 211.846 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 210.583 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 131.181 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.759 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.693 ms | 4.54 | 18.17 | 73% | ok | 1.1x (27%) |
| mixed | gemm | f32>f64 | 512x512x512 | 10.526 ms | 25.50 | 0.40 | 38% | ok | 3.9x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.733 ms | 22.90 | 24.33 | 98% | ok | 1.0x (26%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.800 ms | 5.99 | 23.97 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.377 ms | 12.19 | 24.38 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.679 ms | 24.70 | 24.70 | n/a | ok | 1.0x (26%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.373 ms | 12.22 | 12.22 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.468 ms | 60.08 | 0.70 | n/a | ok | 3.9x (97%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.255 ms | 63.08 | 0.49 | n/a | ok | 3.9x (98%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.193 ms | 64.02 | 0.38 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.944 ms | 284.50 | 1.67 | n/a | ok | 3.8x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.340 ms | 80.36 | 0.47 | n/a | ok | 3.9x (99%) |

