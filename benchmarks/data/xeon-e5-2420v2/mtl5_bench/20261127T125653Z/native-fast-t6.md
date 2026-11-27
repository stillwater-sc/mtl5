## xeon-e5-2420v2 -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 59.1 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 59.1, f32 118.2 GFLOP/s |
| ... this build, SSE4 | f64 59.1, f32 118.1 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 99% | 108% | 0 | 18% |
| blas2 | 10 | 85% | 108% | 0 | 18% |
| blas3 | 10 | 46% | 79% | 0 | 57% |
| factor | 12 | 6% | 42% | 0 | 23% |
| eigen | 8 | 3% | 13% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 103% | 0 | 17% |
| direct | 19 | 1% | 31% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 70% | 82% | 0 | 17% |
| mg | 1 | 65% | 65% | 0 | 15% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 68% | 91% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 79% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 10.867 ms | 1.54 | 12.35 | 108% | ok | 1.1x (18%) |
| blas1 | dot | f32 | n=8388608 | 5.439 ms | 3.08 | 12.34 | 108% | ok | 1.1x (18%) |
| blas1 | axpy | f64 | n=8388608 | 17.846 ms | 0.94 | 11.28 | 99% | ok | 1.1x (18%) |
| blas1 | axpy | f32 | n=8388608 | 8.444 ms | 1.99 | 11.92 | 104% | ok | 1.1x (18%) |
| blas1 | scale | f64 | n=8388608 | 12.141 ms | 0.69 | 11.05 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 5.424 ms | 3.09 | 12.37 | 108% | ok | 1.1x (18%) |
| blas1 | one_norm | f64 | n=8388608 | 9.815 ms | 0.85 | 6.84 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.059 ms | 0.64 | 5.14 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.765 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 272.067 ms | 0.68 | 0.49 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.909 ms | 3.08 | 12.31 | 108% | ok | 1.2x (20%) |
| blas2 | gemv | f32 | n=4096 | 5.452 ms | 6.15 | 12.31 | 108% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 11.058 ms | 3.03 | 12.14 | 106% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.820 ms | 5.77 | 11.54 | 101% | ok | 1.1x (18%) |
| blas2 | symv | f64 | n=4096 | 12.669 ms | 2.65 | 5.30 | 46% | ok | 1.0x (16%) |
| blas2 | trmv | f64 | n=4096 | 7.059 ms | 2.38 | 9.52 | 83% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 3.577 ms | 4.69 | 9.39 | 82% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 6.914 ms | 2.43 | 9.72 | 85% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.809 ms | 4.40 | 8.82 | 77% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 28.222 ms | 1.19 | 9.51 | 83% | ok | 1.1x (18%) |
| blas3 | gemm | f64 | 512x512x512 | 5.860 ms | 45.81 | 1.07 | 77% | ok | 5.1x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 2.883 ms | 93.10 | 1.09 | 79% | ok | 4.9x (82%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.988 ms | 26.91 | 1.26 | 45% | ok | 3.4x (56%) |
| blas3 | symm | f64 | 512x512x512 | 6.903 ms | 38.89 | 0.91 | 66% | ok | 4.2x (69%) |
| blas3 | syrk | f64 | 512x512x512 | 4.925 ms | 27.25 | 0.85 | 46% | ok | 3.4x (57%) |
| blas3 | syr2k | f64 | 512x512x512 | 9.540 ms | 28.14 | 0.66 | 48% | ok | 3.5x (58%) |
| blas3 | trmm | f64 | 512x512x512 | 7.880 ms | 17.03 | 0.67 | 29% | ok | 2.2x (37%) |
| blas3 | trsm | f64 | 512x512x512 | 8.623 ms | 15.56 | 0.61 | 26% | ok | 2.1x (35%) |
| blas3 | trsm | f32 | 512x512x512 | 4.174 ms | 32.15 | 0.63 | 27% | ok | 2.1x (35%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.721 ms | 0.14 | 1.10 | 10% | ok | 1.0x (16%) |
| factor | lu | f64 | 512x512 | 20.435 ms | 4.38 | 0.21 | 7% | ok | 1.4x (24%) |
| factor | lu | f32 | 512x512 | 11.244 ms | 7.96 | 0.19 | 7% | ok | 1.4x (23%) |
| factor | cholesky | f64 | 512x512 | 25.358 ms | 1.76 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 23.878 ms | 1.87 | 0.09 | 2% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 25.561 ms | 1.75 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 13.057 ms | 3.43 | 0.32 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 322.805 ms | 0.55 | 0.01 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 47.859 ms | 3.74 | 0.09 | 6% | ok | 1.5x (26%) |
| factor | hessenberg | f64 | 512x512 | 400.919 ms | 1.12 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.882 ms | 1.19 | 4.76 | 42%\* | ok | 3.6x (60%) |
| factor | inv | f64 | 512x512 | 143.920 ms | 1.87 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.784 ms | 5.33 | 0.25 | 9% | ok | 1.2x (21%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.919 ms | 0.37 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 171.578 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 991.950 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 106.141 ms | 1.58 | 0.01 | 3% | ok | 1.0x (16%) |
| eigen | schur | f64 | 256x256 | 260.247 ms | 1.61 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 260.226 ms | 1.42 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 32.626 ms | 7.71 | 0.03 | 13% | ok | 1.5x (25%) |
| eigen | sylvester | f64 | 256x256 | 582.115 ms | 1.73 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 152.144 ms | 0.06 | 0.88 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 139.537 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 171.860 ms | 0.05 | 0.78 | 7% | ok | 1.0x (16%) |
| elementwise | sqrt | f64 | n=8388608 | 90.077 ms | 0.09 | 1.49 | 13% | ok | 1.1x (18%) |
| elementwise | erf | f64 | n=8388608 | 2499.840 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.300 ms | 1.14 | 11.84 | 103% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.123 ms | 0.84 | 8.72 | 76% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.323 ms | 3.70 | 7.78 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.463 ms | 0.20 | 0.70 | 6%\* | ok | 1.0x (16%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.873 ms | 0.34 | 2.83 | 25% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.864 ms | 0.27 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.705 ms | 0.16 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.151 ms | 0.44 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.767 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 130.195 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.086 ms | 0.47 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 79.138 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.272 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.299 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 127.925 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.543 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.643 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.772 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.276 ms | 0.34 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.854 ms | 0.89 | 3.54 | 31%\* | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.579 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.524 ms | 0.89 | 0.06 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.884 ms | 0.86 | 3.42 | 30%\* | ok | 0.9x (15%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.776 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.937 ms | 0.22 | 0.01 | 0% | ok | 1.1x (18%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.761 ms | 0.20 | 0.80 | 7%\* | ok | 1.2x (20%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.229 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 112.938 ms | 0.20 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.881 ms | 0.19 | 0.78 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.219 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 635.964 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.028 ms | 0.48 | 1.92 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.935 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 206.045 ms | 1.50 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.076 ms | 0.48 | 1.90 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.874 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1720.063 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.667 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.296 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1938.059 ms | 0.32 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.597 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.244 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 164.476 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 59.958 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 31.782 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.223 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 36.992 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 65.666 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 69.337 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.402 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.472 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 39.396 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 38.321 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 44.707 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.128 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.598 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.310 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ic_0 | f64 | 128^2 grid | 2.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.764 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.080 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.219 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.644 ms | 0.72 | 8.05 | 70% | ok | 1.0x (16%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.122 ms | 0.84 | 9.39 | 82% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.805 ms | 0.69 | 7.71 | 67% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 39.232 ms | 0.98 | 7.41 | 65% | ok | 0.9x (15%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 139.864 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 144.545 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 129.901 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.047 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 6.458 ms | 2.60 | 10.39 | 91% | ok | 0.9x (15%) |
| mixed | gemm | f32>f64 | 512x512x512 | 11.733 ms | 22.88 | 0.36 | 39% | ok | 5.3x (88%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.282 ms | 7.35 | 7.81 | 68% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.014 ms | 2.79 | 11.16 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.978 ms | 5.63 | 11.27 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.082 ms | 8.06 | 8.06 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.641 ms | 10.22 | 10.22 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.948 ms | 67.99 | 0.80 | n/a | ok | 4.8x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.541 ms | 59.11 | 0.46 | n/a | ok | 4.9x (81%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.492 ms | 59.76 | 0.35 | n/a | ok | 4.9x (82%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.315 ms | 80.98 | 0.47 | n/a | ok | 4.9x (82%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 2.565 ms | 104.66 | 0.61 | n/a | ok | 4.8x (81%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

