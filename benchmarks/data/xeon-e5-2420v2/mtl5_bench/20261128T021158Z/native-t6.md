## xeon-e5-2420v2 -- native, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 58.8 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 58.8, f32 118.1 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.6, triad 8.4 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 106% | 0 | 84% |
| blas2 | 10 | 85% | 108% | 0 | 17% |
| blas3 | 10 | 7% | 39% | 0 | 17% |
| factor | 12 | 6% | 40% | 0 | 23% |
| eigen | 8 | 3% | 11% | 0 | 18% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 104% | 0 | 17% |
| direct | 19 | 1% | 31% | 0 | 16% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 73% | 80% | 0 | 17% |
| mg | 1 | 72% | 72% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 71% | 0 | 18% |
| integer | 9 | n/a | n/a | 0 | 91% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 32.722 ms | 0.51 | 4.10 | 36% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=8388608 | 5.536 ms | 3.03 | 12.12 | 106% | ok | 5.0x (84%) |
| blas1 | axpy | f64 | n=8388608 | 38.119 ms | 0.44 | 5.28 | 46% | ok | 5.5x (92%) |
| blas1 | axpy | f32 | n=8388608 | 16.786 ms | 1.00 | 6.00 | 53% | ok | 5.4x (91%) |
| blas1 | scale | f64 | n=8388608 | 12.086 ms | 0.69 | 11.11 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 33.035 ms | 0.51 | 2.03 | 18% | ok | 5.6x (94%) |
| blas1 | one_norm | f64 | n=8388608 | 9.771 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.159 ms | 0.64 | 5.10 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.745 ms | 0.86 | 6.89 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 279.997 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.929 ms | 3.07 | 12.29 | 108% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.488 ms | 6.11 | 12.23 | 107% | ok | 1.5x (25%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.919 ms | 3.07 | 12.30 | 108% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.456 ms | 6.15 | 12.31 | 108% | ok | 1.2x (20%) |
| blas2 | symv | f64 | n=4096 | 12.597 ms | 2.66 | 5.33 | 47% | ok | 1.0x (16%) |
| blas2 | trmv | f64 | n=4096 | 7.098 ms | 2.36 | 9.46 | 83% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 5.110 ms | 3.28 | 6.57 | 58% | ok | 0.9x (15%) |
| blas2 | trsv | f64 | n=4096 | 6.905 ms | 2.43 | 9.73 | 85% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.595 ms | 3.65 | 7.31 | 64% | ok | 0.9x (16%) |
| blas2 | ger | f64 | n=4096 | 28.520 ms | 1.18 | 9.41 | 83% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 11.553 ms | 23.23 | 0.54 | 39% | ok | 5.6x (93%) |
| blas3 | gemm | f32 | 512x512x512 | 6.228 ms | 43.10 | 0.51 | 36% | ok | 5.6x (94%) |
| blas3 | gemmt | f64 | 512x512x512 | 36.346 ms | 3.69 | 0.17 | 6% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 65.932 ms | 4.07 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 38.522 ms | 3.48 | 0.11 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 65.750 ms | 4.08 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 32.599 ms | 4.12 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 32.956 ms | 4.07 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 17.190 ms | 7.81 | 0.15 | 7% | ok | 0.9x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.671 ms | 0.14 | 1.10 | 10% | ok | 1.0x (16%) |
| factor | lu | f64 | 512x512 | 21.854 ms | 4.09 | 0.19 | 7% | ok | 1.4x (24%) |
| factor | lu | f32 | 512x512 | 11.907 ms | 7.51 | 0.18 | 6% | ok | 1.4x (23%) |
| factor | cholesky | f64 | 512x512 | 24.868 ms | 1.80 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 23.677 ms | 1.89 | 0.09 | 2% | ok | 1.0x (16%) |
| factor | ldlt | f64 | 512x512 | 25.585 ms | 1.75 | 0.16 | 3% | ok | 1.0x (16%) |
| factor | ldlt_bk | f64 | 512x512 | 13.487 ms | 3.32 | 0.31 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 320.889 ms | 0.56 | 0.01 | 1% | ok | 1.6x (26%) |
| factor | lq | f64 | 512x512 | 49.121 ms | 3.64 | 0.09 | 6% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 402.067 ms | 1.11 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.927 ms | 1.13 | 4.52 | 40%\* | ok | 3.4x (57%) |
| factor | inv | f64 | 512x512 | 143.563 ms | 1.87 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 17.054 ms | 5.25 | 0.25 | 9% | ok | 1.3x (22%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.344 ms | 0.37 | 0.02 | 1% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 171.997 ms | 0.88 | 0.01 | 1% | ok | 1.1x (18%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1010.164 ms | 0.15 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 108.076 ms | 1.55 | 0.01 | 3% | ok | 1.1x (18%) |
| eigen | schur | f64 | 256x256 | 279.598 ms | 1.50 | 0.00 | 3% | ok | 1.0x (16%) |
| eigen | svd | f64 | 256x256 | 321.014 ms | 1.15 | 0.00 | 2% | ok | 0.9x (16%) |
| eigen | expm | f64 | 256x256 | 37.769 ms | 6.66 | 0.03 | 11% | ok | 2.2x (37%) |
| eigen | sylvester | f64 | 256x256 | 600.658 ms | 1.68 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 152.780 ms | 0.05 | 0.88 | 8% | ok | 1.0x (16%) |
| elementwise | log | f64 | n=8388608 | 139.526 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 170.996 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 91.155 ms | 0.09 | 1.47 | 13% | ok | 1.0x (16%) |
| elementwise | erf | f64 | n=8388608 | 2484.854 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.298 ms | 1.14 | 11.85 | 104% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.146 ms | 0.83 | 8.66 | 76% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.321 ms | 3.70 | 7.78 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.609 ms | 0.20 | 0.69 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.913 ms | 0.34 | 2.82 | 25% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 41.467 ms | 0.27 | 0.02 | 0% | ok | 1.0x (16%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 73.050 ms | 0.15 | 0.01 | 0% | ok | 0.9x (16%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.005 ms | 0.45 | 0.03 | 1% | ok | 1.0x (16%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 118.051 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.251 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.559 ms | 0.46 | 0.02 | 1% | ok | 1.0x (16%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 82.587 ms | 0.02 | 0.00 | 0% | ok | 1.0x (16%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.879 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.280 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 126.052 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.523 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.012 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.216 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.431 ms | 0.33 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.853 ms | 0.89 | 3.54 | 31%\* | ok | 1.0x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.372 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.767 ms | 0.88 | 0.06 | 1% | ok | 1.0x (16%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.879 ms | 0.86 | 3.44 | 30%\* | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.308 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.970 ms | 0.22 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.818 ms | 0.20 | 0.79 | 7%\* | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.845 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.643 ms | 0.20 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.780 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.951 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 633.732 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.072 ms | 0.48 | 1.91 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.791 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 205.548 ms | 1.50 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.223 ms | 0.47 | 1.87 | 16%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.510 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1705.496 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.804 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 17.911 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1523.577 ms | 0.41 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.704 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 231.496 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 404.545 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 128.368 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 105.445 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 123.150 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 80.192 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 536.671 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 175.938 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.855 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 202.554 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 192.450 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 213.620 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 307.384 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.365 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 32.849 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.909 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ildl | f64 | 128^2 grid | 2.727 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.093 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.239 ms | 0.00 | 0.00 | n/a | ok | 0.9x (15%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.528 ms | 0.74 | 8.31 | 73% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.217 ms | 0.81 | 9.12 | 80% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.953 ms | 0.66 | 7.42 | 65% | ok | 1.0x (16%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.563 ms | 1.08 | 8.17 | 72% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 142.190 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 146.248 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 128.253 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.448 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 194.190 ms | 0.09 | 0.35 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.546 ms | 13.07 | 0.20 | 22% | ok | 5.6x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.203 ms | 7.62 | 8.09 | 71% | ok | 1.1x (18%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.502 ms | 2.58 | 10.32 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.110 ms | 5.39 | 10.79 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.982 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.975 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.536 ms | 23.27 | 0.27 | n/a | ok | 5.7x (95%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.214 ms | 32.68 | 0.26 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.340 ms | 61.86 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.333 ms | 61.95 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.090 ms | 20.51 | 0.12 | n/a | ok | 5.5x (91%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

