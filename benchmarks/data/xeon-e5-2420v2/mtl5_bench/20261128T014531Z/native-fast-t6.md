## xeon-e5-2420v2 -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.1 GFLOP/s |
| ... this build, SSE4 | f64 58.9, f32 118.0 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 99% | 108% | 0 | 18% |
| blas2 | 10 | 86% | 108% | 0 | 17% |
| blas3 | 10 | 34% | 51% | 0 | 42% |
| factor | 12 | 6% | 40% | 0 | 22% |
| eigen | 8 | 3% | 13% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 105% | 0 | 17% |
| direct | 19 | 1% | 31% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 72% | 77% | 0 | 17% |
| mg | 1 | 72% | 72% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 68% | 93% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 79% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 10.873 ms | 1.54 | 12.34 | 108% | ok | 1.1x (18%) |
| blas1 | dot | f32 | n=8388608 | 5.439 ms | 3.08 | 12.34 | 108% | ok | 1.1x (18%) |
| blas1 | axpy | f64 | n=8388608 | 17.833 ms | 0.94 | 11.29 | 99% | ok | 1.1x (18%) |
| blas1 | axpy | f32 | n=8388608 | 8.430 ms | 1.99 | 11.94 | 105% | ok | 1.2x (19%) |
| blas1 | scale | f64 | n=8388608 | 12.091 ms | 0.69 | 11.10 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 5.423 ms | 3.09 | 12.37 | 108% | ok | 1.1x (18%) |
| blas1 | one_norm | f64 | n=8388608 | 9.715 ms | 0.86 | 6.91 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.050 ms | 0.64 | 5.14 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.727 ms | 0.86 | 6.90 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 269.923 ms | 0.68 | 0.50 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.902 ms | 3.08 | 12.32 | 108% | ok | 1.2x (20%) |
| blas2 | gemv | f32 | n=4096 | 5.453 ms | 6.15 | 12.31 | 108% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 11.056 ms | 3.03 | 12.15 | 106% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.812 ms | 5.77 | 11.55 | 101% | ok | 1.1x (19%) |
| blas2 | symv | f64 | n=4096 | 12.188 ms | 2.75 | 5.51 | 48% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 6.844 ms | 2.45 | 9.81 | 86% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 3.584 ms | 4.68 | 9.37 | 82% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 6.898 ms | 2.43 | 9.74 | 85% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.601 ms | 4.66 | 9.33 | 82% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 28.566 ms | 1.17 | 9.40 | 82% | ok | 1.0x (16%) |
| blas3 | gemm | f64 | 512x512x512 | 8.835 ms | 30.38 | 0.71 | 51% | ok | 3.1x (52%) |
| blas3 | gemm | f32 | 512x512x512 | 4.482 ms | 59.89 | 0.70 | 51% | ok | 3.0x (50%) |
| blas3 | gemmt | f64 | 512x512x512 | 6.849 ms | 19.60 | 0.92 | 33% | ok | 2.5x (41%) |
| blas3 | symm | f64 | 512x512x512 | 9.929 ms | 27.03 | 0.63 | 46% | ok | 2.9x (48%) |
| blas3 | syrk | f64 | 512x512x512 | 6.696 ms | 20.04 | 0.63 | 34% | ok | 2.5x (42%) |
| blas3 | syr2k | f64 | 512x512x512 | 13.173 ms | 20.38 | 0.48 | 35% | ok | 2.5x (42%) |
| blas3 | trmm | f64 | 512x512x512 | 9.296 ms | 14.44 | 0.56 | 24% | ok | 1.9x (32%) |
| blas3 | trsm | f64 | 512x512x512 | 8.361 ms | 16.05 | 0.63 | 27% | ok | 2.2x (37%) |
| blas3 | trsm | f32 | 512x512x512 | 4.165 ms | 32.23 | 0.63 | 27% | ok | 2.1x (35%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 37.931 ms | 0.14 | 1.12 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 21.023 ms | 4.26 | 0.20 | 7% | ok | 1.4x (23%) |
| factor | lu | f32 | 512x512 | 11.714 ms | 7.64 | 0.18 | 6% | ok | 1.3x (22%) |
| factor | cholesky | f64 | 512x512 | 25.783 ms | 1.74 | 0.16 | 3% | ok | 1.0x (16%) |
| factor | cholesky | f32 | 512x512 | 24.366 ms | 1.84 | 0.09 | 2% | ok | 1.0x (16%) |
| factor | ldlt | f64 | 512x512 | 26.044 ms | 1.72 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 12.769 ms | 3.50 | 0.33 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 328.509 ms | 0.54 | 0.01 | 1% | ok | 1.5x (25%) |
| factor | lq | f64 | 512x512 | 49.131 ms | 3.64 | 0.09 | 6% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 400.302 ms | 1.12 | 0.01 | 2% | ok | 1.5x (26%) |
| factor | householder | f64 | 512x512 | 0.916 ms | 1.15 | 4.58 | 40%\* | ok | 3.5x (59%) |
| factor | inv | f64 | 512x512 | 147.875 ms | 1.82 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.899 ms | 5.29 | 0.25 | 9% | ok | 1.3x (21%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.387 ms | 0.37 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 172.515 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 999.956 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 105.214 ms | 1.59 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 260.339 ms | 1.61 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 258.015 ms | 1.43 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 31.919 ms | 7.88 | 0.03 | 13% | ok | 1.6x (26%) |
| eigen | sylvester | f64 | 256x256 | 570.675 ms | 1.76 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 152.139 ms | 0.06 | 0.88 | 8% | ok | 1.0x (16%) |
| elementwise | log | f64 | n=8388608 | 140.983 ms | 0.06 | 0.95 | 8% | ok | 1.0x (16%) |
| elementwise | sin | f64 | n=8388608 | 170.768 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 89.415 ms | 0.09 | 1.50 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2472.308 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.272 ms | 1.15 | 11.98 | 105% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.110 ms | 0.84 | 8.75 | 77% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.317 ms | 3.70 | 7.78 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.087 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.904 ms | 0.34 | 2.82 | 25% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 41.856 ms | 0.27 | 0.02 | 0% | ok | 1.0x (16%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.224 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.218 ms | 0.44 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.340 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 130.631 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.876 ms | 0.47 | 0.02 | 1% | ok | 1.1x (18%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 79.303 ms | 0.02 | 0.00 | 0% | ok | 1.1x (18%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.184 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.280 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 128.261 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.544 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.921 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.563 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.316 ms | 0.34 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.859 ms | 0.88 | 3.52 | 31%\* | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.685 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.743 ms | 0.88 | 0.06 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.881 ms | 0.86 | 3.43 | 30%\* | ok | 0.9x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.473 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 104.640 ms | 0.21 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.750 ms | 0.20 | 0.81 | 7%\* | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.902 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.608 ms | 0.20 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.771 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.308 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 654.943 ms | 0.47 | 0.00 | 1% | ok | 1.0x (16%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.140 ms | 0.47 | 1.89 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.991 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 205.799 ms | 1.50 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.077 ms | 0.48 | 1.90 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.796 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1751.501 ms | 0.35 | 0.00 | 1% | ok | 1.0x (16%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.669 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.149 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1964.629 ms | 0.31 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.500 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.317 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 122.997 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 59.380 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 31.881 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.279 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.055 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 65.311 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 69.632 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.368 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.460 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 39.876 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 38.635 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 44.963 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.644 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.284 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.876 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.779 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.093 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.189 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.589 ms | 0.73 | 8.17 | 72% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.315 ms | 0.79 | 8.85 | 77% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.921 ms | 0.67 | 7.48 | 65% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.565 ms | 1.08 | 8.17 | 72% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 139.140 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 144.651 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 130.203 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.766 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| mixed | dot | f32>f64 | n=8388608 | 6.318 ms | 2.66 | 10.62 | 93% | ok | 0.9x (16%) |
| mixed | gemm | f32>f64 | 512x512x512 | 11.758 ms | 22.83 | 0.36 | 39% | ok | 5.3x (88%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.284 ms | 7.34 | 7.80 | 68% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.037 ms | 2.78 | 11.12 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.972 ms | 5.64 | 11.29 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.083 ms | 8.06 | 8.06 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.637 ms | 10.25 | 10.25 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.946 ms | 68.03 | 0.80 | n/a | ok | 4.7x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.528 ms | 59.28 | 0.46 | n/a | ok | 4.9x (81%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.491 ms | 59.78 | 0.35 | n/a | ok | 4.9x (82%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.315 ms | 80.98 | 0.47 | n/a | ok | 4.9x (81%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 2.565 ms | 104.67 | 0.61 | n/a | ok | 4.8x (80%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

