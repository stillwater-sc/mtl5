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
| bandwidth | read 11.4, update 10.6, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 99% | 108% | 0 | 18% |
| blas2 | 10 | 86% | 108% | 0 | 17% |
| blas3 | 10 | 46% | 79% | 0 | 57% |
| factor | 12 | 5% | 41% | 0 | 22% |
| eigen | 8 | 3% | 13% | 0 | 17% |
| elementwise | 5 | 7% | 13% | 0 | 17% |
| sparse | 5 | 68% | 104% | 0 | 17% |
| direct | 19 | 1% | 31% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 71% | 78% | 0 | 17% |
| mg | 1 | 71% | 71% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 69% | 94% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 80% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 10.873 ms | 1.54 | 12.34 | 108% | ok | 1.1x (18%) |
| blas1 | dot | f32 | n=8388608 | 5.444 ms | 3.08 | 12.33 | 108% | ok | 1.1x (18%) |
| blas1 | axpy | f64 | n=8388608 | 17.855 ms | 0.94 | 11.28 | 99% | ok | 1.1x (18%) |
| blas1 | axpy | f32 | n=8388608 | 8.443 ms | 1.99 | 11.92 | 104% | ok | 1.1x (18%) |
| blas1 | scale | f64 | n=8388608 | 12.078 ms | 0.69 | 11.11 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 5.428 ms | 3.09 | 12.36 | 108% | ok | 1.1x (18%) |
| blas1 | one_norm | f64 | n=8388608 | 9.720 ms | 0.86 | 6.90 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.252 ms | 0.63 | 5.06 | 44% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.703 ms | 0.86 | 6.92 | 61% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 275.128 ms | 0.67 | 0.49 | 4% | ok | 1.0x (16%) |
| blas2 | gemv | f64 | n=4096 | 10.920 ms | 3.07 | 12.30 | 108% | ok | 1.2x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.638 ms | 5.95 | 11.91 | 104% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 11.066 ms | 3.03 | 12.13 | 106% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.812 ms | 5.77 | 11.55 | 101% | ok | 1.1x (19%) |
| blas2 | symv | f64 | n=4096 | 12.233 ms | 2.74 | 5.49 | 48% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 6.924 ms | 2.42 | 9.70 | 85% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 3.601 ms | 4.66 | 9.33 | 82% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 6.871 ms | 2.44 | 9.78 | 86% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.607 ms | 4.65 | 9.31 | 82% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 28.215 ms | 1.19 | 9.52 | 83% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 5.828 ms | 46.06 | 1.08 | 78% | ok | 4.8x (80%) |
| blas3 | gemm | f32 | 512x512x512 | 2.889 ms | 92.91 | 1.09 | 79% | ok | 4.7x (78%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.977 ms | 26.96 | 1.26 | 46% | ok | 3.4x (57%) |
| blas3 | symm | f64 | 512x512x512 | 6.908 ms | 38.86 | 0.91 | 66% | ok | 4.2x (71%) |
| blas3 | syrk | f64 | 512x512x512 | 4.967 ms | 27.02 | 0.84 | 46% | ok | 3.4x (57%) |
| blas3 | syr2k | f64 | 512x512x512 | 9.491 ms | 28.28 | 0.66 | 48% | ok | 3.6x (59%) |
| blas3 | trmm | f64 | 512x512x512 | 7.686 ms | 17.46 | 0.68 | 30% | ok | 2.3x (39%) |
| blas3 | trsm | f64 | 512x512x512 | 8.385 ms | 16.01 | 0.63 | 27% | ok | 2.2x (36%) |
| blas3 | trsm | f32 | 512x512x512 | 4.168 ms | 32.20 | 0.63 | 27% | ok | 2.3x (38%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 37.879 ms | 0.14 | 1.12 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 20.986 ms | 4.26 | 0.20 | 7% | ok | 1.4x (24%) |
| factor | lu | f32 | 512x512 | 11.649 ms | 7.68 | 0.18 | 7% | ok | 1.3x (22%) |
| factor | cholesky | f64 | 512x512 | 25.734 ms | 1.74 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 23.985 ms | 1.87 | 0.09 | 2% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 26.033 ms | 1.72 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 12.728 ms | 3.52 | 0.33 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 324.698 ms | 0.55 | 0.01 | 1% | ok | 1.5x (26%) |
| factor | lq | f64 | 512x512 | 55.191 ms | 3.24 | 0.08 | 5% | ok | 1.3x (22%) |
| factor | hessenberg | f64 | 512x512 | 404.224 ms | 1.11 | 0.01 | 2% | ok | 1.5x (26%) |
| factor | householder | f64 | 512x512 | 0.899 ms | 1.17 | 4.67 | 41%\* | ok | 3.7x (61%) |
| factor | inv | f64 | 512x512 | 149.159 ms | 1.80 | 0.03 | 3% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.999 ms | 5.26 | 0.25 | 9% | ok | 1.2x (21%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.723 ms | 0.37 | 0.02 | 1% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 174.084 ms | 0.87 | 0.01 | 1% | ok | 1.0x (16%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1005.526 ms | 0.15 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 104.378 ms | 1.61 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 260.930 ms | 1.61 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 261.107 ms | 1.41 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 32.359 ms | 7.78 | 0.03 | 13% | ok | 1.6x (26%) |
| eigen | sylvester | f64 | 256x256 | 580.799 ms | 1.73 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 157.394 ms | 0.05 | 0.85 | 7% | ok | 1.0x (16%) |
| elementwise | log | f64 | n=8388608 | 141.253 ms | 0.06 | 0.95 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 172.004 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 90.240 ms | 0.09 | 1.49 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2502.576 ms | 0.00 | 0.05 | 0% | ok | 1.0x (16%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.302 ms | 1.14 | 11.83 | 104% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.119 ms | 0.84 | 8.73 | 76% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.337 ms | 3.69 | 7.77 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.307 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.901 ms | 0.34 | 2.82 | 25% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 42.116 ms | 0.27 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.097 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.043 ms | 0.43 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.547 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.256 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.425 ms | 0.46 | 0.02 | 1% | ok | 1.0x (16%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 79.366 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.135 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.277 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 128.808 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.544 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.380 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.534 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.482 ms | 0.33 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.856 ms | 0.88 | 3.53 | 31%\* | ok | 1.0x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.658 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.701 ms | 0.88 | 0.06 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.881 ms | 0.86 | 3.43 | 30%\* | ok | 0.9x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.563 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.931 ms | 0.22 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.756 ms | 0.20 | 0.81 | 7%\* | ok | 1.1x (18%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.017 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.483 ms | 0.20 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.806 ms | 0.20 | 0.79 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.726 ms | 0.00 | 0.00 | n/a | ok | 0.8x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 637.303 ms | 0.48 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.064 ms | 0.48 | 1.91 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.862 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 203.983 ms | 1.51 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.049 ms | 0.48 | 1.91 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.784 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1714.061 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.740 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.182 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1963.990 ms | 0.31 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.658 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.486 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 124.228 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 60.491 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 32.095 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.553 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.448 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 66.197 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 69.526 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.586 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.802 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 40.061 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 38.686 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 45.200 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.130 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.624 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 35.827 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| pc | ic_0 | f64 | 128^2 grid | 2.870 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.770 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.093 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.393 ms | 0.00 | 0.00 | n/a | ok | 0.9x (14%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.596 ms | 0.73 | 8.15 | 71% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.314 ms | 0.79 | 8.85 | 78% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.936 ms | 0.66 | 7.45 | 65% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.740 ms | 1.08 | 8.13 | 71% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.222 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 144.479 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 130.901 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.523 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| mixed | dot | f32>f64 | n=8388608 | 6.278 ms | 2.67 | 10.69 | 94% | ok | 0.9x (16%) |
| mixed | gemm | f32>f64 | 512x512x512 | 11.766 ms | 22.81 | 0.36 | 39% | ok | 5.2x (87%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.250 ms | 7.46 | 7.92 | 69% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.048 ms | 2.77 | 11.10 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.972 ms | 5.64 | 11.29 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.078 ms | 8.07 | 8.07 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.641 ms | 10.22 | 10.22 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.945 ms | 68.04 | 0.80 | n/a | ok | 4.8x (80%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.535 ms | 59.20 | 0.46 | n/a | ok | 4.9x (81%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.488 ms | 59.81 | 0.35 | n/a | ok | 4.9x (82%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.318 ms | 80.91 | 0.47 | n/a | ok | 4.9x (81%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 2.568 ms | 104.54 | 0.61 | n/a | ok | 4.8x (80%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

