## xeon-e5-2420v2 -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, SSE4 | f64 10.5, f32 21.0 GFLOP/s |
| bandwidth | read 10.5, update 10.4, triad 8.2 GB/s (ceiling 10.5) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 104% | 109% | 0 |
| blas2 | 10 | 90% | 100% | 0 |
| blas3 | 10 | 76% | 90% | 0 |
| factor | 12 | 17% | 41% | 0 |
| eigen | 8 | 15% | 48% | 0 |
| elementwise | 5 | 8% | 13% | 0 |
| sparse | 5 | 56% | 106% | 0 |
| direct | 19 | 3% | 36% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 78% | 89% | 0 |
| mg | 1 | 78% | 78% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 74% | 107% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 12.003 ms | 1.40 | 11.18 | 107% | ok |
| blas1 | dot | f32 | n=8388608 | 5.934 ms | 2.83 | 11.31 | 108% | ok |
| blas1 | axpy | f64 | n=8388608 | 18.806 ms | 0.89 | 10.71 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 8.972 ms | 1.87 | 11.22 | 107% | ok |
| blas1 | scale | f64 | n=8388608 | 12.353 ms | 0.68 | 10.87 | 104% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.894 ms | 2.85 | 11.39 | 109% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.746 ms | 0.86 | 6.89 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.368 ms | 0.63 | 5.02 | 48% | ok |
| blas1 | sum | f64 | n=8388608 | 9.765 ms | 0.86 | 6.87 | 66% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 271.158 ms | 0.68 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 13.358 ms | 2.51 | 10.05 | 96% | ok |
| blas2 | gemv | f32 | n=4096 | 6.684 ms | 5.02 | 10.05 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.080 ms | 2.57 | 10.27 | 98% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.420 ms | 5.23 | 10.46 | 100% | ok |
| blas2 | symv | f64 | n=4096 | 12.280 ms | 2.73 | 5.47 | 52% | ok |
| blas2 | trmv | f64 | n=4096 | 7.078 ms | 2.37 | 9.49 | 90% | ok |
| blas2 | trmv | f32 | n=4096 | 3.650 ms | 4.60 | 9.20 | 88% | ok |
| blas2 | trsv | f64 | n=4096 | 7.157 ms | 2.34 | 9.39 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 3.931 ms | 4.27 | 8.54 | 81% | ok |
| blas2 | ger | f64 | n=4096 | 29.793 ms | 1.13 | 9.01 | 86% | ok |
| blas3 | gemm | f64 | 512x512x512 | 29.679 ms | 9.04 | 0.21 | 86% | ok |
| blas3 | gemm | f32 | 512x512x512 | 14.233 ms | 18.86 | 0.22 | 90% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.833 ms | 7.97 | 0.37 | 76% | ok |
| blas3 | symm | f64 | 512x512x512 | 28.699 ms | 9.35 | 0.22 | 89% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.749 ms | 8.01 | 0.25 | 76% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 33.427 ms | 8.03 | 0.19 | 77% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.431 ms | 7.70 | 0.30 | 73% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.058 ms | 7.43 | 0.29 | 71% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.828 ms | 15.20 | 0.30 | 72% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.244 ms | 0.14 | 1.11 | 11% | ok |
| factor | lu | f64 | 512x512 | 29.307 ms | 3.05 | 0.14 | 29% | ok |
| factor | lu | f32 | 512x512 | 15.340 ms | 5.83 | 0.14 | 28% | ok |
| factor | cholesky | f64 | 512x512 | 25.360 ms | 1.76 | 0.17 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 23.795 ms | 1.88 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.706 ms | 1.74 | 0.16 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.356 ms | 3.35 | 0.31 | 32% | ok |
| factor | qr | f64 | 512x512 | 514.986 ms | 0.35 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 73.535 ms | 2.43 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 624.495 ms | 0.72 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.187 ms | 0.33 | 1.32 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 153.076 ms | 1.75 | 0.03 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 20.927 ms | 4.28 | 0.20 | 41% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.332 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 173.769 ms | 0.87 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1000.933 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 102.697 ms | 1.63 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 258.472 ms | 1.62 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 261.597 ms | 1.41 | 0.00 | 13% | ok |
| eigen | expm | f64 | 256x256 | 49.894 ms | 5.04 | 0.02 | 48% | ok |
| eigen | sylvester | f64 | 256x256 | 577.818 ms | 1.74 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 150.921 ms | 0.06 | 0.89 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 140.938 ms | 0.06 | 0.95 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 169.215 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 95.191 ms | 0.09 | 1.41 | 13% | ok |
| elementwise | erf | f64 | n=8388608 | 2477.572 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.450 ms | 1.07 | 11.11 | 106% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.146 ms | 0.83 | 8.66 | 83% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.085 ms | 2.78 | 5.84 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.061 ms | 0.20 | 0.72 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.854 ms | 0.34 | 2.84 | 27% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.721 ms | 0.27 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.317 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.939 ms | 0.45 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.640 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 129.151 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.761 ms | 0.47 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 78.575 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.076 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.316 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 127.964 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.555 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 39.889 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.338 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.336 ms | 0.34 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.799 ms | 0.95 | 3.79 | 36%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.487 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.588 ms | 0.89 | 0.06 | 8% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.817 ms | 0.92 | 3.70 | 35%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.558 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 110.504 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 4.592 ms | 0.16 | 0.66 | 6%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.578 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 118.474 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.874 ms | 0.20 | 0.78 | 7%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.253 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 638.640 ms | 0.48 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.836 ms | 0.49 | 1.97 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 29.126 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 206.029 ms | 1.50 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.897 ms | 0.49 | 1.95 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.940 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1727.139 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.837 ms | 0.10 | 0.41 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.466 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1958.880 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.855 ms | 0.10 | 0.41 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.886 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 171.315 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 60.126 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 32.039 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.106 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.203 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 64.883 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 69.066 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.583 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.429 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 39.702 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 38.320 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 44.863 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.127 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.661 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 32.962 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.880 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.756 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.081 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.073 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.585 ms | 0.73 | 8.18 | 78% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.126 ms | 0.84 | 9.38 | 89% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.896 ms | 0.67 | 7.53 | 72% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.671 ms | 1.08 | 8.15 | 78% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.512 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 144.666 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 131.666 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.239 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 5.961 ms | 2.81 | 11.26 | 107% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 62.013 ms | 4.33 | 0.07 | 41% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.283 ms | 7.35 | 7.81 | 74% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.048 ms | 2.77 | 11.10 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.010 ms | 5.57 | 11.15 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.098 ms | 8.00 | 8.00 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.650 ms | 10.17 | 10.17 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.780 ms | 14.29 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.092 ms | 12.15 | 0.09 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.185 ms | 12.10 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 16.286 ms | 16.48 | 0.10 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 12.439 ms | 21.58 | 0.13 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

