## i7-12700k -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, AVX2 | f64 78.3, f32 156.5 GFLOP/s |
| bandwidth | read 37.8, update 41.5, triad 31.5 GB/s (ceiling 41.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 88% | 103% | 0 |
| blas2 | 10 | 93% | 96% | 0 |
| blas3 | 10 | 67% | 90% | 0 |
| factor | 12 | 8% | 25% | 0 |
| eigen | 8 | 7% | 28% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 52% | 89% | 0 |
| direct | 19 | 1% | 29% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 22% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 58% | 71% | 0 |
| mg | 1 | 51% | 51% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 82% | 131% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 3.677 ms | 4.56 | 36.50 | 88% | ok |
| blas1 | dot | f32 | n=8388608 | 1.702 ms | 9.86 | 39.43 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.796 ms | 3.50 | 41.98 | 101% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.359 ms | 7.11 | 42.68 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 3.461 ms | 2.42 | 38.78 | 93% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.213 ms | 7.58 | 30.32 | 73% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.052 ms | 2.07 | 16.56 | 40% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.049 ms | 1.19 | 9.52 | 23% | ok |
| blas1 | sum | f64 | n=8388608 | 3.970 ms | 2.11 | 16.91 | 41% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 71.049 ms | 2.60 | 1.89 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.413 ms | 9.83 | 39.35 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 1.690 ms | 19.85 | 39.72 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.475 ms | 9.66 | 38.65 | 93% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.704 ms | 19.69 | 39.39 | 95% | ok |
| blas2 | symv | f64 | n=4096 | 2.932 ms | 11.44 | 22.91 | 55% | ok |
| blas2 | trmv | f64 | n=4096 | 1.783 ms | 9.41 | 37.68 | 91% | ok |
| blas2 | trmv | f32 | n=4096 | 0.861 ms | 19.48 | 39.00 | 94% | ok |
| blas2 | trsv | f64 | n=4096 | 1.821 ms | 9.21 | 36.89 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 0.869 ms | 19.30 | 38.63 | 93% | ok |
| blas2 | ger | f64 | n=4096 | 7.151 ms | 4.69 | 37.55 | 90% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.794 ms | 70.76 | 1.66 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.897 ms | 141.50 | 1.66 | 90% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.626 ms | 51.12 | 2.40 | 65% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.050 ms | 66.28 | 1.55 | 85% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.574 ms | 52.14 | 1.63 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.988 ms | 53.81 | 1.26 | 69% | ok |
| blas3 | trmm | f64 | 512x512x512 | 3.042 ms | 44.12 | 1.72 | 56% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.092 ms | 43.41 | 1.70 | 55% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.433 ms | 93.69 | 1.83 | 60% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.165 ms | 0.40 | 3.23 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.401 ms | 7.85 | 0.37 | 10% | ok |
| factor | lu | f32 | 512x512 | 3.347 ms | 26.73 | 0.63 | 17% | ok |
| factor | cholesky | f64 | 512x512 | 7.117 ms | 6.29 | 0.59 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.555 ms | 6.83 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.269 ms | 6.15 | 0.58 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 4.663 ms | 9.59 | 0.90 | 12% | ok |
| factor | qr | f64 | 512x512 | 231.678 ms | 0.77 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 23.380 ms | 7.65 | 0.18 | 10% | ok |
| factor | hessenberg | f64 | 512x512 | 267.380 ms | 1.67 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.631 ms | 0.64 | 2.57 | 6%\* | ok |
| factor | inv | f64 | 512x512 | 55.163 ms | 4.87 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.549 ms | 19.67 | 0.92 | 25% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.539 ms | 1.28 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 57.435 ms | 2.63 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 750.082 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 31.493 ms | 5.33 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 80.756 ms | 5.19 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 92.816 ms | 3.98 | 0.01 | 5% | ok |
| eigen | expm | f64 | 256x256 | 11.636 ms | 21.63 | 0.09 | 28% | ok |
| eigen | sylvester | f64 | 256x256 | 174.758 ms | 5.76 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.517 ms | 0.21 | 3.40 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 45.657 ms | 0.18 | 2.94 | 7% | ok |
| elementwise | sin | f64 | n=8388608 | 59.998 ms | 0.14 | 2.24 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.005 ms | 0.29 | 4.63 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 202.774 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.739 ms | 3.54 | 36.85 | 89% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.922 ms | 2.84 | 29.54 | 71% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.108 ms | 10.20 | 21.44 | 52% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.834 ms | 0.67 | 2.38 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.311 ms | 1.53 | 12.70 | 31%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 15.099 ms | 0.74 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 26.995 ms | 0.41 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.936 ms | 1.25 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.097 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.579 ms | 0.50 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.808 ms | 1.33 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 25.457 ms | 0.07 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.459 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.027 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.953 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.193 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.515 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.153 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 13.283 ms | 0.84 | 0.06 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.251 ms | 3.01 | 12.06 | 29%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.162 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.472 ms | 2.50 | 0.16 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.273 ms | 2.77 | 11.09 | 27%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.125 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.865 ms | 0.62 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.654 ms | 1.16 | 4.63 | 11%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.484 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.541 ms | 0.58 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.663 ms | 1.14 | 4.56 | 11%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.020 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 212.010 ms | 1.46 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.491 ms | 2.26 | 9.04 | 22%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.759 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 67.706 ms | 4.56 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.903 ms | 1.77 | 7.08 | 17%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.972 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 531.097 ms | 1.16 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.663 ms | 0.44 | 1.76 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.343 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 697.365 ms | 0.89 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.643 ms | 0.44 | 1.76 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.235 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 30.145 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 21.056 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.490 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.774 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.526 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 19.843 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.714 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.463 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.453 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.080 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 13.211 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 14.504 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.600 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.013 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.084 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.025 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.721 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.991 ms | 2.64 | 29.58 | 71% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.221 ms | 2.14 | 24.01 | 58% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.421 ms | 1.84 | 20.64 | 50% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.597 ms | 2.83 | 21.38 | 51% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 28.986 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 29.377 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.716 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.598 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.971 ms | 8.51 | 34.04 | 82% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 6.002 ms | 44.72 | 0.70 | 57% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.328 ms | 51.13 | 54.33 | 131%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.897 ms | 8.84 | 35.38 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.770 ms | 21.80 | 43.59 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.293 ms | 57.29 | 57.29 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.281 ms | 59.72 | 59.72 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.957 ms | 67.85 | 0.80 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.591 ms | 48.01 | 0.38 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.558 ms | 48.29 | 0.28 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 2.101 ms | 127.77 | 0.75 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 1.879 ms | 142.85 | 0.84 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

