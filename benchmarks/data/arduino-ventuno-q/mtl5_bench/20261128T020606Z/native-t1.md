## arduino-ventuno-q -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, scalar | f64 8.4, f32 8.4 GFLOP/s |
| bandwidth | read 22.6, update 24.5, triad 23.2 GB/s (ceiling 24.5) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 85% | 104% | 0 |
| blas2 | 10 | 61% | 70% | 0 |
| blas3 | 10 | 30% | 34% | 0 |
| factor | 12 | 13% | 32% | 0 |
| eigen | 8 | 15% | 22% | 0 |
| elementwise | 5 | 5% | 7% | 0 |
| sparse | 5 | 30% | 68% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 10% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 47% | 57% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.672 ms | 2.96 | 23.66 | 96% | ok |
| blas1 | dot | f32 | n=8388608 | 4.518 ms | 3.71 | 14.85 | 61% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.858 ms | 2.13 | 25.62 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.969 ms | 4.23 | 25.36 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.418 ms | 1.55 | 24.77 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.224 ms | 5.20 | 20.82 | 85% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.173 ms | 1.03 | 8.21 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.324 ms | 0.48 | 3.87 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.566 ms | 0.98 | 7.83 | 32% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 200.306 ms | 0.92 | 0.67 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 7.986 ms | 4.20 | 16.82 | 69% | ok |
| blas2 | gemv | f32 | n=4096 | 5.809 ms | 5.78 | 11.56 | 47% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.852 ms | 3.41 | 13.63 | 56% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.475 ms | 7.50 | 15.00 | 61% | ok |
| blas2 | symv | f64 | n=4096 | 11.716 ms | 2.86 | 5.73 | 23% | ok |
| blas2 | trmv | f64 | n=4096 | 4.509 ms | 3.72 | 14.90 | 61% | ok |
| blas2 | trmv | f32 | n=4096 | 3.249 ms | 5.16 | 10.34 | 42% | ok |
| blas2 | trsv | f64 | n=4096 | 4.415 ms | 3.80 | 15.22 | 62% | ok |
| blas2 | trsv | f32 | n=4096 | 3.086 ms | 5.44 | 10.88 | 44% | ok |
| blas2 | ger | f64 | n=4096 | 15.620 ms | 2.15 | 17.19 | 70% | ok |
| blas3 | gemm | f64 | 512x512x512 | 52.761 ms | 5.09 | 0.12 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 24.086 ms | 11.14 | 0.13 | 33% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 39.547 ms | 3.39 | 0.16 | 20% | ok |
| blas3 | symm | f64 | 512x512x512 | 54.091 ms | 4.96 | 0.12 | 30% | ok |
| blas3 | syrk | f64 | 512x512x512 | 42.106 ms | 3.19 | 0.10 | 19% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 81.945 ms | 3.28 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 26.196 ms | 5.12 | 0.20 | 31% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.587 ms | 5.05 | 0.20 | 30% | ok |
| blas3 | trsm | f32 | 512x512x512 | 11.893 ms | 11.29 | 0.22 | 34% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 32.361 ms | 0.16 | 1.31 | 5% | ok |
| factor | lu | f64 | 512x512 | 24.010 ms | 3.73 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 12.960 ms | 6.90 | 0.16 | 21% | ok |
| factor | cholesky | f64 | 512x512 | 20.816 ms | 2.15 | 0.20 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 19.622 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.530 ms | 1.99 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 11.094 ms | 4.03 | 0.38 | 24% | ok |
| factor | qr | f64 | 512x512 | 249.325 ms | 0.72 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.916 ms | 2.94 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 347.113 ms | 1.29 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.556 ms | 0.68 | 2.70 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.943 ms | 1.84 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.730 ms | 5.35 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.987 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.179 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 602.184 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.332 ms | 2.49 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 149.921 ms | 2.80 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.213 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 66.732 ms | 3.77 | 0.02 | 22% | ok |
| eigen | sylvester | f64 | 256x256 | 369.990 ms | 2.72 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 89.502 ms | 0.09 | 1.50 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 112.186 ms | 0.07 | 1.20 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 149.856 ms | 0.06 | 0.90 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 73.211 ms | 0.11 | 1.83 | 7% | ok |
| elementwise | erf | f64 | n=8388608 | 119.469 ms | 0.07 | 1.12 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.624 ms | 1.61 | 16.77 | 68% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.133 ms | 1.23 | 12.76 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.062 ms | 3.47 | 7.30 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.016 ms | 0.25 | 0.88 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.073 ms | 0.22 | 1.83 | 7% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 37.890 ms | 0.30 | 0.02 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.971 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.415 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 102.428 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 120.038 ms | 0.19 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.094 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 73.666 ms | 0.03 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.374 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.204 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 124.520 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.571 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.372 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.496 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 29.783 ms | 0.38 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.839 ms | 0.41 | 1.65 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.185 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.702 ms | 0.96 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.753 ms | 0.43 | 1.72 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.420 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 87.425 ms | 0.26 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.945 ms | 0.13 | 0.51 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.627 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 101.002 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.867 ms | 0.13 | 0.52 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 11.701 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 679.890 ms | 0.45 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.748 ms | 0.43 | 1.74 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 27.737 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 182.563 ms | 1.69 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.674 ms | 0.44 | 1.76 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.773 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1784.986 ms | 0.35 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 30.502 ms | 0.11 | 0.44 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.766 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1190.063 ms | 0.52 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 30.527 ms | 0.11 | 0.44 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.698 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 116.626 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.317 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.931 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.741 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.380 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 62.754 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.614 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.645 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 39.344 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.898 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.736 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 40.825 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.114 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.996 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.578 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.141 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.056 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.078 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.750 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.086 ms | 1.25 | 14.06 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.533 ms | 1.03 | 11.58 | 47% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.973 ms | 0.88 | 9.86 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 31.933 ms | 1.20 | 9.10 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 231.468 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 231.737 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 146.534 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.336 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.047 ms | 2.08 | 8.34 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.624 ms | 5.52 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.939 ms | 5.71 | 6.07 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.850 ms | 5.89 | 23.55 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.399 ms | 12.00 | 23.99 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.655 ms | 4.59 | 4.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.158 ms | 14.49 | 14.49 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 25.536 ms | 10.51 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.357 ms | 10.18 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.306 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.307 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.516 ms | 6.97 | 0.04 | n/a | ok |

