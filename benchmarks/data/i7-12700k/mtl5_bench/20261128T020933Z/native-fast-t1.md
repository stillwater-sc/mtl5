## i7-12700k -- native-fast, 1 thread, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.5 GFLOP/s |
| ... this build, AVX2 | f64 78.3, f32 156.6 GFLOP/s |
| bandwidth | read 37.6, update 41.2, triad 31.3 GB/s (ceiling 41.2) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 86% | 101% | 0 |
| blas2 | 10 | 91% | 94% | 0 |
| blas3 | 10 | 77% | 93% | 0 |
| factor | 12 | 7% | 15% | 0 |
| eigen | 8 | 3% | 25% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 51% | 75% | 0 |
| direct | 19 | 1% | 16% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 15% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 57% | 65% | 0 |
| mg | 1 | 42% | 42% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 79% | 81% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 15.179 ms | 4.42 | 35.37 | 86% | ok |
| blas1 | dot | f32 | n=33554432 | 7.504 ms | 8.94 | 35.77 | 87% | ok |
| blas1 | axpy | f64 | n=33554432 | 19.452 ms | 3.45 | 41.40 | 100% | ok |
| blas1 | axpy | f32 | n=33554432 | 9.670 ms | 6.94 | 41.64 | 101% | ok |
| blas1 | scale | f64 | n=33554432 | 13.936 ms | 2.41 | 38.52 | 93% | ok |
| blas1 | two_norm | f64 | n=33554432 | 9.587 ms | 7.00 | 28.00 | 68% | ok |
| blas1 | one_norm | f64 | n=33554432 | 16.475 ms | 2.04 | 16.29 | 40% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 28.258 ms | 1.19 | 9.50 | 23% | ok |
| blas1 | sum | f64 | n=33554432 | 16.238 ms | 2.07 | 16.53 | 40% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 300.098 ms | 2.46 | 1.79 | 4% | ok |
| blas2 | gemv | f64 | n=8192 | 13.853 ms | 9.69 | 38.77 | 94% | ok |
| blas2 | gemv | f32 | n=8192 | 6.919 ms | 19.40 | 38.80 | 94% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 14.233 ms | 9.43 | 37.73 | 91% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 7.083 ms | 18.95 | 37.91 | 92% | ok |
| blas2 | symv | f64 | n=8192 | 12.371 ms | 10.85 | 21.71 | 53% | ok |
| blas2 | trmv | f64 | n=8192 | 7.221 ms | 9.29 | 37.19 | 90% | ok |
| blas2 | trmv | f32 | n=8192 | 3.666 ms | 18.31 | 36.63 | 89% | ok |
| blas2 | trsv | f64 | n=8192 | 7.238 ms | 9.27 | 37.11 | 90% | ok |
| blas2 | trsv | f32 | n=8192 | 3.684 ms | 18.22 | 36.45 | 88% | ok |
| blas2 | ger | f64 | n=8192 | 28.616 ms | 4.69 | 37.53 | 91% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 30.187 ms | 71.14 | 0.83 | 91% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 14.784 ms | 145.25 | 0.85 | 93% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 18.962 ms | 56.63 | 1.33 | 72% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 32.722 ms | 65.63 | 0.77 | 84% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 17.611 ms | 60.97 | 0.95 | 78% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 35.836 ms | 59.92 | 0.70 | 77% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 21.872 ms | 49.09 | 0.96 | 63% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 22.469 ms | 47.79 | 0.93 | 61% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 10.556 ms | 101.72 | 0.99 | 65% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 44.062 ms | 0.38 | 3.05 | 7% | ok |
| factor | lu | f64 | 1024x1024 | 100.981 ms | 7.09 | 0.17 | 9% | ok |
| factor | lu | f32 | 1024x1024 | 56.055 ms | 12.77 | 0.15 | 8% | ok |
| factor | cholesky | f64 | 1024x1024 | 68.725 ms | 5.21 | 0.24 | 7% | ok |
| factor | cholesky | f32 | 1024x1024 | 62.246 ms | 5.75 | 0.13 | 4% | ok |
| factor | ldlt | f64 | 1024x1024 | 69.084 ms | 5.18 | 0.24 | 7% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 40.661 ms | 8.80 | 0.41 | 11% | ok |
| factor | qr | f64 | 1024x1024 | 2257.084 ms | 0.63 | 0.01 | 1% | ok |
| factor | lq | f64 | 1024x1024 | 212.880 ms | 6.73 | 0.08 | 9% | ok |
| factor | hessenberg | f64 | 1024x1024 | 2563.502 ms | 1.40 | 0.01 | 2% | ok |
| factor | householder | f64 | 1024x1024 | 6.767 ms | 0.62 | 2.48 | 6%\* | ok |
| factor | inv | f64 | 1024x1024 | 443.943 ms | 4.84 | 0.04 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 61.666 ms | 11.61 | 0.27 | 15% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 270.060 ms | 0.66 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 949.429 ms | 1.27 | 0.00 | 2% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 12249.069 ms | 0.10 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 440.368 ms | 3.05 | 0.01 | 4% | ok |
| eigen | schur | f64 | 512x512 | 1384.994 ms | 2.42 | 0.00 | 3% | ok |
| eigen | svd | f64 | 512x512 | 1091.818 ms | 2.70 | 0.00 | 3% | ok |
| eigen | expm | f64 | 512x512 | 103.623 ms | 19.43 | 0.04 | 25% | ok |
| eigen | sylvester | f64 | 512x512 | 2863.032 ms | 2.81 | 0.00 | 4% | ok |
| elementwise | exp | f64 | n=33554432 | 164.428 ms | 0.20 | 3.27 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 189.045 ms | 0.18 | 2.84 | 7% | ok |
| elementwise | sin | f64 | n=33554432 | 246.677 ms | 0.14 | 2.18 | 5% | ok |
| elementwise | sqrt | f64 | n=33554432 | 121.972 ms | 0.28 | 4.40 | 11% | ok |
| elementwise | erf | f64 | n=33554432 | 814.944 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 3.531 ms | 2.97 | 30.86 | 75% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.161 ms | 2.52 | 26.19 | 64% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 16.837 ms | 9.96 | 20.92 | 51% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 29.848 ms | 0.44 | 1.54 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 14.570 ms | 0.70 | 5.75 | 14% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 104.648 ms | 1.02 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 196.812 ms | 0.54 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 53.525 ms | 2.00 | 0.05 | 3% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 303.558 ms | 0.70 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 328.090 ms | 0.65 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 129.289 ms | 1.66 | 0.02 | 2% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 379.909 ms | 0.06 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 17.144 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 24.033 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 266.752 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.975 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 92.525 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.628 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 97.259 ms | 1.10 | 0.03 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.458 ms | 1.63 | 6.53 | 16%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 18.350 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 32.341 ms | 3.31 | 0.09 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.449 ms | 1.64 | 6.56 | 16%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 8.252 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 279.193 ms | 0.77 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.806 ms | 0.46 | 1.82 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 13.503 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 288.542 ms | 0.74 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.783 ms | 0.46 | 1.83 | 4%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 9.005 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 619.941 ms | 1.56 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 4.846 ms | 1.55 | 6.19 | 15% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 19.392 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 207.903 ms | 4.66 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 5.244 ms | 1.43 | 5.72 | 14% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 9.008 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1781.876 ms | 1.09 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.512 ms | 0.38 | 1.54 | 4% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 13.207 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 2214.990 ms | 0.88 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.473 ms | 0.39 | 1.54 | 4% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 230.277 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 253.264 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 131.516 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 106.506 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 113.154 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 103.577 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 208.839 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 163.192 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 113.227 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 116.802 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 103.002 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 107.140 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 124.791 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.145 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 6.341 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 51.283 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 4.385 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 4.116 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.122 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 2.869 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.410 ms | 2.38 | 26.62 | 65% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.994 ms | 2.10 | 23.50 | 57% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 5.803 ms | 1.81 | 20.23 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 68.002 ms | 2.29 | 17.18 | 42% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 295.222 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 296.065 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 510.910 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 19.243 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 8.235 ms | 8.15 | 32.60 | 79% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 47.454 ms | 45.25 | 0.35 | 58% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.126 ms | 31.57 | 33.55 | 81% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 7.994 ms | 8.39 | 33.58 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 3.908 ms | 17.17 | 34.34 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 2.098 ms | 31.99 | 31.99 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 2.095 ms | 32.04 | 32.04 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 31.357 ms | 68.49 | 0.40 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 44.057 ms | 48.74 | 0.19 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 44.006 ms | 48.80 | 0.14 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 16.361 ms | 131.26 | 0.38 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 14.645 ms | 146.63 | 0.43 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

