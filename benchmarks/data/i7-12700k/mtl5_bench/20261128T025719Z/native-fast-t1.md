## i7-12700k -- native-fast, 1 thread, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.5 GFLOP/s |
| ... this build, AVX2 | f64 78.2, f32 156.4 GFLOP/s |
| bandwidth | read 37.6, update 41.1, triad 31.4 GB/s (ceiling 41.1) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 86% | 101% | 0 |
| blas2 | 10 | 91% | 94% | 0 |
| blas3 | 10 | 76% | 92% | 0 |
| factor | 12 | 7% | 17% | 0 |
| eigen | 8 | 3% | 18% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 37% | 0 |
| direct | 19 | 1% | 15% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 11% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 55% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 78% | 79% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 15.250 ms | 4.40 | 35.20 | 86% | ok |
| blas1 | dot | f32 | n=33554432 | 7.581 ms | 8.85 | 35.41 | 86% | ok |
| blas1 | axpy | f64 | n=33554432 | 19.536 ms | 3.44 | 41.22 | 100% | ok |
| blas1 | axpy | f32 | n=33554432 | 9.725 ms | 6.90 | 41.40 | 101% | ok |
| blas1 | scale | f64 | n=33554432 | 13.999 ms | 2.40 | 38.35 | 93% | ok |
| blas1 | two_norm | f64 | n=33554432 | 9.713 ms | 6.91 | 27.64 | 67% | ok |
| blas1 | one_norm | f64 | n=33554432 | 16.555 ms | 2.03 | 16.21 | 39% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 28.326 ms | 1.18 | 9.48 | 23% | ok |
| blas1 | sum | f64 | n=33554432 | 16.289 ms | 2.06 | 16.48 | 40% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 302.506 ms | 2.44 | 1.77 | 4% | ok |
| blas2 | gemv | f64 | n=8192 | 13.892 ms | 9.66 | 38.66 | 94% | ok |
| blas2 | gemv | f32 | n=8192 | 6.925 ms | 19.38 | 38.77 | 94% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 14.316 ms | 9.38 | 37.51 | 91% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 7.109 ms | 18.88 | 37.77 | 92% | ok |
| blas2 | symv | f64 | n=8192 | 12.387 ms | 10.84 | 21.68 | 53% | ok |
| blas2 | trmv | f64 | n=8192 | 7.259 ms | 9.25 | 37.00 | 90% | ok |
| blas2 | trmv | f32 | n=8192 | 3.688 ms | 18.20 | 36.41 | 89% | ok |
| blas2 | trsv | f64 | n=8192 | 7.387 ms | 9.09 | 36.36 | 89% | ok |
| blas2 | trsv | f32 | n=8192 | 3.727 ms | 18.01 | 36.03 | 88% | ok |
| blas2 | ger | f64 | n=8192 | 28.817 ms | 4.66 | 37.27 | 91% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 30.792 ms | 69.74 | 0.82 | 89% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 14.963 ms | 143.52 | 0.84 | 92% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 19.359 ms | 55.47 | 1.30 | 71% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 33.496 ms | 64.11 | 0.75 | 82% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 17.705 ms | 60.65 | 0.95 | 77% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 36.086 ms | 59.51 | 0.70 | 76% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 22.225 ms | 48.31 | 0.94 | 62% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 22.879 ms | 46.93 | 0.92 | 60% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 10.538 ms | 101.89 | 1.00 | 65% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 44.912 ms | 0.37 | 2.99 | 7% | ok |
| factor | lu | f64 | 1024x1024 | 101.362 ms | 7.06 | 0.17 | 9% | ok |
| factor | lu | f32 | 1024x1024 | 55.916 ms | 12.80 | 0.15 | 8% | ok |
| factor | cholesky | f64 | 1024x1024 | 68.867 ms | 5.20 | 0.24 | 7% | ok |
| factor | cholesky | f32 | 1024x1024 | 62.368 ms | 5.74 | 0.13 | 4% | ok |
| factor | ldlt | f64 | 1024x1024 | 69.250 ms | 5.17 | 0.24 | 7% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 40.301 ms | 8.88 | 0.42 | 11% | ok |
| factor | qr | f64 | 1024x1024 | 2270.336 ms | 0.63 | 0.01 | 1% | ok |
| factor | lq | f64 | 1024x1024 | 213.061 ms | 6.72 | 0.08 | 9% | ok |
| factor | hessenberg | f64 | 1024x1024 | 2584.049 ms | 1.39 | 0.01 | 2% | ok |
| factor | householder | f64 | 1024x1024 | 7.991 ms | 0.53 | 2.10 | 5%\* | ok |
| factor | inv | f64 | 1024x1024 | 463.873 ms | 4.63 | 0.04 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 52.549 ms | 13.62 | 0.32 | 17% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 256.205 ms | 0.70 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 886.876 ms | 1.36 | 0.00 | 2% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 11928.339 ms | 0.10 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 462.052 ms | 2.90 | 0.01 | 4% | ok |
| eigen | schur | f64 | 512x512 | 1411.903 ms | 2.38 | 0.00 | 3% | ok |
| eigen | svd | f64 | 512x512 | 1022.945 ms | 2.89 | 0.00 | 4% | ok |
| eigen | expm | f64 | 512x512 | 145.395 ms | 13.85 | 0.03 | 18% | ok |
| eigen | sylvester | f64 | 512x512 | 3095.851 ms | 2.60 | 0.00 | 3% | ok |
| elementwise | exp | f64 | n=33554432 | 273.473 ms | 0.12 | 1.96 | 5% | ok |
| elementwise | log | f64 | n=33554432 | 286.450 ms | 0.12 | 1.87 | 5% | ok |
| elementwise | sin | f64 | n=33554432 | 378.494 ms | 0.09 | 1.42 | 3% | ok |
| elementwise | sqrt | f64 | n=33554432 | 154.165 ms | 0.22 | 3.48 | 8% | ok |
| elementwise | erf | f64 | n=33554432 | 1083.069 ms | 0.03 | 0.50 | 1% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 7.522 ms | 1.39 | 14.49 | 35% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 7.257 ms | 1.44 | 15.02 | 37% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 28.996 ms | 5.78 | 12.15 | 30% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 44.638 ms | 0.29 | 1.03 | 3% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 22.656 ms | 0.45 | 3.70 | 9% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 165.637 ms | 0.65 | 0.02 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 206.622 ms | 0.52 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 56.631 ms | 1.89 | 0.05 | 2% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 311.663 ms | 0.69 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 330.467 ms | 0.65 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 129.718 ms | 1.65 | 0.02 | 2% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 385.105 ms | 0.06 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 17.515 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 25.175 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 268.570 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.948 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 94.502 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.876 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 97.606 ms | 1.10 | 0.03 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.688 ms | 1.49 | 5.97 | 15%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 18.721 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 32.660 ms | 3.28 | 0.09 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.668 ms | 1.50 | 6.02 | 15%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 8.461 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 280.380 ms | 0.76 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 9.070 ms | 0.44 | 1.77 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 13.709 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 297.360 ms | 0.72 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 9.091 ms | 0.44 | 1.77 | 4%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 9.077 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 727.844 ms | 1.33 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 6.897 ms | 1.09 | 4.35 | 11% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 22.678 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 222.881 ms | 4.35 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 7.145 ms | 1.05 | 4.20 | 10% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 9.835 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 2410.790 ms | 0.80 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 20.502 ms | 0.37 | 1.46 | 4% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 13.491 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 2294.596 ms | 0.84 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 21.940 ms | 0.34 | 1.37 | 3% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 280.855 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 405.971 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 174.036 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 142.344 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 184.582 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 162.165 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 354.436 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 250.091 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 155.031 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 151.261 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 136.983 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 176.245 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 208.809 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.267 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 11.230 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 85.718 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 7.475 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 7.619 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.271 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 5.091 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 6.736 ms | 1.56 | 17.42 | 42% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 5.241 ms | 2.00 | 22.39 | 55% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.009 ms | 1.74 | 19.53 | 48% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 71.716 ms | 2.17 | 16.29 | 40% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 333.285 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 305.222 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 503.649 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 19.177 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 8.408 ms | 7.98 | 31.93 | 78% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 48.388 ms | 44.38 | 0.35 | 57% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.199 ms | 30.51 | 32.42 | 79% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 8.147 ms | 8.24 | 32.95 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 3.979 ms | 16.86 | 33.73 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 2.176 ms | 30.83 | 30.83 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 2.121 ms | 31.64 | 31.64 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 35.641 ms | 60.25 | 0.35 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 48.279 ms | 44.48 | 0.17 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 48.281 ms | 44.48 | 0.13 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 18.240 ms | 117.73 | 0.34 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 15.950 ms | 134.64 | 0.39 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

