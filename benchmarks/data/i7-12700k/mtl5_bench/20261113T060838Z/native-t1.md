## i7-12700k -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.4 GFLOP/s |
| bandwidth | read 33.4, update 38.6, triad 32.6 GB/s (ceiling 38.6) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 97% | 0 |
| blas2 | 10 | 80% | 96% | 0 |
| blas3 | 10 | 17% | 18% | 0 |
| factor | 12 | 8% | 18% | 0 |
| eigen | 8 | 7% | 12% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 95% | 0 |
| direct | 19 | 1% | 32% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 24% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 78% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 36% | 84% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.578 ms | 3.01 | 24.06 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.203 ms | 3.99 | 15.97 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.301 ms | 2.66 | 31.95 | 83% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.686 ms | 3.58 | 21.48 | 56% | ok |
| blas1 | scale | f64 | n=8388608 | 3.568 ms | 2.35 | 37.61 | 97% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.827 ms | 3.48 | 13.90 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 3.987 ms | 2.10 | 16.83 | 44% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.029 ms | 1.19 | 9.55 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.953 ms | 2.12 | 16.98 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.300 ms | 2.42 | 1.76 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.631 ms | 9.24 | 36.98 | 96% | ok |
| blas2 | gemv | f32 | n=4096 | 2.746 ms | 12.22 | 24.45 | 63% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.695 ms | 7.15 | 28.60 | 74% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.187 ms | 15.34 | 30.69 | 80% | ok |
| blas2 | symv | f64 | n=4096 | 6.577 ms | 5.10 | 10.21 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 2.006 ms | 8.36 | 33.48 | 87% | ok |
| blas2 | trmv | f32 | n=4096 | 1.417 ms | 11.84 | 23.70 | 61% | ok |
| blas2 | trsv | f64 | n=4096 | 1.969 ms | 8.52 | 34.12 | 88% | ok |
| blas2 | trsv | f32 | n=4096 | 1.383 ms | 12.13 | 24.29 | 63% | ok |
| blas2 | ger | f64 | n=4096 | 7.681 ms | 4.37 | 34.96 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 19.950 ms | 13.46 | 0.32 | 17% | ok |
| blas3 | gemm | f32 | 512x512x512 | 10.616 ms | 25.29 | 0.30 | 16% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 10.754 ms | 12.48 | 0.59 | 16% | ok |
| blas3 | symm | f64 | 512x512x512 | 18.723 ms | 14.34 | 0.34 | 18% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.637 ms | 9.84 | 0.31 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 24.459 ms | 10.98 | 0.26 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.876 ms | 13.59 | 0.53 | 17% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.767 ms | 13.74 | 0.54 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.176 ms | 25.93 | 0.51 | 17% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.080 ms | 0.41 | 3.25 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.536 ms | 7.76 | 0.36 | 10% | ok |
| factor | lu | f32 | 512x512 | 6.314 ms | 14.17 | 0.33 | 9% | ok |
| factor | cholesky | f64 | 512x512 | 7.301 ms | 6.13 | 0.57 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.574 ms | 6.81 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.416 ms | 6.03 | 0.57 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 45.877 ms | 0.98 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 229.889 ms | 0.78 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 25.827 ms | 6.93 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 269.801 ms | 1.66 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.650 ms | 0.64 | 2.54 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 54.323 ms | 4.94 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.213 ms | 14.40 | 0.68 | 18% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.872 ms | 1.25 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 58.694 ms | 2.57 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 746.021 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 32.512 ms | 5.16 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 82.255 ms | 5.10 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 111.286 ms | 3.32 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 26.338 ms | 9.55 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 188.447 ms | 5.34 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.278 ms | 0.21 | 3.42 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.669 ms | 0.18 | 2.94 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.009 ms | 0.14 | 2.24 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.876 ms | 0.29 | 4.65 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 201.442 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.746 ms | 3.51 | 36.52 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.928 ms | 2.82 | 29.33 | 76% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.636 ms | 9.03 | 18.99 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.843 ms | 0.67 | 2.37 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.383 ms | 1.45 | 12.04 | 31%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.135 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.488 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.897 ms | 1.26 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.069 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.724 ms | 0.42 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.852 ms | 1.33 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.545 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.434 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.078 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.224 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.121 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.355 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 88.624 ms | 0.13 | 0.01 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.247 ms | 3.06 | 12.24 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.493 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.171 ms | 2.68 | 0.18 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.251 ms | 3.01 | 12.05 | 31%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.091 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 36.452 ms | 0.61 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.625 ms | 1.21 | 4.84 | 13%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.932 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.663 ms | 0.47 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.623 ms | 1.21 | 4.86 | 13%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.480 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7431.880 ms | 0.04 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.428 ms | 2.36 | 9.44 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.787 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 67.422 ms | 4.58 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.446 ms | 2.33 | 9.32 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.887 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 554.834 ms | 1.11 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.497 ms | 0.45 | 1.80 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.648 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 644.323 ms | 0.96 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.543 ms | 0.45 | 1.79 | 5%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.249 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 33.274 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.893 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.567 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.091 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.107 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.769 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.389 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.397 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.333 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.539 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.976 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.258 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.595 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.413 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.076 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.010 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.715 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.979 ms | 2.67 | 29.97 | 78% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.217 ms | 2.15 | 24.09 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.562 ms | 1.68 | 18.77 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.274 ms | 2.90 | 21.90 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.793 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 44.903 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 35.809 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.039 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.898 ms | 3.43 | 13.70 | 36% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 39.306 ms | 6.83 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.549 ms | 30.56 | 32.47 | 84%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.285 ms | 7.34 | 29.36 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.987 ms | 16.99 | 33.98 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.721 ms | 9.75 | 9.75 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.721 ms | 9.75 | 9.75 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.455 ms | 15.38 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.803 ms | 12.90 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.224 ms | 26.26 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.230 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.017 ms | 9.94 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

