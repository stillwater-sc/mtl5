## 12th Gen Intel(R) Core(TM) i7-12700K -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `e7497e6e4c47` |
| FMA peak (ceiling) | f64 79.5 GFLOP/s, f32 159.2 GFLOP/s |
| ... hardware, AVX2+FMA | f64 79.5, f32 159.2 GFLOP/s |
| ... this build, AVX2 | f64 79.3, f32 159.1 GFLOP/s |
| bandwidth | read 35.0, update 36.9, triad 29.3 GB/s (ceiling 36.9) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 100% | 116% | 0 |
| blas2 | 6 | 103% | 109% | 0 |
| blas3 | 10 | 57% | 90% | 0 |
| factor | 12 | 8% | 19% | 0 |
| eigen | 8 | 7% | 26% | 0 |
| elementwise | 5 | 8% | 13% | 0 |
| sparse | 5 | 59% | 100% | 0 |
| direct | 7 | 1% | 2% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 81% | 0 |
| mg | 1 | 55% | 55% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 95% | 159% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 3.642 ms | 4.61 | 36.85 | 100% | ok |
| blas1 | dot | f32 | n=8388608 | 1.679 ms | 9.99 | 39.96 | 108% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.752 ms | 3.53 | 42.37 | 115% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.341 ms | 7.17 | 43.00 | 116% | ok |
| blas1 | scale | f64 | n=8388608 | 3.452 ms | 2.43 | 38.88 | 105% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.199 ms | 7.63 | 30.52 | 83% | ok |
| blas1 | one_norm | f64 | n=8388608 | 3.992 ms | 2.10 | 16.81 | 46% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.959 ms | 1.21 | 9.64 | 26% | ok |
| blas1 | sum | f64 | n=8388608 | 3.956 ms | 2.12 | 16.96 | 46% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 70.121 ms | 2.63 | 1.91 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.369 ms | 9.96 | 39.86 | 108% | ok |
| blas2 | gemv | f32 | n=4096 | 1.662 ms | 20.19 | 40.40 | 109% | ok |
| blas2 | symv | f64 | n=4096 | 6.173 ms | 5.44 | 10.88 | 29% | ok |
| blas2 | trmv | f64 | n=4096 | 1.762 ms | 9.52 | 38.13 | 103% | ok |
| blas2 | trsv | f64 | n=4096 | 1.794 ms | 9.35 | 37.43 | 101% | ok |
| blas2 | ger | f64 | n=4096 | 7.124 ms | 4.71 | 37.69 | 102% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.752 ms | 71.55 | 1.68 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.882 ms | 142.63 | 1.67 | 90% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.606 ms | 51.49 | 2.41 | 65% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.991 ms | 67.26 | 1.58 | 85% | ok |
| blas3 | syrk | f64 | 512x512x512 | 10.270 ms | 13.07 | 0.41 | 16% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 17.340 ms | 15.48 | 0.36 | 19% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.979 ms | 45.06 | 1.76 | 57% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.056 ms | 43.92 | 1.72 | 55% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.497 ms | 89.64 | 1.75 | 56% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.948 ms | 0.41 | 3.28 | 9% | ok |
| factor | lu | f64 | 512x512 | 10.277 ms | 8.71 | 0.41 | 11% | ok |
| factor | lu | f32 | 512x512 | 3.656 ms | 24.48 | 0.57 | 15% | ok |
| factor | cholesky | f64 | 512x512 | 6.902 ms | 6.48 | 0.61 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.454 ms | 6.93 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.059 ms | 6.34 | 0.59 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 44.039 ms | 1.02 | 0.10 | 1% | ok |
| factor | qr | f64 | 512x512 | 213.335 ms | 0.84 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 22.619 ms | 7.91 | 0.19 | 10% | ok |
| factor | hessenberg | f64 | 512x512 | 251.907 ms | 1.78 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.500 ms | 0.70 | 2.80 | 8%\* | ok |
| factor | inv | f64 | 512x512 | 63.004 ms | 4.26 | 0.07 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.004 ms | 14.90 | 0.70 | 19% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.429 ms | 1.36 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 56.560 ms | 2.67 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 746.278 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 30.163 ms | 5.56 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 78.935 ms | 5.31 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 90.369 ms | 4.08 | 0.01 | 5% | ok |
| eigen | expm | f64 | 256x256 | 12.306 ms | 20.45 | 0.09 | 26% | ok |
| eigen | sylvester | f64 | 256x256 | 174.692 ms | 5.76 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 38.764 ms | 0.22 | 3.46 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 46.015 ms | 0.18 | 2.92 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 58.790 ms | 0.14 | 2.28 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.525 ms | 0.29 | 4.71 | 13% | ok |
| elementwise | erf | f64 | n=8388608 | 201.148 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.734 ms | 3.56 | 37.07 | 100% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.922 ms | 2.84 | 29.53 | 80% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.032 ms | 10.39 | 21.84 | 59% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.793 ms | 0.68 | 2.40 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.222 ms | 1.64 | 13.62 | 37%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 71.264 ms | 0.16 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 26.851 ms | 0.42 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.842 ms | 1.27 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.262 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 55.531 ms | 0.40 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.562 ms | 1.35 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 88.892 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.362 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.922 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.221 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.041 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 27.630 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.153 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.512 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.447 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.672 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.562 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.223 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.564 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.415 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.542 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.823 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.826 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.529 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 11.810 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.065 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 0.982 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.691 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.976 ms | 2.68 | 30.06 | 81% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.286 ms | 2.04 | 22.80 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.506 ms | 1.74 | 19.47 | 53% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.210 ms | 2.71 | 20.45 | 55% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.754 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 38.760 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 39.083 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.512 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.921 ms | 8.73 | 34.93 | 95% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 5.986 ms | 44.84 | 0.70 | 56% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.304 ms | 55.11 | 58.55 | 159%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.914 ms | 8.77 | 35.07 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.753 ms | 22.29 | 44.57 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.275 ms | 60.94 | 60.94 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.255 ms | 65.75 | 65.75 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.942 ms | 68.10 | 0.80 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.523 ms | 48.60 | 0.38 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.499 ms | 48.81 | 0.29 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 2.084 ms | 128.80 | 0.75 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 1.860 ms | 144.35 | 0.85 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

