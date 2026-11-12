## xeon-e5-2420v2 -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, SSE4 | f64 10.5, f32 20.9 GFLOP/s |
| bandwidth | read 10.4, update 10.2, triad 8.0 GB/s (ceiling 10.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 104% | 109% | 0 |
| blas2 | 10 | 91% | 101% | 0 |
| blas3 | 10 | 73% | 95% | 0 |
| factor | 12 | 17% | 39% | 0 |
| eigen | 8 | 15% | 47% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 107% | 0 |
| direct | 7 | 2% | 4% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 79% | 91% | 0 |
| mg | 1 | 79% | 79% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 76% | 90% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.873 ms | 1.41 | 11.30 | 109% | ok |
| blas1 | dot | f32 | n=8388608 | 5.948 ms | 2.82 | 11.28 | 109% | ok |
| blas1 | axpy | f64 | n=8388608 | 18.714 ms | 0.90 | 10.76 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 9.108 ms | 1.84 | 11.05 | 106% | ok |
| blas1 | scale | f64 | n=8388608 | 12.452 ms | 0.67 | 10.78 | 104% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.929 ms | 2.83 | 11.32 | 109% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.731 ms | 0.86 | 6.90 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.074 ms | 0.64 | 5.13 | 49% | ok |
| blas1 | sum | f64 | n=8388608 | 9.746 ms | 0.86 | 6.89 | 66% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 275.357 ms | 0.67 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 13.002 ms | 2.58 | 10.33 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 6.429 ms | 5.22 | 10.44 | 101% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 37.452 ms | 0.90 | 3.59 | 35% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 16.981 ms | 1.98 | 3.95 | 38% | ok |
| blas2 | symv | f64 | n=4096 | 16.333 ms | 2.05 | 4.11 | 40% | ok |
| blas2 | trmv | f64 | n=4096 | 6.984 ms | 2.40 | 9.62 | 93% | ok |
| blas2 | trmv | f32 | n=4096 | 3.597 ms | 4.66 | 9.34 | 90% | ok |
| blas2 | trsv | f64 | n=4096 | 6.878 ms | 2.44 | 9.77 | 94% | ok |
| blas2 | trsv | f32 | n=4096 | 3.606 ms | 4.65 | 9.32 | 90% | ok |
| blas2 | ger | f64 | n=4096 | 28.433 ms | 1.18 | 9.44 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 27.833 ms | 9.64 | 0.23 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.450 ms | 19.96 | 0.23 | 95% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.922 ms | 7.93 | 0.37 | 76% | ok |
| blas3 | symm | f64 | 512x512x512 | 28.905 ms | 9.29 | 0.22 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 36.965 ms | 3.63 | 0.11 | 35% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 63.544 ms | 4.22 | 0.10 | 40% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.583 ms | 7.63 | 0.30 | 73% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.220 ms | 7.37 | 0.29 | 70% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.841 ms | 15.18 | 0.30 | 72% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.806 ms | 0.14 | 1.09 | 11% | ok |
| factor | lu | f64 | 512x512 | 29.511 ms | 3.03 | 0.14 | 29% | ok |
| factor | lu | f32 | 512x512 | 16.382 ms | 5.46 | 0.13 | 26% | ok |
| factor | cholesky | f64 | 512x512 | 25.477 ms | 1.76 | 0.16 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 24.502 ms | 1.83 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.790 ms | 1.73 | 0.16 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 147.732 ms | 0.30 | 0.03 | 3% | ok |
| factor | qr | f64 | 512x512 | 504.336 ms | 0.35 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 73.246 ms | 2.44 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 613.815 ms | 0.73 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.197 ms | 0.33 | 1.31 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 153.929 ms | 1.74 | 0.03 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.812 ms | 4.10 | 0.19 | 39% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.166 ms | 0.37 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 171.982 ms | 0.88 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 983.557 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 108.691 ms | 1.54 | 0.01 | 15% | ok |
| eigen | schur | f64 | 256x256 | 260.876 ms | 1.61 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 263.758 ms | 1.40 | 0.00 | 13% | ok |
| eigen | expm | f64 | 256x256 | 51.273 ms | 4.91 | 0.02 | 47% | ok |
| eigen | sylvester | f64 | 256x256 | 582.486 ms | 1.73 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 151.940 ms | 0.06 | 0.88 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 139.980 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 171.195 ms | 0.05 | 0.78 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 90.635 ms | 0.09 | 1.48 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2471.853 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.458 ms | 1.06 | 11.08 | 107% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 4.118 ms | 0.64 | 6.61 | 64% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.122 ms | 2.77 | 5.82 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.362 ms | 0.20 | 0.70 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.796 ms | 0.35 | 2.87 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 239.595 ms | 0.05 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.342 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.482 ms | 0.42 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 118.167 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 155.544 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 49.093 ms | 0.46 | 0.01 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 234.179 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.074 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.277 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.649 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.344 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.659 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 112.556 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 64.660 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 32.333 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.260 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.303 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 69.815 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 74.553 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.737 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.895 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 42.497 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 45.418 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 54.655 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.130 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.507 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.337 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.885 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.806 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.060 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.586 ms | 0.73 | 8.18 | 79% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.096 ms | 0.85 | 9.47 | 91% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.806 ms | 0.69 | 7.71 | 74% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.622 ms | 1.08 | 8.16 | 79% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.101 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 181.276 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 148.458 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.226 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 7.161 ms | 2.34 | 9.37 | 90% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 62.050 ms | 4.33 | 0.07 | 41% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.270 ms | 7.39 | 7.85 | 76% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.008 ms | 2.79 | 11.17 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.998 ms | 5.60 | 11.19 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.080 ms | 8.07 | 8.07 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.646 ms | 10.19 | 10.19 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.762 ms | 14.31 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.024 ms | 12.19 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.034 ms | 12.18 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 16.107 ms | 16.67 | 0.10 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 12.410 ms | 21.63 | 0.13 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

