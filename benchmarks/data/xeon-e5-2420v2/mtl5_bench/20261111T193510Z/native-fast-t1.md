## xeon-e5-2420v2 -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.1 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.1 GFLOP/s |
| ... this build, SSE4 | f64 10.5, f32 21.0 GFLOP/s |
| bandwidth | read 10.4, update 10.4, triad 8.1 GB/s (ceiling 10.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 101% | 108% | 0 |
| blas2 | 10 | 91% | 100% | 0 |
| blas3 | 10 | 72% | 94% | 0 |
| factor | 12 | 17% | 40% | 0 |
| eigen | 8 | 15% | 47% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 106% | 0 |
| direct | 7 | 1% | 4% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 79% | 90% | 0 |
| mg | 1 | 78% | 78% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 75% | 89% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.950 ms | 1.40 | 11.23 | 108% | ok |
| blas1 | dot | f32 | n=8388608 | 5.963 ms | 2.81 | 11.25 | 108% | ok |
| blas1 | axpy | f64 | n=8388608 | 18.827 ms | 0.89 | 10.69 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 9.019 ms | 1.86 | 11.16 | 107% | ok |
| blas1 | scale | f64 | n=8388608 | 13.211 ms | 0.63 | 10.16 | 97% | ok |
| blas1 | two_norm | f64 | n=8388608 | 6.388 ms | 2.63 | 10.51 | 101% | ok |
| blas1 | one_norm | f64 | n=8388608 | 10.333 ms | 0.81 | 6.49 | 62% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.777 ms | 0.61 | 4.87 | 47% | ok |
| blas1 | sum | f64 | n=8388608 | 10.209 ms | 0.82 | 6.57 | 63% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 272.662 ms | 0.68 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 13.273 ms | 2.53 | 10.12 | 97% | ok |
| blas2 | gemv | f32 | n=4096 | 6.450 ms | 5.20 | 10.41 | 100% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 40.732 ms | 0.82 | 3.30 | 32% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 17.404 ms | 1.93 | 3.86 | 37% | ok |
| blas2 | symv | f64 | n=4096 | 16.138 ms | 2.08 | 4.16 | 40% | ok |
| blas2 | trmv | f64 | n=4096 | 7.059 ms | 2.38 | 9.52 | 91% | ok |
| blas2 | trmv | f32 | n=4096 | 3.526 ms | 4.76 | 9.53 | 91% | ok |
| blas2 | trsv | f64 | n=4096 | 7.066 ms | 2.37 | 9.51 | 91% | ok |
| blas2 | trsv | f32 | n=4096 | 3.614 ms | 4.64 | 9.29 | 89% | ok |
| blas2 | ger | f64 | n=4096 | 28.123 ms | 1.19 | 9.55 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 27.725 ms | 9.68 | 0.23 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.517 ms | 19.86 | 0.23 | 94% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.977 ms | 7.91 | 0.37 | 75% | ok |
| blas3 | symm | f64 | 512x512x512 | 28.856 ms | 9.30 | 0.22 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 36.868 ms | 3.64 | 0.11 | 35% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 63.609 ms | 4.22 | 0.10 | 40% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.787 ms | 7.55 | 0.29 | 72% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.178 ms | 7.38 | 0.29 | 70% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.881 ms | 15.11 | 0.30 | 72% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.540 ms | 0.14 | 1.10 | 11% | ok |
| factor | lu | f64 | 512x512 | 29.596 ms | 3.02 | 0.14 | 29% | ok |
| factor | lu | f32 | 512x512 | 16.040 ms | 5.58 | 0.13 | 27% | ok |
| factor | cholesky | f64 | 512x512 | 25.624 ms | 1.75 | 0.16 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 24.180 ms | 1.85 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.865 ms | 1.73 | 0.16 | 16% | ok |
| factor | ldlt_bk | f64 | 512x512 | 134.445 ms | 0.33 | 0.03 | 3% | ok |
| factor | qr | f64 | 512x512 | 497.393 ms | 0.36 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 74.031 ms | 2.42 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 614.277 ms | 0.73 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.170 ms | 0.33 | 1.32 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 153.982 ms | 1.74 | 0.03 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.144 ms | 4.23 | 0.20 | 40% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.777 ms | 0.37 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 173.587 ms | 0.87 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1002.472 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 104.061 ms | 1.61 | 0.01 | 15% | ok |
| eigen | schur | f64 | 256x256 | 254.869 ms | 1.65 | 0.00 | 16% | ok |
| eigen | svd | f64 | 256x256 | 259.287 ms | 1.42 | 0.00 | 14% | ok |
| eigen | expm | f64 | 256x256 | 50.632 ms | 4.97 | 0.02 | 47% | ok |
| eigen | sylvester | f64 | 256x256 | 575.084 ms | 1.75 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 151.136 ms | 0.06 | 0.89 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 140.081 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 171.322 ms | 0.05 | 0.78 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 90.254 ms | 0.09 | 1.49 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2470.162 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.464 ms | 1.06 | 11.05 | 106% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 4.124 ms | 0.63 | 6.60 | 63% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.072 ms | 2.78 | 5.84 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.446 ms | 0.20 | 0.70 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.822 ms | 0.34 | 2.86 | 27% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 238.908 ms | 0.05 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 71.511 ms | 0.16 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 27.030 ms | 0.41 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.001 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 154.406 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.760 ms | 0.46 | 0.01 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 234.125 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.063 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.162 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.620 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.620 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 79.629 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 110.372 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 64.425 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 33.612 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 37.815 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 36.870 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 68.906 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 74.107 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.183 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.287 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 41.879 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 44.653 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 53.810 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.129 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.476 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.710 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.866 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.771 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.075 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.068 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.578 ms | 0.73 | 8.20 | 79% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.113 ms | 0.84 | 9.42 | 90% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.799 ms | 0.69 | 7.72 | 74% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.497 ms | 1.08 | 8.19 | 78% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 139.647 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 181.001 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 152.067 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.329 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 7.204 ms | 2.33 | 9.32 | 89% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 62.449 ms | 4.30 | 0.07 | 41% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.266 ms | 7.40 | 7.87 | 75% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.039 ms | 2.78 | 11.11 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.985 ms | 5.62 | 11.24 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.085 ms | 8.05 | 8.05 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.677 ms | 10.01 | 10.01 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.952 ms | 14.16 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.131 ms | 12.13 | 0.09 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.000 ms | 12.20 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 16.194 ms | 16.58 | 0.10 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 12.365 ms | 21.71 | 0.13 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

