## ryzen-7-8845hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 80.0 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.0, f32 160.1 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 43.2, update 53.5, triad 38.8 GB/s (ceiling 53.5) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 110% | 0 |
| blas2 | 10 | 76% | 90% | 0 |
| blas3 | 10 | 71% | 92% | 0 |
| factor | 12 | 13% | 23% | 0 |
| eigen | 8 | 17% | 53% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 25% | 61% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 34% | 37% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 23% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.471 ms | 1.46 | 11.70 | 22% | ok |
| blas1 | dot | f32 | n=8388608 | 9.772 ms | 1.72 | 6.87 | 13% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.040 ms | 1.39 | 16.72 | 31% | ok |
| blas1 | axpy | f32 | n=8388608 | 10.255 ms | 1.64 | 9.82 | 18% | ok |
| blas1 | scale | f64 | n=8388608 | 2.283 ms | 3.67 | 58.78 | 110% | ok |
| blas1 | two_norm | f64 | n=8388608 | 10.584 ms | 1.59 | 6.34 | 12% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.070 ms | 1.65 | 13.24 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 89.225 ms | 2.07 | 1.50 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.217 ms | 10.43 | 41.74 | 78% | ok |
| blas2 | gemv | f32 | n=4096 | 1.573 ms | 21.33 | 42.68 | 80% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.210 ms | 10.45 | 41.83 | 78% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.388 ms | 24.17 | 48.37 | 90% | ok |
| blas2 | symv | f64 | n=4096 | 1.813 ms | 18.51 | 37.05 | 69% | ok |
| blas2 | trmv | f64 | n=4096 | 1.646 ms | 10.19 | 40.82 | 76% | ok |
| blas2 | trmv | f32 | n=4096 | 0.863 ms | 19.44 | 38.91 | 73% | ok |
| blas2 | trsv | f64 | n=4096 | 1.780 ms | 9.42 | 37.73 | 71% | ok |
| blas2 | trsv | f32 | n=4096 | 0.925 ms | 18.13 | 36.30 | 68% | ok |
| blas2 | ger | f64 | n=4096 | 13.593 ms | 2.47 | 19.75 | 37% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.630 ms | 73.96 | 1.73 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.887 ms | 142.27 | 1.67 | 89% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.730 ms | 49.16 | 2.30 | 61% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.777 ms | 71.08 | 1.67 | 89% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.925 ms | 45.89 | 1.43 | 57% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.740 ms | 56.63 | 1.33 | 71% | ok |
| blas3 | trmm | f64 | 512x512x512 | 1.888 ms | 71.07 | 2.78 | 89% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.809 ms | 47.77 | 1.87 | 60% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.328 ms | 101.09 | 1.97 | 63% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.594 ms | 0.55 | 4.43 | 8% | ok |
| factor | lu | f64 | 512x512 | 4.986 ms | 17.94 | 0.84 | 22% | ok |
| factor | lu | f32 | 512x512 | 2.423 ms | 36.94 | 0.87 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 4.366 ms | 10.25 | 0.96 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 1.678 ms | 26.66 | 1.25 | 17% | ok |
| factor | ldlt | f64 | 512x512 | 11.813 ms | 3.79 | 0.36 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 163.637 ms | 0.27 | 0.03 | 0% | ok |
| factor | qr | f64 | 512x512 | 10.232 ms | 17.49 | 0.41 | 22% | ok |
| factor | lq | f64 | 512x512 | 64.712 ms | 2.77 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 741.036 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.894 ms | 0.27 | 1.08 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 60.728 ms | 4.42 | 0.07 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.533 ms | 16.17 | 0.76 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.670 ms | 13.40 | 0.63 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 129.184 ms | 1.17 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 505.111 ms | 0.30 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.748 ms | 10.02 | 0.06 | 13% | ok |
| eigen | schur | f64 | 256x256 | 21.953 ms | 19.11 | 0.05 | 24% | ok |
| eigen | svd | f64 | 256x256 | 8.660 ms | 42.62 | 0.12 | 53% | ok |
| eigen | expm | f64 | 256x256 | 16.114 ms | 15.62 | 0.07 | 20% | ok |
| eigen | sylvester | f64 | 256x256 | 73.897 ms | 13.62 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 28.026 ms | 0.30 | 4.79 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 43.694 ms | 0.19 | 3.07 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 39.203 ms | 0.21 | 3.42 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.901 ms | 0.38 | 6.13 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 35.048 ms | 0.24 | 3.83 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.838 ms | 3.12 | 32.51 | 61% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.260 ms | 2.08 | 21.61 | 40% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.533 ms | 6.41 | 13.48 | 25% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.213 ms | 0.20 | 0.71 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.770 ms | 1.13 | 9.41 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 102.856 ms | 0.11 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.253 ms | 0.37 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 17.319 ms | 0.65 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.818 ms | 0.42 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 72.251 ms | 0.31 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.912 ms | 0.86 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 80.136 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.639 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.563 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.417 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 13.929 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.256 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 45.978 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.183 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.229 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.340 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.973 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 54.827 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.678 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.181 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 24.296 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 23.479 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 26.430 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 32.682 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.414 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 24.125 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.824 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.741 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.316 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.597 ms | 1.64 | 18.36 | 34% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.461 ms | 1.79 | 20.07 | 37% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.783 ms | 1.47 | 16.45 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.889 ms | 2.42 | 18.29 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.001 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.685 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 53.435 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.908 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 11.377 ms | 1.47 | 5.90 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.622 ms | 9.06 | 0.14 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.469 ms | 11.42 | 12.13 | 23% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.214 ms | 5.22 | 20.88 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.880 ms | 5.83 | 11.65 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.387 ms | 12.09 | 12.09 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.387 ms | 12.10 | 12.10 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.328 ms | 32.23 | 0.38 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.305 ms | 8.86 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 30.582 ms | 8.78 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 30.607 ms | 8.77 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.278 ms | 9.84 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

