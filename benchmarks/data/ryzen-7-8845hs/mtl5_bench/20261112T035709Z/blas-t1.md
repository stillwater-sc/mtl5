## ryzen-7-8845hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 80.1 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.1, f32 160.1 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 42.8, update 53.5, triad 38.9 GB/s (ceiling 53.5) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 110% | 0 |
| blas2 | 10 | 77% | 84% | 0 |
| blas3 | 10 | 68% | 92% | 0 |
| factor | 12 | 13% | 23% | 0 |
| eigen | 8 | 17% | 53% | 0 |
| elementwise | 5 | 7% | 12% | 0 |
| sparse | 5 | 21% | 37% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 35% | 36% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 23% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.491 ms | 1.46 | 11.68 | 22% | ok |
| blas1 | dot | f32 | n=8388608 | 10.308 ms | 1.63 | 6.51 | 12% | ok |
| blas1 | axpy | f64 | n=8388608 | 11.930 ms | 1.41 | 16.88 | 32% | ok |
| blas1 | axpy | f32 | n=8388608 | 10.299 ms | 1.63 | 9.77 | 18% | ok |
| blas1 | scale | f64 | n=8388608 | 2.287 ms | 3.67 | 58.69 | 110% | ok |
| blas1 | two_norm | f64 | n=8388608 | 11.013 ms | 1.52 | 6.09 | 11% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.072 ms | 1.65 | 13.23 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 84.572 ms | 2.18 | 1.59 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.219 ms | 10.42 | 41.71 | 78% | ok |
| blas2 | gemv | f32 | n=4096 | 1.567 ms | 21.41 | 42.85 | 80% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.374 ms | 9.94 | 39.80 | 74% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.502 ms | 22.34 | 44.71 | 84% | ok |
| blas2 | symv | f64 | n=4096 | 1.826 ms | 18.38 | 36.79 | 69% | ok |
| blas2 | trmv | f64 | n=4096 | 1.626 ms | 10.32 | 41.32 | 77% | ok |
| blas2 | trmv | f32 | n=4096 | 0.847 ms | 19.81 | 39.67 | 74% | ok |
| blas2 | trsv | f64 | n=4096 | 1.614 ms | 10.40 | 41.63 | 78% | ok |
| blas2 | trsv | f32 | n=4096 | 0.908 ms | 18.49 | 37.01 | 69% | ok |
| blas2 | ger | f64 | n=4096 | 13.588 ms | 2.47 | 19.76 | 37% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.629 ms | 73.97 | 1.73 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.913 ms | 140.30 | 1.64 | 88% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.726 ms | 49.24 | 2.31 | 61% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.806 ms | 70.54 | 1.65 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 3.039 ms | 44.16 | 1.38 | 55% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.897 ms | 54.82 | 1.28 | 68% | ok |
| blas3 | trmm | f64 | 512x512x512 | 1.888 ms | 71.10 | 2.78 | 89% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.842 ms | 47.22 | 1.84 | 59% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.346 ms | 99.69 | 1.95 | 62% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.306 ms | 0.57 | 4.56 | 9% | ok |
| factor | lu | f64 | 512x512 | 4.986 ms | 17.94 | 0.84 | 22% | ok |
| factor | lu | f32 | 512x512 | 2.443 ms | 36.63 | 0.86 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 4.350 ms | 10.28 | 0.96 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 1.676 ms | 26.69 | 1.25 | 17% | ok |
| factor | ldlt | f64 | 512x512 | 11.785 ms | 3.80 | 0.36 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 164.303 ms | 0.27 | 0.03 | 0% | ok |
| factor | qr | f64 | 512x512 | 10.166 ms | 17.60 | 0.41 | 22% | ok |
| factor | lq | f64 | 512x512 | 64.507 ms | 2.77 | 0.07 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 745.299 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.878 ms | 0.27 | 1.08 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 60.512 ms | 4.44 | 0.07 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.538 ms | 16.16 | 0.76 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.678 ms | 13.33 | 0.63 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 129.208 ms | 1.17 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 499.815 ms | 0.30 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.673 ms | 10.06 | 0.06 | 13% | ok |
| eigen | schur | f64 | 256x256 | 21.833 ms | 19.21 | 0.05 | 24% | ok |
| eigen | svd | f64 | 256x256 | 8.649 ms | 42.68 | 0.12 | 53% | ok |
| eigen | expm | f64 | 256x256 | 15.903 ms | 15.82 | 0.07 | 20% | ok |
| eigen | sylvester | f64 | 256x256 | 72.035 ms | 13.97 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 27.847 ms | 0.30 | 4.82 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 43.053 ms | 0.19 | 3.12 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 38.927 ms | 0.22 | 3.45 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.365 ms | 0.39 | 6.28 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 36.179 ms | 0.23 | 3.71 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.424 ms | 1.08 | 11.23 | 21% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.370 ms | 1.91 | 19.88 | 37% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.594 ms | 6.35 | 13.35 | 25% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 17.840 ms | 0.18 | 0.64 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.752 ms | 1.15 | 9.50 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 103.386 ms | 0.11 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.551 ms | 0.37 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 17.140 ms | 0.65 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.899 ms | 0.41 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 73.333 ms | 0.31 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.679 ms | 0.87 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 79.417 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.651 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.609 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.427 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.147 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.253 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 46.171 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.072 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.261 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.413 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.946 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 54.792 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.736 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.221 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 24.339 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 23.512 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 26.343 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 32.786 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.401 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 22.279 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.839 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.764 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.321 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.588 ms | 1.65 | 18.47 | 35% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.540 ms | 1.70 | 19.04 | 36% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.791 ms | 1.46 | 16.38 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.984 ms | 2.41 | 18.18 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.228 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.894 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 57.155 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.998 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 11.424 ms | 1.47 | 5.87 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.514 ms | 9.10 | 0.14 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.478 ms | 11.35 | 12.06 | 23% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.132 ms | 5.36 | 21.42 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.878 ms | 5.83 | 11.66 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.394 ms | 12.04 | 12.04 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.394 ms | 12.04 | 12.04 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.394 ms | 31.98 | 0.37 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.301 ms | 8.86 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 30.554 ms | 8.79 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 30.558 ms | 8.78 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.237 ms | 9.86 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

