## ryzen-7-8845hs -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 79.9 GFLOP/s, f32 159.3 GFLOP/s |
| ... hardware, AVX-512F | f64 79.9, f32 159.3 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 43.1, update 53.2, triad 38.9 GB/s (ceiling 53.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 110% | 0 |
| blas2 | 10 | 53% | 80% | 0 |
| blas3 | 10 | 15% | 21% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 11% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 26% | 60% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 34% | 35% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 14% | 24% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 9.804 ms | 1.71 | 13.69 | 26% | ok |
| blas1 | dot | f32 | n=8388608 | 9.699 ms | 1.73 | 6.92 | 13% | ok |
| blas1 | axpy | f64 | n=8388608 | 11.852 ms | 1.42 | 16.99 | 32% | ok |
| blas1 | axpy | f32 | n=8388608 | 11.804 ms | 1.42 | 8.53 | 16% | ok |
| blas1 | scale | f64 | n=8388608 | 2.297 ms | 3.65 | 58.42 | 110% | ok |
| blas1 | two_norm | f64 | n=8388608 | 9.280 ms | 1.81 | 7.23 | 14% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.075 ms | 1.65 | 13.22 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.063 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 88.197 ms | 2.09 | 1.52 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.174 ms | 10.57 | 42.30 | 80% | ok |
| blas2 | gemv | f32 | n=4096 | 2.379 ms | 14.11 | 28.22 | 53% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.062 ms | 4.75 | 19.02 | 36% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.215 ms | 15.15 | 30.32 | 57% | ok |
| blas2 | symv | f64 | n=4096 | 8.589 ms | 3.91 | 7.82 | 15% | ok |
| blas2 | trmv | f64 | n=4096 | 1.811 ms | 9.26 | 37.09 | 70% | ok |
| blas2 | trmv | f32 | n=4096 | 1.508 ms | 11.13 | 22.28 | 42% | ok |
| blas2 | trsv | f64 | n=4096 | 1.790 ms | 9.37 | 37.54 | 71% | ok |
| blas2 | trsv | f32 | n=4096 | 1.564 ms | 10.73 | 21.48 | 40% | ok |
| blas2 | ger | f64 | n=4096 | 13.479 ms | 2.49 | 19.92 | 37% | ok |
| blas3 | gemm | f64 | 512x512x512 | 15.726 ms | 17.07 | 0.40 | 21% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.255 ms | 32.52 | 0.38 | 20% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.034 ms | 12.16 | 0.57 | 15% | ok |
| blas3 | symm | f64 | 512x512x512 | 15.954 ms | 16.83 | 0.39 | 21% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.480 ms | 9.96 | 0.31 | 12% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 22.836 ms | 11.76 | 0.28 | 15% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.617 ms | 5.24 | 0.20 | 7% | ok |
| blas3 | trsm | f64 | 512x512x512 | 28.121 ms | 4.77 | 0.19 | 6% | ok |
| blas3 | trsm | f32 | 512x512x512 | 27.018 ms | 4.97 | 0.10 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.193 ms | 0.58 | 4.62 | 9% | ok |
| factor | lu | f64 | 512x512 | 32.361 ms | 2.76 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 31.115 ms | 2.88 | 0.07 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.379 ms | 3.93 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 11.036 ms | 4.05 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.464 ms | 3.90 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 162.628 ms | 0.28 | 0.03 | 0% | ok |
| factor | qr | f64 | 512x512 | 645.984 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 64.987 ms | 2.75 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 741.265 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.880 ms | 0.27 | 1.08 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 89.804 ms | 2.99 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 34.068 ms | 2.63 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.480 ms | 0.60 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 125.038 ms | 1.21 | 0.01 | 2% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 507.004 ms | 0.30 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 58.866 ms | 2.85 | 0.02 | 4% | ok |
| eigen | schur | f64 | 256x256 | 115.454 ms | 3.63 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 118.009 ms | 3.13 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 29.955 ms | 8.40 | 0.04 | 11% | ok |
| eigen | sylvester | f64 | 256x256 | 264.371 ms | 3.81 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 32.711 ms | 0.26 | 4.10 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 42.275 ms | 0.20 | 3.17 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 39.091 ms | 0.21 | 3.43 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.923 ms | 0.38 | 6.12 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 31.902 ms | 0.26 | 4.21 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.852 ms | 3.07 | 31.98 | 60% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.258 ms | 2.08 | 21.64 | 41% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.324 ms | 6.62 | 13.92 | 26% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.678 ms | 0.20 | 0.69 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.170 ms | 0.92 | 7.67 | 14%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 106.776 ms | 0.10 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.083 ms | 0.36 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 17.583 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.119 ms | 0.41 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 71.682 ms | 0.31 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 26.848 ms | 0.83 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 75.885 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.611 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.483 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.414 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.017 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 39.164 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 43.702 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 30.540 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.637 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.655 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.615 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 50.818 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 33.763 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.759 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.684 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.838 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.251 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 31.233 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.043 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.333 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 21.896 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.794 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.710 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.651 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.618 ms | 1.62 | 18.12 | 34% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.554 ms | 1.68 | 18.87 | 35% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.785 ms | 1.47 | 16.43 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 16.048 ms | 2.40 | 18.11 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.324 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 46.556 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 56.593 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.729 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 9.288 ms | 1.81 | 7.23 | 14% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.084 ms | 9.23 | 0.14 | 12% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.371 ms | 12.24 | 13.01 | 24% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.122 ms | 5.37 | 21.50 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.877 ms | 5.83 | 11.66 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.374 ms | 12.21 | 12.21 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.375 ms | 12.20 | 12.20 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.044 ms | 33.37 | 0.39 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.301 ms | 8.86 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 30.602 ms | 8.77 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 30.587 ms | 8.78 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.303 ms | 9.83 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

