## ryzen-7-8845hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 630.6 GFLOP/s, f32 1260.3 GFLOP/s |
| ... hardware, AVX-512F | f64 630.6, f32 1260.3 GFLOP/s |
| ... this build, scalar | f64 14.4, f32 14.3 GFLOP/s |
| bandwidth | read 53.1, update 53.4, triad 41.1 GB/s (ceiling 53.4) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 120% | 0 | 42% |
| blas2 | 10 | 71% | 109% | 0 | 13% |
| blas3 | 10 | 2% | 17% | 0 | 12% |
| factor | 12 | 1% | 8% | 0 | 17% |
| eigen | 8 | 0% | 2% | 0 | 12% |
| elementwise | 5 | 8% | 11% | 0 | 12% |
| sparse | 5 | 39% | 256% | 0 | 16% |
| direct | 7 | 0% | 0% | 0 | 12% |
| ordering | 4 | n/a | n/a | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 30% | 33% | 0 | 11% |
| mg | 1 | 35% | 35% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 13% | 24% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 81% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.315 ms | 7.25 | 57.97 | 109% | ok | 4.2x (53%) |
| blas1 | dot | f32 | n=8388608 | 1.349 ms | 12.43 | 49.74 | 93% | ok | 7.2x (90%) |
| blas1 | axpy | f64 | n=8388608 | 3.521 ms | 4.76 | 57.17 | 107% | ok | 3.4x (42%) |
| blas1 | axpy | f32 | n=8388608 | 1.836 ms | 9.14 | 54.82 | 103% | ok | 6.4x (80%) |
| blas1 | scale | f64 | n=8388608 | 2.088 ms | 4.02 | 64.28 | 120% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=8388608 | 1.294 ms | 12.97 | 51.87 | 97% | ok | 7.2x (90%) |
| blas1 | one_norm | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.074 ms | 1.65 | 13.23 | 25% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 87.533 ms | 2.11 | 1.53 | 3% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.362 ms | 14.20 | 56.84 | 106% | ok | 1.3x (17%) |
| blas2 | gemv | f32 | n=4096 | 1.149 ms | 29.20 | 58.42 | 109% | ok | 2.1x (26%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.307 ms | 14.54 | 58.20 | 109% | ok | 3.1x (38%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.169 ms | 28.70 | 57.42 | 107% | ok | 1.9x (24%) |
| blas2 | symv | f64 | n=4096 | 8.528 ms | 3.93 | 7.88 | 15% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.779 ms | 9.43 | 37.77 | 71% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.540 ms | 10.90 | 21.81 | 41% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.764 ms | 9.51 | 38.08 | 71% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.563 ms | 10.73 | 21.48 | 40% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 13.551 ms | 2.48 | 19.81 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.487 ms | 107.94 | 2.53 | 17% | ok | 6.3x (79%) |
| blas3 | gemm | f32 | 512x512x512 | 1.230 ms | 218.26 | 2.56 | 17% | ok | 6.7x (84%) |
| blas3 | gemmt | f64 | 512x512x512 | 11.297 ms | 11.88 | 0.56 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 16.042 ms | 16.73 | 0.39 | 3% | ok | 1.0x (12%) |
| blas3 | syrk | f64 | 512x512x512 | 13.714 ms | 9.79 | 0.31 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 23.268 ms | 11.54 | 0.27 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 25.172 ms | 5.33 | 0.21 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f64 | 512x512x512 | 27.071 ms | 4.96 | 0.19 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 28.089 ms | 4.78 | 0.09 | 0% | ok | 1.0x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.585 ms | 0.55 | 4.43 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 26.602 ms | 3.36 | 0.16 | 1% | ok | 1.2x (15%) |
| factor | lu | f32 | 512x512 | 21.944 ms | 4.08 | 0.10 | 0% | ok | 1.4x (18%) |
| factor | cholesky | f64 | 512x512 | 11.426 ms | 3.92 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.930 ms | 4.09 | 0.19 | 0%\* | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 11.464 ms | 3.90 | 0.37 | 1%\* | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 169.655 ms | 0.26 | 0.02 | 0%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 407.152 ms | 0.44 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 44.795 ms | 4.00 | 0.09 | 1% | ok | 1.5x (18%) |
| factor | hessenberg | f64 | 512x512 | 479.837 ms | 0.93 | 0.01 | 0% | ok | 1.5x (19%) |
| factor | householder | f64 | 512x512 | 1.019 ms | 1.03 | 4.12 | 8%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 78.886 ms | 3.40 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 24.968 ms | 3.58 | 0.17 | 1% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.700 ms | 0.59 | 0.03 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 129.571 ms | 1.17 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 513.566 ms | 0.29 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 59.034 ms | 2.84 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 113.511 ms | 3.70 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 118.902 ms | 3.10 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 19.576 ms | 12.86 | 0.05 | 2% | ok | 1.5x (19%) |
| eigen | sylvester | f64 | 256x256 | 260.615 ms | 3.86 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 31.514 ms | 0.27 | 4.26 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 42.606 ms | 0.20 | 3.15 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 39.222 ms | 0.21 | 3.42 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.911 ms | 0.38 | 6.13 | 11% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 32.216 ms | 0.26 | 4.17 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.199 ms | 13.15 | 136.84 | 256% ! | ok | 4.3x (53%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.293 ms | 2.02 | 21.06 | 39% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.268 ms | 18.47 | 38.82 | 73% | ok | 2.8x (35%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.669 ms | 0.20 | 0.69 | 1%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.682 ms | 1.19 | 9.90 | 19%\* | ok | 1.3x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 104.859 ms | 0.11 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 32.011 ms | 0.35 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 17.339 ms | 0.65 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.218 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 71.693 ms | 0.31 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.643 ms | 0.87 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 76.585 ms | 0.02 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.663 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.553 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.422 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.061 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 39.194 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 43.698 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 30.565 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.630 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.635 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.566 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 51.080 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 33.962 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.814 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.682 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.865 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.315 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 31.071 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.372 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 22.739 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.810 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.754 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.652 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.645 ms | 1.59 | 17.83 | 33% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.843 ms | 1.42 | 15.91 | 30% | ok | 0.8x (11%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.032 ms | 1.29 | 14.43 | 27% | ok | 0.9x (11%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.472 ms | 2.49 | 18.79 | 35% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.123 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 46.676 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 54.638 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.790 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 9.326 ms | 1.80 | 7.20 | 13% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.006 ms | 67.02 | 1.05 | 11% | ok | 7.3x (91%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.386 ms | 12.11 | 12.87 | 24% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.121 ms | 5.38 | 21.51 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.881 ms | 5.82 | 11.65 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.381 ms | 12.15 | 12.15 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.381 ms | 12.15 | 12.15 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.248 ms | 215.06 | 2.52 | n/a | ok | 6.4x (81%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.153 ms | 64.64 | 0.51 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.176 ms | 64.28 | 0.38 | n/a | ok | 7.3x (92%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.178 ms | 64.26 | 0.38 | n/a | ok | 7.3x (92%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.775 ms | 71.10 | 0.42 | n/a | ok | 7.2x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

