## i7-12700k -- native, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 600.6 GFLOP/s, f32 1201.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.6, f32 1201.5 GFLOP/s |
| ... this build, scalar | f64 9.9, f32 10.7 GFLOP/s |
| bandwidth | read 83.5, update 78.6, triad 76.6 GB/s (ceiling 83.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 84% | 123% | 0 | 28% |
| blas2 | 10 | 42% | 138% | 0 | 13% |
| blas3 | 10 | 2% | 16% | 0 | 13% |
| factor | 12 | 1% | 12% | 0 | 13% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 282% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 36% | 0 | 13% |
| mg | 1 | 25% | 25% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.475 ms | 11.37 | 90.97 | 109% | ok | 4.0x (50%) |
| blas1 | dot | f32 | n=8388608 | 0.653 ms | 25.68 | 102.72 | 123% | ok | 6.5x (81%) |
| blas1 | axpy | f64 | n=8388608 | 2.834 ms | 5.92 | 71.04 | 85% | ok | 2.2x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.453 ms | 11.55 | 69.29 | 83% | ok | 3.3x (41%) |
| blas1 | scale | f64 | n=8388608 | 1.910 ms | 4.39 | 70.28 | 84% | ok | 1.9x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.742 ms | 22.62 | 90.47 | 108% | ok | 6.6x (82%) |
| blas1 | one_norm | f64 | n=8388608 | 3.977 ms | 2.11 | 16.87 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.932 ms | 1.21 | 9.68 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.980 ms | 2.11 | 16.86 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.714 ms | 2.44 | 1.77 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.435 ms | 23.39 | 93.59 | 112% | ok | 2.6x (33%) |
| blas2 | gemv | f32 | n=4096 | 0.582 ms | 57.69 | 115.44 | 138% | ok | 4.7x (59%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.560 ms | 21.51 | 86.06 | 103% | ok | 3.1x (39%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.866 ms | 38.73 | 77.51 | 93% | ok | 2.6x (32%) |
| blas2 | symv | f64 | n=4096 | 6.506 ms | 5.16 | 10.32 | 12% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.022 ms | 8.30 | 33.22 | 40% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.433 ms | 11.71 | 23.45 | 28% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 2.019 ms | 8.31 | 33.27 | 40% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.394 ms | 12.04 | 24.09 | 29% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.645 ms | 4.39 | 35.12 | 42% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.796 ms | 96.02 | 2.25 | 16% | ok | 7.2x (90%) |
| blas3 | gemm | f32 | 512x512x512 | 1.553 ms | 172.80 | 2.03 | 14% | ok | 6.8x (85%) |
| blas3 | gemmt | f64 | 512x512x512 | 10.229 ms | 13.12 | 0.62 | 2% | ok | 1.1x (13%) |
| blas3 | symm | f64 | 512x512x512 | 19.274 ms | 13.93 | 0.33 | 2% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 12.753 ms | 10.52 | 0.33 | 2% | ok | 1.0x (13%) |
| blas3 | syr2k | f64 | 512x512x512 | 20.393 ms | 13.16 | 0.31 | 2% | ok | 1.2x (14%) |
| blas3 | trmm | f64 | 512x512x512 | 10.655 ms | 12.60 | 0.49 | 2% | ok | 0.9x (11%) |
| blas3 | trsm | f64 | 512x512x512 | 10.717 ms | 12.52 | 0.49 | 2% | ok | 0.9x (11%) |
| blas3 | trsm | f32 | 512x512x512 | 5.508 ms | 24.37 | 0.48 | 2% | ok | 0.9x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.892 ms | 0.41 | 3.29 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 7.486 ms | 11.95 | 0.56 | 2% | ok | 1.6x (19%) |
| factor | lu | f32 | 512x512 | 4.133 ms | 21.65 | 0.51 | 2% | ok | 1.0x (13%) |
| factor | cholesky | f64 | 512x512 | 7.098 ms | 6.30 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.453 ms | 6.93 | 0.32 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.189 ms | 6.22 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 45.056 ms | 0.99 | 0.09 | 0% | ok | 1.1x (13%) |
| factor | qr | f64 | 512x512 | 128.974 ms | 1.39 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 17.508 ms | 10.22 | 0.24 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 165.489 ms | 2.70 | 0.03 | 0% | ok | 1.7x (21%) |
| factor | householder | f64 | 512x512 | 0.421 ms | 2.50 | 9.97 | 12%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 51.398 ms | 5.22 | 0.08 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.092 ms | 14.69 | 0.69 | 2% | ok | 1.0x (13%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.763 ms | 1.26 | 0.06 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.948 ms | 2.61 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 736.881 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 32.627 ms | 5.14 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 80.316 ms | 5.22 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 106.789 ms | 3.46 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 11.613 ms | 21.67 | 0.09 | 4% | ok | 2.3x (28%) |
| eigen | sylvester | f64 | 256x256 | 173.171 ms | 5.81 | 0.01 | 1% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.840 ms | 0.22 | 3.46 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.429 ms | 0.18 | 2.95 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.971 ms | 0.14 | 2.28 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.520 ms | 0.29 | 4.71 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 200.401 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.116 ms | 22.60 | 235.16 | 282% ! | ok | 6.4x (81%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.904 ms | 2.90 | 30.14 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.411 ms | 29.68 | 62.41 | 75% | ok | 3.3x (41%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.817 ms | 0.68 | 2.39 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.387 ms | 1.45 | 12.01 | 14%\* | ok | 0.9x (12%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 90.206 ms | 0.12 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.387 ms | 0.44 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.009 ms | 1.24 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.926 ms | 0.56 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 52.716 ms | 0.42 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 21.140 ms | 1.06 | 0.03 | 0% | ok | 0.8x (10%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 90.302 ms | 0.02 | 0.00 | 0% | ok | 1.1x (14%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.383 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.849 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.226 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.831 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.473 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.177 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.466 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.218 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.611 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.713 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.332 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.570 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.908 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.870 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.153 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.591 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 20.939 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.515 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 12.056 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.048 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.985 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.020 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.675 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.972 ms | 2.69 | 30.17 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.199 ms | 2.18 | 24.45 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.536 ms | 1.70 | 19.09 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.745 ms | 2.80 | 21.15 | 25% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.658 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 43.775 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 34.846 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.921 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.805 ms | 3.49 | 13.97 | 17% | ok | 1.1x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.211 ms | 51.52 | 0.80 | 9% | ok | 7.5x (94%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.533 ms | 31.46 | 33.43 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.299 ms | 7.30 | 29.20 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.997 ms | 16.82 | 33.65 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.686 ms | 9.95 | 9.95 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.686 ms | 9.95 | 9.95 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.497 ms | 107.49 | 1.26 | n/a | ok | 7.0x (87%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.802 ms | 95.80 | 0.75 | n/a | ok | 7.4x (93%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.94 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.357 ms | 197.89 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.532 ms | 76.01 | 0.45 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

