## i7-12700k -- native, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 600.8 GFLOP/s, f32 1201.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.8, f32 1201.6 GFLOP/s |
| ... this build, scalar | f64 9.9, f32 10.7 GFLOP/s |
| bandwidth | read 84.1, update 79.1, triad 77.1 GB/s (ceiling 84.1) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 83% | 122% | 0 | 28% |
| blas2 | 10 | 41% | 137% | 0 | 13% |
| blas3 | 10 | 2% | 15% | 0 | 14% |
| factor | 12 | 1% | 12% | 0 | 19% |
| eigen | 8 | 1% | 4% | 0 | 21% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 288% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 36% | 0 | 13% |
| mg | 1 | 25% | 25% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 16% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.584 ms | 10.59 | 84.76 | 101% | ok | 3.6x (45%) |
| blas1 | dot | f32 | n=8388608 | 0.655 ms | 25.62 | 102.47 | 122% | ok | 6.6x (82%) |
| blas1 | axpy | f64 | n=8388608 | 2.859 ms | 5.87 | 70.42 | 84% | ok | 2.2x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.440 ms | 11.65 | 69.90 | 83% | ok | 3.3x (41%) |
| blas1 | scale | f64 | n=8388608 | 1.922 ms | 4.36 | 69.82 | 83% | ok | 1.9x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.741 ms | 22.64 | 90.55 | 108% | ok | 6.5x (81%) |
| blas1 | one_norm | f64 | n=8388608 | 3.962 ms | 2.12 | 16.94 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.935 ms | 1.21 | 9.68 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.918 ms | 2.14 | 17.13 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 74.506 ms | 2.48 | 1.80 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.425 ms | 23.55 | 94.26 | 112% | ok | 2.6x (32%) |
| blas2 | gemv | f32 | n=4096 | 0.583 ms | 57.59 | 115.23 | 137% | ok | 4.7x (58%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.553 ms | 21.60 | 86.46 | 103% | ok | 3.0x (37%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.861 ms | 38.97 | 77.98 | 93% | ok | 2.6x (32%) |
| blas2 | symv | f64 | n=4096 | 6.496 ms | 5.17 | 10.34 | 12% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.006 ms | 8.36 | 33.49 | 40% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.507 ms | 11.13 | 22.29 | 27% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.995 ms | 8.41 | 33.68 | 40% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.513 ms | 11.09 | 22.20 | 26% | ok | 0.9x (12%) |
| blas2 | ger | f64 | n=4096 | 7.868 ms | 4.26 | 34.13 | 41% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.897 ms | 92.66 | 2.17 | 15% | ok | 7.0x (87%) |
| blas3 | gemm | f32 | 512x512x512 | 1.563 ms | 171.74 | 2.01 | 14% | ok | 6.8x (85%) |
| blas3 | gemmt | f64 | 512x512x512 | 9.045 ms | 14.84 | 0.70 | 2% | ok | 1.2x (16%) |
| blas3 | symm | f64 | 512x512x512 | 16.892 ms | 15.89 | 0.37 | 3% | ok | 1.2x (14%) |
| blas3 | syrk | f64 | 512x512x512 | 11.452 ms | 11.72 | 0.37 | 2% | ok | 1.1x (14%) |
| blas3 | syr2k | f64 | 512x512x512 | 19.362 ms | 13.86 | 0.32 | 2% | ok | 1.2x (15%) |
| blas3 | trmm | f64 | 512x512x512 | 9.169 ms | 14.64 | 0.57 | 2% | ok | 1.1x (13%) |
| blas3 | trsm | f64 | 512x512x512 | 9.205 ms | 14.58 | 0.57 | 2% | ok | 1.1x (14%) |
| blas3 | trsm | f32 | 512x512x512 | 5.041 ms | 26.62 | 0.52 | 2% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.737 ms | 0.42 | 3.33 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 7.464 ms | 11.99 | 0.56 | 2% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 3.911 ms | 22.88 | 0.54 | 2% | ok | 1.0x (13%) |
| factor | cholesky | f64 | 512x512 | 7.080 ms | 6.32 | 0.59 | 1% | ok | 1.1x (13%) |
| factor | cholesky | f32 | 512x512 | 6.442 ms | 6.95 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.193 ms | 6.22 | 0.58 | 1% | ok | 1.1x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 45.577 ms | 0.98 | 0.09 | 0% | ok | 1.1x (14%) |
| factor | qr | f64 | 512x512 | 128.138 ms | 1.40 | 0.03 | 0% | ok | 1.7x (22%) |
| factor | lq | f64 | 512x512 | 17.514 ms | 10.22 | 0.24 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 167.391 ms | 2.67 | 0.03 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 0.413 ms | 2.54 | 10.16 | 12%\* | ok | 4.0x (50%) |
| factor | inv | f64 | 512x512 | 51.461 ms | 5.22 | 0.08 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.021 ms | 14.86 | 0.70 | 2% | ok | 1.5x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 18.560 ms | 1.21 | 0.06 | 0% | ok | 1.7x (21%) |
| eigen | eigen_symmetric | f64 | 256x256 | 58.958 ms | 2.56 | 0.02 | 0% | ok | 1.7x (22%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 747.431 ms | 0.20 | 0.00 | 0% | ok | 1.5x (19%) |
| eigen | eigenvalue | f64 | 256x256 | 33.426 ms | 5.02 | 0.03 | 1% | ok | 1.5x (19%) |
| eigen | schur | f64 | 256x256 | 81.829 ms | 5.13 | 0.01 | 1% | ok | 1.8x (22%) |
| eigen | svd | f64 | 256x256 | 109.066 ms | 3.38 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 11.682 ms | 21.54 | 0.09 | 4% | ok | 2.4x (30%) |
| eigen | sylvester | f64 | 256x256 | 173.356 ms | 5.81 | 0.01 | 1% | ok | 1.1x (14%) |
| elementwise | exp | f64 | n=8388608 | 38.182 ms | 0.22 | 3.52 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 44.372 ms | 0.19 | 3.02 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.290 ms | 0.14 | 2.30 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.020 ms | 0.30 | 4.79 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 200.326 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.112 ms | 23.30 | 242.38 | 288% ! | ok | 6.7x (83%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.908 ms | 2.88 | 29.98 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.403 ms | 29.85 | 62.76 | 75% | ok | 3.3x (42%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.849 ms | 0.67 | 2.37 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.393 ms | 1.44 | 11.95 | 14%\* | ok | 0.9x (12%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 90.099 ms | 0.12 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.922 ms | 0.43 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.933 ms | 1.25 | 0.08 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.282 ms | 0.56 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.511 ms | 0.42 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.431 ms | 1.28 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 88.744 ms | 0.02 | 0.00 | 0% | ok | 1.1x (14%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.295 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.893 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.225 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.849 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.617 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 31.909 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.555 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.263 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.637 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.708 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.165 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.809 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.987 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.944 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.068 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.534 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.103 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.033 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.527 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 12.057 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.983 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.020 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.678 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.972 ms | 2.69 | 30.19 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.199 ms | 2.18 | 24.46 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.534 ms | 1.71 | 19.11 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.654 ms | 2.82 | 21.29 | 25% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.183 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 43.610 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 39.107 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.977 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.860 ms | 3.45 | 13.81 | 16% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.201 ms | 51.61 | 0.81 | 9% | ok | 7.5x (94%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.532 ms | 31.55 | 33.52 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.243 ms | 7.48 | 29.92 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.961 ms | 17.47 | 34.93 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.685 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.522 ms | 106.45 | 1.25 | n/a | ok | 6.9x (87%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.838 ms | 94.57 | 0.74 | n/a | ok | 7.3x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.91 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.357 ms | 197.82 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.527 ms | 76.11 | 0.45 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

