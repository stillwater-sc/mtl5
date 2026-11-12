## i7-12700k -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 600.4 GFLOP/s, f32 1200.9 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.4, f32 1200.9 GFLOP/s |
| ... this build, scalar | f64 10.1, f32 10.0 GFLOP/s |
| bandwidth | read 82.1, update 77.5, triad 75.7 GB/s (ceiling 82.1) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 86% | 125% | 0 | 28% |
| blas2 | 10 | 119% | 197% | 0 | 31% |
| blas3 | 10 | 66% | 85% | 0 | 78% |
| factor | 12 | 5% | 12% | 0 | 17% |
| eigen | 8 | 3% | 10% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 294% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 36% | 0 | 13% |
| mg | 1 | 25% | 25% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 39% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 90% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.463 ms | 11.46 | 91.72 | 112% | ok | 3.8x (48%) |
| blas1 | dot | f32 | n=8388608 | 0.655 ms | 25.60 | 102.40 | 125% | ok | 6.4x (81%) |
| blas1 | axpy | f64 | n=8388608 | 2.831 ms | 5.93 | 71.12 | 87% | ok | 2.3x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.426 ms | 11.76 | 70.57 | 86% | ok | 3.5x (43%) |
| blas1 | scale | f64 | n=8388608 | 1.910 ms | 4.39 | 70.26 | 86% | ok | 1.9x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.751 ms | 22.35 | 89.38 | 109% | ok | 6.5x (81%) |
| blas1 | one_norm | f64 | n=8388608 | 4.025 ms | 2.08 | 16.67 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.961 ms | 1.21 | 9.64 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.976 ms | 2.11 | 16.88 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 76.839 ms | 2.40 | 1.75 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.441 ms | 23.28 | 93.16 | 113% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.557 ms | 60.21 | 120.49 | 147% | ok | 2.9x (37%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.448 ms | 23.17 | 92.71 | 113% | ok | 2.5x (31%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.690 ms | 48.65 | 97.35 | 119% | ok | 2.4x (30%) |
| blas2 | symv | f64 | n=4096 | 0.626 ms | 53.58 | 107.27 | 131% | ok | 3.1x (38%) |
| blas2 | trmv | f64 | n=4096 | 0.605 ms | 27.74 | 111.08 | 135% | ok | 3.1x (38%) |
| blas2 | trmv | f32 | n=4096 | 0.207 ms | 80.89 | 161.93 | 197% ! | ok | 4.5x (57%) |
| blas2 | trsv | f64 | n=4096 | 1.761 ms | 9.53 | 38.14 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.856 ms | 19.61 | 39.26 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.657 ms | 4.38 | 35.07 | 43% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.539 ms | 497.80 | 11.67 | 83% | ok | 6.9x (86%) |
| blas3 | gemm | f32 | 512x512x512 | 0.264 ms | 1017.39 | 11.92 | 85% | ok | 6.9x (87%) |
| blas3 | gemmt | f64 | 512x512x512 | 0.706 ms | 190.21 | 8.92 | 32% | ok | 3.6x (46%) |
| blas3 | symm | f64 | 512x512x512 | 0.540 ms | 497.44 | 11.66 | 83% | ok | 6.9x (86%) |
| blas3 | syrk | f64 | 512x512x512 | 0.749 ms | 179.08 | 5.60 | 30% | ok | 3.4x (42%) |
| blas3 | syr2k | f64 | 512x512x512 | 1.123 ms | 239.07 | 5.60 | 40% | ok | 4.2x (53%) |
| blas3 | trmm | f64 | 512x512x512 | 0.341 ms | 393.78 | 15.38 | 66% | ok | 6.2x (78%) |
| blas3 | trsm | f64 | 512x512x512 | 0.338 ms | 396.51 | 15.49 | 66% | ok | 6.3x (79%) |
| blas3 | trsm | f32 | 512x512x512 | 0.160 ms | 838.72 | 16.38 | 70% | ok | 6.0x (75%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.021 ms | 0.41 | 3.26 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 2.126 ms | 42.09 | 1.97 | 7% | ok | 1.5x (19%) |
| factor | lu | f32 | 512x512 | 1.252 ms | 71.47 | 1.68 | 6% | ok | 1.5x (19%) |
| factor | cholesky | f64 | 512x512 | 1.533 ms | 29.19 | 2.74 | 5% | ok | 1.3x (17%) |
| factor | cholesky | f32 | 512x512 | 0.943 ms | 47.43 | 2.22 | 4% | ok | 1.2x (15%) |
| factor | ldlt | f64 | 512x512 | 7.270 ms | 6.15 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 43.304 ms | 1.03 | 0.10 | 0% | ok | 1.1x (13%) |
| factor | qr | f64 | 512x512 | 4.146 ms | 43.16 | 1.01 | 7% | ok | 1.4x (17%) |
| factor | lq | f64 | 512x512 | 17.482 ms | 10.24 | 0.24 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 164.842 ms | 2.71 | 0.03 | 0% | ok | 1.6x (21%) |
| factor | householder | f64 | 512x512 | 0.431 ms | 2.44 | 9.73 | 12%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 46.394 ms | 5.79 | 0.09 | 1% | ok | 1.0x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 2.759 ms | 32.43 | 1.52 | 5% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.641 ms | 13.63 | 0.64 | 2% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.936 ms | 2.61 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 739.098 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 17.426 ms | 9.63 | 0.06 | 2% | ok | 0.9x (12%) |
| eigen | schur | f64 | 256x256 | 21.686 ms | 19.34 | 0.05 | 3% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 6.151 ms | 60.01 | 0.17 | 10% | ok | 1.1x (14%) |
| eigen | expm | f64 | 256x256 | 8.753 ms | 28.75 | 0.12 | 5% | ok | 1.3x (16%) |
| eigen | sylvester | f64 | 256x256 | 55.162 ms | 18.25 | 0.03 | 3% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 39.075 ms | 0.21 | 3.43 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 46.105 ms | 0.18 | 2.91 | 4% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 59.795 ms | 0.14 | 2.24 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.700 ms | 0.29 | 4.68 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 202.511 ms | 0.04 | 0.66 | 1% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.23 | 241.72 | 294% ! | ok | 6.6x (83%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.912 ms | 2.87 | 29.86 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.418 ms | 29.54 | 62.10 | 76% | ok | 3.3x (41%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.820 ms | 0.68 | 2.39 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.378 ms | 1.46 | 12.08 | 15%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 90.181 ms | 0.12 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.378 ms | 0.44 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.963 ms | 1.25 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.411 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 52.932 ms | 0.42 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.012 ms | 1.32 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 89.053 ms | 0.02 | 0.00 | 0% | ok | 1.1x (14%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.344 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.918 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.215 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.919 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.712 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.214 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.606 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.252 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.797 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.770 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.226 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.897 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.993 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.932 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.658 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.538 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 12.528 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.054 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.125 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.993 ms | 2.64 | 29.54 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.209 ms | 2.17 | 24.26 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.595 ms | 1.64 | 18.39 | 22% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.204 ms | 2.71 | 20.46 | 25% | ok | 0.9x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.596 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.574 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 39.274 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.096 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 4.916 ms | 3.41 | 13.65 | 17% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 8.815 ms | 30.45 | 0.48 | 5% | ok | 4.4x (55%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.562 ms | 29.84 | 31.71 | 39%\* | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.323 ms | 7.22 | 28.89 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.004 ms | 16.71 | 33.42 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.684 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.686 ms | 9.95 | 9.95 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.434 ms | 110.29 | 1.29 | n/a | ok | 7.2x (90%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.827 ms | 94.96 | 0.74 | n/a | ok | 7.4x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.357 ms | 197.84 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.358 ms | 197.65 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.538 ms | 75.87 | 0.44 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

