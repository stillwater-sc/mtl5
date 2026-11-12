## xeon-e5-2420v2 -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 58.9 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 58.9, f32 118.2 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.5, triad 8.4 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 81% |
| blas2 | 10 | 103% | 108% | 0 | 20% |
| blas3 | 10 | 129% | 159% | 0 | 77% |
| factor | 12 | 18% | 38% | 0 | 25% |
| eigen | 8 | 9% | 24% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 103% | 0 | 17% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 67% | 82% | 0 | 16% |
| mg | 1 | 73% | 73% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 68% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 90% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 32.690 ms | 0.51 | 4.11 | 36% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=8388608 | 5.521 ms | 3.04 | 12.16 | 107% | ok | 4.9x (81%) |
| blas1 | axpy | f64 | n=8388608 | 38.180 ms | 0.44 | 5.27 | 46% | ok | 5.5x (92%) |
| blas1 | axpy | f32 | n=8388608 | 16.570 ms | 1.01 | 6.08 | 53% | ok | 5.5x (92%) |
| blas1 | scale | f64 | n=8388608 | 12.164 ms | 0.69 | 11.03 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 32.840 ms | 0.51 | 2.04 | 18% | ok | 5.7x (95%) |
| blas1 | one_norm | f64 | n=8388608 | 9.692 ms | 0.87 | 6.92 | 61% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.066 ms | 0.64 | 5.14 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.710 ms | 0.86 | 6.91 | 61% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 277.510 ms | 0.67 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 11.307 ms | 2.97 | 11.88 | 105% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.466 ms | 6.14 | 12.28 | 108% | ok | 1.1x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.924 ms | 3.07 | 12.29 | 108% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.691 ms | 5.90 | 11.80 | 104% | ok | 1.2x (19%) |
| blas2 | symv | f64 | n=4096 | 5.779 ms | 5.81 | 11.62 | 102% | ok | 1.2x (20%) |
| blas2 | trmv | f64 | n=4096 | 5.725 ms | 2.93 | 11.73 | 103% | ok | 1.3x (22%) |
| blas2 | trmv | f32 | n=4096 | 2.968 ms | 5.65 | 11.32 | 100% | ok | 1.3x (21%) |
| blas2 | trsv | f64 | n=4096 | 7.433 ms | 2.26 | 9.04 | 80% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.476 ms | 4.83 | 9.66 | 85% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 27.922 ms | 1.20 | 9.62 | 85% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 2.870 ms | 93.55 | 2.19 | 159% | ok | 4.9x (82%) |
| blas3 | gemm | f32 | 512x512x512 | 1.762 ms | 152.37 | 1.79 | 129% | ok | 4.9x (81%) |
| blas3 | gemmt | f64 | 512x512x512 | 2.536 ms | 52.93 | 2.48 | 90% | ok | 3.5x (58%) |
| blas3 | symm | f64 | 512x512x512 | 2.863 ms | 93.76 | 2.20 | 159% | ok | 5.0x (83%) |
| blas3 | syrk | f64 | 512x512x512 | 2.517 ms | 53.33 | 1.67 | 91% | ok | 3.1x (52%) |
| blas3 | syr2k | f64 | 512x512x512 | 3.736 ms | 71.85 | 1.68 | 122% | ok | 4.0x (67%) |
| blas3 | trmm | f64 | 512x512x512 | 1.617 ms | 83.02 | 3.24 | 141% | ok | 4.6x (77%) |
| blas3 | trsm | f64 | 512x512x512 | 1.773 ms | 75.69 | 2.96 | 129% | ok | 4.6x (77%) |
| blas3 | trsm | f32 | 512x512x512 | 1.251 ms | 107.26 | 2.09 | 91% | ok | 4.4x (74%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.657 ms | 0.14 | 1.10 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.593 ms | 16.00 | 0.75 | 27% | ok | 1.7x (28%) |
| factor | lu | f32 | 512x512 | 3.865 ms | 23.15 | 0.54 | 20% | ok | 1.7x (28%) |
| factor | cholesky | f64 | 512x512 | 4.129 ms | 10.84 | 1.02 | 18% | ok | 1.4x (23%) |
| factor | cholesky | f32 | 512x512 | 3.086 ms | 14.50 | 0.68 | 12% | ok | 1.3x (22%) |
| factor | ldlt | f64 | 512x512 | 24.996 ms | 1.79 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 137.762 ms | 0.32 | 0.03 | 1% | ok | 1.1x (18%) |
| factor | qr | f64 | 512x512 | 13.176 ms | 13.58 | 0.32 | 23% | ok | 1.5x (25%) |
| factor | lq | f64 | 512x512 | 51.909 ms | 3.45 | 0.08 | 6% | ok | 1.4x (24%) |
| factor | hessenberg | f64 | 512x512 | 415.183 ms | 1.08 | 0.01 | 2% | ok | 1.5x (26%) |
| factor | householder | f64 | 512x512 | 0.967 ms | 1.09 | 4.34 | 38%\* | ok | 3.4x (56%) |
| factor | inv | f64 | 512x512 | 131.496 ms | 2.04 | 0.03 | 3% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.442 ms | 10.60 | 0.50 | 18% | ok | 1.5x (25%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.764 ms | 3.88 | 0.18 | 7% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 171.931 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 995.176 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 56.534 ms | 2.97 | 0.02 | 5% | ok | 0.9x (16%) |
| eigen | schur | f64 | 256x256 | 75.580 ms | 5.55 | 0.01 | 9% | ok | 1.0x (16%) |
| eigen | svd | f64 | 256x256 | 26.537 ms | 13.91 | 0.04 | 24% | ok | 1.1x (19%) |
| eigen | expm | f64 | 256x256 | 27.345 ms | 9.20 | 0.04 | 16% | ok | 1.3x (22%) |
| eigen | sylvester | f64 | 256x256 | 194.172 ms | 5.18 | 0.01 | 9% | ok | 1.0x (16%) |
| elementwise | exp | f64 | n=8388608 | 152.246 ms | 0.06 | 0.88 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 140.359 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 170.935 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 90.533 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2470.984 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.317 ms | 1.13 | 11.75 | 103% | ok | 1.0x (17%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.097 ms | 0.85 | 8.79 | 77% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.339 ms | 3.69 | 7.76 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.120 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.816 ms | 0.35 | 2.86 | 25% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 274.497 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.834 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.474 ms | 0.42 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.364 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 154.712 ms | 0.14 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.117 ms | 0.46 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 228.643 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.869 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.299 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.614 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.338 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 230.409 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 392.404 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 134.126 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 103.190 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 121.995 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 78.769 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 517.377 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 180.982 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 90.716 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 135.858 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 136.541 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 171.868 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 248.053 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.441 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.003 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.830 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.717 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.077 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.542 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.877 ms | 0.68 | 7.56 | 67% | ok | 0.9x (15%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.156 ms | 0.83 | 9.29 | 82% | ok | 1.0x (16%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.846 ms | 0.68 | 7.63 | 67% | ok | 1.0x (16%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 34.876 ms | 1.10 | 8.33 | 73% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.690 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 147.380 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 143.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.687 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 192.821 ms | 0.09 | 0.35 | 3% | ok | 1.0x (16%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.562 ms | 13.05 | 0.20 | 22% | ok | 5.6x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.298 ms | 7.30 | 7.76 | 68% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.532 ms | 2.57 | 10.27 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.138 ms | 5.35 | 10.69 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.013 ms | 3.35 | 3.35 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.987 ms | 3.36 | 3.36 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.489 ms | 25.59 | 0.30 | n/a | ok | 5.5x (91%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.236 ms | 32.59 | 0.25 | n/a | ok | 5.5x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.336 ms | 61.91 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.330 ms | 62.00 | 0.36 | n/a | ok | 5.4x (90%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.075 ms | 20.53 | 0.12 | n/a | ok | 5.4x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

