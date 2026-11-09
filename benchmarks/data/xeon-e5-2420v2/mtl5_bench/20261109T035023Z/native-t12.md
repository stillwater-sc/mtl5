## Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz -- native, 12 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-42-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `73d11400a0e3` |
| FMA peak (ceiling) | f64 56.2 GFLOP/s, f32 117.7 GFLOP/s |
| ... hardware, SSE2 | f64 56.2, f32 117.7 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.1 GFLOP/s |
| bandwidth | read 11.3, update 10.6, triad 8.4 GB/s (ceiling 11.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 54% | 101% | 0 | 40% |
| blas2 | 6 | 82% | 107% | 0 | 9% |
| blas3 | 10 | 7% | 46% | 0 | 8% |
| factor | 12 | 3% | 37% | 0 | 9% |
| eigen | 8 | 2% | 10% | 0 | 9% |
| elementwise | 5 | 8% | 13% | 0 | 8% |
| sparse | 5 | 66% | 102% | 0 | 9% |
| direct | 7 | 0% | 1% | 0 | 8% |
| ordering | 4 | n/a | n/a | 0 | 8% |
| krylov | 13 | n/a | n/a | 0 | 8% |
| pc | 7 | n/a | n/a | 0 | 8% |
| smoother | 3 | 69% | 80% | 0 | 8% |
| mg | 1 | 69% | 69% | 0 | 8% |
| sparse_eig | 4 | n/a | n/a | 0 | 8% |
| mixed | 3 | 23% | 56% | 0 | 8% |
| integer | 9 | n/a | n/a | 0 | 43% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 29.704 ms | 0.56 | 4.52 | 40% | ok | 6.7x (56%) |
| blas1 | dot | f32 | n=8388608 | 5.843 ms | 2.87 | 11.48 | 101% | ok | 4.8x (40%) |
| blas1 | axpy | f64 | n=8388608 | 32.837 ms | 0.51 | 6.13 | 54% | ok | 6.8x (57%) |
| blas1 | axpy | f32 | n=8388608 | 11.495 ms | 1.46 | 8.76 | 77% | ok | 8.3x (69%) |
| blas1 | scale | f64 | n=8388608 | 12.484 ms | 0.67 | 10.75 | 95% | ok | 1.1x (9%) |
| blas1 | two_norm | f64 | n=8388608 | 28.924 ms | 0.58 | 2.32 | 20% | ok | 6.6x (55%) |
| blas1 | one_norm | f64 | n=8388608 | 11.744 ms | 0.71 | 5.71 | 50% | ok | 0.9x (7%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.548 ms | 0.62 | 4.95 | 44% | ok | 1.0x (8%) |
| blas1 | sum | f64 | n=8388608 | 10.156 ms | 0.83 | 6.61 | 58% | ok | 1.0x (8%) |
| blas1 | dot_k3 | f64 | n=8388608 | 293.663 ms | 0.63 | 0.46 | 4% | ok | 1.0x (8%) |
| blas2 | gemv | f64 | n=4096 | 11.115 ms | 3.02 | 12.08 | 107% | ok | 1.8x (15%) |
| blas2 | gemv | f32 | n=4096 | 5.740 ms | 5.85 | 11.70 | 103% | ok | 3.5x (29%) |
| blas2 | symv | f64 | n=4096 | 34.082 ms | 0.98 | 1.97 | 17% | ok | 0.5x (4%) |
| blas2 | trmv | f64 | n=4096 | 10.767 ms | 1.56 | 6.24 | 55% | ok | 1.0x (8%) |
| blas2 | trsv | f64 | n=4096 | 11.587 ms | 1.45 | 5.80 | 51% | ok | 0.9x (8%) |
| blas2 | ger | f64 | n=4096 | 28.784 ms | 1.17 | 9.33 | 82% | ok | 1.0x (9%) |
| blas3 | gemm | f64 | 512x512x512 | 10.443 ms | 25.70 | 0.60 | 46% | ok | 6.4x (53%) |
| blas3 | gemm | f32 | 512x512x512 | 6.027 ms | 44.54 | 0.52 | 38% | ok | 6.0x (50%) |
| blas3 | gemmt | f64 | 512x512x512 | 35.633 ms | 3.77 | 0.18 | 7% | ok | 0.9x (7%) |
| blas3 | symm | f64 | 512x512x512 | 62.577 ms | 4.29 | 0.10 | 8% | ok | 1.0x (8%) |
| blas3 | syrk | f64 | 512x512x512 | 39.642 ms | 3.39 | 0.11 | 6% | ok | 1.0x (8%) |
| blas3 | syr2k | f64 | 512x512x512 | 69.485 ms | 3.86 | 0.09 | 7% | ok | 1.0x (8%) |
| blas3 | trmm | f64 | 512x512x512 | 34.180 ms | 3.93 | 0.15 | 7% | ok | 1.0x (9%) |
| blas3 | trsm | f64 | 512x512x512 | 33.895 ms | 3.96 | 0.15 | 7% | ok | 1.0x (8%) |
| blas3 | trsm | f32 | 512x512x512 | 17.061 ms | 7.87 | 0.15 | 7% | ok | 1.0x (8%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.623 ms | 0.14 | 1.10 | 10% | ok | 1.0x (8%) |
| factor | lu | f64 | 512x512 | 26.055 ms | 3.43 | 0.16 | 6% | ok | 1.2x (10%) |
| factor | lu | f32 | 512x512 | 15.740 ms | 5.68 | 0.13 | 5% | ok | 1.1x (9%) |
| factor | cholesky | f64 | 512x512 | 55.426 ms | 0.81 | 0.08 | 1% | ok | 0.5x (4%) |
| factor | cholesky | f32 | 512x512 | 24.554 ms | 1.82 | 0.09 | 2% | ok | 1.0x (8%) |
| factor | ldlt | f64 | 512x512 | 26.380 ms | 1.70 | 0.16 | 3% | ok | 1.0x (8%) |
| factor | ldlt_bk | f64 | 512x512 | 140.140 ms | 0.32 | 0.03 | 1% | ok | 1.0x (8%) |
| factor | qr | f64 | 512x512 | 346.492 ms | 0.52 | 0.01 | 1% | ok | 1.5x (13%) |
| factor | lq | f64 | 512x512 | 54.204 ms | 3.30 | 0.08 | 6% | ok | 1.4x (12%) |
| factor | hessenberg | f64 | 512x512 | 467.108 ms | 0.96 | 0.01 | 2% | ok | 1.4x (12%) |
| factor | householder | f64 | 512x512 | 0.988 ms | 1.06 | 4.25 | 37%\* | ok | 3.4x (29%) |
| factor | inv | f64 | 512x512 | 194.016 ms | 1.38 | 0.02 | 2% | ok | 1.0x (9%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.095 ms | 3.87 | 0.18 | 7% | ok | 1.0x (8%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 63.615 ms | 0.35 | 0.02 | 1% | ok | 1.0x (8%) |
| eigen | eigen_symmetric | f64 | 256x256 | 177.025 ms | 0.85 | 0.01 | 2% | ok | 1.0x (9%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1047.826 ms | 0.14 | 0.00 | 0% | ok | 1.0x (8%) |
| eigen | eigenvalue | f64 | 256x256 | 186.422 ms | 0.90 | 0.01 | 2% | ok | 0.6x (5%) |
| eigen | schur | f64 | 256x256 | 277.157 ms | 1.51 | 0.00 | 3% | ok | 1.0x (8%) |
| eigen | svd | f64 | 256x256 | 313.352 ms | 1.18 | 0.00 | 2% | ok | 1.0x (9%) |
| eigen | expm | f64 | 256x256 | 42.970 ms | 5.86 | 0.02 | 10% | ok | 2.2x (19%) |
| eigen | sylvester | f64 | 256x256 | 594.847 ms | 1.69 | 0.00 | 3% | ok | 1.1x (9%) |
| elementwise | exp | f64 | n=8388608 | 157.376 ms | 0.05 | 0.85 | 8% | ok | 1.0x (8%) |
| elementwise | log | f64 | n=8388608 | 144.081 ms | 0.06 | 0.93 | 8% | ok | 1.0x (8%) |
| elementwise | sin | f64 | n=8388608 | 175.478 ms | 0.05 | 0.76 | 7% | ok | 1.0x (8%) |
| elementwise | sqrt | f64 | n=8388608 | 92.801 ms | 0.09 | 1.45 | 13% | ok | 1.0x (8%) |
| elementwise | erf | f64 | n=8388608 | 2552.635 ms | 0.00 | 0.05 | 0% | ok | 1.0x (9%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.367 ms | 1.11 | 11.51 | 102% | ok | 1.1x (9%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.296 ms | 0.79 | 8.26 | 73% | ok | 1.0x (8%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.853 ms | 3.53 | 7.43 | 66% | ok | 1.4x (11%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.726 ms | 0.17 | 0.61 | 5%\* | ok | 0.9x (8%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 6.037 ms | 0.33 | 2.76 | 24% | ok | 1.1x (9%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 283.581 ms | 0.04 | 0.00 | 0% | ok | 1.0x (9%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 71.766 ms | 0.16 | 0.01 | 0% | ok | 1.0x (8%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 58.662 ms | 0.19 | 0.01 | 0% | ok | 0.5x (4%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 119.379 ms | 0.19 | 0.01 | 0% | ok | 1.0x (8%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 160.539 ms | 0.14 | 0.00 | 0% | ok | 1.0x (8%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 55.346 ms | 0.40 | 0.01 | 1% | ok | 0.9x (8%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 238.642 ms | 0.01 | 0.00 | 0% | ok | 1.0x (8%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.325 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.526 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.626 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.503 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 238.580 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 407.430 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 130.672 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 106.152 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 134.780 ms | 0.00 | 0.00 | n/a | ok | 0.9x (8%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 81.901 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 536.164 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 179.608 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 93.759 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 140.135 ms | 0.00 | 0.00 | n/a | ok | 1.2x (10%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 142.570 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 175.695 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 257.813 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.132 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.576 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ilut | f64 | 128^2 grid | 34.005 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ic_0 | f64 | 128^2 grid | 2.992 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ildl | f64 | 128^2 grid | 2.823 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ssor | f64 | 128^2 grid | 0.078 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.264 ms | 0.00 | 0.00 | n/a | ok | 0.9x (8%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.736 ms | 0.70 | 7.85 | 69% | ok | 1.0x (8%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.240 ms | 0.81 | 9.05 | 80% | ok | 1.0x (8%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.075 ms | 0.64 | 7.20 | 64% | ok | 1.0x (8%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.987 ms | 1.04 | 7.86 | 69% | ok | 1.0x (8%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 149.920 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 197.737 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 152.407 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 44.059 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| mixed | dot | f32>f64 | n=8388608 | 195.020 ms | 0.09 | 0.34 | 3% | ok | 1.0x (8%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.586 ms | 13.04 | 0.20 | 23% | ok | 5.7x (48%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.821 ms | 5.95 | 6.32 | 56% | ok | 0.9x (7%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.725 ms | 2.49 | 9.98 | n/a | ok | 1.0x (8%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.100 ms | 5.41 | 10.82 | n/a | ok | 1.1x (9%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.208 ms | 3.22 | 3.22 | n/a | ok | 1.1x (9%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 5.202 ms | 3.22 | 3.22 | n/a | ok | 1.0x (9%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.831 ms | 27.31 | 0.32 | n/a | ok | 6.1x (51%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 9.129 ms | 29.40 | 0.23 | n/a | ok | 5.1x (43%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.506 ms | 59.57 | 0.35 | n/a | ok | 5.4x (45%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.229 ms | 63.47 | 0.37 | n/a | ok | 5.8x (48%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.913 ms | 19.29 | 0.11 | n/a | ok | 5.3x (45%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

