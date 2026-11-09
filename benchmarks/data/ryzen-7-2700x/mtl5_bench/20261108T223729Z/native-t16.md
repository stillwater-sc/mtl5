## AMD Ryzen 7 2700X Eight-Core Processor -- native, 16 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 2700X Eight-Core Processor (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36256 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `e096dbbc6788` |
| FMA peak (ceiling) | f64 248.9 GFLOP/s, f32 498.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 248.9, f32 498.5 GFLOP/s |
| ... this build, scalar | f64 14.9, f32 17.0 GFLOP/s |
| bandwidth | read 26.3, update 27.4, triad 21.2 GB/s (ceiling 27.4) |
| last-level cache | 8 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 91% | 112% | 0 | 14% |
| blas2 | 6 | 41% | 108% | 0 | 6% |
| blas3 | 10 | 4% | 30% | 0 | 6% |
| factor | 12 | 1% | 20% | 0 | 10% |
| eigen | 8 | 1% | 3% | 0 | 6% |
| elementwise | 5 | 8% | 17% | 0 | 6% |
| sparse | 5 | 32% | 103% | 0 | 6% |
| direct | 7 | 0% | 0% | 0 | 6% |
| ordering | 4 | n/a | n/a | 0 | 6% |
| krylov | 13 | n/a | n/a | 0 | 6% |
| pc | 7 | n/a | n/a | 0 | 6% |
| smoother | 3 | 47% | 54% | 0 | 6% |
| mg | 1 | 41% | 41% | 0 | 6% |
| sparse_eig | 4 | n/a | n/a | 0 | 5% |
| mixed | 3 | 16% | 21% | 0 | 6% |
| integer | 9 | n/a | n/a | 0 | 29% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.706 ms | 3.57 | 28.52 | 104% | ok | 3.0x (19%) |
| blas1 | dot | f32 | n=8388608 | 2.809 ms | 5.97 | 23.89 | 87% | ok | 5.1x (32%) |
| blas1 | axpy | f64 | n=8388608 | 7.158 ms | 2.34 | 28.13 | 103% | ok | 2.3x (14%) |
| blas1 | axpy | f32 | n=8388608 | 3.499 ms | 4.79 | 28.77 | 105% | ok | 4.3x (27%) |
| blas1 | scale | f64 | n=8388608 | 4.374 ms | 1.92 | 30.69 | 112% | ok | 1.6x (10%) |
| blas1 | two_norm | f64 | n=8388608 | 2.687 ms | 6.25 | 24.98 | 91% | ok | 5.4x (34%) |
| blas1 | one_norm | f64 | n=8388608 | 6.604 ms | 1.27 | 10.16 | 37% | ok | 1.0x (6%) |
| blas1 | infinity_norm | f64 | n=8388608 | 8.787 ms | 0.95 | 7.64 | 28% | ok | 1.0x (6%) |
| blas1 | sum | f64 | n=8388608 | 6.611 ms | 1.27 | 10.15 | 37% | ok | 1.0x (6%) |
| blas1 | dot_k3 | f64 | n=8388608 | 115.573 ms | 1.60 | 1.16 | 4% | ok | 1.0x (6%) |
| blas2 | gemv | f64 | n=4096 | 4.541 ms | 7.39 | 29.57 | 108% | ok | 3.1x (20%) |
| blas2 | gemv | f32 | n=4096 | 2.621 ms | 12.80 | 25.61 | 94% | ok | 5.0x (31%) |
| blas2 | symv | f64 | n=4096 | 10.069 ms | 3.33 | 6.67 | 24% | ok | 1.0x (6%) |
| blas2 | trmv | f64 | n=4096 | 7.113 ms | 2.36 | 9.44 | 35% | ok | 1.0x (6%) |
| blas2 | trsv | f64 | n=4096 | 7.081 ms | 2.37 | 9.49 | 35% | ok | 1.0x (6%) |
| blas2 | ger | f64 | n=4096 | 23.955 ms | 1.40 | 11.21 | 41% | ok | 1.0x (6%) |
| blas3 | gemm | f64 | 512x512x512 | 3.622 ms | 74.11 | 1.74 | 30% | ok | 7.9x (49%) |
| blas3 | gemm | f32 | 512x512x512 | 2.715 ms | 98.88 | 1.16 | 20% | ok | 4.9x (31%) |
| blas3 | gemmt | f64 | 512x512x512 | 15.902 ms | 8.44 | 0.40 | 3% | ok | 1.0x (6%) |
| blas3 | symm | f64 | 512x512x512 | 28.242 ms | 9.50 | 0.22 | 4% | ok | 1.0x (6%) |
| blas3 | syrk | f64 | 512x512x512 | 18.815 ms | 7.13 | 0.22 | 3% | ok | 1.0x (6%) |
| blas3 | syr2k | f64 | 512x512x512 | 28.495 ms | 9.42 | 0.22 | 4% | ok | 1.0x (6%) |
| blas3 | trmm | f64 | 512x512x512 | 44.256 ms | 3.03 | 0.12 | 1% | ok | 1.0x (6%) |
| blas3 | trsm | f64 | 512x512x512 | 46.392 ms | 2.89 | 0.11 | 1% | ok | 1.0x (6%) |
| blas3 | trsm | f32 | 512x512x512 | 44.436 ms | 3.02 | 0.06 | 1% | ok | 1.1x (7%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 17.021 ms | 0.31 | 2.50 | 9% | ok | 1.1x (7%) |
| factor | lu | f64 | 512x512 | 50.534 ms | 1.77 | 0.08 | 1% | ok | 2.9x (18%) |
| factor | lu | f32 | 512x512 | 58.603 ms | 1.53 | 0.04 | 0% | ok | 2.6x (16%) |
| factor | cholesky | f64 | 512x512 | 16.523 ms | 2.71 | 0.25 | 1% | ok | 1.0x (6%) |
| factor | cholesky | f32 | 512x512 | 16.316 ms | 2.74 | 0.13 | 1% | ok | 1.0x (6%) |
| factor | ldlt | f64 | 512x512 | 15.902 ms | 2.81 | 0.26 | 1% | ok | 1.0x (6%) |
| factor | ldlt_bk | f64 | 512x512 | 159.553 ms | 0.28 | 0.03 | 0% | ok | 1.0x (6%) |
| factor | qr | f64 | 512x512 | 310.657 ms | 0.58 | 0.01 | 0% | ok | 1.7x (10%) |
| factor | lq | f64 | 512x512 | 72.861 ms | 2.46 | 0.06 | 1% | ok | 2.1x (13%) |
| factor | hessenberg | f64 | 512x512 | 418.485 ms | 1.07 | 0.01 | 0% | ok | 1.6x (10%) |
| factor | householder | f64 | 512x512 | 0.778 ms | 1.35 | 5.39 | 20%\* | ok | 4.2x (26%) |
| factor | inv | f64 | 512x512 | 159.077 ms | 1.69 | 0.03 | 1% | ok | 1.1x (7%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 63.831 ms | 1.40 | 0.07 | 1% | ok | 1.0x (6%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 33.550 ms | 0.67 | 0.03 | 0% | ok | 1.0x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 104.711 ms | 1.44 | 0.01 | 1% | ok | 1.1x (7%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 583.927 ms | 0.26 | 0.00 | 0% | ok | 1.0x (6%) |
| eigen | eigenvalue | f64 | 256x256 | 68.055 ms | 2.47 | 0.02 | 1% | ok | 1.1x (7%) |
| eigen | schur | f64 | 256x256 | 233.611 ms | 1.80 | 0.00 | 1% | ok | 0.7x (5%) |
| eigen | svd | f64 | 256x256 | 173.886 ms | 2.12 | 0.01 | 1% | ok | 1.0x (6%) |
| eigen | expm | f64 | 256x256 | 36.989 ms | 6.80 | 0.03 | 3% | ok | 1.7x (10%) |
| eigen | sylvester | f64 | 256x256 | 445.949 ms | 2.26 | 0.00 | 1% | ok | 1.0x (6%) |
| elementwise | exp | f64 | n=8388608 | 54.608 ms | 0.15 | 2.46 | 9% | ok | 1.0x (7%) |
| elementwise | log | f64 | n=8388608 | 68.645 ms | 0.12 | 1.96 | 7% | ok | 1.1x (7%) |
| elementwise | sin | f64 | n=8388608 | 58.953 ms | 0.14 | 2.28 | 8% | ok | 1.0x (6%) |
| elementwise | sqrt | f64 | n=8388608 | 28.626 ms | 0.29 | 4.69 | 17% | ok | 1.0x (6%) |
| elementwise | erf | f64 | n=8388608 | 64.030 ms | 0.13 | 2.10 | 8% | ok | 1.0x (6%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.969 ms | 2.70 | 28.10 | 103% | ok | 1.4x (9%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.153 ms | 0.83 | 8.64 | 32% | ok | 1.0x (6%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.561 ms | 9.18 | 19.30 | 71% | ok | 1.7x (10%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.681 ms | 0.16 | 0.56 | 2% | ok | 1.0x (6%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 3.881 ms | 0.52 | 4.29 | 16% | ok | 0.8x (5%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 174.271 ms | 0.06 | 0.00 | 0% | ok | 1.1x (7%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.205 ms | 0.21 | 0.01 | 0% | ok | 1.0x (6%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 33.784 ms | 0.33 | 0.02 | 0% | ok | 1.0x (6%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 109.599 ms | 0.20 | 0.01 | 0% | ok | 1.0x (6%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 134.579 ms | 0.17 | 0.01 | 0% | ok | 0.9x (6%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 50.168 ms | 0.45 | 0.01 | 0% | ok | 0.9x (6%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 420.798 ms | 0.00 | 0.00 | 0% | ok | 0.8x (5%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 6.478 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 10.788 ms | 0.00 | 0.00 | n/a | ok | 0.9x (5%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.866 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 32.297 ms | 0.00 | 0.00 | n/a | ok | 0.8x (5%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 56.006 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 70.935 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 46.187 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 25.129 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 32.520 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 25.480 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 87.694 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.748 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.736 ms | 0.00 | 0.00 | n/a | ok | 0.9x (5%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 48.518 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 53.815 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 93.213 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 79.002 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.076 ms | 0.00 | 0.00 | n/a | ok | 0.8x (5%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.579 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ilut | f64 | 128^2 grid | 49.207 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ic_0 | f64 | 128^2 grid | 3.460 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ildl | f64 | 128^2 grid | 3.748 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok | 1.7x (11%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.272 ms | 1.15 | 12.91 | 47% | ok | 1.0x (6%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.002 ms | 1.31 | 14.65 | 54% | ok | 1.0x (6%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.406 ms | 1.09 | 12.19 | 45% | ok | 1.0x (6%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 25.987 ms | 1.48 | 11.18 | 41% | ok | 0.9x (6%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 55.326 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 101.304 ms | 0.00 | 0.00 | n/a | ok | 0.7x (4%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 152.520 ms | 0.00 | 0.00 | n/a | ok | 0.8x (5%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 14.543 ms | 0.00 | 0.00 | n/a | ok | 0.8x (5%) |
| mixed | dot | f32>f64 | n=8388608 | 15.628 ms | 1.07 | 4.29 | 16% | ok | 0.9x (6%) |
| mixed | gemm | f32>f64 | 512x512x512 | 7.016 ms | 38.26 | 0.60 | 15% | ok | 7.6x (47%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 3.078 ms | 5.45 | 5.79 | 21% | ok | 0.7x (5%) |
| integer | dot | i32 | n=8388608, |v| <= 15 | 4.302 ms | 3.90 | 15.60 | n/a | ok | 1.0x (6%) |
| integer | dot | i16>i32 | n=8388608, |v| <= 15 | 3.894 ms | 4.31 | 8.62 | n/a | ok | 1.0x (6%) |
| integer | dot | i8>i32 | n=8388608, |v| <= 15 | 2.174 ms | 7.72 | 7.72 | n/a | ok | 1.2x (8%) |
| integer | dot | u8i8>i32 | n=8388608, |v| <= 15 | 2.165 ms | 7.75 | 7.75 | n/a | ok | 1.0x (6%) |
| integer | gemm | i32 | 512^3, |v| <= 2047 | 2.704 ms | 99.26 | 1.16 | n/a | ok | 6.8x (42%) |
| integer | gemm | i16>i32 | 512^3, |v| <= 2047 | 8.559 ms | 31.36 | 0.25 | n/a | ok | 5.5x (35%) |
| integer | gemm | i8>i32 | 512^3, |v| <= 127 | 8.393 ms | 31.98 | 0.19 | n/a | ok | 4.6x (29%) |
| integer | gemm_quad | i8>i32 | 512^3, |v| <= 127 | 7.651 ms | 35.08 | 0.21 | n/a | ok | 5.2x (32%) |
| integer | gemm_quad | u8i8>i32 | 512^3, |v| <= 127 | 7.734 ms | 34.71 | 0.20 | n/a | ok | 5.0x (31%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

