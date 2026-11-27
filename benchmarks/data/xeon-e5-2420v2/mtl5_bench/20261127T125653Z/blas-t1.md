## xeon-e5-2420v2 -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 20.9 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 20.9 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.7 GFLOP/s |
| bandwidth | read 10.2, update 10.3, triad 8.1 GB/s (ceiling 10.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 105% | 0 |
| blas2 | 10 | 92% | 104% | 0 |
| blas3 | 10 | 163% | 182% | 0 |
| factor | 12 | 55% | 92% | 0 |
| eigen | 8 | 50% | 119% | 0 |
| elementwise | 5 | 8% | 14% | 0 |
| sparse | 5 | 56% | 110% | 0 |
| direct | 19 | 3% | 35% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 79% | 91% | 0 |
| mg | 1 | 73% | 73% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 22% | 77% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 198.936 ms | 0.08 | 0.67 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 27.457 ms | 0.61 | 2.44 | 24% | ok |
| blas1 | axpy | f64 | n=8388608 | 212.927 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 93.529 ms | 0.18 | 1.08 | 10% | ok |
| blas1 | scale | f64 | n=8388608 | 12.445 ms | 0.67 | 10.79 | 105% | ok |
| blas1 | two_norm | f64 | n=8388608 | 194.668 ms | 0.09 | 0.34 | 3% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.793 ms | 0.86 | 6.85 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.049 ms | 0.64 | 5.14 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 9.784 ms | 0.86 | 6.86 | 67% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 279.348 ms | 0.66 | 0.48 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 14.367 ms | 2.34 | 9.35 | 91% | ok |
| blas2 | gemv | f32 | n=4096 | 6.242 ms | 5.38 | 10.76 | 104% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 12.948 ms | 2.59 | 10.37 | 101% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.561 ms | 5.11 | 10.23 | 99% | ok |
| blas2 | symv | f64 | n=4096 | 7.131 ms | 4.71 | 9.42 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 7.594 ms | 2.21 | 8.85 | 86% | ok |
| blas2 | trmv | f32 | n=4096 | 3.751 ms | 4.47 | 8.96 | 87% | ok |
| blas2 | trsv | f64 | n=4096 | 7.489 ms | 2.24 | 8.97 | 87% | ok |
| blas2 | trsv | f32 | n=4096 | 3.549 ms | 4.73 | 9.46 | 92% | ok |
| blas2 | ger | f64 | n=4096 | 28.428 ms | 1.18 | 9.45 | 92% | ok |
| blas3 | gemm | f64 | 512x512x512 | 14.146 ms | 18.98 | 0.44 | 181% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.046 ms | 33.36 | 0.39 | 160% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 8.794 ms | 15.26 | 0.72 | 146% | ok |
| blas3 | symm | f64 | 512x512x512 | 14.106 ms | 19.03 | 0.45 | 182% | ok |
| blas3 | syrk | f64 | 512x512x512 | 7.857 ms | 17.08 | 0.53 | 163% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 14.957 ms | 17.95 | 0.42 | 171% | ok |
| blas3 | trmm | f64 | 512x512x512 | 7.521 ms | 17.85 | 0.70 | 170% | ok |
| blas3 | trsm | f64 | 512x512x512 | 8.137 ms | 16.50 | 0.64 | 157% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.269 ms | 25.47 | 0.50 | 122% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.265 ms | 0.14 | 1.11 | 11% | ok |
| factor | lu | f64 | 512x512 | 9.254 ms | 9.67 | 0.45 | 92% | ok |
| factor | lu | f32 | 512x512 | 6.321 ms | 14.16 | 0.33 | 68% | ok |
| factor | cholesky | f64 | 512x512 | 5.820 ms | 7.69 | 0.72 | 73% | ok |
| factor | cholesky | f32 | 512x512 | 3.896 ms | 11.48 | 0.54 | 55% | ok |
| factor | ldlt | f64 | 512x512 | 25.500 ms | 1.75 | 0.16 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.359 ms | 3.35 | 0.31 | 32% | ok |
| factor | qr | f64 | 512x512 | 19.958 ms | 8.97 | 0.21 | 86% | ok |
| factor | lq | f64 | 512x512 | 74.984 ms | 2.39 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 614.689 ms | 0.73 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.258 ms | 0.32 | 1.29 | 12%\* | ok |
| factor | inv | f64 | 512x512 | 129.693 ms | 2.07 | 0.03 | 20% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.887 ms | 6.94 | 0.33 | 66% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 6.256 ms | 3.58 | 0.17 | 34% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 172.461 ms | 0.88 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1002.267 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 53.769 ms | 3.12 | 0.02 | 30% | ok |
| eigen | schur | f64 | 256x256 | 72.185 ms | 5.81 | 0.01 | 55% | ok |
| eigen | svd | f64 | 256x256 | 29.665 ms | 12.44 | 0.04 | 119% | ok |
| eigen | expm | f64 | 256x256 | 35.852 ms | 7.02 | 0.03 | 67% | ok |
| eigen | sylvester | f64 | 256x256 | 192.807 ms | 5.22 | 0.01 | 50% | ok |
| elementwise | exp | f64 | n=8388608 | 159.244 ms | 0.05 | 0.84 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 139.871 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 169.619 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 90.266 ms | 0.09 | 1.49 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2470.324 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.404 ms | 1.09 | 11.33 | 110% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.099 ms | 0.84 | 8.79 | 85% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.172 ms | 2.76 | 5.80 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.328 ms | 0.20 | 0.70 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.916 ms | 0.34 | 2.81 | 27% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.704 ms | 0.27 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 68.634 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.040 ms | 0.45 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.065 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 129.846 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.343 ms | 0.47 | 0.02 | 5% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 82.696 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.988 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.366 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 126.136 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.544 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 44.271 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.419 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.888 ms | 0.33 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.843 ms | 0.90 | 3.59 | 35%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.559 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.892 ms | 0.87 | 0.06 | 8% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.836 ms | 0.90 | 3.62 | 35%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.620 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 104.238 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.830 ms | 0.20 | 0.79 | 8%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.903 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 112.362 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.705 ms | 0.20 | 0.82 | 8%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.104 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 635.925 ms | 0.49 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.849 ms | 0.49 | 1.97 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 29.952 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 207.496 ms | 1.49 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.817 ms | 0.49 | 1.98 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.915 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1722.964 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.644 ms | 0.10 | 0.41 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.490 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1537.634 ms | 0.40 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.903 ms | 0.10 | 0.41 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 233.067 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 587.067 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 128.018 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 103.846 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 122.591 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 80.448 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 536.049 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.816 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.941 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 202.284 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 192.984 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 211.189 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 305.954 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.374 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.712 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.904 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.736 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.078 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.520 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.597 ms | 0.73 | 8.15 | 79% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.113 ms | 0.84 | 9.42 | 91% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.024 ms | 0.65 | 7.29 | 71% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 38.671 ms | 0.99 | 7.52 | 73% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 146.332 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 153.998 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 130.006 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.801 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 187.480 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 115.851 ms | 2.32 | 0.04 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.255 ms | 7.44 | 7.90 | 77% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.513 ms | 2.58 | 10.30 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.151 ms | 5.33 | 10.65 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.992 ms | 3.36 | 3.36 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.985 ms | 3.37 | 3.37 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 57.418 ms | 4.68 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.086 ms | 5.95 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 23.603 ms | 11.37 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.706 ms | 11.32 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 73.734 ms | 3.64 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

