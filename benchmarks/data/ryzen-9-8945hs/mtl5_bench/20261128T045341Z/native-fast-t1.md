## ryzen-9-8945hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 63.3 GFLOP/s, f32 126.6 GFLOP/s |
| ... hardware, AVX-512F | f64 63.3, f32 126.6 GFLOP/s |
| ... this build, SSE2 | f64 40.8, f32 81.6 GFLOP/s |
| bandwidth | read 45.8, update 51.9, triad 37.9 GB/s (ceiling 51.9) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 104% | 111% | 0 |
| blas2 | 10 | 64% | 74% | 0 |
| blas3 | 10 | 23% | 31% | 0 |
| factor | 12 | 3% | 6% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 21% | 53% | 0 |
| direct | 19 | 1% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 12% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 25% | 32% | 0 |
| mg | 1 | 25% | 25% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 60% | 60% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.490 ms | 6.74 | 53.90 | 104% | ok |
| blas1 | dot | f32 | n=8388608 | 1.252 ms | 13.40 | 53.59 | 103% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.608 ms | 4.65 | 55.80 | 107% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.743 ms | 9.63 | 57.75 | 111% | ok |
| blas1 | scale | f64 | n=8388608 | 2.336 ms | 3.59 | 57.46 | 111% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.184 ms | 14.17 | 56.69 | 109% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.392 ms | 1.31 | 10.50 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.404 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.422 ms | 1.31 | 10.45 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 103.116 ms | 1.79 | 1.30 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.519 ms | 9.54 | 38.16 | 74% | ok |
| blas2 | gemv | f32 | n=4096 | 1.831 ms | 18.33 | 36.68 | 71% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.043 ms | 8.30 | 33.21 | 64% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.156 ms | 15.56 | 31.14 | 60% | ok |
| blas2 | symv | f64 | n=4096 | 3.389 ms | 9.90 | 19.82 | 38% | ok |
| blas2 | trmv | f64 | n=4096 | 1.889 ms | 8.88 | 35.56 | 68% | ok |
| blas2 | trmv | f32 | n=4096 | 1.015 ms | 16.54 | 33.10 | 64% | ok |
| blas2 | trsv | f64 | n=4096 | 1.922 ms | 8.73 | 34.95 | 67% | ok |
| blas2 | trsv | f32 | n=4096 | 1.010 ms | 16.61 | 33.26 | 64% | ok |
| blas2 | ger | f64 | n=4096 | 16.268 ms | 2.06 | 16.50 | 32% | ok |
| blas3 | gemm | f64 | 512x512x512 | 13.816 ms | 19.43 | 0.46 | 31% | ok |
| blas3 | gemm | f32 | 512x512x512 | 6.934 ms | 38.71 | 0.45 | 31% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 9.298 ms | 14.43 | 0.68 | 23% | ok |
| blas3 | symm | f64 | 512x512x512 | 15.034 ms | 17.86 | 0.42 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.227 ms | 14.55 | 0.45 | 23% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 18.009 ms | 14.91 | 0.35 | 24% | ok |
| blas3 | trmm | f64 | 512x512x512 | 12.069 ms | 11.12 | 0.43 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 12.303 ms | 10.91 | 0.43 | 17% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.090 ms | 16.59 | 0.32 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.253 ms | 0.47 | 3.77 | 7% | ok |
| factor | lu | f64 | 512x512 | 40.050 ms | 2.23 | 0.10 | 4% | ok |
| factor | lu | f32 | 512x512 | 38.025 ms | 2.35 | 0.06 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.309 ms | 3.13 | 0.29 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 13.685 ms | 3.27 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.396 ms | 3.11 | 0.29 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.798 ms | 3.50 | 0.33 | 6% | ok |
| factor | qr | f64 | 512x512 | 813.169 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 87.337 ms | 2.05 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 932.805 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.906 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 111.639 ms | 2.40 | 0.04 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 42.073 ms | 2.13 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 49.386 ms | 0.45 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 163.919 ms | 0.92 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 651.508 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 73.971 ms | 2.27 | 0.01 | 4% | ok |
| eigen | schur | f64 | 256x256 | 143.711 ms | 2.92 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 149.242 ms | 2.47 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 32.163 ms | 7.82 | 0.03 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 323.579 ms | 3.11 | 0.00 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 33.422 ms | 0.25 | 4.02 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 51.987 ms | 0.16 | 2.58 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 49.289 ms | 0.17 | 2.72 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.495 ms | 0.32 | 5.07 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 38.591 ms | 0.22 | 3.48 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.993 ms | 2.64 | 27.43 | 53% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.536 ms | 1.70 | 17.72 | 34% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 8.074 ms | 5.19 | 10.90 | 21% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 22.016 ms | 0.15 | 0.52 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.923 ms | 1.04 | 8.66 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.703 ms | 0.47 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 36.062 ms | 0.31 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 18.034 ms | 0.62 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 64.737 ms | 0.35 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 70.193 ms | 0.32 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.768 ms | 0.73 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 44.259 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.515 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.930 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 76.522 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.226 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.513 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.604 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.701 ms | 0.63 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.353 ms | 2.14 | 8.57 | 17%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.045 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.298 ms | 1.53 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.350 ms | 2.16 | 8.64 | 17%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.083 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.754 ms | 0.39 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.311 ms | 0.58 | 2.31 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.931 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 60.366 ms | 0.37 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.266 ms | 0.60 | 2.39 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.943 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 345.709 ms | 0.89 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.222 ms | 1.52 | 6.07 | 12%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 16.929 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.744 ms | 3.64 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.320 ms | 1.45 | 5.81 | 11%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.615 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1099.721 ms | 0.56 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.142 ms | 0.30 | 1.21 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.586 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 609.193 ms | 1.01 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.101 ms | 0.30 | 1.21 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.204 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.827 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 36.024 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.093 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.749 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 19.926 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 43.431 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 38.215 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.014 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.703 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.300 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 22.354 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 27.200 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.058 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.915 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 27.799 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.215 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.168 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.816 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.741 ms | 1.50 | 16.85 | 32% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.256 ms | 1.16 | 13.00 | 25% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.220 ms | 1.18 | 13.21 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 22.019 ms | 1.75 | 13.20 | 25% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.396 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.189 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 65.109 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.085 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 2.172 ms | 7.72 | 30.90 | 60% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 19.335 ms | 13.88 | 0.22 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.575 ms | 29.17 | 31.00 | 60% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.452 ms | 11.55 | 46.21 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.586 ms | 28.61 | 57.22 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.430 ms | 38.98 | 38.98 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.388 ms | 43.20 | 43.20 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.846 ms | 14.24 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 21.980 ms | 12.21 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.435 ms | 11.97 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.592 ms | 58.46 | 0.34 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.779 ms | 56.16 | 0.33 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

