## ryzen-9-8945hs -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 126.6 GFLOP/s |
| ... hardware, AVX-512F | f64 67.5, f32 126.6 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 36.6, update 51.4, triad 37.9 GB/s (ceiling 51.4) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 99% | 0 |
| blas2 | 10 | 46% | 80% | 0 |
| blas3 | 10 | 14% | 21% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 10% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 22% | 58% | 0 |
| direct | 19 | 1% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 12% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 29% | 33% | 0 |
| mg | 1 | 25% | 25% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 16% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 14.454 ms | 1.16 | 9.29 | 18% | ok |
| blas1 | dot | f32 | n=8388608 | 12.291 ms | 1.37 | 5.46 | 11% | ok |
| blas1 | axpy | f64 | n=8388608 | 15.000 ms | 1.12 | 13.42 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 14.954 ms | 1.12 | 6.73 | 13% | ok |
| blas1 | scale | f64 | n=8388608 | 2.645 ms | 3.17 | 50.75 | 99% | ok |
| blas1 | two_norm | f64 | n=8388608 | 13.303 ms | 1.26 | 5.04 | 10% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.391 ms | 1.31 | 10.50 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.423 ms | 1.31 | 10.45 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.414 ms | 1.31 | 10.46 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 113.510 ms | 1.63 | 1.18 | 2% | ok |
| blas2 | gemv | f64 | n=4096 | 3.266 ms | 10.27 | 41.11 | 80% | ok |
| blas2 | gemv | f32 | n=4096 | 2.837 ms | 11.83 | 23.66 | 46% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.745 ms | 4.33 | 17.34 | 34% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.490 ms | 13.48 | 26.97 | 52% | ok |
| blas2 | symv | f64 | n=4096 | 3.482 ms | 9.64 | 19.29 | 38% | ok |
| blas2 | trmv | f64 | n=4096 | 1.884 ms | 8.91 | 35.66 | 69% | ok |
| blas2 | trmv | f32 | n=4096 | 1.767 ms | 9.50 | 19.01 | 37% | ok |
| blas2 | trsv | f64 | n=4096 | 1.935 ms | 8.67 | 34.72 | 68% | ok |
| blas2 | trsv | f32 | n=4096 | 1.940 ms | 8.65 | 17.31 | 34% | ok |
| blas2 | ger | f64 | n=4096 | 16.257 ms | 2.06 | 16.52 | 32% | ok |
| blas3 | gemm | f64 | 512x512x512 | 19.996 ms | 13.42 | 0.31 | 20% | ok |
| blas3 | gemm | f32 | 512x512x512 | 9.943 ms | 27.00 | 0.32 | 21% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 13.878 ms | 9.67 | 0.45 | 14% | ok |
| blas3 | symm | f64 | 512x512x512 | 20.605 ms | 13.03 | 0.31 | 19% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.795 ms | 7.99 | 0.25 | 12% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 27.743 ms | 9.68 | 0.23 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 30.774 ms | 4.36 | 0.17 | 6% | ok |
| blas3 | trsm | f64 | 512x512x512 | 35.275 ms | 3.80 | 0.15 | 6% | ok |
| blas3 | trsm | f32 | 512x512x512 | 32.088 ms | 4.18 | 0.08 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.296 ms | 0.47 | 3.76 | 7% | ok |
| factor | lu | f64 | 512x512 | 39.894 ms | 2.24 | 0.11 | 3% | ok |
| factor | lu | f32 | 512x512 | 37.897 ms | 2.36 | 0.06 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.309 ms | 3.13 | 0.29 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 13.711 ms | 3.26 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.390 ms | 3.11 | 0.29 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.995 ms | 3.44 | 0.32 | 5% | ok |
| factor | qr | f64 | 512x512 | 815.495 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 80.556 ms | 2.22 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 931.200 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.889 ms | 0.21 | 0.86 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 111.231 ms | 2.41 | 0.04 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 41.339 ms | 2.16 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 48.426 ms | 0.46 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 165.469 ms | 0.91 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 647.244 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 76.220 ms | 2.20 | 0.01 | 3% | ok |
| eigen | schur | f64 | 256x256 | 142.628 ms | 2.94 | 0.01 | 4% | ok |
| eigen | svd | f64 | 256x256 | 149.291 ms | 2.47 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 37.071 ms | 6.79 | 0.03 | 10% | ok |
| eigen | sylvester | f64 | 256x256 | 333.307 ms | 3.02 | 0.00 | 4% | ok |
| elementwise | exp | f64 | n=8388608 | 34.903 ms | 0.24 | 3.85 | 7% | ok |
| elementwise | log | f64 | n=8388608 | 51.687 ms | 0.16 | 2.60 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 48.745 ms | 0.17 | 2.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.465 ms | 0.32 | 5.07 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 38.549 ms | 0.22 | 3.48 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.915 ms | 2.86 | 29.76 | 58% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.728 ms | 1.51 | 15.76 | 31% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.767 ms | 5.39 | 11.34 | 22% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.887 ms | 0.15 | 0.53 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.931 ms | 1.04 | 8.62 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.937 ms | 0.47 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.756 ms | 0.29 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 20.077 ms | 0.56 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.431 ms | 0.34 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 70.257 ms | 0.32 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.387 ms | 0.74 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 44.011 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.585 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.012 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 76.067 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.272 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.879 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.607 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.509 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.364 ms | 2.08 | 8.31 | 16%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.082 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.292 ms | 1.53 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.345 ms | 2.19 | 8.77 | 17%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.127 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.613 ms | 0.39 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.302 ms | 0.58 | 2.32 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.949 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.945 ms | 0.37 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.282 ms | 0.59 | 2.36 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.980 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 339.949 ms | 0.91 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.249 ms | 1.50 | 5.99 | 12%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.131 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.248 ms | 3.66 | 0.01 | 5% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.265 ms | 1.49 | 5.95 | 12%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.654 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1106.584 ms | 0.56 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.276 ms | 0.30 | 1.20 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.652 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 611.081 ms | 1.01 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.124 ms | 0.30 | 1.21 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 51.144 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 77.659 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.799 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.976 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.433 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.861 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 73.807 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 44.409 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.639 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.471 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.713 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.267 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 45.219 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.054 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.937 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.000 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.257 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.164 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.800 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.720 ms | 1.52 | 17.05 | 33% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.987 ms | 1.32 | 14.76 | 29% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.247 ms | 1.16 | 13.05 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 22.703 ms | 1.69 | 12.80 | 25% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.572 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.200 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 65.417 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.890 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 14.384 ms | 1.17 | 4.67 | 9% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 36.700 ms | 7.31 | 0.11 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.181 ms | 7.69 | 8.17 | 16% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.198 ms | 2.71 | 10.83 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.681 ms | 4.56 | 9.11 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.72 | 9.72 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.864 ms | 27.21 | 0.32 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.300 ms | 7.01 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 34.544 ms | 7.77 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 34.554 ms | 7.77 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.467 ms | 7.79 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

