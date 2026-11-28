## ryzen-9-8945hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 63.4 GFLOP/s, f32 126.5 GFLOP/s |
| ... hardware, AVX-512F | f64 63.4, f32 126.5 GFLOP/s |
| ... this build, SSE2 | f64 40.8, f32 81.4 GFLOP/s |
| bandwidth | read 44.4, update 48.5, triad 37.4 GB/s (ceiling 48.5) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 111% | 119% | 0 |
| blas2 | 10 | 66% | 77% | 0 |
| blas3 | 10 | 23% | 30% | 0 |
| factor | 12 | 3% | 6% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 22% | 54% | 0 |
| direct | 19 | 1% | 8% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 32% | 39% | 0 |
| mg | 1 | 30% | 30% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 64% | 64% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.516 ms | 6.67 | 53.34 | 110% | ok |
| blas1 | dot | f32 | n=8388608 | 1.249 ms | 13.43 | 53.73 | 111% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.648 ms | 4.60 | 55.19 | 114% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.753 ms | 9.57 | 57.42 | 119% | ok |
| blas1 | scale | f64 | n=8388608 | 2.344 ms | 3.58 | 57.25 | 118% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.169 ms | 14.35 | 57.40 | 118% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.399 ms | 1.31 | 10.49 | 22% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.406 ms | 1.31 | 10.48 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 6.423 ms | 1.31 | 10.45 | 22% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 101.404 ms | 1.82 | 1.32 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.580 ms | 9.37 | 37.51 | 77% | ok |
| blas2 | gemv | f32 | n=4096 | 1.801 ms | 18.63 | 37.27 | 77% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.230 ms | 7.93 | 31.75 | 66% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.316 ms | 14.49 | 28.99 | 60% | ok |
| blas2 | symv | f64 | n=4096 | 3.606 ms | 9.31 | 18.63 | 38% | ok |
| blas2 | trmv | f64 | n=4096 | 2.078 ms | 8.07 | 32.32 | 67% | ok |
| blas2 | trmv | f32 | n=4096 | 1.143 ms | 14.67 | 29.38 | 61% | ok |
| blas2 | trsv | f64 | n=4096 | 2.098 ms | 8.00 | 32.02 | 66% | ok |
| blas2 | trsv | f32 | n=4096 | 1.226 ms | 13.69 | 27.41 | 57% | ok |
| blas2 | ger | f64 | n=4096 | 16.269 ms | 2.06 | 16.50 | 34% | ok |
| blas3 | gemm | f64 | 512x512x512 | 14.016 ms | 19.15 | 0.45 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 6.961 ms | 38.56 | 0.45 | 30% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 9.395 ms | 14.29 | 0.67 | 23% | ok |
| blas3 | symm | f64 | 512x512x512 | 15.194 ms | 17.67 | 0.41 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.299 ms | 14.43 | 0.45 | 23% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 18.056 ms | 14.87 | 0.35 | 23% | ok |
| blas3 | trmm | f64 | 512x512x512 | 11.804 ms | 11.37 | 0.44 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 12.012 ms | 11.17 | 0.44 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 7.988 ms | 16.80 | 0.33 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.558 ms | 0.50 | 4.02 | 8% | ok |
| factor | lu | f64 | 512x512 | 40.443 ms | 2.21 | 0.10 | 3% | ok |
| factor | lu | f32 | 512x512 | 38.978 ms | 2.30 | 0.05 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.299 ms | 3.13 | 0.29 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 13.714 ms | 3.26 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.436 ms | 3.10 | 0.29 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.679 ms | 3.53 | 0.33 | 6% | ok |
| factor | qr | f64 | 512x512 | 823.534 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 81.683 ms | 2.19 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 939.169 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.888 ms | 0.21 | 0.86 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 110.040 ms | 2.44 | 0.04 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 44.106 ms | 2.03 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 48.625 ms | 0.46 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 162.039 ms | 0.93 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 643.712 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 75.110 ms | 2.23 | 0.01 | 4% | ok |
| eigen | schur | f64 | 256x256 | 147.411 ms | 2.85 | 0.01 | 4% | ok |
| eigen | svd | f64 | 256x256 | 151.419 ms | 2.44 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 32.272 ms | 7.80 | 0.03 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 329.892 ms | 3.05 | 0.00 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 36.458 ms | 0.23 | 3.68 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 50.928 ms | 0.16 | 2.64 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 48.061 ms | 0.17 | 2.79 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 25.489 ms | 0.33 | 5.27 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 37.700 ms | 0.22 | 3.56 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.042 ms | 2.51 | 26.12 | 54% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.555 ms | 1.68 | 17.51 | 36% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 8.177 ms | 5.12 | 10.77 | 22% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.800 ms | 0.16 | 0.58 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.947 ms | 1.03 | 8.55 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.915 ms | 0.47 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 36.053 ms | 0.31 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 17.899 ms | 0.62 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.004 ms | 0.34 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.871 ms | 0.32 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 29.862 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 44.265 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.559 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.061 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 78.159 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.231 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.596 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.614 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.522 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.764 ms | 0.99 | 3.96 | 8%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.117 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.076 ms | 1.58 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.948 ms | 0.80 | 3.19 | 7%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.177 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 58.854 ms | 0.38 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.591 ms | 0.48 | 1.90 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.093 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.207 ms | 0.38 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.125 ms | 0.67 | 2.69 | 6%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 7.066 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 348.877 ms | 0.88 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.189 ms | 1.54 | 6.16 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 16.935 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 86.666 ms | 3.56 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.239 ms | 1.50 | 6.02 | 12%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.795 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1147.161 ms | 0.54 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.218 ms | 0.30 | 1.20 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.569 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 624.397 ms | 0.99 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.209 ms | 0.30 | 1.20 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.372 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.803 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 36.163 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.102 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.707 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 20.044 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 43.385 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.052 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 21.947 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.689 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.209 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 22.369 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 27.403 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.058 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.929 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.658 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.245 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.141 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.822 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.567 ms | 1.67 | 18.71 | 39% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.911 ms | 1.37 | 15.35 | 32% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.236 ms | 1.17 | 13.11 | 27% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.845 ms | 1.94 | 14.65 | 30% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 49.780 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.512 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 71.936 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.075 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 2.175 ms | 7.71 | 30.86 | 64% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 19.334 ms | 13.88 | 0.22 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.572 ms | 29.33 | 31.16 | 64% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.479 ms | 11.35 | 45.39 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.599 ms | 28.02 | 56.05 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.430 ms | 38.98 | 38.98 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.444 ms | 37.79 | 37.79 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 19.127 ms | 14.03 | 0.16 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.103 ms | 12.14 | 0.09 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.445 ms | 11.96 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.580 ms | 58.61 | 0.34 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.553 ms | 58.96 | 0.35 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

