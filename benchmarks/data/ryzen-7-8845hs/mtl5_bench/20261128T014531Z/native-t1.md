## ryzen-7-8845hs -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 80.0 GFLOP/s, f32 159.9 GFLOP/s |
| ... hardware, AVX-512F | f64 80.0, f32 159.9 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 43.0, update 52.8, triad 38.5 GB/s (ceiling 52.8) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 111% | 0 |
| blas2 | 10 | 46% | 85% | 0 |
| blas3 | 10 | 14% | 21% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 10% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 26% | 61% | 0 |
| direct | 19 | 1% | 21% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 36% | 39% | 0 |
| mg | 1 | 33% | 33% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 12% | 23% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.507 ms | 1.46 | 11.66 | 22% | ok |
| blas1 | dot | f32 | n=8388608 | 9.822 ms | 1.71 | 6.83 | 13% | ok |
| blas1 | axpy | f64 | n=8388608 | 11.908 ms | 1.41 | 16.91 | 32% | ok |
| blas1 | axpy | f32 | n=8388608 | 11.871 ms | 1.41 | 8.48 | 16% | ok |
| blas1 | scale | f64 | n=8388608 | 2.292 ms | 3.66 | 58.57 | 111% | ok |
| blas1 | two_norm | f64 | n=8388608 | 11.014 ms | 1.52 | 6.09 | 12% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.075 ms | 1.65 | 13.22 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 85.865 ms | 2.15 | 1.56 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 2.989 ms | 11.23 | 44.93 | 85% | ok |
| blas2 | gemv | f32 | n=4096 | 2.732 ms | 12.28 | 24.58 | 46% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.471 ms | 5.18 | 20.75 | 39% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.244 ms | 14.95 | 29.92 | 57% | ok |
| blas2 | symv | f64 | n=4096 | 2.909 ms | 11.53 | 23.09 | 44% | ok |
| blas2 | trmv | f64 | n=4096 | 1.713 ms | 9.79 | 39.21 | 74% | ok |
| blas2 | trmv | f32 | n=4096 | 1.547 ms | 10.84 | 21.71 | 41% | ok |
| blas2 | trsv | f64 | n=4096 | 1.757 ms | 9.55 | 38.24 | 72% | ok |
| blas2 | trsv | f32 | n=4096 | 1.503 ms | 11.16 | 22.35 | 42% | ok |
| blas2 | ger | f64 | n=4096 | 13.435 ms | 2.50 | 19.98 | 38% | ok |
| blas3 | gemm | f64 | 512x512x512 | 16.006 ms | 16.77 | 0.39 | 21% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.166 ms | 32.87 | 0.39 | 21% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.329 ms | 11.85 | 0.56 | 15% | ok |
| blas3 | symm | f64 | 512x512x512 | 16.616 ms | 16.16 | 0.38 | 20% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.589 ms | 9.88 | 0.31 | 12% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 24.526 ms | 10.94 | 0.26 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.039 ms | 5.36 | 0.21 | 7% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.839 ms | 5.00 | 0.20 | 6% | ok |
| blas3 | trsm | f32 | 512x512x512 | 25.965 ms | 5.17 | 0.10 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.536 ms | 0.62 | 4.97 | 9% | ok |
| factor | lu | f64 | 512x512 | 32.566 ms | 2.75 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 30.846 ms | 2.90 | 0.07 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.357 ms | 3.94 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 10.952 ms | 4.09 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.426 ms | 3.92 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.289 ms | 4.35 | 0.41 | 5% | ok |
| factor | qr | f64 | 512x512 | 647.893 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 64.630 ms | 2.77 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 744.249 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.047 ms | 0.26 | 1.04 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 88.592 ms | 3.03 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 33.763 ms | 2.65 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.216 ms | 0.60 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 129.477 ms | 1.17 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 513.378 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 59.803 ms | 2.81 | 0.02 | 4% | ok |
| eigen | schur | f64 | 256x256 | 109.246 ms | 3.84 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 118.074 ms | 3.13 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 29.965 ms | 8.40 | 0.03 | 10% | ok |
| eigen | sylvester | f64 | 256x256 | 263.305 ms | 3.82 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 26.722 ms | 0.31 | 5.02 | 10% | ok |
| elementwise | log | f64 | n=8388608 | 41.559 ms | 0.20 | 3.23 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 38.701 ms | 0.22 | 3.47 | 7% | ok |
| elementwise | sqrt | f64 | n=8388608 | 20.576 ms | 0.41 | 6.52 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 30.618 ms | 0.27 | 4.38 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.839 ms | 3.12 | 32.47 | 61% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.409 ms | 1.86 | 19.32 | 37% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.505 ms | 6.44 | 13.54 | 26% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.216 ms | 0.20 | 0.71 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.699 ms | 1.18 | 9.80 | 19%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.769 ms | 0.60 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.515 ms | 0.35 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 14.804 ms | 0.76 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 52.482 ms | 0.43 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 55.901 ms | 0.40 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 26.636 ms | 0.84 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 35.800 ms | 0.05 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.625 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.577 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 61.361 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.230 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 27.721 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.987 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.004 ms | 0.80 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.290 ms | 2.61 | 10.44 | 20%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.521 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.649 ms | 1.98 | 0.13 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.274 ms | 2.76 | 11.04 | 21%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.261 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 46.932 ms | 0.48 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.141 ms | 0.66 | 2.65 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.711 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.238 ms | 0.46 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.105 ms | 0.68 | 2.74 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.553 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 253.227 ms | 1.22 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.951 ms | 1.73 | 6.91 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.635 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 68.344 ms | 4.52 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.053 ms | 1.64 | 6.57 | 12%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.038 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 888.570 ms | 0.69 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.492 ms | 0.32 | 1.28 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.395 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 496.105 ms | 1.24 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.465 ms | 0.32 | 1.29 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.574 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 60.658 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 32.280 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.367 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.489 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 18.026 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 58.416 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.195 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.279 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 27.311 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 25.882 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 27.042 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 35.748 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.380 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 22.052 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.807 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.734 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.647 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.441 ms | 1.82 | 20.35 | 39% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.560 ms | 1.68 | 18.79 | 36% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.826 ms | 1.43 | 16.06 | 30% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 16.764 ms | 2.29 | 17.34 | 33% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.056 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.457 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 51.722 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.989 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 11.387 ms | 1.47 | 5.89 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.061 ms | 9.24 | 0.14 | 12% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.468 ms | 11.43 | 12.14 | 23% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.201 ms | 5.24 | 20.96 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.880 ms | 5.82 | 11.65 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.385 ms | 12.12 | 12.12 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.388 ms | 12.09 | 12.09 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.156 ms | 32.91 | 0.39 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.275 ms | 8.87 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 27.303 ms | 9.83 | 0.06 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 27.305 ms | 9.83 | 0.06 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.277 ms | 9.84 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

