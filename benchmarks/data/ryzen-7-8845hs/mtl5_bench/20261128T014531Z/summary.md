# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy power scheme copy of Balanced with processor state 100-100% and EPP 0, boost aggressive; AC, 2026-11-28T01:57:11Z.

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

## ryzen-7-8845hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 628.0 GFLOP/s, f32 1255.4 GFLOP/s |
| ... hardware, AVX-512F | f64 628.0, f32 1255.4 GFLOP/s |
| ... this build, scalar | f64 12.4, f32 14.4 GFLOP/s |
| bandwidth | read 53.1, update 53.7, triad 41.2 GB/s (ceiling 53.7) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 94% | 127% | 0 | 43% |
| blas2 | 10 | 74% | 109% | 0 | 13% |
| blas3 | 10 | 2% | 18% | 0 | 12% |
| factor | 12 | 1% | 8% | 0 | 19% |
| eigen | 8 | 0% | 2% | 0 | 12% |
| elementwise | 5 | 8% | 12% | 0 | 12% |
| sparse | 5 | 36% | 289% | 0 | 13% |
| direct | 19 | 0% | 20% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 12% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 31% | 35% | 0 | 12% |
| mg | 1 | 31% | 31% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 10% |
| mixed | 3 | 10% | 19% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 54% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.269 ms | 7.39 | 59.14 | 110% | ok | 5.1x (63%) |
| blas1 | dot | f32 | n=8388608 | 1.334 ms | 12.57 | 50.30 | 94% | ok | 7.4x (92%) |
| blas1 | axpy | f64 | n=8388608 | 3.500 ms | 4.79 | 57.53 | 107% | ok | 3.4x (43%) |
| blas1 | axpy | f32 | n=8388608 | 1.765 ms | 9.50 | 57.03 | 106% | ok | 6.7x (84%) |
| blas1 | scale | f64 | n=8388608 | 1.965 ms | 4.27 | 68.31 | 127% | ok | 1.2x (15%) |
| blas1 | two_norm | f64 | n=8388608 | 1.487 ms | 11.29 | 45.14 | 84% | ok | 7.4x (93%) |
| blas1 | one_norm | f64 | n=8388608 | 5.065 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.189 ms | 1.62 | 12.93 | 24% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 83.357 ms | 2.21 | 1.61 | 3% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.367 ms | 14.18 | 56.74 | 106% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.187 ms | 28.27 | 56.57 | 105% | ok | 2.3x (29%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.288 ms | 14.67 | 58.70 | 109% | ok | 2.8x (35%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.152 ms | 29.13 | 58.29 | 109% | ok | 1.9x (24%) |
| blas2 | symv | f64 | n=4096 | 3.018 ms | 11.12 | 22.26 | 41% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.680 ms | 9.98 | 39.98 | 74% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.546 ms | 10.85 | 21.72 | 40% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.683 ms | 9.97 | 39.92 | 74% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.611 ms | 10.41 | 20.85 | 39% | ok | 0.9x (12%) |
| blas2 | ger | f64 | n=4096 | 13.601 ms | 2.47 | 19.74 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.476 ms | 108.41 | 2.54 | 17% | ok | 6.5x (81%) |
| blas3 | gemm | f32 | 512x512x512 | 1.197 ms | 224.35 | 2.63 | 18% | ok | 6.8x (85%) |
| blas3 | gemmt | f64 | 512x512x512 | 11.433 ms | 11.74 | 0.55 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 17.021 ms | 15.77 | 0.37 | 3% | ok | 1.0x (12%) |
| blas3 | syrk | f64 | 512x512x512 | 13.679 ms | 9.81 | 0.31 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 22.882 ms | 11.73 | 0.27 | 2% | ok | 1.1x (13%) |
| blas3 | trmm | f64 | 512x512x512 | 25.288 ms | 5.31 | 0.21 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f64 | 512x512x512 | 26.927 ms | 4.98 | 0.19 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f32 | 512x512x512 | 26.234 ms | 5.12 | 0.10 | 0% | ok | 1.0x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.910 ms | 0.60 | 4.77 | 9% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 21.862 ms | 4.09 | 0.19 | 1% | ok | 1.5x (19%) |
| factor | lu | f32 | 512x512 | 20.784 ms | 4.31 | 0.10 | 0% | ok | 1.5x (19%) |
| factor | cholesky | f64 | 512x512 | 11.364 ms | 3.94 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.911 ms | 4.10 | 0.19 | 0%\* | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 11.440 ms | 3.91 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.410 ms | 4.30 | 0.40 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 397.690 ms | 0.45 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 41.640 ms | 4.30 | 0.10 | 1% | ok | 1.6x (19%) |
| factor | hessenberg | f64 | 512x512 | 457.013 ms | 0.98 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.010 ms | 1.04 | 4.15 | 8%\* | ok | 4.0x (50%) |
| factor | inv | f64 | 512x512 | 78.585 ms | 3.42 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.347 ms | 3.83 | 0.18 | 1% | ok | 1.4x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.423 ms | 0.60 | 0.03 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 129.979 ms | 1.16 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 511.502 ms | 0.30 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 60.275 ms | 2.78 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 114.301 ms | 3.67 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 118.784 ms | 3.11 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 18.532 ms | 13.58 | 0.06 | 2% | ok | 1.6x (20%) |
| eigen | sylvester | f64 | 256x256 | 260.969 ms | 3.86 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 32.121 ms | 0.26 | 4.18 | 8% | ok | 0.8x (10%) |
| elementwise | log | f64 | n=8388608 | 41.906 ms | 0.20 | 3.20 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 39.417 ms | 0.21 | 3.41 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 20.620 ms | 0.41 | 6.51 | 12% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 31.274 ms | 0.27 | 4.29 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.176 ms | 14.90 | 154.98 | 289% ! | ok | 4.8x (60%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.427 ms | 1.83 | 19.08 | 36% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.237 ms | 18.72 | 39.36 | 73% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.197 ms | 0.20 | 0.71 | 1%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.599 ms | 1.26 | 10.41 | 19%\* | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.467 ms | 0.57 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.562 ms | 0.35 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.198 ms | 0.74 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.285 ms | 0.42 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.566 ms | 0.40 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 24.764 ms | 0.90 | 0.03 | 0% | ok | 1.1x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 36.117 ms | 0.05 | 0.01 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.733 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.671 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 61.539 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.227 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.290 ms | 0.00 | 0.00 | n/a | ok | 1.9x (24%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.040 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.294 ms | 0.78 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.277 ms | 2.73 | 10.91 | 20%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.702 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.685 ms | 1.97 | 0.13 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.280 ms | 2.70 | 10.81 | 20%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.327 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.056 ms | 0.48 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.139 ms | 0.66 | 2.66 | 5%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.780 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.877 ms | 0.46 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.115 ms | 0.68 | 2.71 | 5%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 257.247 ms | 1.20 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.024 ms | 1.66 | 6.66 | 12%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.753 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.116 ms | 4.46 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.107 ms | 1.60 | 6.39 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.082 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 914.859 ms | 0.67 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.864 ms | 0.31 | 1.24 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.813 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 543.510 ms | 1.14 | 0.00 | 0% | ok | 0.9x (11%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 12.045 ms | 0.28 | 1.12 | 2%\* | ok | 0.9x (11%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 45.511 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 67.749 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 36.663 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 20.141 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 26.011 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 20.831 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 60.298 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.792 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.452 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 27.974 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 26.358 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 27.750 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 36.427 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.434 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 24.120 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| pc | ic_0 | f64 | 128^2 grid | 1.863 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.738 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.698 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.790 ms | 1.46 | 16.38 | 31% | ok | 0.8x (10%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.552 ms | 1.69 | 18.89 | 35% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.931 ms | 1.36 | 15.19 | 28% | ok | 0.9x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 17.439 ms | 2.20 | 16.67 | 31% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 34.631 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 42.447 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 77.355 ms | 0.00 | 0.00 | n/a | ok | 0.7x (8%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 10.113 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| mixed | dot | f32>f64 | n=8388608 | 12.254 ms | 1.37 | 5.48 | 10% | ok | 0.9x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.640 ms | 47.60 | 0.74 | 8% | ok | 5.2x (64%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.745 ms | 9.62 | 10.22 | 19% | ok | 0.8x (11%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.239 ms | 5.18 | 20.72 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.921 ms | 5.74 | 11.49 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.418 ms | 11.83 | 11.83 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.418 ms | 11.83 | 11.83 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.902 ms | 141.12 | 1.65 | n/a | ok | 4.3x (54%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.601 ms | 40.67 | 0.32 | n/a | ok | 4.6x (57%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.949 ms | 45.13 | 0.26 | n/a | ok | 4.6x (57%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.879 ms | 45.66 | 0.27 | n/a | ok | 4.6x (58%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 6.052 ms | 44.36 | 0.26 | n/a | ok | 4.5x (56%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## ryzen-7-8845hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 79.2 GFLOP/s, f32 159.0 GFLOP/s |
| ... hardware, AVX-512F | f64 79.2, f32 159.0 GFLOP/s |
| ... this build, SSE2 | f64 39.6, f32 79.9 GFLOP/s |
| bandwidth | read 40.9, update 46.5, triad 34.1 GB/s (ceiling 46.5) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 100% | 117% | 0 |
| blas2 | 10 | 76% | 81% | 0 |
| blas3 | 10 | 23% | 30% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 29% | 66% | 0 |
| direct | 19 | 1% | 22% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 15% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 41% | 41% | 0 |
| mg | 1 | 39% | 39% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 81% | 81% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.895 ms | 5.80 | 46.37 | 100% | ok |
| blas1 | dot | f32 | n=8388608 | 1.519 ms | 11.04 | 44.18 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.203 ms | 3.99 | 47.90 | 103% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.022 ms | 8.30 | 49.78 | 107% | ok |
| blas1 | scale | f64 | n=8388608 | 2.461 ms | 3.41 | 54.54 | 117% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.377 ms | 12.18 | 48.73 | 105% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.127 ms | 1.64 | 13.09 | 28% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.244 ms | 1.34 | 10.75 | 23% | ok |
| blas1 | sum | f64 | n=8388608 | 5.124 ms | 1.64 | 13.10 | 28% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 98.609 ms | 1.87 | 1.36 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.692 ms | 9.09 | 36.38 | 78% | ok |
| blas2 | gemv | f32 | n=4096 | 1.775 ms | 18.90 | 37.82 | 81% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.972 ms | 8.45 | 33.81 | 73% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.997 ms | 16.80 | 33.62 | 72% | ok |
| blas2 | symv | f64 | n=4096 | 2.730 ms | 12.29 | 24.60 | 53% | ok |
| blas2 | trmv | f64 | n=4096 | 1.898 ms | 8.84 | 35.39 | 76% | ok |
| blas2 | trmv | f32 | n=4096 | 0.956 ms | 17.55 | 35.14 | 76% | ok |
| blas2 | trsv | f64 | n=4096 | 1.832 ms | 9.16 | 36.66 | 79% | ok |
| blas2 | trsv | f32 | n=4096 | 0.963 ms | 17.42 | 34.87 | 75% | ok |
| blas2 | ger | f64 | n=4096 | 13.712 ms | 2.45 | 19.58 | 42% | ok |
| blas3 | gemm | f64 | 512x512x512 | 11.219 ms | 23.93 | 0.56 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 5.659 ms | 47.44 | 0.56 | 30% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 7.710 ms | 17.41 | 0.82 | 22% | ok |
| blas3 | symm | f64 | 512x512x512 | 12.211 ms | 21.98 | 0.52 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 7.474 ms | 17.96 | 0.56 | 23% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 14.459 ms | 18.57 | 0.44 | 23% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.702 ms | 13.83 | 0.54 | 17% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.688 ms | 13.85 | 0.54 | 17% | ok |
| blas3 | trsm | f32 | 512x512x512 | 6.460 ms | 20.78 | 0.41 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.893 ms | 0.60 | 4.78 | 10% | ok |
| factor | lu | f64 | 512x512 | 32.895 ms | 2.72 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 32.398 ms | 2.76 | 0.06 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.391 ms | 3.93 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 10.895 ms | 4.11 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.431 ms | 3.91 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.414 ms | 4.30 | 0.40 | 5% | ok |
| factor | qr | f64 | 512x512 | 650.452 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 64.487 ms | 2.78 | 0.07 | 4% | ok |
| factor | hessenberg | f64 | 512x512 | 745.886 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.937 ms | 0.27 | 1.07 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 88.567 ms | 3.03 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 33.827 ms | 2.65 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.326 ms | 0.60 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 131.031 ms | 1.15 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 509.109 ms | 0.30 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 60.823 ms | 2.76 | 0.02 | 3% | ok |
| eigen | schur | f64 | 256x256 | 110.037 ms | 3.81 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 118.573 ms | 3.11 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 25.940 ms | 9.70 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 263.817 ms | 3.82 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 27.009 ms | 0.31 | 4.97 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 41.384 ms | 0.20 | 3.24 | 7% | ok |
| elementwise | sin | f64 | n=8388608 | 38.934 ms | 0.22 | 3.45 | 7% | ok |
| elementwise | sqrt | f64 | n=8388608 | 20.509 ms | 0.41 | 6.54 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 30.969 ms | 0.27 | 4.33 | 9% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.887 ms | 2.95 | 30.70 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.321 ms | 1.98 | 20.62 | 44% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.603 ms | 6.34 | 13.33 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.151 ms | 0.20 | 0.71 | 2%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.719 ms | 1.17 | 9.68 | 21%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.747 ms | 0.60 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 28.465 ms | 0.39 | 0.03 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 13.624 ms | 0.82 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 55.662 ms | 0.40 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.700 ms | 0.39 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 24.507 ms | 0.91 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.194 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.655 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.595 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.527 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.223 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.056 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.924 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.373 ms | 0.78 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.290 ms | 2.60 | 10.42 | 22%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.562 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.587 ms | 2.00 | 0.13 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.292 ms | 2.59 | 10.35 | 22%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.269 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.393 ms | 0.47 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.167 ms | 0.65 | 2.59 | 6%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.746 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.768 ms | 0.46 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.120 ms | 0.67 | 2.70 | 6%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.586 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 258.664 ms | 1.19 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.980 ms | 1.70 | 6.81 | 15%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.527 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.409 ms | 4.45 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.015 ms | 1.67 | 6.69 | 14%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.072 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 897.791 ms | 0.69 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.621 ms | 0.32 | 1.27 | 3%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.411 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 496.157 ms | 1.24 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.494 ms | 0.32 | 1.28 | 3%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.625 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 45.816 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 28.540 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.511 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.985 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.779 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 34.474 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.511 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.522 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.722 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 17.670 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 19.438 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.572 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.392 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 22.727 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.818 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.745 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.669 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.536 ms | 1.70 | 19.09 | 41% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.545 ms | 1.69 | 18.98 | 41% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.796 ms | 1.46 | 16.33 | 35% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 16.142 ms | 2.38 | 18.01 | 39% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.215 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.735 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.640 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.516 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.786 ms | 9.39 | 37.56 | 81% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 15.285 ms | 17.56 | 0.27 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.476 ms | 35.25 | 37.45 | 81% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.244 ms | 13.49 | 53.96 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.471 ms | 35.59 | 71.18 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.349 ms | 48.14 | 48.14 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.318 ms | 52.81 | 52.81 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 14.999 ms | 17.90 | 0.21 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 17.363 ms | 15.46 | 0.12 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 17.736 ms | 15.14 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.722 ms | 72.12 | 0.42 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.698 ms | 72.59 | 0.43 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-7-8845hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 624.5 GFLOP/s, f32 1248.4 GFLOP/s |
| ... hardware, AVX-512F | f64 624.5, f32 1248.4 GFLOP/s |
| ... this build, SSE2 | f64 312.2, f32 624.2 GFLOP/s |
| bandwidth | read 53.1, update 53.4, triad 41.0 GB/s (ceiling 53.4) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 112% | 128% | 0 | 15% |
| blas2 | 10 | 68% | 111% | 0 | 13% |
| blas3 | 10 | 9% | 19% | 0 | 39% |
| factor | 12 | 1% | 8% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 12% |
| elementwise | 5 | 8% | 12% | 0 | 12% |
| sparse | 5 | 39% | 291% | 0 | 13% |
| direct | 19 | 0% | 20% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 12% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 35% | 35% | 0 | 12% |
| mg | 1 | 36% | 36% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 69% | 70% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.236 ms | 7.50 | 60.03 | 112% | ok | 1.3x (16%) |
| blas1 | dot | f32 | n=8388608 | 0.982 ms | 17.08 | 68.32 | 128% | ok | 1.5x (19%) |
| blas1 | axpy | f64 | n=8388608 | 3.464 ms | 4.84 | 58.12 | 109% | ok | 1.2x (15%) |
| blas1 | axpy | f32 | n=8388608 | 1.610 ms | 10.42 | 62.52 | 117% | ok | 1.3x (16%) |
| blas1 | scale | f64 | n=8388608 | 2.028 ms | 4.14 | 66.17 | 124% | ok | 1.2x (15%) |
| blas1 | two_norm | f64 | n=8388608 | 1.107 ms | 15.15 | 60.61 | 113% | ok | 1.2x (16%) |
| blas1 | one_norm | f64 | n=8388608 | 5.079 ms | 1.65 | 13.21 | 25% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.100 ms | 1.64 | 13.16 | 25% | ok | 1.2x (15%) |
| blas1 | sum | f64 | n=8388608 | 5.063 ms | 1.66 | 13.26 | 25% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 82.224 ms | 2.24 | 1.63 | 3% | ok | 1.2x (15%) |
| blas2 | gemv | f64 | n=4096 | 2.283 ms | 14.70 | 58.82 | 110% | ok | 1.6x (20%) |
| blas2 | gemv | f32 | n=4096 | 1.131 ms | 29.67 | 59.37 | 111% | ok | 1.6x (20%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.316 ms | 14.49 | 57.97 | 109% | ok | 1.7x (21%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.193 ms | 28.12 | 56.26 | 105% | ok | 1.7x (21%) |
| blas2 | symv | f64 | n=4096 | 2.780 ms | 12.07 | 24.16 | 45% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.865 ms | 9.00 | 36.02 | 67% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.959 ms | 17.49 | 35.02 | 66% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.919 ms | 8.74 | 35.01 | 66% | ok | 1.0x (12%) |
| blas2 | trsv | f32 | n=4096 | 0.925 ms | 18.14 | 36.31 | 68% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 13.764 ms | 2.44 | 19.51 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.227 ms | 120.55 | 2.83 | 19% | ok | 5.0x (63%) |
| blas3 | gemm | f32 | 512x512x512 | 1.219 ms | 220.26 | 2.58 | 18% | ok | 4.6x (58%) |
| blas3 | gemmt | f64 | 512x512x512 | 2.867 ms | 46.81 | 2.19 | 7% | ok | 2.7x (34%) |
| blas3 | symm | f64 | 512x512x512 | 2.972 ms | 90.32 | 2.12 | 14% | ok | 4.1x (51%) |
| blas3 | syrk | f64 | 512x512x512 | 2.405 ms | 55.82 | 1.74 | 9% | ok | 3.1x (39%) |
| blas3 | syr2k | f64 | 512x512x512 | 4.607 ms | 58.27 | 1.37 | 9% | ok | 3.1x (39%) |
| blas3 | trmm | f64 | 512x512x512 | 5.398 ms | 24.86 | 0.97 | 4% | ok | 1.8x (22%) |
| blas3 | trsm | f64 | 512x512x512 | 5.648 ms | 23.76 | 0.93 | 4% | ok | 1.7x (21%) |
| blas3 | trsm | f32 | 512x512x512 | 4.444 ms | 30.20 | 0.59 | 2% | ok | 1.5x (18%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.005 ms | 0.59 | 4.72 | 9% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 22.276 ms | 4.02 | 0.19 | 1% | ok | 1.5x (18%) |
| factor | lu | f32 | 512x512 | 20.731 ms | 4.32 | 0.10 | 0% | ok | 1.6x (20%) |
| factor | cholesky | f64 | 512x512 | 11.441 ms | 3.91 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.941 ms | 4.09 | 0.19 | 0%\* | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 11.497 ms | 3.89 | 0.36 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.682 ms | 4.19 | 0.39 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 395.325 ms | 0.45 | 0.01 | 0% | ok | 1.6x (21%) |
| factor | lq | f64 | 512x512 | 42.593 ms | 4.20 | 0.10 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 456.070 ms | 0.98 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.013 ms | 1.04 | 4.14 | 8%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 78.930 ms | 3.40 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.808 ms | 3.76 | 0.18 | 1% | ok | 1.4x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 38.733 ms | 0.58 | 0.03 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 128.775 ms | 1.17 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 514.689 ms | 0.29 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 60.959 ms | 2.75 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 116.123 ms | 3.61 | 0.01 | 1% | ok | 0.9x (12%) |
| eigen | svd | f64 | 256x256 | 121.805 ms | 3.03 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 18.672 ms | 13.48 | 0.06 | 2% | ok | 1.4x (17%) |
| eigen | sylvester | f64 | 256x256 | 267.665 ms | 3.76 | 0.01 | 1% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 27.340 ms | 0.31 | 4.91 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 42.035 ms | 0.20 | 3.19 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 39.578 ms | 0.21 | 3.39 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 20.617 ms | 0.41 | 6.51 | 12% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 31.541 ms | 0.27 | 4.26 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.175 ms | 14.96 | 155.60 | 291% ! | ok | 5.1x (63%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.292 ms | 2.03 | 21.07 | 39% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.244 ms | 18.66 | 39.24 | 73% | ok | 2.9x (37%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.203 ms | 0.20 | 0.71 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.621 ms | 1.24 | 10.27 | 19%\* | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.168 ms | 0.62 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 28.247 ms | 0.40 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 13.162 ms | 0.85 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.662 ms | 0.42 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.826 ms | 0.39 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.594 ms | 0.87 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 39.429 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.738 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.729 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 62.384 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.226 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.149 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.202 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.634 ms | 0.76 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.284 ms | 2.66 | 10.63 | 20%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.695 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.786 ms | 1.93 | 0.13 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.287 ms | 2.63 | 10.52 | 20%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.342 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.270 ms | 0.47 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.128 ms | 0.67 | 2.68 | 5%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.825 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.983 ms | 0.46 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.142 ms | 0.66 | 2.65 | 5%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.808 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 261.534 ms | 1.18 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.038 ms | 1.65 | 6.61 | 12%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.816 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.588 ms | 4.43 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.055 ms | 1.64 | 6.56 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 926.196 ms | 0.67 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.573 ms | 0.32 | 1.27 | 2%\* | ok | 1.0x (13%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.517 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 495.423 ms | 1.25 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.574 ms | 0.32 | 1.27 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 45.928 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 28.943 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.532 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 18.055 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.790 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 34.746 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.576 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.436 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.732 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 17.631 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.699 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.662 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.508 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 22.790 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.826 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.763 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.674 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.566 ms | 1.67 | 18.73 | 35% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.554 ms | 1.68 | 18.87 | 35% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.820 ms | 1.44 | 16.11 | 30% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.248 ms | 2.52 | 19.06 | 36% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.379 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.809 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.320 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.491 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.793 ms | 9.36 | 37.44 | 70% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 2.547 ms | 105.41 | 1.65 | 17% | ok | 6.0x (75%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.482 ms | 34.82 | 37.00 | 69% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.262 ms | 13.29 | 53.17 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.475 ms | 35.30 | 70.60 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.351 ms | 47.83 | 47.83 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.319 ms | 52.54 | 52.54 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.365 ms | 113.48 | 1.33 | n/a | ok | 6.3x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.524 ms | 106.36 | 0.83 | n/a | ok | 6.9x (86%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 2.548 ms | 105.37 | 0.62 | n/a | ok | 7.0x (87%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.667 ms | 402.63 | 2.36 | n/a | ok | 5.6x (70%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.664 ms | 404.15 | 2.37 | n/a | ok | 5.6x (70%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## ryzen-7-8845hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 80.1 GFLOP/s, f32 160.0 GFLOP/s |
| ... hardware, AVX-512F | f64 80.1, f32 160.0 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 43.1, update 53.3, triad 39.0 GB/s (ceiling 53.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 110% | 0 |
| blas2 | 10 | 73% | 89% | 0 |
| blas3 | 10 | 70% | 92% | 0 |
| factor | 12 | 13% | 23% | 0 |
| eigen | 8 | 17% | 54% | 0 |
| elementwise | 5 | 7% | 12% | 0 |
| sparse | 5 | 25% | 57% | 0 |
| direct | 19 | 1% | 20% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 35% | 36% | 0 |
| mg | 1 | 33% | 33% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 13% | 22% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 9.546 ms | 1.76 | 14.06 | 26% | ok |
| blas1 | dot | f32 | n=8388608 | 10.727 ms | 1.56 | 6.26 | 12% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.060 ms | 1.39 | 16.69 | 31% | ok |
| blas1 | axpy | f32 | n=8388608 | 12.027 ms | 1.39 | 8.37 | 16% | ok |
| blas1 | scale | f64 | n=8388608 | 2.296 ms | 3.65 | 58.45 | 110% | ok |
| blas1 | two_norm | f64 | n=8388608 | 9.852 ms | 1.70 | 6.81 | 13% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.065 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.089 ms | 1.65 | 13.19 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.065 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 81.623 ms | 2.26 | 1.64 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.464 ms | 9.69 | 38.76 | 73% | ok |
| blas2 | gemv | f32 | n=4096 | 1.577 ms | 21.27 | 42.56 | 80% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.721 ms | 9.02 | 36.08 | 68% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.409 ms | 23.81 | 47.65 | 89% | ok |
| blas2 | symv | f64 | n=4096 | 1.827 ms | 18.36 | 36.76 | 69% | ok |
| blas2 | trmv | f64 | n=4096 | 1.785 ms | 9.40 | 37.63 | 71% | ok |
| blas2 | trmv | f32 | n=4096 | 0.898 ms | 18.69 | 37.42 | 70% | ok |
| blas2 | trsv | f64 | n=4096 | 1.592 ms | 10.54 | 42.19 | 79% | ok |
| blas2 | trsv | f32 | n=4096 | 1.013 ms | 16.56 | 33.14 | 62% | ok |
| blas2 | ger | f64 | n=4096 | 6.366 ms | 5.27 | 42.18 | 79% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.657 ms | 73.40 | 1.72 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.906 ms | 140.83 | 1.65 | 88% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.740 ms | 48.98 | 2.30 | 61% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.795 ms | 70.72 | 1.66 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.950 ms | 45.49 | 1.42 | 57% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.770 ms | 56.28 | 1.32 | 70% | ok |
| blas3 | trmm | f64 | 512x512x512 | 1.898 ms | 70.72 | 2.76 | 88% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.845 ms | 47.18 | 1.84 | 59% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.344 ms | 99.88 | 1.95 | 62% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.849 ms | 0.60 | 4.80 | 9% | ok |
| factor | lu | f64 | 512x512 | 4.954 ms | 18.06 | 0.85 | 23% | ok |
| factor | lu | f32 | 512x512 | 2.456 ms | 36.43 | 0.85 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 4.338 ms | 10.31 | 0.97 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 1.710 ms | 26.16 | 1.23 | 16% | ok |
| factor | ldlt | f64 | 512x512 | 11.742 ms | 3.81 | 0.36 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.602 ms | 4.22 | 0.40 | 5% | ok |
| factor | qr | f64 | 512x512 | 10.323 ms | 17.33 | 0.41 | 22% | ok |
| factor | lq | f64 | 512x512 | 64.713 ms | 2.77 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 742.697 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.774 ms | 0.28 | 1.11 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 61.261 ms | 4.38 | 0.07 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.518 ms | 16.22 | 0.76 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.697 ms | 13.18 | 0.62 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 130.519 ms | 1.16 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 513.407 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.889 ms | 9.93 | 0.06 | 12% | ok |
| eigen | schur | f64 | 256x256 | 22.149 ms | 18.94 | 0.05 | 24% | ok |
| eigen | svd | f64 | 256x256 | 8.602 ms | 42.91 | 0.12 | 54% | ok |
| eigen | expm | f64 | 256x256 | 15.544 ms | 16.19 | 0.07 | 20% | ok |
| eigen | sylvester | f64 | 256x256 | 73.573 ms | 13.68 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 27.086 ms | 0.31 | 4.96 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 42.518 ms | 0.20 | 3.16 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 38.474 ms | 0.22 | 3.49 | 7% | ok |
| elementwise | sqrt | f64 | n=8388608 | 20.682 ms | 0.41 | 6.49 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 33.999 ms | 0.25 | 3.95 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.897 ms | 2.92 | 30.37 | 57% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.296 ms | 2.02 | 21.00 | 39% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.636 ms | 6.31 | 13.27 | 25% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.142 ms | 0.20 | 0.71 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.695 ms | 1.18 | 9.82 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.973 ms | 0.59 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.757 ms | 0.36 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 14.853 ms | 0.75 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.152 ms | 0.42 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.277 ms | 0.39 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 24.544 ms | 0.91 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 37.369 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.668 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.610 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.829 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.230 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.433 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.942 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.349 ms | 0.78 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.292 ms | 2.59 | 10.36 | 19%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.591 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.663 ms | 1.98 | 0.13 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.277 ms | 2.73 | 10.90 | 20%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.302 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 46.960 ms | 0.48 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.146 ms | 0.66 | 2.64 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.747 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.628 ms | 0.46 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.128 ms | 0.67 | 2.68 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.582 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 255.858 ms | 1.21 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.986 ms | 1.70 | 6.79 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.622 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.690 ms | 4.43 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.022 ms | 1.67 | 6.66 | 13%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.063 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 901.880 ms | 0.68 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.496 ms | 0.32 | 1.28 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.438 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 509.710 ms | 1.21 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.489 ms | 0.32 | 1.28 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 39.038 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.322 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.899 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.821 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.911 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.798 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 54.276 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.471 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.897 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 26.209 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 24.795 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.593 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.284 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.395 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 23.106 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.816 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.751 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.341 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.566 ms | 1.67 | 18.73 | 35% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.545 ms | 1.69 | 18.98 | 36% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.797 ms | 1.46 | 16.32 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 16.330 ms | 2.35 | 17.80 | 33% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.721 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.891 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.104 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.819 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 9.818 ms | 1.71 | 6.84 | 13% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.556 ms | 9.08 | 0.14 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.487 ms | 11.28 | 11.99 | 22% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.311 ms | 5.07 | 20.27 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.876 ms | 5.83 | 11.67 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.391 ms | 12.06 | 12.06 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.392 ms | 12.06 | 12.06 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.442 ms | 31.80 | 0.37 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.468 ms | 8.81 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 27.401 ms | 9.80 | 0.06 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 27.415 ms | 9.79 | 0.06 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.405 ms | 9.80 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

## ryzen-7-8845hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 623.9 GFLOP/s, f32 1248.4 GFLOP/s |
| ... hardware, AVX-512F | f64 623.9, f32 1248.4 GFLOP/s |
| ... this build, scalar | f64 14.4, f32 14.4 GFLOP/s |
| bandwidth | read 52.3, update 52.7, triad 40.4 GB/s (ceiling 52.7) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 95% | 128% | 0 | 43% |
| blas2 | 10 | 102% | 112% | 0 | 17% |
| blas3 | 10 | 40% | 71% | 0 | 63% |
| factor | 12 | 2% | 8% | 0 | 12% |
| eigen | 8 | 2% | 5% | 0 | 11% |
| elementwise | 5 | 7% | 12% | 0 | 12% |
| sparse | 5 | 23% | 295% | 0 | 13% |
| direct | 19 | 0% | 20% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 13% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 31% | 40% | 0 | 12% |
| mg | 1 | 36% | 36% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 13% | 22% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 88% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.335 ms | 7.18 | 57.48 | 109% | ok | 4.1x (51%) |
| blas1 | dot | f32 | n=8388608 | 1.347 ms | 12.45 | 49.81 | 95% | ok | 8.0x (100%) |
| blas1 | axpy | f64 | n=8388608 | 3.514 ms | 4.77 | 57.29 | 109% | ok | 3.4x (43%) |
| blas1 | axpy | f32 | n=8388608 | 1.774 ms | 9.46 | 56.75 | 108% | ok | 6.8x (85%) |
| blas1 | scale | f64 | n=8388608 | 1.984 ms | 4.23 | 67.64 | 128% | ok | 1.2x (14%) |
| blas1 | two_norm | f64 | n=8388608 | 1.356 ms | 12.37 | 49.49 | 94% | ok | 7.3x (91%) |
| blas1 | one_norm | f64 | n=8388608 | 5.068 ms | 1.66 | 13.24 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.114 ms | 1.64 | 13.12 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 88.120 ms | 2.09 | 1.52 | 3% | ok | 0.9x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.551 ms | 13.15 | 52.64 | 100% | ok | 1.4x (17%) |
| blas2 | gemv | f32 | n=4096 | 1.170 ms | 28.69 | 57.40 | 109% | ok | 1.3x (17%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.625 ms | 12.78 | 51.15 | 97% | ok | 1.4x (18%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.266 ms | 26.50 | 53.02 | 101% | ok | 1.1x (14%) |
| blas2 | symv | f64 | n=4096 | 1.250 ms | 26.85 | 53.76 | 102% | ok | 1.5x (18%) |
| blas2 | trmv | f64 | n=4096 | 1.232 ms | 13.62 | 54.52 | 104% | ok | 1.4x (18%) |
| blas2 | trmv | f32 | n=4096 | 0.571 ms | 29.40 | 58.86 | 112% | ok | 1.6x (20%) |
| blas2 | trsv | f64 | n=4096 | 1.780 ms | 9.42 | 37.73 | 72% | ok | 0.9x (11%) |
| blas2 | trsv | f32 | n=4096 | 0.882 ms | 19.03 | 38.09 | 72% | ok | 1.1x (14%) |
| blas2 | ger | f64 | n=4096 | 4.754 ms | 7.06 | 56.48 | 107% | ok | 1.3x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 0.607 ms | 441.87 | 10.36 | 71% | ok | 6.0x (75%) |
| blas3 | gemm | f32 | 512x512x512 | 0.339 ms | 792.31 | 9.28 | 63% | ok | 5.6x (70%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.348 ms | 99.55 | 4.67 | 16% | ok | 2.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 0.611 ms | 439.70 | 10.31 | 70% | ok | 6.2x (78%) |
| blas3 | syrk | f64 | 512x512x512 | 1.724 ms | 77.85 | 2.43 | 12% | ok | 1.7x (21%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.388 ms | 112.41 | 2.63 | 18% | ok | 2.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 0.390 ms | 344.15 | 13.44 | 55% | ok | 4.9x (61%) |
| blas3 | trsm | f64 | 512x512x512 | 0.538 ms | 249.24 | 9.74 | 40% | ok | 5.3x (66%) |
| blas3 | trsm | f32 | 512x512x512 | 0.268 ms | 501.37 | 9.79 | 40% | ok | 5.0x (63%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.699 ms | 0.61 | 4.88 | 9% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 9.088 ms | 9.85 | 0.46 | 2% | ok | 0.5x (7%) |
| factor | lu | f32 | 512x512 | 4.300 ms | 20.81 | 0.49 | 2% | ok | 0.6x (7%) |
| factor | cholesky | f64 | 512x512 | 5.979 ms | 7.48 | 0.70 | 1%\* | ok | 0.7x (9%) |
| factor | cholesky | f32 | 512x512 | 2.259 ms | 19.81 | 0.93 | 2%\* | ok | 0.8x (9%) |
| factor | ldlt | f64 | 512x512 | 11.874 ms | 3.77 | 0.35 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.709 ms | 4.18 | 0.39 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 11.780 ms | 15.19 | 0.36 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 42.052 ms | 4.26 | 0.10 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 459.745 ms | 0.97 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.014 ms | 1.04 | 4.14 | 8%\* | ok | 3.7x (47%) |
| factor | inv | f64 | 512x512 | 65.489 ms | 4.10 | 0.06 | 1% | ok | 0.9x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.228 ms | 12.38 | 0.58 | 2% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 3.597 ms | 6.22 | 0.29 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 134.195 ms | 1.13 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 506.985 ms | 0.30 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 21.383 ms | 7.85 | 0.05 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 26.235 ms | 15.99 | 0.04 | 3% | ok | 0.8x (11%) |
| eigen | svd | f64 | 256x256 | 12.021 ms | 30.71 | 0.09 | 5% | ok | 0.7x (9%) |
| eigen | expm | f64 | 256x256 | 17.730 ms | 14.19 | 0.06 | 2% | ok | 0.9x (11%) |
| eigen | sylvester | f64 | 256x256 | 81.335 ms | 12.38 | 0.02 | 2% | ok | 0.9x (11%) |
| elementwise | exp | f64 | n=8388608 | 27.825 ms | 0.30 | 4.82 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 43.398 ms | 0.19 | 3.09 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 39.050 ms | 0.21 | 3.44 | 7% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 20.668 ms | 0.41 | 6.49 | 12% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 34.476 ms | 0.24 | 3.89 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.175 ms | 14.96 | 155.60 | 295% ! | ok | 5.1x (64%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.218 ms | 1.18 | 12.28 | 23% | ok | 0.6x (7%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.263 ms | 18.50 | 38.90 | 74% | ok | 2.9x (37%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.717 ms | 0.19 | 0.69 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.639 ms | 1.22 | 10.15 | 19%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.496 ms | 0.57 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.396 ms | 0.36 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.454 ms | 0.72 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.681 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.340 ms | 0.39 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.246 ms | 0.89 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.253 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.757 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.681 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 61.252 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.230 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.478 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.985 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.664 ms | 0.76 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.283 ms | 2.67 | 10.68 | 20%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.681 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.715 ms | 1.96 | 0.13 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.284 ms | 2.66 | 10.66 | 20%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.325 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.032 ms | 0.47 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.124 ms | 0.67 | 2.69 | 5%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.808 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.053 ms | 0.46 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.152 ms | 0.66 | 2.63 | 5%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.732 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 259.842 ms | 1.19 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.021 ms | 1.67 | 6.67 | 13%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.742 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 70.064 ms | 4.40 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.075 ms | 1.62 | 6.49 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.147 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 906.269 ms | 0.68 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.556 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.500 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 510.693 ms | 1.21 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.586 ms | 0.32 | 1.27 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 39.205 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.877 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.837 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.772 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.767 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.765 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 54.630 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.723 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.870 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 26.272 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 24.709 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.729 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.624 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.407 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 22.754 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.809 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.782 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.357 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.406 ms | 1.86 | 20.86 | 40% | ok | 1.1x (14%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.768 ms | 1.48 | 16.59 | 31% | ok | 0.9x (11%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.828 ms | 1.43 | 16.04 | 30% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.381 ms | 2.50 | 18.90 | 36% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.668 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.159 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 51.113 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.872 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 9.917 ms | 1.69 | 6.77 | 13% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.083 ms | 65.75 | 1.03 | 11% | ok | 7.2x (90%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.505 ms | 11.15 | 11.84 | 22% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.268 ms | 5.13 | 20.53 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.876 ms | 5.83 | 11.67 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.404 ms | 11.95 | 11.95 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.403 ms | 11.96 | 11.96 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.204 ms | 222.99 | 2.61 | n/a | ok | 7.0x (88%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.199 ms | 63.92 | 0.50 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.787 ms | 70.89 | 0.42 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.790 ms | 70.83 | 0.42 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.812 ms | 70.42 | 0.41 | n/a | ok | 7.2x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

