## ryzen-9-8945hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 460.4 GFLOP/s, f32 1005.2 GFLOP/s |
| ... hardware, AVX-512F | f64 460.4, f32 1005.2 GFLOP/s |
| ... this build, SSE2 | f64 319.8, f32 638.1 GFLOP/s |
| bandwidth | read 52.6, update 52.8, triad 40.6 GB/s (ceiling 52.8) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 112% | 127% | 0 | 13% |
| blas2 | 10 | 67% | 121% | 0 | 13% |
| blas3 | 10 | 10% | 21% | 0 | 39% |
| factor | 12 | 1% | 6% | 0 | 19% |
| eigen | 8 | 1% | 2% | 0 | 13% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 34% | 291% | 0 | 13% |
| direct | 19 | 0% | 17% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 11% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 10% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 28% | 30% | 0 | 13% |
| mg | 1 | 30% | 30% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 58% | 59% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 74% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.266 ms | 7.40 | 59.24 | 112% | ok | 1.1x (14%) |
| blas1 | dot | f32 | n=8388608 | 1.012 ms | 16.58 | 66.31 | 126% | ok | 1.2x (15%) |
| blas1 | axpy | f64 | n=8388608 | 3.471 ms | 4.83 | 58.00 | 110% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=8388608 | 1.618 ms | 10.37 | 62.22 | 118% | ok | 1.0x (13%) |
| blas1 | scale | f64 | n=8388608 | 2.003 ms | 4.19 | 67.02 | 127% | ok | 1.2x (15%) |
| blas1 | two_norm | f64 | n=8388608 | 1.012 ms | 16.58 | 66.31 | 126% | ok | 1.2x (15%) |
| blas1 | one_norm | f64 | n=8388608 | 6.397 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.408 ms | 1.31 | 10.47 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.50 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 109.407 ms | 1.69 | 1.23 | 2% | ok | 0.9x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.306 ms | 14.55 | 58.24 | 110% | ok | 1.5x (19%) |
| blas2 | gemv | f32 | n=4096 | 1.049 ms | 31.99 | 64.01 | 121% | ok | 1.8x (22%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.332 ms | 14.39 | 57.59 | 109% | ok | 1.7x (22%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.201 ms | 27.93 | 55.89 | 106% | ok | 1.8x (23%) |
| blas2 | symv | f64 | n=4096 | 3.777 ms | 8.88 | 17.79 | 34% | ok | 0.9x (11%) |
| blas2 | trmv | f64 | n=4096 | 1.925 ms | 8.72 | 34.90 | 66% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.028 ms | 16.32 | 32.68 | 62% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.908 ms | 8.79 | 35.21 | 67% | ok | 1.1x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.004 ms | 16.71 | 33.46 | 63% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 16.273 ms | 2.06 | 16.50 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.768 ms | 96.97 | 2.27 | 21% | ok | 5.0x (62%) |
| blas3 | gemm | f32 | 512x512x512 | 1.498 ms | 179.14 | 2.10 | 18% | ok | 4.7x (58%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.011 ms | 33.46 | 1.57 | 7% | ok | 2.3x (29%) |
| blas3 | symm | f64 | 512x512x512 | 3.718 ms | 72.19 | 1.69 | 16% | ok | 4.0x (50%) |
| blas3 | syrk | f64 | 512x512x512 | 2.951 ms | 45.48 | 1.42 | 10% | ok | 3.2x (39%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.546 ms | 48.40 | 1.13 | 11% | ok | 3.3x (41%) |
| blas3 | trmm | f64 | 512x512x512 | 6.539 ms | 20.52 | 0.80 | 4% | ok | 1.8x (23%) |
| blas3 | trsm | f64 | 512x512x512 | 7.212 ms | 18.61 | 0.73 | 4% | ok | 1.7x (22%) |
| blas3 | trsm | f32 | 512x512x512 | 5.736 ms | 23.40 | 0.46 | 2% | ok | 1.5x (18%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.240 ms | 0.52 | 4.15 | 8% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 25.892 ms | 3.46 | 0.16 | 1% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 24.410 ms | 3.67 | 0.09 | 0% | ok | 1.5x (19%) |
| factor | cholesky | f64 | 512x512 | 14.349 ms | 3.12 | 0.29 | 1% | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 13.723 ms | 3.26 | 0.15 | 0% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 14.442 ms | 3.10 | 0.29 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 12.609 ms | 3.55 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 500.235 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 50.586 ms | 3.54 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 579.084 ms | 0.77 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.253 ms | 0.84 | 3.35 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 96.576 ms | 2.78 | 0.04 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 28.231 ms | 3.17 | 0.15 | 1% | ok | 1.5x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 47.694 ms | 0.47 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 163.226 ms | 0.93 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 630.339 ms | 0.24 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 74.373 ms | 2.26 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 147.416 ms | 2.85 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 150.546 ms | 2.45 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 22.154 ms | 11.36 | 0.05 | 2% | ok | 1.5x (18%) |
| eigen | sylvester | f64 | 256x256 | 330.640 ms | 3.04 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 32.449 ms | 0.26 | 4.14 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 50.645 ms | 0.17 | 2.65 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 47.002 ms | 0.18 | 2.86 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 25.430 ms | 0.33 | 5.28 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 38.331 ms | 0.22 | 3.50 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.177 ms | 14.77 | 153.67 | 291% ! | ok | 5.3x (66%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.518 ms | 1.72 | 17.94 | 34% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.279 ms | 18.38 | 38.64 | 73% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.677 ms | 0.17 | 0.58 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.990 ms | 1.01 | 8.36 | 16%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 21.989 ms | 0.51 | 0.03 | 0% | ok | 1.1x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 34.261 ms | 0.33 | 0.02 | 0% | ok | 1.1x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.864 ms | 0.71 | 0.05 | 0% | ok | 1.1x (14%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 68.897 ms | 0.32 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 67.626 ms | 0.33 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.192 ms | 0.74 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.979 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.627 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.997 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 74.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.239 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.323 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.698 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.661 ms | 0.63 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.345 ms | 2.19 | 8.77 | 17%\* | ok | 1.1x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.181 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.947 ms | 1.61 | 0.11 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.345 ms | 2.19 | 8.76 | 17%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.061 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.248 ms | 0.39 | 0.01 | 0% | ok | 1.1x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.282 ms | 0.59 | 2.36 | 4%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.922 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 58.644 ms | 0.38 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.541 ms | 0.49 | 1.96 | 4%\* | ok | 0.8x (10%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 7.005 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 349.585 ms | 0.88 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.306 ms | 1.46 | 5.84 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.085 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.563 ms | 3.65 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.391 ms | 1.41 | 5.64 | 11%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.561 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1106.888 ms | 0.56 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.276 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.580 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 615.742 ms | 1.00 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.280 ms | 0.30 | 1.19 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.471 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 83.242 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 36.045 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 20.844 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.446 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 24.108 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.864 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.780 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.984 ms | 0.00 | 0.00 | n/a | ok | 0.8x (11%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 24.874 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 26.646 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 26.795 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 34.301 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.091 ms | 0.00 | 0.00 | n/a | ok | 0.6x (8%) |
| pc | ilu_0 | f64 | 128^2 grid | 3.005 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 29.049 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.212 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 2.130 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.816 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.975 ms | 1.33 | 14.85 | 28% | ok | 0.9x (11%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.826 ms | 1.43 | 16.06 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.232 ms | 1.17 | 13.14 | 25% | ok | 1.2x (15%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.501 ms | 2.08 | 15.71 | 30% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.891 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.824 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 77.187 ms | 0.00 | 0.00 | n/a | ok | 0.8x (11%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.173 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 2.175 ms | 7.71 | 30.86 | 58% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 3.089 ms | 86.89 | 1.36 | 19% | ok | 6.2x (78%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.577 ms | 29.08 | 30.89 | 59% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.547 ms | 10.85 | 43.38 | n/a | ok | 0.9x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.593 ms | 28.28 | 56.56 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.431 ms | 38.90 | 38.90 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.390 ms | 43.02 | 43.02 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.812 ms | 95.46 | 1.12 | n/a | ok | 6.7x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 3.020 ms | 88.88 | 0.69 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.058 ms | 87.77 | 0.51 | n/a | ok | 7.3x (92%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.774 ms | 346.82 | 2.03 | n/a | ok | 5.9x (74%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.770 ms | 348.53 | 2.04 | n/a | ok | 5.9x (74%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

