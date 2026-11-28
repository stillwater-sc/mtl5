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

