## ryzen-7-8845hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 625.9 GFLOP/s, f32 1248.4 GFLOP/s |
| ... hardware, AVX-512F | f64 625.9, f32 1248.4 GFLOP/s |
| ... this build, SSE2 | f64 312.6, f32 624.5 GFLOP/s |
| bandwidth | read 53.2, update 53.6, triad 41.2 GB/s (ceiling 53.6) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 112% | 129% | 0 | 13% |
| blas2 | 10 | 70% | 112% | 0 | 13% |
| blas3 | 10 | 9% | 18% | 0 | 38% |
| factor | 12 | 1% | 8% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 8% | 12% | 0 | 12% |
| sparse | 5 | 39% | 289% | 0 | 36% |
| direct | 19 | 0% | 20% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 13% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 14% |
| smoother | 3 | 34% | 36% | 0 | 12% |
| mg | 1 | 36% | 36% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 69% | 70% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.235 ms | 7.51 | 60.04 | 112% | ok | 1.1x (13%) |
| blas1 | dot | f32 | n=8388608 | 0.973 ms | 17.25 | 69.00 | 129% | ok | 1.2x (15%) |
| blas1 | axpy | f64 | n=8388608 | 3.481 ms | 4.82 | 57.83 | 108% | ok | 1.0x (12%) |
| blas1 | axpy | f32 | n=8388608 | 1.621 ms | 10.35 | 62.09 | 116% | ok | 1.0x (13%) |
| blas1 | scale | f64 | n=8388608 | 1.964 ms | 4.27 | 68.33 | 127% | ok | 1.0x (12%) |
| blas1 | two_norm | f64 | n=8388608 | 1.082 ms | 15.51 | 62.03 | 116% | ok | 1.1x (13%) |
| blas1 | one_norm | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.098 ms | 1.65 | 13.16 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 89.277 ms | 2.07 | 1.50 | 3% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.289 ms | 14.66 | 58.67 | 109% | ok | 1.4x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.121 ms | 29.95 | 59.92 | 112% | ok | 1.6x (20%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.285 ms | 14.69 | 58.77 | 110% | ok | 1.7x (21%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.183 ms | 28.36 | 56.76 | 106% | ok | 1.9x (24%) |
| blas2 | symv | f64 | n=4096 | 2.951 ms | 11.37 | 22.77 | 42% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.889 ms | 8.88 | 35.56 | 66% | ok | 0.9x (12%) |
| blas2 | trmv | f32 | n=4096 | 0.945 ms | 17.76 | 35.55 | 66% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.891 ms | 8.87 | 35.53 | 66% | ok | 1.0x (12%) |
| blas2 | trsv | f32 | n=4096 | 0.894 ms | 18.76 | 37.57 | 70% | ok | 1.1x (13%) |
| blas2 | ger | f64 | n=4096 | 13.580 ms | 2.47 | 19.77 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.337 ms | 114.84 | 2.69 | 18% | ok | 4.7x (59%) |
| blas3 | gemm | f32 | 512x512x512 | 1.231 ms | 218.03 | 2.56 | 17% | ok | 4.5x (57%) |
| blas3 | gemmt | f64 | 512x512x512 | 2.799 ms | 47.96 | 2.25 | 8% | ok | 2.7x (33%) |
| blas3 | symm | f64 | 512x512x512 | 3.063 ms | 87.63 | 2.05 | 14% | ok | 3.9x (49%) |
| blas3 | syrk | f64 | 512x512x512 | 2.427 ms | 55.31 | 1.73 | 9% | ok | 3.1x (38%) |
| blas3 | syr2k | f64 | 512x512x512 | 4.638 ms | 57.87 | 1.36 | 9% | ok | 3.1x (39%) |
| blas3 | trmm | f64 | 512x512x512 | 5.533 ms | 24.26 | 0.95 | 4% | ok | 1.8x (22%) |
| blas3 | trsm | f64 | 512x512x512 | 5.750 ms | 23.34 | 0.91 | 4% | ok | 1.7x (21%) |
| blas3 | trsm | f32 | 512x512x512 | 4.447 ms | 30.18 | 0.59 | 2% | ok | 1.5x (18%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.391 ms | 0.57 | 4.52 | 8% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 22.152 ms | 4.04 | 0.19 | 1% | ok | 1.5x (18%) |
| factor | lu | f32 | 512x512 | 21.033 ms | 4.25 | 0.10 | 0% | ok | 1.5x (18%) |
| factor | cholesky | f64 | 512x512 | 11.418 ms | 3.92 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.925 ms | 4.09 | 0.19 | 0%\* | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 11.483 ms | 3.90 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.565 ms | 4.23 | 0.40 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 398.693 ms | 0.45 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 41.964 ms | 4.26 | 0.10 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 458.252 ms | 0.98 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.021 ms | 1.03 | 4.11 | 8%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 78.354 ms | 3.43 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.696 ms | 3.78 | 0.18 | 1% | ok | 1.4x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 38.166 ms | 0.59 | 0.03 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 128.378 ms | 1.18 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 510.010 ms | 0.30 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 57.047 ms | 2.94 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 109.391 ms | 3.83 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 119.360 ms | 3.09 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 18.057 ms | 13.94 | 0.06 | 2% | ok | 1.4x (18%) |
| eigen | sylvester | f64 | 256x256 | 259.216 ms | 3.88 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 28.085 ms | 0.30 | 4.78 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 42.629 ms | 0.20 | 3.15 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 40.250 ms | 0.21 | 3.33 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.360 ms | 0.39 | 6.28 | 12% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 31.820 ms | 0.26 | 4.22 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.176 ms | 14.90 | 154.98 | 289% ! | ok | 5.0x (62%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.319 ms | 1.99 | 20.65 | 39% | ok | 3.9x (49%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.248 ms | 18.63 | 39.17 | 73% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.214 ms | 0.18 | 0.63 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.659 ms | 1.21 | 10.03 | 19%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.384 ms | 0.61 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 28.164 ms | 0.40 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 13.281 ms | 0.84 | 0.06 | 0% | ok | 1.1x (14%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.221 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.548 ms | 0.39 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.145 ms | 0.89 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.311 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.745 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.683 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 61.983 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.227 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.366 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.004 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.531 ms | 0.77 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.286 ms | 2.64 | 10.56 | 20%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.690 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.031 ms | 1.85 | 0.12 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.282 ms | 2.69 | 10.74 | 20%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.415 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.002 ms | 0.47 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.106 ms | 0.68 | 2.74 | 5%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.836 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.264 ms | 0.45 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.137 ms | 0.66 | 2.66 | 5%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.636 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 260.486 ms | 1.18 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.001 ms | 1.68 | 6.73 | 13%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 14.141 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.599 ms | 4.43 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.070 ms | 1.63 | 6.51 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.187 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 923.971 ms | 0.67 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.530 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.564 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 503.477 ms | 1.23 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.499 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.629 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 46.085 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 28.534 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.499 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.958 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.715 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 34.541 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.290 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.336 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.611 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 17.552 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.688 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.555 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.043 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.457 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ilut | f64 | 128^2 grid | 22.789 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ic_0 | f64 | 128^2 grid | 1.800 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| pc | ildl | f64 | 128^2 grid | 1.755 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.672 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.620 ms | 1.62 | 18.10 | 34% | ok | 0.9x (11%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.521 ms | 1.72 | 19.29 | 36% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.794 ms | 1.46 | 16.34 | 30% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.266 ms | 2.52 | 19.04 | 36% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.617 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.062 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 54.904 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.549 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.784 ms | 9.40 | 37.61 | 70% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 2.627 ms | 102.17 | 1.60 | 16% | ok | 5.8x (72%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.480 ms | 34.92 | 37.11 | 69% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.213 ms | 13.83 | 55.33 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.476 ms | 35.28 | 70.55 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.349 ms | 48.00 | 48.00 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.318 ms | 52.69 | 52.69 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.421 ms | 110.88 | 1.30 | n/a | ok | 6.2x (77%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.526 ms | 106.26 | 0.83 | n/a | ok | 6.9x (86%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 2.588 ms | 103.71 | 0.61 | n/a | ok | 6.9x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.665 ms | 403.60 | 2.36 | n/a | ok | 5.6x (70%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.664 ms | 404.03 | 2.37 | n/a | ok | 5.6x (70%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

