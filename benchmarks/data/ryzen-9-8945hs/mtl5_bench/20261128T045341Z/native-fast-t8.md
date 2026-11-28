## ryzen-9-8945hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 506.1 GFLOP/s, f32 1011.8 GFLOP/s |
| ... hardware, AVX-512F | f64 506.1, f32 1011.8 GFLOP/s |
| ... this build, SSE2 | f64 320.1, f32 637.8 GFLOP/s |
| bandwidth | read 52.6, update 53.1, triad 40.9 GB/s (ceiling 53.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 112% | 128% | 0 | 13% |
| blas2 | 10 | 65% | 117% | 0 | 13% |
| blas3 | 10 | 9% | 19% | 0 | 39% |
| factor | 12 | 1% | 6% | 0 | 19% |
| eigen | 8 | 0% | 2% | 0 | 12% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 34% | 293% | 0 | 13% |
| direct | 19 | 0% | 16% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 11% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 29% | 31% | 0 | 12% |
| mg | 1 | 29% | 29% | 0 | 15% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 58% | 58% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 74% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.255 ms | 7.44 | 59.51 | 112% | ok | 1.1x (14%) |
| blas1 | dot | f32 | n=8388608 | 0.987 ms | 16.99 | 67.97 | 128% | ok | 1.3x (16%) |
| blas1 | axpy | f64 | n=8388608 | 3.459 ms | 4.85 | 58.20 | 110% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=8388608 | 1.641 ms | 10.22 | 61.35 | 116% | ok | 1.1x (13%) |
| blas1 | scale | f64 | n=8388608 | 2.053 ms | 4.09 | 65.39 | 123% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=8388608 | 1.086 ms | 15.45 | 61.82 | 117% | ok | 1.1x (14%) |
| blas1 | one_norm | f64 | n=8388608 | 6.434 ms | 1.30 | 10.43 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.408 ms | 1.31 | 10.47 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.392 ms | 1.31 | 10.50 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 103.027 ms | 1.79 | 1.30 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.301 ms | 14.58 | 58.36 | 110% | ok | 1.5x (19%) |
| blas2 | gemv | f32 | n=4096 | 1.081 ms | 31.03 | 62.09 | 117% | ok | 1.7x (21%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.341 ms | 14.33 | 57.36 | 108% | ok | 1.7x (22%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.202 ms | 27.92 | 55.86 | 105% | ok | 1.8x (22%) |
| blas2 | symv | f64 | n=4096 | 3.845 ms | 8.73 | 17.47 | 33% | ok | 0.9x (11%) |
| blas2 | trmv | f64 | n=4096 | 1.951 ms | 8.60 | 34.43 | 65% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.025 ms | 16.36 | 32.76 | 62% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 2.069 ms | 8.11 | 32.46 | 61% | ok | 0.9x (12%) |
| blas2 | trsv | f32 | n=4096 | 1.008 ms | 16.64 | 33.31 | 63% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 16.279 ms | 2.06 | 16.49 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.737 ms | 98.06 | 2.30 | 19% | ok | 5.0x (63%) |
| blas3 | gemm | f32 | 512x512x512 | 1.525 ms | 176.05 | 2.06 | 17% | ok | 4.5x (57%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.353 ms | 30.83 | 1.45 | 6% | ok | 2.1x (27%) |
| blas3 | symm | f64 | 512x512x512 | 3.889 ms | 69.02 | 1.62 | 14% | ok | 3.9x (48%) |
| blas3 | syrk | f64 | 512x512x512 | 2.957 ms | 45.39 | 1.42 | 9% | ok | 3.1x (39%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.602 ms | 47.91 | 1.12 | 9% | ok | 3.2x (40%) |
| blas3 | trmm | f64 | 512x512x512 | 6.626 ms | 20.26 | 0.79 | 4% | ok | 1.8x (23%) |
| blas3 | trsm | f64 | 512x512x512 | 6.963 ms | 19.28 | 0.75 | 4% | ok | 1.8x (22%) |
| blas3 | trsm | f32 | 512x512x512 | 5.377 ms | 24.96 | 0.49 | 2% | ok | 1.5x (19%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.243 ms | 0.47 | 3.78 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 26.038 ms | 3.44 | 0.16 | 1% | ok | 1.5x (19%) |
| factor | lu | f32 | 512x512 | 23.910 ms | 3.74 | 0.09 | 0% | ok | 1.6x (20%) |
| factor | cholesky | f64 | 512x512 | 14.276 ms | 3.13 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 13.718 ms | 3.26 | 0.15 | 0% | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 14.398 ms | 3.11 | 0.29 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 13.094 ms | 3.42 | 0.32 | 1% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 501.373 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 50.230 ms | 3.56 | 0.08 | 1% | ok | 1.7x (22%) |
| factor | hessenberg | f64 | 512x512 | 572.191 ms | 0.78 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.252 ms | 0.84 | 3.35 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 97.393 ms | 2.76 | 0.04 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 27.610 ms | 3.24 | 0.15 | 1% | ok | 1.5x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 47.391 ms | 0.47 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 164.438 ms | 0.92 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 661.082 ms | 0.23 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 75.780 ms | 2.21 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 141.331 ms | 2.97 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 149.553 ms | 2.47 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 22.056 ms | 11.41 | 0.05 | 2% | ok | 1.5x (18%) |
| eigen | sylvester | f64 | 256x256 | 324.445 ms | 3.10 | 0.00 | 1% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 33.700 ms | 0.25 | 3.98 | 8% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 51.718 ms | 0.16 | 2.60 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 49.157 ms | 0.17 | 2.73 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.616 ms | 0.32 | 5.04 | 10% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 38.507 ms | 0.22 | 3.49 | 7% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.175 ms | 14.96 | 155.69 | 293% ! | ok | 5.7x (71%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.523 ms | 1.72 | 17.88 | 34% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.287 ms | 18.31 | 38.50 | 73% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.924 ms | 0.16 | 0.55 | 1%\* | ok | 1.1x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.888 ms | 1.06 | 8.82 | 17%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 22.027 ms | 0.51 | 0.03 | 0% | ok | 1.1x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 34.084 ms | 0.33 | 0.02 | 0% | ok | 1.1x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 16.443 ms | 0.68 | 0.04 | 0% | ok | 1.1x (14%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.123 ms | 0.34 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 68.618 ms | 0.33 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.126 ms | 0.74 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.803 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.556 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.946 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 77.139 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.224 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.541 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.689 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.733 ms | 0.63 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.377 ms | 2.00 | 8.01 | 15%\* | ok | 0.9x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.140 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.241 ms | 1.54 | 0.10 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.347 ms | 2.18 | 8.71 | 16%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.079 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 58.326 ms | 0.38 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.305 ms | 0.58 | 2.32 | 4%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.909 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.886 ms | 0.37 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.273 ms | 0.59 | 2.38 | 4%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.940 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 346.637 ms | 0.89 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.270 ms | 1.48 | 5.94 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.110 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.557 ms | 3.65 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.325 ms | 1.45 | 5.80 | 11%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.601 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1090.466 ms | 0.57 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.302 ms | 0.30 | 1.19 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.571 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 607.792 ms | 1.02 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.258 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.757 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 36.214 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.133 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.811 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 19.933 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 43.274 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.107 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 21.958 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.593 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.260 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 22.303 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 27.199 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.059 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 3.022 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 27.984 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.214 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.163 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.819 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.808 ms | 1.45 | 16.22 | 31% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.912 ms | 1.37 | 15.34 | 29% | ok | 1.2x (15%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.939 ms | 0.89 | 9.98 | 19% | ok | 0.8x (9%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.619 ms | 2.07 | 15.61 | 29% | ok | 1.2x (15%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.478 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.198 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 66.756 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.141 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 2.171 ms | 7.73 | 30.91 | 58% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 3.133 ms | 85.67 | 1.34 | 17% | ok | 6.2x (77%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.576 ms | 29.13 | 30.95 | 58% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.458 ms | 11.51 | 46.03 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.598 ms | 28.05 | 56.10 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.428 ms | 39.15 | 39.15 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.387 ms | 43.40 | 43.40 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.886 ms | 93.01 | 1.09 | n/a | ok | 6.5x (82%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 3.010 ms | 89.17 | 0.70 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.042 ms | 88.23 | 0.52 | n/a | ok | 7.4x (92%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.772 ms | 347.58 | 2.04 | n/a | ok | 5.9x (74%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.768 ms | 349.30 | 2.05 | n/a | ok | 6.2x (78%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

