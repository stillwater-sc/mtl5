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

