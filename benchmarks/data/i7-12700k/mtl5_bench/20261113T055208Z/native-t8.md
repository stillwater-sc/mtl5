## i7-12700k -- native, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 600.8 GFLOP/s, f32 1201.7 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.8, f32 1201.7 GFLOP/s |
| ... this build, scalar | f64 9.8, f32 10.5 GFLOP/s |
| bandwidth | read 83.9, update 78.8, triad 77.3 GB/s (ceiling 83.9) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 84% | 123% | 0 | 28% |
| blas2 | 10 | 42% | 136% | 0 | 13% |
| blas3 | 10 | 2% | 15% | 0 | 15% |
| factor | 12 | 1% | 12% | 0 | 14% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 288% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 11% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 36% | 0 | 13% |
| mg | 1 | 25% | 25% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.490 ms | 11.26 | 90.08 | 107% | ok | 3.8x (47%) |
| blas1 | dot | f32 | n=8388608 | 0.650 ms | 25.83 | 103.31 | 123% | ok | 6.5x (81%) |
| blas1 | axpy | f64 | n=8388608 | 2.832 ms | 5.92 | 71.10 | 85% | ok | 2.2x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.444 ms | 11.62 | 69.70 | 83% | ok | 3.2x (41%) |
| blas1 | scale | f64 | n=8388608 | 1.915 ms | 4.38 | 70.10 | 84% | ok | 1.9x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.742 ms | 22.61 | 90.45 | 108% | ok | 6.5x (82%) |
| blas1 | one_norm | f64 | n=8388608 | 3.943 ms | 2.13 | 17.02 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.916 ms | 1.21 | 9.70 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.918 ms | 2.14 | 17.13 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.132 ms | 2.46 | 1.79 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.445 ms | 23.22 | 92.92 | 111% | ok | 2.5x (31%) |
| blas2 | gemv | f32 | n=4096 | 0.588 ms | 57.02 | 114.09 | 136% | ok | 4.7x (59%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.573 ms | 21.33 | 85.37 | 102% | ok | 3.0x (37%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.866 ms | 38.75 | 77.53 | 92% | ok | 2.5x (32%) |
| blas2 | symv | f64 | n=4096 | 6.562 ms | 5.11 | 10.24 | 12% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.998 ms | 8.40 | 33.62 | 40% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.430 ms | 11.73 | 23.48 | 28% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.958 ms | 8.57 | 34.31 | 41% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.398 ms | 12.00 | 24.03 | 29% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 7.715 ms | 4.35 | 34.80 | 42% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.894 ms | 92.77 | 2.17 | 15% | ok | 6.9x (87%) |
| blas3 | gemm | f32 | 512x512x512 | 1.525 ms | 176.03 | 2.06 | 15% | ok | 7.0x (87%) |
| blas3 | gemmt | f64 | 512x512x512 | 8.940 ms | 15.01 | 0.70 | 2% | ok | 1.2x (15%) |
| blas3 | symm | f64 | 512x512x512 | 16.659 ms | 16.11 | 0.38 | 3% | ok | 1.1x (14%) |
| blas3 | syrk | f64 | 512x512x512 | 11.566 ms | 11.60 | 0.36 | 2% | ok | 1.2x (15%) |
| blas3 | syr2k | f64 | 512x512x512 | 19.514 ms | 13.76 | 0.32 | 2% | ok | 1.3x (16%) |
| blas3 | trmm | f64 | 512x512x512 | 9.126 ms | 14.71 | 0.57 | 2% | ok | 1.1x (14%) |
| blas3 | trsm | f64 | 512x512x512 | 9.279 ms | 14.47 | 0.57 | 2% | ok | 1.1x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 5.041 ms | 26.63 | 0.52 | 2% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.865 ms | 0.41 | 3.30 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 7.439 ms | 12.03 | 0.56 | 2% | ok | 1.6x (19%) |
| factor | lu | f32 | 512x512 | 4.011 ms | 22.31 | 0.52 | 2% | ok | 1.1x (14%) |
| factor | cholesky | f64 | 512x512 | 7.096 ms | 6.30 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.454 ms | 6.93 | 0.32 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.189 ms | 6.22 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 45.606 ms | 0.98 | 0.09 | 0% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 129.421 ms | 1.38 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 17.287 ms | 10.35 | 0.24 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 169.442 ms | 2.64 | 0.02 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 0.415 ms | 2.53 | 10.11 | 12%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 50.850 ms | 5.28 | 0.08 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.274 ms | 14.26 | 0.67 | 2% | ok | 1.0x (12%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.460 ms | 1.28 | 0.06 | 0% | ok | 1.1x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.630 ms | 2.62 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 735.479 ms | 0.21 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 31.847 ms | 5.27 | 0.03 | 1% | ok | 1.1x (13%) |
| eigen | schur | f64 | 256x256 | 82.477 ms | 5.09 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 109.077 ms | 3.38 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 11.690 ms | 21.53 | 0.09 | 4% | ok | 2.3x (28%) |
| eigen | sylvester | f64 | 256x256 | 177.212 ms | 5.68 | 0.01 | 1% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.541 ms | 0.22 | 3.48 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.037 ms | 0.19 | 2.98 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.773 ms | 0.14 | 2.28 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.424 ms | 0.30 | 4.72 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 199.195 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.21 | 241.51 | 288% ! | ok | 6.6x (82%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.908 ms | 2.88 | 30.00 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.409 ms | 29.73 | 62.51 | 75% | ok | 3.3x (41%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.802 ms | 0.68 | 2.39 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.367 ms | 1.47 | 12.17 | 15%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 89.593 ms | 0.12 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.009 ms | 0.45 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.234 ms | 1.21 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.446 ms | 0.55 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.086 ms | 0.42 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.757 ms | 1.34 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 85.311 ms | 0.02 | 0.00 | 0% | ok | 1.2x (15%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.249 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.848 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.220 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.723 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.270 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 87.246 ms | 0.13 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.271 ms | 2.79 | 11.15 | 13%\* | ok | 0.9x (11%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.350 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.144 ms | 2.70 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.262 ms | 2.88 | 11.53 | 14%\* | ok | 0.9x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.022 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 36.698 ms | 0.61 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.598 ms | 1.27 | 5.06 | 6%\* | ok | 1.1x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.772 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.324 ms | 0.47 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.606 ms | 1.25 | 4.99 | 6%\* | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.275 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7286.320 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.525 ms | 2.21 | 8.84 | 11%\* | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.256 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.516 ms | 4.44 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.536 ms | 2.19 | 8.78 | 10%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.743 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 537.356 ms | 1.15 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.059 ms | 0.42 | 1.67 | 2%\* | ok | 0.9x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.307 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 626.857 ms | 0.98 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.056 ms | 0.42 | 1.67 | 2%\* | ok | 0.9x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.483 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.118 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.443 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.213 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.626 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.698 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.160 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.682 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.946 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.830 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.080 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.472 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 20.776 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.575 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilut | f64 | 128^2 grid | 12.131 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.981 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.020 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.703 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.968 ms | 2.70 | 30.30 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.197 ms | 2.19 | 24.49 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.533 ms | 1.71 | 19.13 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.714 ms | 2.80 | 21.19 | 25% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.139 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 43.953 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 34.207 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.891 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.809 ms | 3.49 | 13.96 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.171 ms | 51.92 | 0.81 | 9% | ok | 7.5x (94%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.532 ms | 31.55 | 33.52 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.307 ms | 7.27 | 29.09 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.978 ms | 17.15 | 34.31 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.684 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.684 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.503 ms | 107.25 | 1.26 | n/a | ok | 7.0x (87%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.748 ms | 97.70 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.97 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.357 ms | 197.88 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.607 ms | 74.42 | 0.44 | n/a | ok | 7.4x (93%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

