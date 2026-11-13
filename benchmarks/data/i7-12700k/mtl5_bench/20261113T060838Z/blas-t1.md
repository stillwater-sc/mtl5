## i7-12700k -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.4, update 38.5, triad 32.7 GB/s (ceiling 38.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 98% | 0 |
| blas2 | 10 | 98% | 107% | 0 |
| blas3 | 10 | 81% | 94% | 0 |
| factor | 12 | 22% | 40% | 0 |
| eigen | 8 | 24% | 71% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 48% | 95% | 0 |
| direct | 31 | 3% | 32% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 21 | 25% | 70% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 78% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 36% | 83% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.580 ms | 3.01 | 24.06 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.214 ms | 3.98 | 15.93 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.438 ms | 2.61 | 31.27 | 81% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.930 ms | 3.40 | 20.42 | 53% | ok |
| blas1 | scale | f64 | n=8388608 | 3.570 ms | 2.35 | 37.59 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.827 ms | 3.48 | 13.90 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 3.978 ms | 2.11 | 16.87 | 44% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.029 ms | 1.19 | 9.55 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.947 ms | 2.13 | 17.00 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.460 ms | 2.41 | 1.76 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.489 ms | 9.62 | 38.49 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 1.649 ms | 20.35 | 40.72 | 106% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.610 ms | 9.30 | 37.20 | 97% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.635 ms | 20.52 | 41.06 | 107% | ok |
| blas2 | symv | f64 | n=4096 | 1.916 ms | 17.52 | 35.07 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 1.851 ms | 9.06 | 36.29 | 94% | ok |
| blas2 | trmv | f32 | n=4096 | 0.943 ms | 17.80 | 35.63 | 92% | ok |
| blas2 | trsv | f64 | n=4096 | 1.785 ms | 9.40 | 37.63 | 98% | ok |
| blas2 | trsv | f32 | n=4096 | 0.872 ms | 19.24 | 38.52 | 100% | ok |
| blas2 | ger | f64 | n=4096 | 7.664 ms | 4.38 | 35.03 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.704 ms | 72.47 | 1.70 | 93% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.832 ms | 146.56 | 1.72 | 94% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.578 ms | 52.07 | 2.44 | 66% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.711 ms | 72.33 | 1.70 | 92% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.546 ms | 52.72 | 1.65 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.744 ms | 56.58 | 1.33 | 72% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.113 ms | 63.52 | 2.48 | 81% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.134 ms | 62.91 | 2.46 | 80% | ok |
| blas3 | trsm | f32 | 512x512x512 | 0.963 ms | 139.32 | 2.72 | 89% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.096 ms | 0.41 | 3.24 | 8% | ok |
| factor | lu | f64 | 512x512 | 3.339 ms | 26.80 | 1.26 | 34% | ok |
| factor | lu | f32 | 512x512 | 1.781 ms | 50.25 | 1.18 | 32% | ok |
| factor | cholesky | f64 | 512x512 | 2.050 ms | 21.83 | 2.05 | 28% | ok |
| factor | cholesky | f32 | 512x512 | 1.304 ms | 34.30 | 1.61 | 22% | ok |
| factor | ldlt | f64 | 512x512 | 7.465 ms | 5.99 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 46.657 ms | 0.96 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 5.750 ms | 31.12 | 0.73 | 40% | ok |
| factor | lq | f64 | 512x512 | 25.573 ms | 7.00 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 268.325 ms | 1.67 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.633 ms | 0.64 | 2.57 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 45.725 ms | 5.87 | 0.09 | 7% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 3.786 ms | 23.63 | 1.11 | 30% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.699 ms | 13.16 | 0.62 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 58.924 ms | 2.56 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 769.140 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.021 ms | 10.47 | 0.07 | 13% | ok |
| eigen | schur | f64 | 256x256 | 21.208 ms | 19.78 | 0.05 | 25% | ok |
| eigen | svd | f64 | 256x256 | 6.675 ms | 55.29 | 0.16 | 71% | ok |
| eigen | expm | f64 | 256x256 | 11.222 ms | 22.43 | 0.09 | 29% | ok |
| eigen | sylvester | f64 | 256x256 | 53.574 ms | 18.79 | 0.03 | 24% | ok |
| elementwise | exp | f64 | n=8388608 | 39.227 ms | 0.21 | 3.42 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.541 ms | 0.18 | 2.95 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 59.963 ms | 0.14 | 2.24 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.845 ms | 0.29 | 4.65 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 201.570 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.741 ms | 3.53 | 36.75 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.930 ms | 2.81 | 29.27 | 76% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.769 ms | 8.78 | 18.46 | 48% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.878 ms | 0.67 | 2.36 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.244 ms | 1.61 | 13.38 | 35%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.045 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.539 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.916 ms | 1.25 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.490 ms | 0.57 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.607 ms | 0.42 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.784 ms | 1.33 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.843 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.384 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.968 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.221 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.133 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.361 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 88.448 ms | 0.13 | 0.01 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.246 ms | 3.07 | 12.30 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.526 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.183 ms | 2.67 | 0.17 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.248 ms | 3.05 | 12.19 | 32%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.090 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.413 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.643 ms | 1.18 | 4.71 | 12%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.929 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.630 ms | 0.47 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.666 ms | 1.14 | 4.54 | 12%\* | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.637 ms | 1.98 | 0.13 | 3% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.882 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.385 ms | 3.30 | 0.22 | 4% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.327 ms | 2.31 | 9.25 | 24%\* | ok |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 4.524 ms | 2.47 | 0.16 | 3% | ok |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 2.110 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 1.901 ms | 5.88 | 0.38 | 8% | ok |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.462 ms | 1.64 | 6.55 | 17%\* | ok |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 7.448 ms | 3.00 | 0.10 | 4% | ok |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.591 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 3.223 ms | 6.94 | 0.23 | 9% | ok |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.575 ms | 1.32 | 5.26 | 14%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.497 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7433.298 ms | 0.04 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.416 ms | 2.38 | 9.52 | 25%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.719 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.644 ms | 4.43 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.444 ms | 2.33 | 9.33 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.878 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 551.574 ms | 1.12 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.505 ms | 0.45 | 1.80 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.629 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 666.424 ms | 0.93 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.509 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.912 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 11.397 ms | 27.08 | 0.08 | 35% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.501 ms | 6.72 | 26.88 | 70%\* | ok |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.942 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 9.680 ms | 31.88 | 0.09 | 41% | ok |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.870 ms | 3.87 | 15.49 | 40%\* | ok |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 9.045 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 19.119 ms | 32.28 | 0.04 | 41% | ok |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.052 ms | 3.20 | 12.81 | 33%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.158 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 33.001 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.926 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.557 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.047 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.058 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.718 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.410 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.360 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.264 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.550 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.946 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.225 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.565 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.248 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.073 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.015 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.170 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.976 ms | 2.68 | 30.04 | 78% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.217 ms | 2.15 | 24.09 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.562 ms | 1.68 | 18.78 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.238 ms | 2.90 | 21.96 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.869 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.238 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 34.205 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.054 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.850 ms | 3.46 | 13.84 | 36% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.939 ms | 6.89 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.554 ms | 30.26 | 32.15 | 83%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.294 ms | 7.31 | 29.26 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.996 ms | 16.84 | 33.69 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.721 ms | 9.75 | 9.75 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.720 ms | 9.75 | 9.75 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.442 ms | 15.39 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.845 ms | 12.88 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.220 ms | 26.27 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.219 ms | 26.27 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.814 ms | 10.01 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

