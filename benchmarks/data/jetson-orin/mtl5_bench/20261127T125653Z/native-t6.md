## jetson-orin -- native, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 64.2 GFLOP/s, f32 128.5 GFLOP/s |
| ... hardware, NEON | f64 64.2, f32 128.5 GFLOP/s |
| ... this build, scalar | f64 32.1, f32 32.1 GFLOP/s |
| bandwidth | read 28.0, update 36.6, triad 32.7 GB/s (ceiling 36.6) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 83% | 129% | 0 | 37% |
| blas2 | 10 | 55% | 81% | 0 | 19% |
| blas3 | 10 | 6% | 19% | 0 | 17% |
| factor | 12 | 4% | 21% | 0 | 22% |
| eigen | 8 | 3% | 8% | 0 | 17% |
| elementwise | 5 | 3% | 7% | 0 | 17% |
| sparse | 5 | 24% | 88% | 0 | 17% |
| direct | 19 | 0% | 4% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 1% | 4% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 21% | 27% | 0 | 17% |
| mg | 1 | 19% | 19% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 15% | 32% | 0 | 20% |
| integer | 9 | n/a | n/a | 0 | 71% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.573 ms | 3.67 | 29.35 | 80% | ok | 2.5x (42%) |
| blas1 | dot | f32 | n=8388608 | 2.196 ms | 7.64 | 30.55 | 83% | ok | 2.7x (46%) |
| blas1 | axpy | f64 | n=8388608 | 5.526 ms | 3.04 | 36.43 | 99% | ok | 2.2x (37%) |
| blas1 | axpy | f32 | n=8388608 | 2.464 ms | 6.81 | 40.86 | 112% | ok | 2.4x (41%) |
| blas1 | scale | f64 | n=8388608 | 2.845 ms | 2.95 | 47.18 | 129% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.184 ms | 7.68 | 30.72 | 84% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.508 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.764 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.510 ms | 0.67 | 5.36 | 15% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 192.019 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.217 ms | 6.43 | 25.74 | 70% | ok | 2.5x (41%) |
| blas2 | gemv | f32 | n=4096 | 2.277 ms | 14.74 | 29.49 | 81% | ok | 3.4x (57%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.477 ms | 5.18 | 20.73 | 57% | ok | 2.6x (43%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.269 ms | 10.27 | 20.54 | 56% | ok | 2.0x (33%) |
| blas2 | symv | f64 | n=4096 | 22.963 ms | 1.46 | 2.93 | 8% | ok | 1.1x (19%) |
| blas2 | trmv | f64 | n=4096 | 8.883 ms | 1.89 | 7.56 | 21% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 5.187 ms | 3.23 | 6.48 | 18% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 8.649 ms | 1.94 | 7.77 | 21% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.760 ms | 3.52 | 7.06 | 19% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.397 ms | 2.50 | 20.04 | 55% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 21.692 ms | 12.37 | 0.29 | 19% | ok | 3.5x (58%) |
| blas3 | gemm | f32 | 512x512x512 | 12.347 ms | 21.74 | 0.25 | 17% | ok | 3.0x (50%) |
| blas3 | gemmt | f64 | 512x512x512 | 48.709 ms | 2.76 | 0.13 | 4% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 75.095 ms | 3.57 | 0.08 | 6% | ok | 1.0x (16%) |
| blas3 | syrk | f64 | 512x512x512 | 50.497 ms | 2.66 | 0.08 | 4% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 89.628 ms | 3.00 | 0.07 | 5% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 37.917 ms | 3.54 | 0.14 | 6% | ok | 1.0x (16%) |
| blas3 | trsm | f64 | 512x512x512 | 38.687 ms | 3.47 | 0.14 | 5% | ok | 1.0x (16%) |
| blas3 | trsm | f32 | 512x512x512 | 19.177 ms | 7.00 | 0.14 | 5% | ok | 1.0x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.347 ms | 0.57 | 4.54 | 12% | ok | 1.1x (18%) |
| factor | lu | f64 | 512x512 | 24.712 ms | 3.62 | 0.17 | 6% | ok | 1.4x (24%) |
| factor | lu | f32 | 512x512 | 14.423 ms | 6.20 | 0.15 | 5% | ok | 1.4x (24%) |
| factor | cholesky | f64 | 512x512 | 31.679 ms | 1.41 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.141 ms | 1.48 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 32.152 ms | 1.39 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 16.144 ms | 2.77 | 0.26 | 4% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 208.991 ms | 0.86 | 0.02 | 1% | ok | 1.6x (26%) |
| factor | lq | f64 | 512x512 | 69.639 ms | 2.57 | 0.06 | 4% | ok | 1.3x (22%) |
| factor | hessenberg | f64 | 512x512 | 302.294 ms | 1.48 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.548 ms | 1.92 | 7.66 | 21% | ok | 3.7x (61%) |
| factor | inv | f64 | 512x512 | 181.520 ms | 1.48 | 0.02 | 2% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.359 ms | 4.62 | 0.22 | 7% | ok | 1.3x (22%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.981 ms | 0.36 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 145.442 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 900.055 ms | 0.17 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 99.734 ms | 1.68 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 215.530 ms | 1.95 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 358.628 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 48.682 ms | 5.17 | 0.02 | 8% | ok | 2.1x (35%) |
| eigen | sylvester | f64 | 256x256 | 504.577 ms | 2.00 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 75.923 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 108.014 ms | 0.08 | 1.24 | 3% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 177.911 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 55.188 ms | 0.15 | 2.43 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 127.016 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.844 ms | 3.10 | 32.24 | 88% | ok | 2.7x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.039 ms | 0.86 | 8.96 | 24% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.915 ms | 7.08 | 14.89 | 41% | ok | 3.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.970 ms | 0.16 | 0.55 | 1% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.804 ms | 0.17 | 1.41 | 4% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 53.793 ms | 0.21 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 90.079 ms | 0.12 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.205 ms | 0.31 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 135.245 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 155.324 ms | 0.14 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.960 ms | 0.37 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 106.088 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.165 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 180.802 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.896 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.925 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.861 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 43.746 ms | 0.26 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.349 ms | 0.32 | 1.29 | 4% | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.677 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.798 ms | 0.67 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.134 ms | 0.35 | 1.42 | 4% | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.629 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 116.046 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.718 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.317 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 132.761 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.844 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 17.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 847.535 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.393 ms | 0.32 | 1.30 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 37.060 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 256.956 ms | 1.20 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.298 ms | 0.33 | 1.31 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.241 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1941.512 ms | 0.32 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.849 ms | 0.09 | 0.36 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.458 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1504.730 ms | 0.41 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.022 ms | 0.09 | 0.35 | 1% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.147 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 229.342 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.990 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.791 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.925 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.847 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.846 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.715 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.708 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.609 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.508 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.513 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.625 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.168 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.811 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.711 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.494 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.386 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.097 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.349 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.994 ms | 0.87 | 9.80 | 27% | ok | 1.0x (16%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.736 ms | 0.70 | 7.85 | 21% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 8.143 ms | 0.32 | 3.60 | 10% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.021 ms | 0.92 | 6.92 | 19% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 350.011 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 353.801 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 188.813 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.505 ms | 1.34 | 5.37 | 15% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.894 ms | 20.82 | 0.33 | 32% | ok | 5.9x (98%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.567 ms | 3.67 | 3.90 | 11% | ok | 1.2x (20%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.611 ms | 2.99 | 11.96 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.712 ms | 6.19 | 12.37 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.701 ms | 2.94 | 2.94 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.761 ms | 9.53 | 9.53 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.985 ms | 29.88 | 0.35 | n/a | ok | 4.2x (71%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.081 ms | 37.91 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.490 ms | 48.90 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.490 ms | 48.90 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.299 ms | 26.06 | 0.15 | n/a | ok | 5.8x (97%) |

