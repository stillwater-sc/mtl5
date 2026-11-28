## jetson-orin -- native, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.4 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.4 GFLOP/s |
| ... this build, scalar | f64 32.2, f32 32.2 GFLOP/s |
| bandwidth | read 29.7, update 34.7, triad 33.1 GB/s (ceiling 34.7) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 89% | 121% | 0 | 36% |
| blas2 | 10 | 58% | 84% | 0 | 17% |
| blas3 | 10 | 6% | 21% | 0 | 17% |
| factor | 12 | 4% | 23% | 0 | 23% |
| eigen | 8 | 3% | 8% | 0 | 17% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 26% | 93% | 0 | 18% |
| direct | 19 | 0% | 4% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 1% | 4% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 29% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 15% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 71% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.276 ms | 3.92 | 31.39 | 91% | ok | 2.7x (45%) |
| blas1 | dot | f32 | n=8388608 | 2.214 ms | 7.58 | 30.31 | 87% | ok | 3.0x (50%) |
| blas1 | axpy | f64 | n=8388608 | 5.564 ms | 3.02 | 36.18 | 104% | ok | 2.2x (37%) |
| blas1 | axpy | f32 | n=8388608 | 2.737 ms | 6.13 | 36.78 | 106% | ok | 2.2x (36%) |
| blas1 | scale | f64 | n=8388608 | 3.191 ms | 2.63 | 42.06 | 121% | ok | 2.0x (33%) |
| blas1 | two_norm | f64 | n=8388608 | 2.182 ms | 7.69 | 30.75 | 89% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.504 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.757 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.504 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 192.495 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.608 ms | 5.98 | 23.95 | 69% | ok | 2.3x (39%) |
| blas2 | gemv | f32 | n=4096 | 2.296 ms | 14.61 | 29.24 | 84% | ok | 3.4x (56%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.508 ms | 5.16 | 20.63 | 60% | ok | 2.5x (42%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.191 ms | 10.51 | 21.04 | 61% | ok | 2.0x (34%) |
| blas2 | symv | f64 | n=4096 | 23.549 ms | 1.42 | 2.85 | 8% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 8.861 ms | 1.89 | 7.58 | 22% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 5.289 ms | 3.17 | 6.35 | 18% | ok | 1.0x (16%) |
| blas2 | trsv | f64 | n=4096 | 8.769 ms | 1.91 | 7.66 | 22% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.902 ms | 3.42 | 6.85 | 20% | ok | 1.0x (16%) |
| blas2 | ger | f64 | n=4096 | 13.410 ms | 2.50 | 20.02 | 58% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 21.299 ms | 12.60 | 0.30 | 20% | ok | 3.5x (59%) |
| blas3 | gemm | f32 | 512x512x512 | 10.165 ms | 26.41 | 0.31 | 21% | ok | 3.8x (63%) |
| blas3 | gemmt | f64 | 512x512x512 | 48.543 ms | 2.76 | 0.13 | 4% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 73.654 ms | 3.64 | 0.09 | 6% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 51.002 ms | 2.63 | 0.08 | 4% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 91.177 ms | 2.94 | 0.07 | 5% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 37.221 ms | 3.61 | 0.14 | 6% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 37.896 ms | 3.54 | 0.14 | 6% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 19.334 ms | 6.94 | 0.14 | 5% | ok | 1.0x (17%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.838 ms | 0.54 | 4.32 | 12% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 24.731 ms | 3.62 | 0.17 | 6% | ok | 1.5x (24%) |
| factor | lu | f32 | 512x512 | 14.443 ms | 6.20 | 0.15 | 5% | ok | 1.4x (24%) |
| factor | cholesky | f64 | 512x512 | 31.725 ms | 1.41 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.130 ms | 1.48 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 32.199 ms | 1.39 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 15.995 ms | 2.80 | 0.26 | 4% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 206.089 ms | 0.87 | 0.02 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 66.692 ms | 2.68 | 0.06 | 4% | ok | 1.4x (23%) |
| factor | hessenberg | f64 | 512x512 | 299.870 ms | 1.49 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.534 ms | 1.97 | 7.85 | 23% | ok | 3.8x (63%) |
| factor | inv | f64 | 512x512 | 181.256 ms | 1.48 | 0.02 | 2% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.500 ms | 4.59 | 0.22 | 7% | ok | 1.3x (22%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.484 ms | 0.36 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 145.126 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 937.497 ms | 0.16 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 99.017 ms | 1.69 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 214.985 ms | 1.95 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 358.486 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 49.010 ms | 5.13 | 0.02 | 8% | ok | 2.1x (35%) |
| eigen | sylvester | f64 | 256x256 | 507.513 ms | 1.98 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 75.789 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 108.273 ms | 0.08 | 1.24 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 177.825 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 55.105 ms | 0.15 | 2.44 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.611 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.847 ms | 3.09 | 32.15 | 93% | ok | 2.7x (45%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.004 ms | 0.87 | 9.06 | 26% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.748 ms | 8.82 | 18.54 | 54% | ok | 3.6x (60%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.116 ms | 0.15 | 0.54 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.073 ms | 0.18 | 1.50 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 53.226 ms | 0.21 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 97.518 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.170 ms | 0.31 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 134.285 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 156.325 ms | 0.14 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.994 ms | 0.37 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 113.324 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.095 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.111 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 181.549 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.890 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.595 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.796 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 43.817 ms | 0.26 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.147 ms | 0.35 | 1.41 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.634 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.771 ms | 0.67 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.109 ms | 0.36 | 1.43 | 4% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.651 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 117.392 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.593 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.406 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 131.220 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.663 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.891 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 834.263 ms | 0.37 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.777 ms | 0.34 | 1.38 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 36.916 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 255.914 ms | 1.21 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 9.883 ms | 0.34 | 1.36 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.230 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1939.034 ms | 0.32 | 0.00 | 0% | ok | 1.0x (16%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.149 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.431 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1517.790 ms | 0.41 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.141 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.872 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 166.788 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.609 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.699 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.930 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.765 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.545 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.730 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.602 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.508 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.259 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.430 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.545 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.771 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.796 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.548 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.344 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.110 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.362 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.885 ms | 0.91 | 10.16 | 29% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.737 ms | 0.70 | 7.85 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.525 ms | 0.58 | 6.48 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.201 ms | 0.91 | 6.89 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 357.403 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 360.948 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 188.410 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.109 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.552 ms | 1.34 | 5.35 | 15% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.953 ms | 20.72 | 0.32 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.566 ms | 3.67 | 3.90 | 11% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.641 ms | 2.97 | 11.90 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.681 ms | 6.26 | 12.52 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.698 ms | 2.94 | 2.94 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.962 ms | 29.95 | 0.35 | n/a | ok | 4.2x (71%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.071 ms | 37.96 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.483 ms | 48.95 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.482 ms | 48.96 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.299 ms | 26.06 | 0.15 | n/a | ok | 5.8x (97%) |

