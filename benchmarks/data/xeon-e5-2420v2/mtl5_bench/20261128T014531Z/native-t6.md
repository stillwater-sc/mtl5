## xeon-e5-2420v2 -- native, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 58.9 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 58.9, f32 118.2 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 84% |
| blas2 | 10 | 85% | 107% | 0 | 17% |
| blas3 | 10 | 7% | 39% | 0 | 17% |
| factor | 12 | 6% | 40% | 0 | 22% |
| eigen | 8 | 3% | 11% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 105% | 0 | 17% |
| direct | 19 | 1% | 30% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 74% | 79% | 0 | 17% |
| mg | 1 | 72% | 72% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 69% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 84% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 32.745 ms | 0.51 | 4.10 | 36% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=8388608 | 5.492 ms | 3.05 | 12.22 | 107% | ok | 5.1x (84%) |
| blas1 | axpy | f64 | n=8388608 | 38.030 ms | 0.44 | 5.29 | 46% | ok | 5.6x (93%) |
| blas1 | axpy | f32 | n=8388608 | 16.746 ms | 1.00 | 6.01 | 53% | ok | 5.5x (91%) |
| blas1 | scale | f64 | n=8388608 | 12.144 ms | 0.69 | 11.05 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 33.137 ms | 0.51 | 2.03 | 18% | ok | 5.6x (93%) |
| blas1 | one_norm | f64 | n=8388608 | 9.719 ms | 0.86 | 6.91 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.065 ms | 0.64 | 5.14 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.712 ms | 0.86 | 6.91 | 61% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 277.277 ms | 0.67 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.946 ms | 3.07 | 12.27 | 107% | ok | 1.2x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.498 ms | 6.10 | 12.21 | 107% | ok | 1.6x (26%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 11.235 ms | 2.99 | 11.95 | 105% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.480 ms | 6.12 | 12.25 | 107% | ok | 1.2x (20%) |
| blas2 | symv | f64 | n=4096 | 12.461 ms | 2.69 | 5.39 | 47% | ok | 1.0x (16%) |
| blas2 | trmv | f64 | n=4096 | 7.221 ms | 2.32 | 9.30 | 81% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 4.920 ms | 3.41 | 6.83 | 60% | ok | 0.9x (16%) |
| blas2 | trsv | f64 | n=4096 | 6.954 ms | 2.41 | 9.66 | 85% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.704 ms | 3.57 | 7.14 | 63% | ok | 0.9x (16%) |
| blas2 | ger | f64 | n=4096 | 28.553 ms | 1.18 | 9.40 | 82% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 11.546 ms | 23.25 | 0.54 | 39% | ok | 5.6x (94%) |
| blas3 | gemm | f32 | 512x512x512 | 6.252 ms | 42.94 | 0.50 | 36% | ok | 5.5x (92%) |
| blas3 | gemmt | f64 | 512x512x512 | 36.165 ms | 3.71 | 0.17 | 6% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 65.870 ms | 4.08 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 38.103 ms | 3.52 | 0.11 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 65.772 ms | 4.08 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 32.449 ms | 4.14 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 33.187 ms | 4.04 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 17.096 ms | 7.85 | 0.15 | 7% | ok | 1.0x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.329 ms | 0.14 | 1.11 | 10% | ok | 1.1x (18%) |
| factor | lu | f64 | 512x512 | 21.911 ms | 4.08 | 0.19 | 7% | ok | 1.4x (24%) |
| factor | lu | f32 | 512x512 | 12.106 ms | 7.39 | 0.17 | 6% | ok | 1.3x (22%) |
| factor | cholesky | f64 | 512x512 | 25.039 ms | 1.79 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 23.897 ms | 1.87 | 0.09 | 2% | ok | 1.0x (16%) |
| factor | ldlt | f64 | 512x512 | 25.403 ms | 1.76 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 13.460 ms | 3.32 | 0.31 | 6% | ok | 1.0x (16%) |
| factor | qr | f64 | 512x512 | 323.232 ms | 0.55 | 0.01 | 1% | ok | 1.6x (26%) |
| factor | lq | f64 | 512x512 | 49.135 ms | 3.64 | 0.09 | 6% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 401.253 ms | 1.11 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.910 ms | 1.15 | 4.61 | 40%\* | ok | 3.6x (61%) |
| factor | inv | f64 | 512x512 | 143.211 ms | 1.87 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.975 ms | 5.27 | 0.25 | 9% | ok | 1.3x (21%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.981 ms | 0.36 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 172.488 ms | 0.88 | 0.01 | 1% | ok | 1.0x (16%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 989.790 ms | 0.15 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 107.657 ms | 1.56 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 264.611 ms | 1.59 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 302.166 ms | 1.22 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 37.624 ms | 6.69 | 0.03 | 11% | ok | 2.2x (37%) |
| eigen | sylvester | f64 | 256x256 | 595.989 ms | 1.69 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 152.448 ms | 0.06 | 0.88 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 144.359 ms | 0.06 | 0.93 | 8% | ok | 1.0x (16%) |
| elementwise | sin | f64 | n=8388608 | 170.535 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 90.137 ms | 0.09 | 1.49 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2479.223 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.271 ms | 1.15 | 11.99 | 105% | ok | 1.1x (19%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.121 ms | 0.84 | 8.73 | 76% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.343 ms | 3.69 | 7.76 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.590 ms | 0.20 | 0.69 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.953 ms | 0.34 | 2.80 | 24% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.981 ms | 0.27 | 0.02 | 0% | ok | 1.0x (16%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 68.613 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.964 ms | 0.45 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 124.298 ms | 0.18 | 0.01 | 0% | ok | 0.9x (15%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.575 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.252 ms | 0.46 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 82.463 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.924 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.979 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.524 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 44.278 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.238 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.247 ms | 0.34 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.872 ms | 0.87 | 3.47 | 30%\* | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.363 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.483 ms | 0.90 | 0.06 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.908 ms | 0.83 | 3.33 | 29%\* | ok | 0.9x (15%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.283 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.981 ms | 0.22 | 0.01 | 0% | ok | 1.1x (18%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.767 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.789 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 114.934 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.773 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.788 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 633.652 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.968 ms | 0.48 | 1.93 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.870 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 204.500 ms | 1.51 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.042 ms | 0.48 | 1.91 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.410 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1712.214 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.554 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.042 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1527.563 ms | 0.40 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.544 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 230.933 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 401.282 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 127.825 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 103.319 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 121.844 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 80.536 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 533.111 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.165 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 90.975 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 201.302 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 194.830 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 211.890 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 310.407 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.184 ms | 0.00 | 0.00 | n/a | ok | 0.9x (15%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.380 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 32.697 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.874 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.723 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.093 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.197 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.469 ms | 0.75 | 8.45 | 74% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.239 ms | 0.81 | 9.05 | 79% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.850 ms | 0.68 | 7.62 | 67% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.589 ms | 1.08 | 8.17 | 72% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 141.479 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 146.627 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 127.291 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.450 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 187.857 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.522 ms | 13.08 | 0.20 | 22% | ok | 5.5x (92%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.277 ms | 7.37 | 7.83 | 69% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.400 ms | 2.62 | 10.49 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.114 ms | 5.39 | 10.77 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.977 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.980 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.514 ms | 23.31 | 0.27 | n/a | ok | 5.0x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.247 ms | 32.55 | 0.25 | n/a | ok | 5.5x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.324 ms | 62.07 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.328 ms | 62.03 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.067 ms | 20.54 | 0.12 | n/a | ok | 5.4x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

