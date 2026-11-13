## i7-12700k -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 600.9 GFLOP/s, f32 1201.9 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.9, f32 1201.9 GFLOP/s |
| ... this build, AVX2 | f64 600.8, f32 1201.7 GFLOP/s |
| bandwidth | read 83.0, update 78.9, triad 77.5 GB/s (ceiling 83.0) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 87% | 146% | 0 | 22% |
| blas2 | 10 | 48% | 148% | 0 | 13% |
| blas3 | 10 | 12% | 67% | 0 | 18% |
| factor | 12 | 1% | 12% | 0 | 13% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 291% | 0 | 13% |
| direct | 19 | 0% | 13% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 11% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 28% | 37% | 0 | 13% |
| mg | 1 | 24% | 24% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 49% | 72% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 78% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.394 ms | 12.04 | 96.30 | 116% | ok | 2.6x (33%) |
| blas1 | dot | f32 | n=8388608 | 0.554 ms | 30.29 | 121.18 | 146% | ok | 3.0x (38%) |
| blas1 | axpy | f64 | n=8388608 | 2.774 ms | 6.05 | 72.58 | 87% | ok | 1.7x (21%) |
| blas1 | axpy | f32 | n=8388608 | 1.362 ms | 12.32 | 73.93 | 89% | ok | 1.7x (22%) |
| blas1 | scale | f64 | n=8388608 | 1.903 ms | 4.41 | 70.53 | 85% | ok | 1.8x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.563 ms | 29.78 | 119.13 | 144% | ok | 3.9x (49%) |
| blas1 | one_norm | f64 | n=8388608 | 3.927 ms | 2.14 | 17.09 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.904 ms | 1.22 | 9.72 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.878 ms | 2.16 | 17.30 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 69.276 ms | 2.66 | 1.94 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.438 ms | 23.34 | 93.39 | 113% | ok | 2.4x (29%) |
| blas2 | gemv | f32 | n=4096 | 0.546 ms | 61.45 | 122.97 | 148% | ok | 3.1x (39%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.449 ms | 23.15 | 92.64 | 112% | ok | 2.4x (30%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.675 ms | 49.75 | 99.54 | 120% | ok | 2.5x (31%) |
| blas2 | symv | f64 | n=4096 | 6.192 ms | 5.42 | 10.85 | 13% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.750 ms | 9.59 | 38.38 | 46% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.836 ms | 20.07 | 40.18 | 48% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.780 ms | 9.42 | 37.73 | 45% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.850 ms | 19.74 | 39.51 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.099 ms | 4.73 | 37.82 | 46% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.671 ms | 399.90 | 9.37 | 67% | ok | 5.7x (71%) |
| blas3 | gemm | f32 | 512x512x512 | 0.354 ms | 757.34 | 8.88 | 63% | ok | 5.3x (67%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.248 ms | 107.57 | 5.04 | 18% | ok | 2.1x (26%) |
| blas3 | symm | f64 | 512x512x512 | 1.120 ms | 239.66 | 5.62 | 40% | ok | 3.6x (45%) |
| blas3 | syrk | f64 | 512x512x512 | 10.766 ms | 12.47 | 0.39 | 2% | ok | 0.9x (11%) |
| blas3 | syr2k | f64 | 512x512x512 | 16.674 ms | 16.10 | 0.38 | 3% | ok | 1.2x (15%) |
| blas3 | trmm | f64 | 512x512x512 | 1.884 ms | 71.22 | 2.78 | 12% | ok | 1.4x (18%) |
| blas3 | trsm | f64 | 512x512x512 | 2.142 ms | 62.67 | 2.45 | 10% | ok | 1.3x (16%) |
| blas3 | trsm | f32 | 512x512x512 | 1.105 ms | 121.43 | 2.37 | 10% | ok | 1.2x (15%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.845 ms | 0.41 | 3.31 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.129 ms | 14.60 | 0.68 | 2% | ok | 1.8x (22%) |
| factor | lu | f32 | 512x512 | 3.570 ms | 25.07 | 0.59 | 2% | ok | 1.0x (13%) |
| factor | cholesky | f64 | 512x512 | 6.930 ms | 6.46 | 0.61 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.407 ms | 6.98 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.042 ms | 6.35 | 0.60 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 43.169 ms | 1.04 | 0.10 | 0% | ok | 1.1x (13%) |
| factor | qr | f64 | 512x512 | 126.396 ms | 1.42 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 15.133 ms | 11.83 | 0.28 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 158.763 ms | 2.82 | 0.03 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 0.412 ms | 2.55 | 10.19 | 12%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 51.663 ms | 5.20 | 0.08 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.780 ms | 18.72 | 0.88 | 3% | ok | 1.0x (13%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.493 ms | 1.36 | 0.06 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 55.973 ms | 2.70 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 735.554 ms | 0.21 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 30.866 ms | 5.44 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 78.618 ms | 5.34 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 91.303 ms | 4.04 | 0.01 | 1% | ok | 1.1x (14%) |
| eigen | expm | f64 | 256x256 | 10.026 ms | 25.10 | 0.10 | 4% | ok | 1.2x (15%) |
| eigen | sylvester | f64 | 256x256 | 169.264 ms | 5.95 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.474 ms | 0.22 | 3.49 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.719 ms | 0.18 | 2.94 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.512 ms | 0.14 | 2.29 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.293 ms | 0.30 | 4.74 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 198.832 ms | 0.04 | 0.68 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.23 | 241.72 | 291% ! | ok | 6.5x (82%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.911 ms | 2.87 | 29.89 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.362 ms | 30.75 | 64.66 | 78% | ok | 3.0x (37%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.784 ms | 0.68 | 2.40 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.386 ms | 1.45 | 12.01 | 14%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 71.877 ms | 0.16 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 26.997 ms | 0.41 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.113 ms | 1.23 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.264 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 55.389 ms | 0.40 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.803 ms | 1.33 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 85.698 ms | 0.02 | 0.00 | 0% | ok | 1.2x (15%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.358 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.877 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.222 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.865 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.279 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 68.926 ms | 0.16 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.275 ms | 2.75 | 11.00 | 13%\* | ok | 0.9x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.372 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.280 ms | 2.61 | 0.17 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.289 ms | 2.62 | 10.48 | 13%\* | ok | 0.9x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.056 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.433 ms | 0.63 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.611 ms | 1.24 | 4.95 | 6%\* | ok | 1.1x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.795 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.245 ms | 0.45 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.613 ms | 1.23 | 4.94 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.262 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 5212.481 ms | 0.06 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.537 ms | 2.19 | 8.77 | 11%\* | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.259 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 64.492 ms | 4.79 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.776 ms | 1.90 | 7.59 | 9%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.830 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 511.821 ms | 1.21 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.021 ms | 0.42 | 1.68 | 2%\* | ok | 0.9x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.364 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 643.745 ms | 0.96 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.028 ms | 0.42 | 1.68 | 2%\* | ok | 0.9x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.138 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.367 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.665 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.390 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.646 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.521 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.490 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.426 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.519 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.765 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.848 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 17.909 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.527 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.809 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.068 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.996 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.721 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.954 ms | 2.74 | 30.74 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.284 ms | 2.04 | 22.85 | 28% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.480 ms | 1.77 | 19.81 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.888 ms | 2.58 | 19.52 | 24% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 28.577 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 39.682 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.695 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.505 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.855 ms | 9.04 | 36.18 | 44% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 0.911 ms | 294.56 | 4.60 | 49% | ok | 6.6x (82%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.297 ms | 56.45 | 59.98 | 72%\* | ok | 1.1x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.885 ms | 8.90 | 35.61 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.739 ms | 22.72 | 45.44 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.266 ms | 63.04 | 63.04 | n/a | ok | 1.1x (14%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.251 ms | 66.81 | 66.81 | n/a | ok | 1.1x (14%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 0.624 ms | 430.46 | 5.04 | n/a | ok | 6.4x (80%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 0.819 ms | 327.93 | 2.56 | n/a | ok | 6.9x (86%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 0.820 ms | 327.41 | 1.92 | n/a | ok | 7.0x (87%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.336 ms | 798.49 | 4.68 | n/a | ok | 6.3x (79%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.307 ms | 874.79 | 5.13 | n/a | ok | 6.2x (78%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

