## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 67.8 GFLOP/s, f32 135.2 GFLOP/s |
| ... hardware, NEON | f64 67.5, f32 135.2 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.8, f32 135.1 GFLOP/s |
| bandwidth | read 23.1, update 25.0, triad 22.8 GB/s (ceiling 25.0) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 113% | 0 | 28% |
| blas2 | 10 | 56% | 88% | 0 | 26% |
| blas3 | 10 | 25% | 55% | 0 | 57% |
| factor | 12 | 3% | 39% | 0 | 35% |
| eigen | 8 | 4% | 13% | 0 | 28% |
| elementwise | 5 | 5% | 9% | 0 | 26% |
| sparse | 5 | 55% | 91% | 0 | 28% |
| direct | 19 | 0% | 6% | 0 | 27% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 45% | 60% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 38% | 74% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 97% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.555 ms | 3.02 | 24.16 | 97% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.805 ms | 5.98 | 23.92 | 96% | ok | 1.0x (26%) |
| blas1 | axpy | f64 | n=8388608 | 8.157 ms | 2.06 | 24.68 | 99% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.979 ms | 4.22 | 25.30 | 101% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.759 ms | 1.76 | 28.21 | 113% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.634 ms | 6.37 | 25.48 | 102% | ok | 1.2x (29%) |
| blas1 | one_norm | f64 | n=8388608 | 7.882 ms | 1.06 | 8.51 | 34% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.624 ms | 0.54 | 4.30 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 8.041 ms | 1.04 | 8.35 | 33% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 180.048 ms | 1.03 | 0.75 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.835 ms | 4.91 | 19.65 | 79% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 3.048 ms | 11.01 | 22.03 | 88% | ok | 1.3x (34%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.715 ms | 4.35 | 17.41 | 70% | ok | 1.0x (25%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.925 ms | 6.81 | 13.63 | 55% | ok | 0.8x (20%) |
| blas2 | symv | f64 | n=4096 | 15.467 ms | 2.17 | 4.34 | 17% | ok | 1.1x (28%) |
| blas2 | trmv | f64 | n=4096 | 4.837 ms | 3.47 | 13.89 | 56% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 2.616 ms | 6.41 | 12.84 | 51% | ok | 1.0x (25%) |
| blas2 | trsv | f64 | n=4096 | 4.849 ms | 3.46 | 13.85 | 55% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.558 ms | 6.56 | 13.13 | 53% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.982 ms | 2.24 | 17.92 | 72% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.404 ms | 36.26 | 0.85 | 53% | ok | 3.8x (95%) |
| blas3 | gemm | f32 | 512x512x512 | 3.586 ms | 74.86 | 0.88 | 55% | ok | 3.9x (96%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.923 ms | 22.66 | 1.06 | 33% | ok | 3.0x (76%) |
| blas3 | symm | f64 | 512x512x512 | 8.127 ms | 33.03 | 0.77 | 49% | ok | 3.6x (89%) |
| blas3 | syrk | f64 | 512x512x512 | 39.616 ms | 3.39 | 0.11 | 5% | ok | 1.1x (26%) |
| blas3 | syr2k | f64 | 512x512x512 | 85.766 ms | 3.13 | 0.07 | 5% | ok | 0.9x (24%) |
| blas3 | trmm | f64 | 512x512x512 | 7.799 ms | 17.21 | 0.67 | 25% | ok | 2.3x (57%) |
| blas3 | trsm | f64 | 512x512x512 | 8.073 ms | 16.63 | 0.65 | 25% | ok | 2.2x (55%) |
| blas3 | trsm | f32 | 512x512x512 | 4.984 ms | 26.93 | 0.53 | 20% | ok | 1.7x (43%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.373 ms | 0.19 | 1.50 | 6% | ok | 0.9x (23%) |
| factor | lu | f64 | 512x512 | 17.217 ms | 5.20 | 0.24 | 8% | ok | 1.4x (35%) |
| factor | lu | f32 | 512x512 | 8.926 ms | 10.02 | 0.23 | 7% | ok | 1.4x (36%) |
| factor | cholesky | f64 | 512x512 | 19.979 ms | 2.24 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.569 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.323 ms | 2.20 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 61.355 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 146.966 ms | 1.22 | 0.03 | 2% | ok | 1.7x (43%) |
| factor | lq | f64 | 512x512 | 36.790 ms | 4.86 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 199.962 ms | 2.24 | 0.02 | 3% | ok | 1.8x (44%) |
| factor | householder | f64 | 512x512 | 0.427 ms | 2.46 | 9.82 | 39% | ok | 3.6x (90%) |
| factor | inv | f64 | 512x512 | 126.047 ms | 2.13 | 0.03 | 3% | ok | 1.1x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.228 ms | 7.32 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.813 ms | 0.62 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.016 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 527.104 ms | 0.29 | 0.00 | 0% | ok | 1.2x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 60.291 ms | 2.78 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.671 ms | 3.11 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.773 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 29.124 ms | 8.64 | 0.04 | 13% | ok | 1.6x (41%) |
| eigen | sylvester | f64 | 256x256 | 314.992 ms | 3.20 | 0.00 | 5% | ok | 1.1x (28%) |
| elementwise | exp | f64 | n=8388608 | 80.767 ms | 0.10 | 1.66 | 7% | ok | 1.0x (26%) |
| elementwise | log | f64 | n=8388608 | 100.787 ms | 0.08 | 1.33 | 5% | ok | 1.0x (26%) |
| elementwise | sin | f64 | n=8388608 | 134.424 ms | 0.06 | 1.00 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 62.521 ms | 0.13 | 2.15 | 9% | ok | 1.1x (26%) |
| elementwise | erf | f64 | n=8388608 | 101.749 ms | 0.08 | 1.32 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.202 ms | 2.18 | 22.66 | 91% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.998 ms | 1.31 | 13.63 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.107 ms | 6.86 | 14.42 | 58% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.140 ms | 0.27 | 0.95 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.694 ms | 0.23 | 1.91 | 8% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 164.721 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 55.331 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.518 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 94.575 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 127.652 ms | 0.18 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.940 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 189.474 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.694 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.153 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.631 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.827 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.420 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 156.392 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.016 ms | 0.38 | 1.50 | 6% | ok | 0.9x (22%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.743 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.387 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.995 ms | 0.38 | 1.52 | 6% | ok | 0.9x (22%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.997 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 81.417 ms | 0.27 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.574 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.027 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 109.094 ms | 0.21 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.566 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.153 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 15065.494 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.679 ms | 0.44 | 1.75 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 42.413 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 185.491 ms | 1.66 | 0.00 | 2% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.976 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.721 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1587.984 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.301 ms | 0.11 | 0.46 | 2% | ok | 0.9x (23%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.273 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2354.770 ms | 0.26 | 0.00 | 0% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.791 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.252 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 96.839 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.857 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.094 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.334 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.549 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.174 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.790 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.506 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.726 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.454 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.586 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.179 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.582 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.480 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.866 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.739 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.592 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.971 ms | 1.33 | 14.88 | 60% | ok | 1.1x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.161 ms | 0.63 | 7.05 | 28% | ok | 1.1x (29%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.638 ms | 0.99 | 11.12 | 45% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.485 ms | 1.18 | 8.95 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 211.902 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 244.706 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 135.230 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.798 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.637 ms | 4.61 | 18.45 | 74% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 10.514 ms | 25.53 | 0.40 | 38% | ok | 4.0x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.380 ms | 7.05 | 7.49 | 30% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.795 ms | 6.00 | 24.01 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.372 ms | 12.23 | 24.46 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.102 ms | 7.98 | 7.98 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 3.810 ms | 4.40 | 4.40 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.463 ms | 60.14 | 0.70 | n/a | ok | 3.9x (97%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.255 ms | 63.09 | 0.49 | n/a | ok | 4.0x (100%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.193 ms | 64.02 | 0.38 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.958 ms | 45.06 | 0.26 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 12.734 ms | 21.08 | 0.12 | n/a | ok | 4.0x (100%) |

