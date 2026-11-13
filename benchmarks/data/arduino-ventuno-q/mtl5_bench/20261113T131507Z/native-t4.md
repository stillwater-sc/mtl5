## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 67.4 GFLOP/s, f32 135.5 GFLOP/s |
| ... hardware, NEON | f64 67.4, f32 135.5 GFLOP/s |
| ... this build, scalar | f64 25.9, f32 33.9 GFLOP/s |
| bandwidth | read 23.0, update 24.8, triad 22.5 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 114% | 0 | 28% |
| blas2 | 10 | 61% | 90% | 0 | 27% |
| blas3 | 10 | 8% | 17% | 0 | 28% |
| factor | 12 | 4% | 38% | 0 | 36% |
| eigen | 8 | 4% | 10% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 55% | 92% | 0 | 27% |
| direct | 19 | 0% | 6% | 0 | 27% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 27% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 60% | 0 | 27% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 32% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 60% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.579 ms | 3.01 | 24.06 | 97% | ok | 1.0x (25%) |
| blas1 | dot | f32 | n=8388608 | 2.806 ms | 5.98 | 23.92 | 96% | ok | 1.6x (40%) |
| blas1 | axpy | f64 | n=8388608 | 8.023 ms | 2.09 | 25.09 | 101% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.944 ms | 4.25 | 25.52 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.744 ms | 1.77 | 28.29 | 114% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.627 ms | 6.39 | 25.54 | 103% | ok | 1.3x (31%) |
| blas1 | one_norm | f64 | n=8388608 | 7.299 ms | 1.15 | 9.19 | 37% | ok | 1.2x (29%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.473 ms | 0.54 | 4.34 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.804 ms | 1.07 | 8.60 | 35% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 180.731 ms | 1.02 | 0.74 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.744 ms | 4.98 | 19.91 | 80% | ok | 1.2x (29%) |
| blas2 | gemv | f32 | n=4096 | 3.000 ms | 11.19 | 22.38 | 90% | ok | 1.9x (48%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.733 ms | 4.34 | 17.37 | 70% | ok | 1.3x (33%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.652 ms | 5.94 | 11.88 | 48% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 15.999 ms | 2.10 | 4.20 | 17% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.418 ms | 3.80 | 15.20 | 61% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 4.447 ms | 3.77 | 7.55 | 30% | ok | 1.1x (27%) |
| blas2 | trsv | f64 | n=4096 | 4.420 ms | 3.80 | 15.20 | 61% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 4.333 ms | 3.87 | 7.75 | 31% | ok | 1.1x (28%) |
| blas2 | ger | f64 | n=4096 | 14.995 ms | 2.24 | 17.91 | 72% | ok | 1.0x (25%) |
| blas3 | gemm | f64 | 512x512x512 | 26.781 ms | 10.02 | 0.23 | 15% | ok | 1.9x (48%) |
| blas3 | gemm | f32 | 512x512x512 | 11.715 ms | 22.91 | 0.27 | 17% | ok | 2.1x (53%) |
| blas3 | gemmt | f64 | 512x512x512 | 39.569 ms | 3.39 | 0.16 | 5% | ok | 1.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 48.281 ms | 5.56 | 0.13 | 8% | ok | 1.1x (28%) |
| blas3 | syrk | f64 | 512x512x512 | 39.616 ms | 3.39 | 0.11 | 5% | ok | 1.1x (26%) |
| blas3 | syr2k | f64 | 512x512x512 | 82.559 ms | 3.25 | 0.08 | 5% | ok | 1.0x (24%) |
| blas3 | trmm | f64 | 512x512x512 | 23.591 ms | 5.69 | 0.22 | 8% | ok | 1.1x (28%) |
| blas3 | trsm | f64 | 512x512x512 | 23.256 ms | 5.77 | 0.23 | 9% | ok | 1.2x (29%) |
| blas3 | trsm | f32 | 512x512x512 | 11.113 ms | 12.08 | 0.24 | 9% | ok | 1.3x (32%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.305 ms | 0.19 | 1.50 | 6% | ok | 1.0x (24%) |
| factor | lu | f64 | 512x512 | 17.167 ms | 5.21 | 0.24 | 8% | ok | 1.4x (36%) |
| factor | lu | f32 | 512x512 | 8.864 ms | 10.09 | 0.24 | 7% | ok | 1.5x (37%) |
| factor | cholesky | f64 | 512x512 | 18.759 ms | 2.38 | 0.22 | 4% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.588 ms | 2.54 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.176 ms | 2.22 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.924 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 146.976 ms | 1.22 | 0.03 | 2% | ok | 1.7x (43%) |
| factor | lq | f64 | 512x512 | 36.524 ms | 4.90 | 0.11 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 201.107 ms | 2.22 | 0.02 | 3% | ok | 1.8x (44%) |
| factor | householder | f64 | 512x512 | 0.439 ms | 2.39 | 9.55 | 38% | ok | 3.5x (89%) |
| factor | inv | f64 | 512x512 | 126.031 ms | 2.13 | 0.03 | 3% | ok | 1.2x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.009 ms | 7.45 | 0.35 | 11% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.514 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.740 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 526.984 ms | 0.29 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 59.946 ms | 2.80 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.099 ms | 3.13 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.482 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 38.413 ms | 6.55 | 0.03 | 10% | ok | 1.8x (44%) |
| eigen | sylvester | f64 | 256x256 | 317.446 ms | 3.17 | 0.00 | 5% | ok | 1.2x (29%) |
| elementwise | exp | f64 | n=8388608 | 76.502 ms | 0.11 | 1.75 | 7% | ok | 1.2x (29%) |
| elementwise | log | f64 | n=8388608 | 101.108 ms | 0.08 | 1.33 | 5% | ok | 1.1x (26%) |
| elementwise | sin | f64 | n=8388608 | 130.148 ms | 0.06 | 1.03 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 65.947 ms | 0.13 | 2.04 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 104.495 ms | 0.08 | 1.28 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.194 ms | 2.19 | 22.80 | 92% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.987 ms | 1.32 | 13.70 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.115 ms | 6.85 | 14.40 | 58% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.861 ms | 0.27 | 0.97 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.697 ms | 0.23 | 1.91 | 8% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.728 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.682 ms | 0.19 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.470 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 93.495 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 125.966 ms | 0.18 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.207 ms | 0.57 | 0.02 | 1% | ok | 1.1x (27%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 187.084 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.704 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.212 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.634 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.373 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.412 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 155.293 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.014 ms | 0.38 | 1.50 | 6% | ok | 0.9x (23%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.673 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.348 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.996 ms | 0.38 | 1.52 | 6% | ok | 0.9x (22%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.960 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.261 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.568 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.017 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 107.678 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.574 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.308 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14705.103 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.659 ms | 0.44 | 1.76 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 42.084 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 187.753 ms | 1.64 | 0.00 | 2% | ok | 1.2x (29%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.995 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.635 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1589.976 ms | 0.39 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.617 ms | 0.12 | 0.49 | 2% | ok | 1.0x (25%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.561 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2194.938 ms | 0.28 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.332 ms | 0.12 | 0.49 | 2% | ok | 1.0x (25%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.152 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.049 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 51.535 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 27.309 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 32.543 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 31.863 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 58.379 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 60.465 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 32.467 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.785 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 33.360 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 36.552 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 43.130 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.489 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.539 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.879 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.727 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.591 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.971 ms | 1.33 | 14.88 | 60% | ok | 1.1x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.262 ms | 1.16 | 12.97 | 52% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.677 ms | 0.98 | 10.95 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.660 ms | 1.18 | 8.90 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 209.041 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 241.219 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 136.173 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.858 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.191 ms | 2.33 | 9.33 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.308 ms | 21.81 | 0.34 | 32% | ok | 4.0x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.638 ms | 6.36 | 6.76 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.804 ms | 5.98 | 23.94 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.378 ms | 12.17 | 24.34 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.274 ms | 5.12 | 5.12 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.048 ms | 16.00 | 16.00 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.600 ms | 23.14 | 0.27 | n/a | ok | 2.4x (60%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.682 ms | 40.18 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.138 ms | 52.24 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.137 ms | 52.26 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.738 ms | 27.56 | 0.16 | n/a | ok | 4.0x (99%) |

