## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 67.8 GFLOP/s, f32 135.6 GFLOP/s |
| ... hardware, NEON | f64 67.8, f32 135.6 GFLOP/s |
| ... this build, scalar | f64 33.8, f32 33.9 GFLOP/s |
| bandwidth | read 22.9, update 24.7, triad 22.3 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 114% | 0 | 28% |
| blas2 | 10 | 63% | 87% | 0 | 26% |
| blas3 | 10 | 8% | 17% | 0 | 28% |
| factor | 12 | 6% | 37% | 0 | 34% |
| eigen | 8 | 4% | 10% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 56% | 92% | 0 | 28% |
| direct | 19 | 1% | 6% | 0 | 27% |
| ordering | 5 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 61% | 0 | 27% |
| mg | 1 | 36% | 36% | 0 | 24% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 31% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 55% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.577 ms | 3.01 | 24.06 | 97% | ok | 1.0x (25%) |
| blas1 | dot | f32 | n=8388608 | 2.822 ms | 5.95 | 23.78 | 96% | ok | 1.6x (40%) |
| blas1 | axpy | f64 | n=8388608 | 8.030 ms | 2.09 | 25.07 | 101% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.994 ms | 4.20 | 25.20 | 102% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.761 ms | 1.76 | 28.19 | 114% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.645 ms | 6.34 | 25.38 | 103% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.715 ms | 1.09 | 8.70 | 35% | ok | 1.1x (26%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.422 ms | 0.54 | 4.35 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.535 ms | 1.11 | 8.91 | 36% | ok | 1.1x (28%) |
| blas1 | dot_k3 | f64 | n=8388608 | 181.910 ms | 1.01 | 0.74 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.591 ms | 5.09 | 20.38 | 82% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 3.112 ms | 10.78 | 21.57 | 87% | ok | 1.9x (47%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.931 ms | 4.23 | 16.93 | 68% | ok | 1.2x (31%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.503 ms | 6.10 | 12.20 | 49% | ok | 0.8x (20%) |
| blas2 | symv | f64 | n=4096 | 11.628 ms | 2.89 | 5.78 | 23% | ok | 1.0x (25%) |
| blas2 | trmv | f64 | n=4096 | 4.303 ms | 3.90 | 15.61 | 63% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 3.099 ms | 5.41 | 10.84 | 44% | ok | 1.0x (26%) |
| blas2 | trsv | f64 | n=4096 | 4.364 ms | 3.84 | 15.39 | 62% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.861 ms | 5.86 | 11.74 | 47% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 15.307 ms | 2.19 | 17.54 | 71% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 26.735 ms | 10.04 | 0.24 | 15% | ok | 2.0x (49%) |
| blas3 | gemm | f32 | 512x512x512 | 11.386 ms | 23.58 | 0.28 | 17% | ok | 2.1x (53%) |
| blas3 | gemmt | f64 | 512x512x512 | 37.207 ms | 3.61 | 0.17 | 5% | ok | 1.1x (27%) |
| blas3 | symm | f64 | 512x512x512 | 47.929 ms | 5.60 | 0.13 | 8% | ok | 1.1x (28%) |
| blas3 | syrk | f64 | 512x512x512 | 38.496 ms | 3.49 | 0.11 | 5% | ok | 1.1x (27%) |
| blas3 | syr2k | f64 | 512x512x512 | 78.682 ms | 3.41 | 0.08 | 5% | ok | 1.0x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 23.205 ms | 5.78 | 0.23 | 9% | ok | 1.1x (28%) |
| blas3 | trsm | f64 | 512x512x512 | 24.385 ms | 5.50 | 0.22 | 8% | ok | 1.1x (27%) |
| blas3 | trsm | f32 | 512x512x512 | 10.691 ms | 12.55 | 0.25 | 9% | ok | 1.1x (28%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.879 ms | 0.18 | 1.42 | 6% | ok | 1.1x (27%) |
| factor | lu | f64 | 512x512 | 17.917 ms | 4.99 | 0.23 | 7% | ok | 1.3x (34%) |
| factor | lu | f32 | 512x512 | 9.030 ms | 9.91 | 0.23 | 7% | ok | 1.4x (36%) |
| factor | cholesky | f64 | 512x512 | 18.618 ms | 2.40 | 0.23 | 4% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.642 ms | 2.54 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.899 ms | 2.14 | 0.20 | 3% | ok | 1.1x (27%) |
| factor | ldlt_bk | f64 | 512x512 | 10.945 ms | 4.09 | 0.38 | 6% | ok | 1.0x (25%) |
| factor | qr | f64 | 512x512 | 148.434 ms | 1.21 | 0.03 | 2% | ok | 1.7x (42%) |
| factor | lq | f64 | 512x512 | 36.803 ms | 4.86 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 202.719 ms | 2.21 | 0.02 | 3% | ok | 1.7x (43%) |
| factor | householder | f64 | 512x512 | 0.459 ms | 2.29 | 9.14 | 37% | ok | 3.4x (85%) |
| factor | inv | f64 | 512x512 | 133.299 ms | 2.01 | 0.03 | 3% | ok | 1.1x (27%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.199 ms | 7.34 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.901 ms | 0.62 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.199 ms | 1.73 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 528.306 ms | 0.29 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.346 ms | 2.78 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.184 ms | 3.13 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.616 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 38.082 ms | 6.61 | 0.03 | 10% | ok | 1.8x (44%) |
| eigen | sylvester | f64 | 256x256 | 318.779 ms | 3.16 | 0.00 | 5% | ok | 1.2x (29%) |
| elementwise | exp | f64 | n=8388608 | 82.795 ms | 0.10 | 1.62 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 101.575 ms | 0.08 | 1.32 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 136.070 ms | 0.06 | 0.99 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 67.027 ms | 0.13 | 2.00 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 108.346 ms | 0.08 | 1.24 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.201 ms | 2.18 | 22.67 | 92% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.959 ms | 1.34 | 13.90 | 56% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.256 ms | 6.69 | 14.07 | 57% | ok | 1.9x (48%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.537 ms | 0.28 | 1.00 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.221 ms | 0.22 | 1.81 | 7% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 33.991 ms | 0.33 | 0.02 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 54.164 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.388 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 95.121 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 111.566 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.175 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 66.060 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.684 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.108 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 112.148 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.516 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.245 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.005 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 26.786 ms | 0.42 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.240 ms | 0.34 | 1.35 | 5% | ok | 0.8x (21%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.267 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.867 ms | 1.03 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.165 ms | 0.35 | 1.40 | 6% | ok | 0.8x (20%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.999 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 81.069 ms | 0.28 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.700 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.904 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 93.886 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.707 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 10.986 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 618.654 ms | 0.50 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.945 ms | 0.42 | 1.70 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 24.928 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 164.881 ms | 1.87 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.960 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.608 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1579.636 ms | 0.39 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.088 ms | 0.12 | 0.48 | 2% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 14.913 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1138.275 ms | 0.54 | 0.00 | 1% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.563 ms | 0.11 | 0.46 | 2% | ok | 1.0x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.834 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 107.356 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.643 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.360 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.531 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 31.105 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.875 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.591 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.353 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.506 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.621 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.953 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.652 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.579 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.240 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.859 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.784 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.073 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.611 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.953 ms | 1.34 | 15.02 | 61% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.301 ms | 1.14 | 12.75 | 52% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.708 ms | 0.97 | 10.83 | 44% | ok | 1.1x (27%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.646 ms | 1.18 | 8.90 | 36% | ok | 1.0x (24%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 207.643 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 208.603 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 133.598 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.935 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.195 ms | 2.33 | 9.33 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.757 ms | 21.04 | 0.33 | 31% | ok | 3.8x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.636 ms | 6.36 | 6.76 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.827 ms | 5.93 | 23.73 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.385 ms | 12.11 | 24.23 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.275 ms | 5.12 | 5.12 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.046 ms | 16.03 | 16.03 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.690 ms | 22.96 | 0.27 | n/a | ok | 2.2x (55%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.675 ms | 40.21 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.138 ms | 52.25 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.138 ms | 52.24 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.736 ms | 27.57 | 0.16 | n/a | ok | 4.0x (99%) |

