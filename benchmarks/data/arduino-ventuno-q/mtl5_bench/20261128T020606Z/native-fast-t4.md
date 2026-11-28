## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 135.4 GFLOP/s |
| ... hardware, NEON | f64 67.5, f32 135.4 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.5, f32 135.0 GFLOP/s |
| bandwidth | read 23.0, update 25.1, triad 22.8 GB/s (ceiling 25.1) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 95% | 111% | 0 | 27% |
| blas2 | 10 | 57% | 85% | 0 | 26% |
| blas3 | 10 | 38% | 56% | 0 | 78% |
| factor | 12 | 7% | 37% | 0 | 35% |
| eigen | 8 | 4% | 13% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 27% |
| sparse | 5 | 56% | 90% | 0 | 28% |
| direct | 19 | 1% | 6% | 0 | 28% |
| ordering | 5 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 50% | 61% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 24% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 74% | 97% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 50% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.605 ms | 2.99 | 23.95 | 96% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.814 ms | 5.96 | 23.85 | 95% | ok | 1.0x (26%) |
| blas1 | axpy | f64 | n=8388608 | 8.152 ms | 2.06 | 24.70 | 99% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.209 ms | 3.99 | 23.92 | 95% | ok | 0.9x (24%) |
| blas1 | scale | f64 | n=8388608 | 4.831 ms | 1.74 | 27.78 | 111% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.654 ms | 6.32 | 25.28 | 101% | ok | 1.2x (29%) |
| blas1 | one_norm | f64 | n=8388608 | 7.868 ms | 1.07 | 8.53 | 34% | ok | 1.1x (27%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.683 ms | 0.53 | 4.28 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.661 ms | 1.09 | 8.76 | 35% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 181.876 ms | 1.01 | 0.74 | 3% | ok | 1.1x (27%) |
| blas2 | gemv | f64 | n=4096 | 6.804 ms | 4.93 | 19.74 | 79% | ok | 1.2x (31%) |
| blas2 | gemv | f32 | n=4096 | 3.136 ms | 10.70 | 21.41 | 85% | ok | 1.3x (32%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.679 ms | 4.37 | 17.49 | 70% | ok | 1.0x (26%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.939 ms | 6.79 | 13.59 | 54% | ok | 0.8x (20%) |
| blas2 | symv | f64 | n=4096 | 11.346 ms | 2.96 | 5.92 | 24% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.709 ms | 3.56 | 14.27 | 57% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 2.654 ms | 6.32 | 12.65 | 51% | ok | 1.0x (26%) |
| blas2 | trsv | f64 | n=4096 | 4.845 ms | 3.46 | 13.86 | 55% | ok | 1.0x (26%) |
| blas2 | trsv | f32 | n=4096 | 2.596 ms | 6.46 | 12.94 | 52% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 14.694 ms | 2.28 | 18.27 | 73% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.302 ms | 36.76 | 0.86 | 54% | ok | 3.9x (97%) |
| blas3 | gemm | f32 | 512x512x512 | 3.568 ms | 75.23 | 0.88 | 56% | ok | 3.9x (96%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.857 ms | 22.91 | 1.07 | 34% | ok | 3.1x (77%) |
| blas3 | symm | f64 | 512x512x512 | 8.157 ms | 32.91 | 0.77 | 49% | ok | 3.6x (90%) |
| blas3 | syrk | f64 | 512x512x512 | 5.281 ms | 25.42 | 0.79 | 38% | ok | 3.1x (78%) |
| blas3 | syr2k | f64 | 512x512x512 | 10.425 ms | 25.75 | 0.60 | 38% | ok | 3.1x (78%) |
| blas3 | trmm | f64 | 512x512x512 | 7.493 ms | 17.91 | 0.70 | 27% | ok | 2.4x (59%) |
| blas3 | trsm | f64 | 512x512x512 | 7.726 ms | 17.37 | 0.68 | 26% | ok | 2.3x (58%) |
| blas3 | trsm | f32 | 512x512x512 | 3.567 ms | 37.63 | 0.73 | 28% | ok | 2.4x (59%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.916 ms | 0.18 | 1.42 | 6% | ok | 1.1x (27%) |
| factor | lu | f64 | 512x512 | 17.413 ms | 5.14 | 0.24 | 8% | ok | 1.4x (35%) |
| factor | lu | f32 | 512x512 | 9.039 ms | 9.90 | 0.23 | 7% | ok | 1.4x (36%) |
| factor | cholesky | f64 | 512x512 | 19.865 ms | 2.25 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 18.816 ms | 2.38 | 0.11 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 18.844 ms | 2.37 | 0.22 | 4% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 10.085 ms | 4.44 | 0.42 | 7% | ok | 1.1x (27%) |
| factor | qr | f64 | 512x512 | 145.985 ms | 1.23 | 0.03 | 2% | ok | 1.7x (41%) |
| factor | lq | f64 | 512x512 | 36.818 ms | 4.86 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 200.275 ms | 2.23 | 0.02 | 3% | ok | 1.7x (42%) |
| factor | householder | f64 | 512x512 | 0.456 ms | 2.30 | 9.21 | 37% | ok | 3.3x (83%) |
| factor | inv | f64 | 512x512 | 131.046 ms | 2.05 | 0.03 | 3% | ok | 1.1x (28%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.072 ms | 7.41 | 0.35 | 11% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.693 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.179 ms | 1.73 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 535.455 ms | 0.28 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.542 ms | 2.77 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 135.481 ms | 3.10 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.444 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 28.587 ms | 8.80 | 0.04 | 13% | ok | 1.7x (42%) |
| eigen | sylvester | f64 | 256x256 | 316.738 ms | 3.18 | 0.00 | 5% | ok | 1.1x (29%) |
| elementwise | exp | f64 | n=8388608 | 82.344 ms | 0.10 | 1.63 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 101.929 ms | 0.08 | 1.32 | 5% | ok | 1.1x (27%) |
| elementwise | sin | f64 | n=8388608 | 136.118 ms | 0.06 | 0.99 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 67.861 ms | 0.12 | 1.98 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 108.246 ms | 0.08 | 1.24 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.206 ms | 2.17 | 22.57 | 90% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.943 ms | 1.35 | 14.01 | 56% | ok | 1.1x (28%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.110 ms | 6.85 | 14.41 | 58% | ok | 2.0x (50%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.974 ms | 0.27 | 0.96 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.921 ms | 0.22 | 1.87 | 7% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 34.447 ms | 0.32 | 0.02 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 52.971 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.351 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 93.970 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 113.003 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.369 ms | 0.58 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 70.740 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.728 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.471 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 111.829 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.514 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.404 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.079 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 27.034 ms | 0.41 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.246 ms | 0.34 | 1.35 | 5% | ok | 0.8x (20%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.076 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.803 ms | 1.04 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.127 ms | 0.36 | 1.42 | 6% | ok | 0.8x (20%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.941 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 80.311 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.566 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.899 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 95.418 ms | 0.23 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.570 ms | 0.14 | 0.54 | 2% | ok | 1.1x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 11.221 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 622.025 ms | 0.50 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.707 ms | 0.44 | 1.75 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 24.896 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 168.355 ms | 1.83 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.666 ms | 0.44 | 1.76 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.772 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1594.991 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.269 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 15.143 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1102.450 ms | 0.56 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.041 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.782 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 108.043 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.758 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.346 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.555 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.779 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.327 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.357 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.403 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.345 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.476 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.830 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.691 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.101 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.585 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.811 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.837 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.735 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.584 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.931 ms | 1.36 | 15.19 | 61% | ok | 1.1x (28%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.320 ms | 1.13 | 12.64 | 50% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.692 ms | 0.97 | 10.89 | 43% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.627 ms | 1.18 | 8.91 | 36% | ok | 1.0x (24%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 208.930 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 208.217 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 129.434 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.815 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.643 ms | 4.60 | 18.42 | 74% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 10.499 ms | 25.57 | 0.40 | 38% | ok | 3.9x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.737 ms | 22.78 | 24.20 | 97% | ok | 1.0x (26%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.796 ms | 6.00 | 24.00 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.384 ms | 12.12 | 24.24 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.681 ms | 24.64 | 24.64 | n/a | ok | 1.0x (26%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.361 ms | 12.33 | 12.33 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.460 ms | 60.19 | 0.71 | n/a | ok | 3.9x (97%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.253 ms | 63.12 | 0.49 | n/a | ok | 3.9x (98%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.192 ms | 64.04 | 0.38 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.824 ms | 147.20 | 0.86 | n/a | ok | 2.0x (50%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.338 ms | 80.42 | 0.47 | n/a | ok | 4.0x (99%) |

