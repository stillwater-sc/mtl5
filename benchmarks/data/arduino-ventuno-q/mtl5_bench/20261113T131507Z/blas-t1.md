## arduino-ventuno-q -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, scalar | f64 8.4, f32 8.4 GFLOP/s |
| bandwidth | read 22.2, update 24.2, triad 22.7 GB/s (ceiling 24.2) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 85% | 106% | 0 |
| blas2 | 10 | 60% | 74% | 0 |
| blas3 | 10 | 83% | 92% | 0 |
| factor | 12 | 41% | 58% | 0 |
| eigen | 8 | 34% | 74% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 67% | 0 |
| direct | 23 | 2% | 11% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 15 | 7% | 41% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 41% | 57% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 35% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.683 ms | 2.95 | 23.62 | 98% | ok |
| blas1 | dot | f32 | n=8388608 | 4.511 ms | 3.72 | 14.88 | 61% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.862 ms | 2.13 | 25.61 | 106% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.987 ms | 4.21 | 25.25 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 5.480 ms | 1.53 | 24.49 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.253 ms | 5.16 | 20.63 | 85% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.662 ms | 0.97 | 7.75 | 32% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.609 ms | 0.48 | 3.81 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.117 ms | 1.03 | 8.27 | 34% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 187.560 ms | 0.98 | 0.72 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 9.037 ms | 3.71 | 14.86 | 61% | ok |
| blas2 | gemv | f32 | n=4096 | 3.856 ms | 8.70 | 17.41 | 72% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.275 ms | 3.62 | 14.48 | 60% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.733 ms | 8.99 | 17.99 | 74% | ok |
| blas2 | symv | f64 | n=4096 | 11.580 ms | 2.90 | 5.80 | 24% | ok |
| blas2 | trmv | f64 | n=4096 | 6.276 ms | 2.67 | 10.70 | 44% | ok |
| blas2 | trmv | f32 | n=4096 | 3.282 ms | 5.11 | 10.23 | 42% | ok |
| blas2 | trsv | f64 | n=4096 | 6.391 ms | 2.63 | 10.51 | 43% | ok |
| blas2 | trsv | f32 | n=4096 | 3.230 ms | 5.19 | 10.40 | 43% | ok |
| blas2 | ger | f64 | n=4096 | 15.349 ms | 2.19 | 17.49 | 72% | ok |
| blas3 | gemm | f64 | 512x512x512 | 17.798 ms | 15.08 | 0.35 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.654 ms | 31.02 | 0.36 | 92% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.401 ms | 11.77 | 0.55 | 70% | ok |
| blas3 | symm | f64 | 512x512x512 | 18.111 ms | 14.82 | 0.35 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.842 ms | 13.64 | 0.43 | 81% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 19.251 ms | 13.94 | 0.33 | 83% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.089 ms | 14.77 | 0.58 | 88% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.747 ms | 13.77 | 0.54 | 82% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.019 ms | 26.74 | 0.52 | 80% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 25.886 ms | 0.21 | 1.64 | 7% | ok |
| factor | lu | f64 | 512x512 | 9.162 ms | 9.77 | 0.46 | 58% | ok |
| factor | lu | f32 | 512x512 | 5.538 ms | 16.16 | 0.38 | 48% | ok |
| factor | cholesky | f64 | 512x512 | 5.409 ms | 8.27 | 0.78 | 49% | ok |
| factor | cholesky | f32 | 512x512 | 3.244 ms | 13.79 | 0.65 | 41% | ok |
| factor | ldlt | f64 | 512x512 | 22.645 ms | 1.98 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 69.422 ms | 0.64 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 19.899 ms | 8.99 | 0.21 | 54% | ok |
| factor | lq | f64 | 512x512 | 60.503 ms | 2.96 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 350.045 ms | 1.28 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.563 ms | 0.67 | 2.68 | 11% | ok |
| factor | inv | f64 | 512x512 | 131.041 ms | 2.05 | 0.03 | 12% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 10.300 ms | 8.69 | 0.41 | 52% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.370 ms | 3.04 | 0.14 | 18% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 96.991 ms | 1.56 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 593.306 ms | 0.25 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 45.104 ms | 3.72 | 0.02 | 22% | ok |
| eigen | schur | f64 | 256x256 | 63.626 ms | 6.59 | 0.02 | 39% | ok |
| eigen | svd | f64 | 256x256 | 29.496 ms | 12.51 | 0.04 | 74% | ok |
| eigen | expm | f64 | 256x256 | 38.029 ms | 6.62 | 0.03 | 39% | ok |
| eigen | sylvester | f64 | 256x256 | 176.078 ms | 5.72 | 0.01 | 34% | ok |
| elementwise | exp | f64 | n=8388608 | 89.263 ms | 0.09 | 1.50 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 103.467 ms | 0.08 | 1.30 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 148.006 ms | 0.06 | 0.91 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 69.891 ms | 0.12 | 1.92 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 113.324 ms | 0.07 | 1.18 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.675 ms | 1.56 | 16.26 | 67% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.159 ms | 1.21 | 12.61 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.035 ms | 3.48 | 7.32 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.862 ms | 0.25 | 0.89 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.812 ms | 0.23 | 1.89 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 181.825 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 57.716 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.437 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 105.182 ms | 0.21 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 143.052 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 44.241 ms | 0.51 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 200.412 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.781 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.321 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.695 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.529 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.871 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 173.801 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.787 ms | 0.42 | 1.69 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.696 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.420 ms | 0.98 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.707 ms | 0.44 | 1.77 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.434 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 89.319 ms | 0.25 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.949 ms | 0.13 | 0.51 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.997 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 121.561 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.818 ms | 0.13 | 0.52 | 2% | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 16.663 ms | 0.67 | 0.04 | 4% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.888 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 11.162 ms | 1.00 | 0.07 | 6% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.163 ms | 0.65 | 2.60 | 11% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.448 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16466.380 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.721 ms | 0.44 | 1.75 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 46.209 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 204.997 ms | 1.51 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.545 ms | 0.45 | 1.79 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.820 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1779.822 ms | 0.35 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.306 ms | 0.11 | 0.46 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.055 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2457.092 ms | 0.25 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.600 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 7.481 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 45.079 ms | 6.85 | 0.02 | 41% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.411 ms | 1.40 | 5.59 | 23% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 71.014 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.175 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.866 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.932 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.542 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.970 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.892 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.649 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.541 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.520 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.956 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 40.111 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.485 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.113 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.063 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.314 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.131 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.026 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.968 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.133 ms | 1.23 | 13.75 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.727 ms | 0.55 | 6.20 | 26% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.969 ms | 0.88 | 9.88 | 41% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.703 ms | 1.18 | 8.89 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 229.827 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 235.554 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 144.709 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.342 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.027 ms | 2.09 | 8.36 | 35% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.624 ms | 5.52 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.938 ms | 5.71 | 6.07 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.845 ms | 5.90 | 23.59 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.392 ms | 12.06 | 24.11 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.652 ms | 4.59 | 4.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.166 ms | 14.39 | 14.39 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.641 ms | 10.89 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.386 ms | 10.17 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.300 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.302 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.385 ms | 6.99 | 0.04 | n/a | ok |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

