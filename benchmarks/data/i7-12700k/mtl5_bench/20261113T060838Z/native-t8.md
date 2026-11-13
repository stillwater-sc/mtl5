## i7-12700k -- native, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 600.8 GFLOP/s, f32 1201.8 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.8, f32 1201.8 GFLOP/s |
| ... this build, scalar | f64 9.8, f32 10.5 GFLOP/s |
| bandwidth | read 83.8, update 78.7, triad 76.9 GB/s (ceiling 83.8) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 84% | 123% | 0 | 28% |
| blas2 | 10 | 42% | 137% | 0 | 13% |
| blas3 | 10 | 2% | 16% | 0 | 15% |
| factor | 12 | 1% | 12% | 0 | 19% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 280% | 0 | 13% |
| direct | 19 | 0% | 13% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 10% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 36% | 0 | 13% |
| mg | 1 | 25% | 25% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 92% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.618 ms | 10.37 | 82.95 | 99% | ok | 3.4x (43%) |
| blas1 | dot | f32 | n=8388608 | 0.653 ms | 25.71 | 102.85 | 123% | ok | 6.4x (81%) |
| blas1 | axpy | f64 | n=8388608 | 2.853 ms | 5.88 | 70.57 | 84% | ok | 2.2x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.452 ms | 11.56 | 69.34 | 83% | ok | 3.2x (40%) |
| blas1 | scale | f64 | n=8388608 | 1.912 ms | 4.39 | 70.18 | 84% | ok | 1.9x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.740 ms | 22.66 | 90.64 | 108% | ok | 6.5x (81%) |
| blas1 | one_norm | f64 | n=8388608 | 3.920 ms | 2.14 | 17.12 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.903 ms | 1.22 | 9.72 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.891 ms | 2.16 | 17.25 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.065 ms | 2.46 | 1.79 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.450 ms | 23.13 | 92.58 | 110% | ok | 2.5x (31%) |
| blas2 | gemv | f32 | n=4096 | 0.583 ms | 57.57 | 115.20 | 137% | ok | 4.7x (59%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.573 ms | 21.33 | 85.35 | 102% | ok | 3.0x (37%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.868 ms | 38.66 | 77.36 | 92% | ok | 2.5x (32%) |
| blas2 | symv | f64 | n=4096 | 6.465 ms | 5.19 | 10.39 | 12% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.992 ms | 8.42 | 33.72 | 40% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.397 ms | 12.01 | 24.04 | 29% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.953 ms | 8.59 | 34.39 | 41% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.361 ms | 12.32 | 24.67 | 29% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.623 ms | 4.40 | 35.22 | 42% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.733 ms | 98.23 | 2.30 | 16% | ok | 7.3x (91%) |
| blas3 | gemm | f32 | 512x512x512 | 1.529 ms | 175.53 | 2.06 | 15% | ok | 6.9x (87%) |
| blas3 | gemmt | f64 | 512x512x512 | 9.014 ms | 14.89 | 0.70 | 2% | ok | 1.2x (15%) |
| blas3 | symm | f64 | 512x512x512 | 16.644 ms | 16.13 | 0.38 | 3% | ok | 1.1x (14%) |
| blas3 | syrk | f64 | 512x512x512 | 11.603 ms | 11.57 | 0.36 | 2% | ok | 1.2x (15%) |
| blas3 | syr2k | f64 | 512x512x512 | 19.509 ms | 13.76 | 0.32 | 2% | ok | 1.3x (16%) |
| blas3 | trmm | f64 | 512x512x512 | 9.117 ms | 14.72 | 0.58 | 2% | ok | 1.1x (14%) |
| blas3 | trsm | f64 | 512x512x512 | 9.270 ms | 14.48 | 0.57 | 2% | ok | 1.1x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 5.050 ms | 26.58 | 0.52 | 2% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.825 ms | 0.41 | 3.31 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 7.199 ms | 12.43 | 0.58 | 2% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 3.998 ms | 22.38 | 0.52 | 2% | ok | 1.6x (20%) |
| factor | cholesky | f64 | 512x512 | 7.058 ms | 6.34 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.446 ms | 6.94 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.180 ms | 6.23 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 45.293 ms | 0.99 | 0.09 | 0% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 128.122 ms | 1.40 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 17.425 ms | 10.27 | 0.24 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 165.708 ms | 2.70 | 0.03 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 0.434 ms | 2.42 | 9.66 | 12%\* | ok | 3.8x (47%) |
| factor | inv | f64 | 512x512 | 51.055 ms | 5.26 | 0.08 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.165 ms | 14.51 | 0.68 | 2% | ok | 1.0x (13%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.932 ms | 1.25 | 0.06 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.476 ms | 2.63 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 743.840 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 31.619 ms | 5.31 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 81.222 ms | 5.16 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 109.493 ms | 3.37 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 11.631 ms | 21.64 | 0.09 | 4% | ok | 2.3x (28%) |
| eigen | sylvester | f64 | 256x256 | 175.079 ms | 5.75 | 0.01 | 1% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.436 ms | 0.22 | 3.49 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 44.713 ms | 0.19 | 3.00 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.771 ms | 0.14 | 2.28 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.255 ms | 0.30 | 4.75 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 199.387 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.116 ms | 22.58 | 234.90 | 280% ! | ok | 6.4x (80%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.898 ms | 2.92 | 30.34 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.316 ms | 31.82 | 66.90 | 80% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.768 ms | 0.68 | 2.41 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.380 ms | 1.45 | 12.06 | 14%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 89.346 ms | 0.13 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.028 ms | 0.45 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.880 ms | 1.26 | 0.08 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.399 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 52.301 ms | 0.43 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.766 ms | 1.33 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 85.428 ms | 0.02 | 0.00 | 0% | ok | 1.2x (15%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.262 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.863 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.220 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.714 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.279 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 86.680 ms | 0.13 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.268 ms | 2.82 | 11.29 | 13%\* | ok | 0.9x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.360 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.137 ms | 2.70 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.272 ms | 2.78 | 11.13 | 13%\* | ok | 0.9x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.123 ms | 0.64 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.599 ms | 1.26 | 5.05 | 6%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.783 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 46.658 ms | 0.48 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.605 ms | 1.25 | 5.00 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.248 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7281.000 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.533 ms | 2.20 | 8.79 | 10%\* | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.283 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 66.073 ms | 4.67 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.555 ms | 2.17 | 8.67 | 10%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.754 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 542.204 ms | 1.14 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.970 ms | 0.42 | 1.69 | 2%\* | ok | 0.9x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.332 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 655.625 ms | 0.94 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.013 ms | 0.42 | 1.68 | 2%\* | ok | 0.9x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.454 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.020 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.392 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.201 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.605 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.677 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.683 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.909 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.766 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.457 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 20.696 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.530 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 12.087 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.987 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.020 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.699 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.963 ms | 2.72 | 30.44 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.197 ms | 2.19 | 24.50 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.533 ms | 1.71 | 19.13 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.759 ms | 2.79 | 21.12 | 25% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.497 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 44.271 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.634 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.901 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.786 ms | 3.51 | 14.02 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.187 ms | 51.75 | 0.81 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.534 ms | 31.40 | 33.37 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.252 ms | 7.45 | 29.80 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.977 ms | 17.17 | 34.35 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.684 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.379 ms | 112.85 | 1.32 | n/a | ok | 7.3x (92%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.749 ms | 97.65 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.95 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.96 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.607 ms | 74.41 | 0.44 | n/a | ok | 7.5x (94%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

