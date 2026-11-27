## i7-12700k -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 600.6 GFLOP/s, f32 1201.4 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.4, f32 1201.2 GFLOP/s |
| ... this build, AVX2 | f64 600.6, f32 1201.4 GFLOP/s |
| bandwidth | read 81.6, update 78.7, triad 76.9 GB/s (ceiling 81.6) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 93% | 150% | 0 | 23% |
| blas2 | 10 | 49% | 149% | 0 | 13% |
| blas3 | 10 | 18% | 66% | 0 | 27% |
| factor | 12 | 2% | 13% | 0 | 14% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 304% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 10% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 28% | 37% | 0 | 13% |
| mg | 1 | 24% | 24% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 49% | 71% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 76% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.400 ms | 11.98 | 95.88 | 117% | ok | 2.6x (33%) |
| blas1 | dot | f32 | n=8388608 | 0.547 ms | 30.69 | 122.78 | 150% ! | ok | 3.1x (39%) |
| blas1 | axpy | f64 | n=8388608 | 2.664 ms | 6.30 | 75.57 | 93% | ok | 1.8x (22%) |
| blas1 | axpy | f32 | n=8388608 | 1.250 ms | 13.42 | 80.50 | 99% | ok | 1.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 1.763 ms | 4.76 | 76.13 | 93% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.563 ms | 29.79 | 119.17 | 146% | ok | 3.9x (49%) |
| blas1 | one_norm | f64 | n=8388608 | 3.962 ms | 2.12 | 16.94 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.951 ms | 1.21 | 9.65 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.926 ms | 2.14 | 17.09 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 69.893 ms | 2.64 | 1.92 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.403 ms | 23.92 | 95.74 | 117% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.552 ms | 60.74 | 121.54 | 149% | ok | 3.1x (38%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.441 ms | 23.28 | 93.18 | 114% | ok | 2.4x (30%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.674 ms | 49.80 | 99.66 | 122% | ok | 2.5x (31%) |
| blas2 | symv | f64 | n=4096 | 2.868 ms | 11.70 | 23.42 | 29% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.761 ms | 9.53 | 38.15 | 47% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.841 ms | 19.95 | 39.94 | 49% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.793 ms | 9.36 | 37.46 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.855 ms | 19.62 | 39.28 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.122 ms | 4.71 | 37.70 | 46% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.674 ms | 398.10 | 9.33 | 66% | ok | 5.6x (70%) |
| blas3 | gemm | f32 | 512x512x512 | 0.350 ms | 767.30 | 8.99 | 64% | ok | 5.4x (68%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.231 ms | 109.03 | 5.11 | 18% | ok | 2.1x (27%) |
| blas3 | symm | f64 | 512x512x512 | 0.917 ms | 292.73 | 6.86 | 49% | ok | 4.4x (55%) |
| blas3 | syrk | f64 | 512x512x512 | 1.277 ms | 105.12 | 3.29 | 18% | ok | 2.0x (25%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.305 ms | 116.45 | 2.73 | 19% | ok | 2.2x (27%) |
| blas3 | trmm | f64 | 512x512x512 | 1.800 ms | 74.55 | 2.91 | 12% | ok | 1.5x (19%) |
| blas3 | trsm | f64 | 512x512x512 | 1.854 ms | 72.40 | 2.83 | 12% | ok | 1.5x (19%) |
| blas3 | trsm | f32 | 512x512x512 | 0.945 ms | 141.98 | 2.77 | 12% | ok | 1.5x (19%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.941 ms | 0.41 | 3.28 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.309 ms | 14.18 | 0.66 | 2% | ok | 1.7x (22%) |
| factor | lu | f32 | 512x512 | 2.858 ms | 31.31 | 0.73 | 3% | ok | 1.2x (14%) |
| factor | cholesky | f64 | 512x512 | 6.888 ms | 6.50 | 0.61 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.391 ms | 7.00 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.010 ms | 6.38 | 0.60 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.311 ms | 10.38 | 0.97 | 2% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 130.161 ms | 1.37 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 14.479 ms | 12.36 | 0.29 | 2% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 149.999 ms | 2.98 | 0.03 | 0% | ok | 1.7x (22%) |
| factor | householder | f64 | 512x512 | 0.402 ms | 2.61 | 10.42 | 13%\* | ok | 4.1x (51%) |
| factor | inv | f64 | 512x512 | 49.781 ms | 5.39 | 0.08 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.242 ms | 21.09 | 0.99 | 4% | ok | 1.2x (14%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.139 ms | 1.31 | 0.06 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 56.511 ms | 2.67 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 735.961 ms | 0.21 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 30.362 ms | 5.53 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 77.946 ms | 5.38 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 90.615 ms | 4.07 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 9.346 ms | 26.93 | 0.11 | 4% | ok | 1.2x (16%) |
| eigen | sylvester | f64 | 256x256 | 170.888 ms | 5.89 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 37.301 ms | 0.22 | 3.60 | 4% | ok | 1.1x (13%) |
| elementwise | log | f64 | n=8388608 | 44.831 ms | 0.19 | 2.99 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 59.702 ms | 0.14 | 2.25 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.935 ms | 0.29 | 4.64 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 201.190 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.110 ms | 23.85 | 248.15 | 304% ! | ok | 6.7x (84%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.921 ms | 2.84 | 29.55 | 36% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.307 ms | 32.03 | 67.35 | 82% | ok | 3.3x (41%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.853 ms | 0.67 | 2.37 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.537 ms | 1.31 | 10.83 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 15.008 ms | 0.75 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 26.577 ms | 0.42 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.915 ms | 1.25 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.088 ms | 0.56 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.025 ms | 0.51 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.567 ms | 1.35 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.636 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.357 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.907 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.266 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.397 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.073 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.429 ms | 0.90 | 0.06 | 0% | ok | 1.1x (14%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.257 ms | 2.95 | 11.79 | 14%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.465 ms | 2.51 | 0.16 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.279 ms | 2.71 | 10.83 | 13%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.708 ms | 0.63 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.628 ms | 1.20 | 4.82 | 6%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.391 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.359 ms | 0.58 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.623 ms | 1.21 | 4.85 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.863 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 198.022 ms | 1.56 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.627 ms | 2.07 | 8.28 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.434 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 65.950 ms | 4.68 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.829 ms | 1.84 | 7.37 | 9%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.823 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 513.208 ms | 1.20 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.801 ms | 0.43 | 1.73 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.303 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 697.752 ms | 0.88 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.696 ms | 0.44 | 1.75 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.131 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 40.511 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.891 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.408 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.682 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.475 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 19.844 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.660 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.384 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.108 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 13.714 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 12.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 14.358 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.543 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.854 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.062 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.006 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.708 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.965 ms | 2.71 | 30.38 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.287 ms | 2.03 | 22.78 | 28% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.484 ms | 1.76 | 19.76 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.669 ms | 2.62 | 19.81 | 24% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.899 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 28.342 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 32.714 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.513 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.880 ms | 8.93 | 35.70 | 44% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 0.911 ms | 294.58 | 4.60 | 49% | ok | 6.6x (82%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.308 ms | 54.54 | 57.95 | 71%\* | ok | 1.1x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.890 ms | 8.88 | 35.50 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.743 ms | 22.58 | 45.16 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.276 ms | 60.76 | 60.76 | n/a | ok | 1.1x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.263 ms | 63.89 | 63.89 | n/a | ok | 1.1x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 0.627 ms | 428.31 | 5.02 | n/a | ok | 6.3x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 0.826 ms | 324.92 | 2.54 | n/a | ok | 6.8x (84%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 0.806 ms | 332.98 | 1.95 | n/a | ok | 6.9x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.337 ms | 797.37 | 4.67 | n/a | ok | 6.2x (78%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.308 ms | 870.24 | 5.10 | n/a | ok | 6.1x (76%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

