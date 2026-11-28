## xeon-e5-2420v2 -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.1 GFLOP/s, f32 118.3 GFLOP/s |
| ... hardware, SSE2 | f64 59.1, f32 118.3 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.6, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 85% |
| blas2 | 10 | 101% | 108% | 0 | 19% |
| blas3 | 10 | 125% | 157% | 0 | 75% |
| factor | 12 | 17% | 40% | 0 | 25% |
| eigen | 8 | 9% | 23% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 104% | 0 | 17% |
| direct | 19 | 1% | 30% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 73% | 78% | 0 | 17% |
| mg | 1 | 72% | 72% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 16% |
| mixed | 3 | 22% | 69% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 83% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 33.406 ms | 0.50 | 4.02 | 35% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=8388608 | 5.510 ms | 3.04 | 12.18 | 107% | ok | 5.1x (85%) |
| blas1 | axpy | f64 | n=8388608 | 38.806 ms | 0.43 | 5.19 | 45% | ok | 5.5x (91%) |
| blas1 | axpy | f32 | n=8388608 | 16.768 ms | 1.00 | 6.00 | 53% | ok | 5.5x (91%) |
| blas1 | scale | f64 | n=8388608 | 12.084 ms | 0.69 | 11.11 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 33.018 ms | 0.51 | 2.03 | 18% | ok | 5.6x (93%) |
| blas1 | one_norm | f64 | n=8388608 | 9.757 ms | 0.86 | 6.88 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.072 ms | 0.64 | 5.13 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.698 ms | 0.86 | 6.92 | 61% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 278.755 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 11.310 ms | 2.97 | 11.87 | 104% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.462 ms | 6.14 | 12.29 | 108% | ok | 1.2x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 11.218 ms | 2.99 | 11.97 | 105% | ok | 1.2x (19%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 7.521 ms | 4.46 | 8.93 | 78% | ok | 0.9x (15%) |
| blas2 | symv | f64 | n=4096 | 5.796 ms | 5.79 | 11.59 | 101% | ok | 1.2x (20%) |
| blas2 | trmv | f64 | n=4096 | 5.733 ms | 2.93 | 11.72 | 103% | ok | 1.3x (22%) |
| blas2 | trmv | f32 | n=4096 | 2.954 ms | 5.68 | 11.37 | 99% | ok | 1.2x (21%) |
| blas2 | trsv | f64 | n=4096 | 7.481 ms | 2.24 | 8.98 | 79% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.548 ms | 4.73 | 9.47 | 83% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 25.917 ms | 1.29 | 10.36 | 91% | ok | 1.1x (19%) |
| blas3 | gemm | f64 | 512x512x512 | 2.897 ms | 92.66 | 2.17 | 157% | ok | 4.8x (81%) |
| blas3 | gemm | f32 | 512x512x512 | 1.818 ms | 147.61 | 1.73 | 125% | ok | 4.5x (75%) |
| blas3 | gemmt | f64 | 512x512x512 | 2.583 ms | 51.97 | 2.44 | 88% | ok | 3.4x (57%) |
| blas3 | symm | f64 | 512x512x512 | 2.929 ms | 91.66 | 2.15 | 155% | ok | 4.8x (81%) |
| blas3 | syrk | f64 | 512x512x512 | 2.530 ms | 53.04 | 1.66 | 90% | ok | 3.1x (52%) |
| blas3 | syr2k | f64 | 512x512x512 | 3.724 ms | 72.08 | 1.69 | 122% | ok | 4.1x (68%) |
| blas3 | trmm | f64 | 512x512x512 | 1.618 ms | 82.94 | 3.24 | 140% | ok | 4.6x (77%) |
| blas3 | trsm | f64 | 512x512x512 | 1.765 ms | 76.06 | 2.97 | 129% | ok | 4.6x (77%) |
| blas3 | trsm | f32 | 512x512x512 | 1.245 ms | 107.80 | 2.11 | 91% | ok | 4.2x (70%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.847 ms | 0.14 | 1.09 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.595 ms | 15.99 | 0.75 | 27% | ok | 1.7x (28%) |
| factor | lu | f32 | 512x512 | 4.111 ms | 21.77 | 0.51 | 18% | ok | 1.6x (26%) |
| factor | cholesky | f64 | 512x512 | 4.354 ms | 10.27 | 0.96 | 17% | ok | 1.3x (22%) |
| factor | cholesky | f32 | 512x512 | 3.106 ms | 14.40 | 0.68 | 12% | ok | 1.3x (21%) |
| factor | ldlt | f64 | 512x512 | 26.513 ms | 1.69 | 0.16 | 3% | ok | 1.0x (16%) |
| factor | ldlt_bk | f64 | 512x512 | 14.209 ms | 3.15 | 0.30 | 5% | ok | 1.0x (16%) |
| factor | qr | f64 | 512x512 | 13.250 ms | 13.51 | 0.32 | 23% | ok | 1.5x (25%) |
| factor | lq | f64 | 512x512 | 48.679 ms | 3.68 | 0.09 | 6% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 401.665 ms | 1.11 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.912 ms | 1.15 | 4.60 | 40%\* | ok | 3.5x (58%) |
| factor | inv | f64 | 512x512 | 132.722 ms | 2.02 | 0.03 | 3% | ok | 1.0x (16%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.703 ms | 10.28 | 0.48 | 17% | ok | 1.5x (24%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.890 ms | 3.80 | 0.18 | 6% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 171.284 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 982.781 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 56.911 ms | 2.95 | 0.02 | 5% | ok | 0.9x (16%) |
| eigen | schur | f64 | 256x256 | 76.159 ms | 5.51 | 0.01 | 9% | ok | 1.0x (16%) |
| eigen | svd | f64 | 256x256 | 26.850 ms | 13.75 | 0.04 | 23% | ok | 1.1x (18%) |
| eigen | expm | f64 | 256x256 | 27.410 ms | 9.18 | 0.04 | 16% | ok | 1.3x (22%) |
| eigen | sylvester | f64 | 256x256 | 194.627 ms | 5.17 | 0.01 | 9% | ok | 1.0x (16%) |
| elementwise | exp | f64 | n=8388608 | 152.118 ms | 0.06 | 0.88 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 140.269 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 173.758 ms | 0.05 | 0.77 | 7% | ok | 1.0x (16%) |
| elementwise | sqrt | f64 | n=8388608 | 90.046 ms | 0.09 | 1.49 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2499.109 ms | 0.00 | 0.05 | 0% | ok | 1.0x (16%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.289 ms | 1.14 | 11.90 | 104% | ok | 1.0x (17%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.112 ms | 0.84 | 8.75 | 77% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.318 ms | 3.70 | 7.78 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.265 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.934 ms | 0.34 | 2.81 | 25% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 41.171 ms | 0.27 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.138 ms | 0.16 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.072 ms | 0.45 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 121.649 ms | 0.18 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.078 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.544 ms | 0.47 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 82.030 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.875 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.131 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.450 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.571 ms | 0.00 | 0.00 | n/a | ok | 0.9x (15%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.141 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.168 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.425 ms | 0.33 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.876 ms | 0.86 | 3.45 | 30%\* | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.345 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.684 ms | 0.88 | 0.06 | 1% | ok | 1.0x (16%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.895 ms | 0.85 | 3.38 | 30%\* | ok | 0.9x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.312 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 106.019 ms | 0.21 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.772 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.747 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 114.651 ms | 0.20 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.800 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.878 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 636.133 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 8.131 ms | 0.41 | 1.66 | 15%\* | ok | 0.8x (14%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 29.378 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 203.709 ms | 1.51 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.056 ms | 0.48 | 1.91 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.510 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1710.838 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.652 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 17.959 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1519.827 ms | 0.41 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.889 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 232.016 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 408.953 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 129.086 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 103.886 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 122.972 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 80.665 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 538.021 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 175.622 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.095 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 201.668 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 193.502 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 218.373 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 307.802 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.367 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| pc | ilut | f64 | 128^2 grid | 32.693 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.872 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.715 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.093 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.462 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.517 ms | 0.74 | 8.34 | 73% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.303 ms | 0.79 | 8.88 | 78% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.808 ms | 0.69 | 7.70 | 67% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.533 ms | 1.08 | 8.18 | 72% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.798 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 147.837 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 129.230 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.507 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 187.491 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.558 ms | 13.06 | 0.20 | 22% | ok | 5.5x (92%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.246 ms | 7.47 | 7.94 | 69% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 10.650 ms | 1.58 | 6.30 | n/a | ok | 0.6x (10%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.132 ms | 5.36 | 10.71 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.993 ms | 3.36 | 3.36 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.982 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.525 ms | 23.29 | 0.27 | n/a | ok | 5.0x (83%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.278 ms | 32.43 | 0.25 | n/a | ok | 5.4x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.326 ms | 62.05 | 0.36 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.323 ms | 62.10 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.063 ms | 20.55 | 0.12 | n/a | ok | 5.5x (92%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

