## i7-12700k -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.1, update 38.5, triad 32.5 GB/s (ceiling 38.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 97% | 0 |
| blas2 | 10 | 100% | 106% | 0 |
| blas3 | 10 | 81% | 94% | 0 |
| factor | 12 | 25% | 40% | 0 |
| eigen | 8 | 24% | 70% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 94% | 0 |
| direct | 31 | 3% | 32% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 21 | 24% | 69% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 77% | 0 |
| mg | 1 | 56% | 56% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 36% | 84% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.625 ms | 2.98 | 23.86 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.229 ms | 3.97 | 15.87 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.438 ms | 2.61 | 31.27 | 81% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.727 ms | 3.55 | 21.30 | 55% | ok |
| blas1 | scale | f64 | n=8388608 | 3.578 ms | 2.34 | 37.51 | 97% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.881 ms | 3.44 | 13.75 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.013 ms | 2.09 | 16.72 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.061 ms | 1.19 | 9.50 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 4.001 ms | 2.10 | 16.77 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.779 ms | 2.40 | 1.75 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.506 ms | 9.57 | 38.30 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 1.649 ms | 20.35 | 40.72 | 106% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.631 ms | 9.24 | 36.99 | 96% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.639 ms | 20.48 | 40.97 | 106% | ok |
| blas2 | symv | f64 | n=4096 | 1.924 ms | 17.44 | 34.91 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 1.858 ms | 9.03 | 36.15 | 94% | ok |
| blas2 | trmv | f32 | n=4096 | 0.945 ms | 17.75 | 35.53 | 92% | ok |
| blas2 | trsv | f64 | n=4096 | 1.797 ms | 9.33 | 37.37 | 97% | ok |
| blas2 | trsv | f32 | n=4096 | 0.871 ms | 19.26 | 38.56 | 100% | ok |
| blas2 | ger | f64 | n=4096 | 6.828 ms | 4.91 | 39.32 | 102% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.708 ms | 72.40 | 1.70 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.833 ms | 146.46 | 1.72 | 94% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.573 ms | 52.16 | 2.44 | 67% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.715 ms | 72.25 | 1.69 | 92% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.544 ms | 52.75 | 1.65 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.706 ms | 57.04 | 1.34 | 73% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.109 ms | 63.64 | 2.49 | 81% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.135 ms | 62.86 | 2.46 | 80% | ok |
| blas3 | trsm | f32 | 512x512x512 | 0.961 ms | 139.62 | 2.73 | 89% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.223 ms | 0.40 | 3.21 | 8% | ok |
| factor | lu | f64 | 512x512 | 3.293 ms | 27.17 | 1.27 | 35% | ok |
| factor | lu | f32 | 512x512 | 1.880 ms | 47.60 | 1.12 | 30% | ok |
| factor | cholesky | f64 | 512x512 | 2.059 ms | 21.73 | 2.04 | 28% | ok |
| factor | cholesky | f32 | 512x512 | 1.153 ms | 38.80 | 1.82 | 25% | ok |
| factor | ldlt | f64 | 512x512 | 7.507 ms | 5.96 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 5.002 ms | 8.94 | 0.84 | 11% | ok |
| factor | qr | f64 | 512x512 | 5.744 ms | 31.15 | 0.73 | 40% | ok |
| factor | lq | f64 | 512x512 | 25.994 ms | 6.88 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 266.121 ms | 1.68 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.638 ms | 0.64 | 2.56 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 46.074 ms | 5.83 | 0.09 | 7% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 3.820 ms | 23.42 | 1.10 | 30% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.734 ms | 12.90 | 0.60 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 58.920 ms | 2.56 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 756.369 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 15.991 ms | 10.49 | 0.07 | 13% | ok |
| eigen | schur | f64 | 256x256 | 21.077 ms | 19.90 | 0.05 | 25% | ok |
| eigen | svd | f64 | 256x256 | 6.709 ms | 55.02 | 0.16 | 70% | ok |
| eigen | expm | f64 | 256x256 | 11.181 ms | 22.51 | 0.09 | 29% | ok |
| eigen | sylvester | f64 | 256x256 | 53.131 ms | 18.95 | 0.03 | 24% | ok |
| elementwise | exp | f64 | n=8388608 | 39.475 ms | 0.21 | 3.40 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 46.358 ms | 0.18 | 2.90 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.298 ms | 0.14 | 2.23 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.128 ms | 0.29 | 4.61 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 202.211 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.750 ms | 3.49 | 36.29 | 94% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.925 ms | 2.83 | 29.44 | 77% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.676 ms | 8.96 | 18.83 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.853 ms | 0.67 | 2.37 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.323 ms | 1.52 | 12.59 | 33%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.441 ms | 0.77 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.975 ms | 0.45 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.432 ms | 1.33 | 0.09 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.599 ms | 0.55 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.500 ms | 0.50 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.022 ms | 1.31 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.772 ms | 0.08 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.374 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.028 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.114 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.186 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.186 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.105 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.104 ms | 0.92 | 0.06 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.247 ms | 3.06 | 12.23 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.071 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.085 ms | 2.74 | 0.18 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.249 ms | 3.03 | 12.14 | 32%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.086 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.712 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.631 ms | 1.20 | 4.79 | 12%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.437 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.371 ms | 0.58 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.641 ms | 1.18 | 4.72 | 12%\* | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.738 ms | 1.95 | 0.13 | 2% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.891 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.469 ms | 3.22 | 0.21 | 4% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.329 ms | 2.30 | 9.19 | 24%\* | ok |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 4.561 ms | 2.45 | 0.16 | 3% | ok |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 2.121 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 1.912 ms | 5.85 | 0.38 | 7% | ok |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.461 ms | 1.64 | 6.56 | 17%\* | ok |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 7.585 ms | 2.95 | 0.10 | 4% | ok |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.596 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 3.248 ms | 6.89 | 0.23 | 9% | ok |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.577 ms | 1.31 | 5.24 | 14%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.929 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 206.424 ms | 1.49 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.483 ms | 2.27 | 9.09 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.671 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 59.930 ms | 5.15 | 0.01 | 7% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.478 ms | 2.28 | 9.12 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.870 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 557.055 ms | 1.11 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.547 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.253 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 453.313 ms | 1.36 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.525 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.898 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 11.452 ms | 26.95 | 0.07 | 34% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.508 ms | 6.63 | 26.52 | 69%\* | ok |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.993 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 9.684 ms | 31.87 | 0.09 | 41% | ok |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.874 ms | 3.85 | 15.42 | 40%\* | ok |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 9.048 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 19.144 ms | 32.24 | 0.04 | 41% | ok |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.036 ms | 3.25 | 13.00 | 34%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.411 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 35.168 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.181 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.696 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.192 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.183 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.683 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.296 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.397 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.341 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.806 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.276 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 19.224 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.041 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.559 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 11.652 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.034 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.157 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.987 ms | 2.65 | 29.72 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.221 ms | 2.14 | 24.03 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.563 ms | 1.67 | 18.77 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.391 ms | 2.87 | 21.71 | 56% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.619 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.920 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.494 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.999 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.866 ms | 3.45 | 13.79 | 36% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 39.000 ms | 6.88 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.554 ms | 30.27 | 32.16 | 84%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.309 ms | 7.27 | 29.06 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.996 ms | 16.84 | 33.69 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.724 ms | 9.73 | 9.73 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.73 | 9.73 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.446 ms | 15.39 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.837 ms | 12.88 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.234 ms | 26.23 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.232 ms | 26.23 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.866 ms | 9.99 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

