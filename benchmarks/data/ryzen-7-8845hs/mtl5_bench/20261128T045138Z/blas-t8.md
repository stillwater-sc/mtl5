## ryzen-7-8845hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 624.5 GFLOP/s, f32 1249.3 GFLOP/s |
| ... hardware, AVX-512F | f64 624.5, f32 1249.3 GFLOP/s |
| ... this build, scalar | f64 14.4, f32 14.4 GFLOP/s |
| bandwidth | read 53.3, update 53.7, triad 41.2 GB/s (ceiling 53.7) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 94% | 125% | 0 | 43% |
| blas2 | 10 | 100% | 109% | 0 | 17% |
| blas3 | 10 | 40% | 71% | 0 | 63% |
| factor | 12 | 2% | 8% | 0 | 11% |
| eigen | 8 | 2% | 5% | 0 | 11% |
| elementwise | 5 | 7% | 12% | 0 | 12% |
| sparse | 5 | 19% | 296% | 0 | 13% |
| direct | 19 | 0% | 20% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 12% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 35% | 35% | 0 | 12% |
| mg | 1 | 31% | 31% | 0 | 11% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 13% | 22% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 89% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.264 ms | 7.41 | 59.30 | 110% | ok | 4.2x (53%) |
| blas1 | dot | f32 | n=8388608 | 1.335 ms | 12.57 | 50.27 | 94% | ok | 7.4x (92%) |
| blas1 | axpy | f64 | n=8388608 | 3.501 ms | 4.79 | 57.50 | 107% | ok | 3.4x (43%) |
| blas1 | axpy | f32 | n=8388608 | 1.748 ms | 9.60 | 57.57 | 107% | ok | 6.8x (85%) |
| blas1 | scale | f64 | n=8388608 | 1.992 ms | 4.21 | 67.37 | 125% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=8388608 | 1.346 ms | 12.47 | 49.87 | 93% | ok | 7.3x (91%) |
| blas1 | one_norm | f64 | n=8388608 | 5.063 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.096 ms | 1.65 | 13.17 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 83.179 ms | 2.22 | 1.61 | 3% | ok | 1.1x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.571 ms | 13.05 | 52.22 | 97% | ok | 1.4x (17%) |
| blas2 | gemv | f32 | n=4096 | 1.158 ms | 28.97 | 57.98 | 108% | ok | 1.2x (15%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.567 ms | 13.07 | 52.32 | 97% | ok | 1.3x (16%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.276 ms | 26.29 | 52.60 | 98% | ok | 1.1x (14%) |
| blas2 | symv | f64 | n=4096 | 1.245 ms | 26.95 | 53.95 | 100% | ok | 1.5x (18%) |
| blas2 | trmv | f64 | n=4096 | 1.208 ms | 13.89 | 55.62 | 104% | ok | 1.5x (18%) |
| blas2 | trmv | f32 | n=4096 | 0.574 ms | 29.22 | 58.49 | 109% | ok | 1.7x (22%) |
| blas2 | trsv | f64 | n=4096 | 1.701 ms | 9.86 | 39.48 | 74% | ok | 0.9x (12%) |
| blas2 | trsv | f32 | n=4096 | 0.849 ms | 19.77 | 39.57 | 74% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 4.661 ms | 7.20 | 57.60 | 107% | ok | 1.4x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 0.608 ms | 441.36 | 10.34 | 71% | ok | 6.0x (75%) |
| blas3 | gemm | f32 | 512x512x512 | 0.338 ms | 793.72 | 9.30 | 64% | ok | 5.6x (70%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.354 ms | 99.13 | 4.65 | 16% | ok | 2.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 0.611 ms | 439.63 | 10.30 | 70% | ok | 6.2x (77%) |
| blas3 | syrk | f64 | 512x512x512 | 1.723 ms | 77.89 | 2.43 | 12% | ok | 1.7x (21%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.386 ms | 112.51 | 2.64 | 18% | ok | 2.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 0.388 ms | 345.65 | 13.50 | 55% | ok | 4.9x (61%) |
| blas3 | trsm | f64 | 512x512x512 | 0.536 ms | 250.17 | 9.77 | 40% | ok | 5.3x (66%) |
| blas3 | trsm | f32 | 512x512x512 | 0.267 ms | 503.44 | 9.83 | 40% | ok | 5.0x (63%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.246 ms | 0.57 | 4.59 | 9% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 9.191 ms | 9.74 | 0.46 | 2% | ok | 0.5x (7%) |
| factor | lu | f32 | 512x512 | 4.346 ms | 20.59 | 0.48 | 2% | ok | 0.6x (7%) |
| factor | cholesky | f64 | 512x512 | 5.932 ms | 7.54 | 0.71 | 1%\* | ok | 0.7x (9%) |
| factor | cholesky | f32 | 512x512 | 2.343 ms | 19.09 | 0.89 | 2%\* | ok | 0.7x (9%) |
| factor | ldlt | f64 | 512x512 | 11.807 ms | 3.79 | 0.36 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.253 ms | 4.36 | 0.41 | 1%\* | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 11.934 ms | 15.00 | 0.35 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 42.651 ms | 4.20 | 0.10 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 456.085 ms | 0.98 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.014 ms | 1.04 | 4.14 | 8%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 66.556 ms | 4.03 | 0.06 | 1% | ok | 0.9x (11%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.288 ms | 12.28 | 0.58 | 2% | ok | 0.8x (9%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 3.628 ms | 6.17 | 0.29 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 132.594 ms | 1.14 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 522.308 ms | 0.29 | 0.00 | 0% | ok | 0.9x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 21.066 ms | 7.96 | 0.05 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 26.487 ms | 15.84 | 0.04 | 3% | ok | 0.8x (10%) |
| eigen | svd | f64 | 256x256 | 11.978 ms | 30.82 | 0.09 | 5% | ok | 0.7x (9%) |
| eigen | expm | f64 | 256x256 | 17.867 ms | 14.08 | 0.06 | 2% | ok | 0.9x (11%) |
| eigen | sylvester | f64 | 256x256 | 82.444 ms | 12.21 | 0.02 | 2% | ok | 0.9x (11%) |
| elementwise | exp | f64 | n=8388608 | 28.249 ms | 0.30 | 4.75 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 43.826 ms | 0.19 | 3.06 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 39.359 ms | 0.21 | 3.41 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.328 ms | 0.39 | 6.29 | 12% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 34.988 ms | 0.24 | 3.84 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.171 ms | 15.27 | 158.87 | 296% ! | ok | 5.0x (62%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.239 ms | 0.81 | 8.41 | 16% | ok | 0.4x (5%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.253 ms | 18.58 | 39.07 | 73% | ok | 2.9x (37%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.265 ms | 0.18 | 0.63 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.636 ms | 1.23 | 10.18 | 19%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.830 ms | 0.56 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.590 ms | 0.35 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 16.563 ms | 0.68 | 0.04 | 0% | ok | 0.9x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.973 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 58.087 ms | 0.39 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.114 ms | 0.89 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 36.642 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.762 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.738 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 61.555 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.229 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.370 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.981 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.412 ms | 0.78 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.283 ms | 2.68 | 10.70 | 20%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 7.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.029 ms | 1.86 | 0.12 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.283 ms | 2.67 | 10.69 | 20%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.399 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.639 ms | 0.45 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.143 ms | 0.66 | 2.65 | 5%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.837 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 50.035 ms | 0.45 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.079 ms | 0.70 | 2.80 | 5%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.649 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 258.522 ms | 1.19 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.010 ms | 1.68 | 6.70 | 12%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 14.158 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 70.168 ms | 4.40 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.050 ms | 1.64 | 6.57 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.227 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 900.301 ms | 0.69 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.537 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.572 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 514.710 ms | 1.20 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.681 ms | 0.32 | 1.26 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 39.165 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.248 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.742 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.717 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.730 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.640 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 54.691 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.468 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.010 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 26.113 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 24.681 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.562 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.400 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.410 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 22.775 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.834 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.794 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.337 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.567 ms | 1.67 | 18.71 | 35% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.559 ms | 1.68 | 18.82 | 35% | ok | 1.1x (14%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.801 ms | 1.45 | 16.28 | 30% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 17.678 ms | 2.18 | 16.44 | 31% | ok | 0.9x (11%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.718 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.182 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.764 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.810 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 9.853 ms | 1.70 | 6.81 | 13% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.069 ms | 65.97 | 1.03 | 11% | ok | 7.3x (91%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.498 ms | 11.20 | 11.90 | 22% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.261 ms | 5.14 | 20.58 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.870 ms | 5.85 | 11.69 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.394 ms | 12.03 | 12.03 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.394 ms | 12.04 | 12.04 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.185 ms | 226.55 | 2.65 | n/a | ok | 7.1x (89%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.185 ms | 64.14 | 0.50 | n/a | ok | 7.2x (90%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.797 ms | 70.69 | 0.41 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.800 ms | 70.63 | 0.41 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.801 ms | 70.62 | 0.41 | n/a | ok | 7.2x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

