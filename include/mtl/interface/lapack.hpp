#pragma once
// MTL5 -- Optional LAPACK interface for factorizations and eigensolvers
// Port from MTL4: boost/numeric/mtl/interface/lapack.hpp
// Guarded by MTL5_HAS_LAPACK (set by CMake when MTL5_ENABLE_LAPACK=ON)

#ifdef MTL5_HAS_LAPACK

#include <cstddef>

// -- Fortran LAPACK declarations ----------------------------------------
extern "C" {

// Cholesky factorization
void spotrf_(const char* uplo, const int* n, float* A, const int* lda, int* info);
void dpotrf_(const char* uplo, const int* n, double* A, const int* lda, int* info);

// LU factorization with partial pivoting
void sgetrf_(const int* m, const int* n, float* A, const int* lda,
             int* ipiv, int* info);
void dgetrf_(const int* m, const int* n, double* A, const int* lda,
             int* ipiv, int* info);

// QR factorization
void sgeqrf_(const int* m, const int* n, float* A, const int* lda,
             float* tau, float* work, const int* lwork, int* info);
void dgeqrf_(const int* m, const int* n, double* A, const int* lda,
             double* tau, double* work, const int* lwork, int* info);

// SVD (divide and conquer)
void sgesdd_(const char* jobz, const int* m, const int* n,
             float* A, const int* lda, float* S,
             float* U, const int* ldu, float* VT, const int* ldvt,
             float* work, const int* lwork, int* iwork, int* info);
void dgesdd_(const char* jobz, const int* m, const int* n,
             double* A, const int* lda, double* S,
             double* U, const int* ldu, double* VT, const int* ldvt,
             double* work, const int* lwork, int* iwork, int* info);

// Symmetric eigenvalue (real symmetric)
void ssyev_(const char* jobz, const char* uplo, const int* n,
            float* A, const int* lda, float* W,
            float* work, const int* lwork, int* info);
void dsyev_(const char* jobz, const char* uplo, const int* n,
            double* A, const int* lda, double* W,
            double* work, const int* lwork, int* info);

} // extern "C"

// -- C++ wrapper functions ----------------------------------------------

namespace mtl::interface::lapack {

// -- Cholesky -----------------------------------------------------------

inline int potrf(char uplo, int n, float* A, int lda) {
    int info = 0;
    spotrf_(&uplo, &n, A, &lda, &info);
    return info;
}
inline int potrf(char uplo, int n, double* A, int lda) {
    int info = 0;
    dpotrf_(&uplo, &n, A, &lda, &info);
    return info;
}

// -- LU -----------------------------------------------------------------

inline int getrf(int m, int n, float* A, int lda, int* ipiv) {
    int info = 0;
    sgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}
inline int getrf(int m, int n, double* A, int lda, int* ipiv) {
    int info = 0;
    dgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

// -- QR -----------------------------------------------------------------

inline int geqrf(int m, int n, float* A, int lda, float* tau,
                 float* work, int lwork) {
    int info = 0;
    sgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info);
    return info;
}
inline int geqrf(int m, int n, double* A, int lda, double* tau,
                 double* work, int lwork) {
    int info = 0;
    dgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info);
    return info;
}

// -- SVD ----------------------------------------------------------------

inline int gesdd(char jobz, int m, int n, float* A, int lda,
                 float* S, float* U, int ldu, float* VT, int ldvt,
                 float* work, int lwork, int* iwork) {
    int info = 0;
    sgesdd_(&jobz, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt,
            work, &lwork, iwork, &info);
    return info;
}
inline int gesdd(char jobz, int m, int n, double* A, int lda,
                 double* S, double* U, int ldu, double* VT, int ldvt,
                 double* work, int lwork, int* iwork) {
    int info = 0;
    dgesdd_(&jobz, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt,
            work, &lwork, iwork, &info);
    return info;
}

// -- Symmetric eigenvalue -----------------------------------------------

inline int syev(char jobz, char uplo, int n, float* A, int lda,
                float* W, float* work, int lwork) {
    int info = 0;
    ssyev_(&jobz, &uplo, &n, A, &lda, W, work, &lwork, &info);
    return info;
}
inline int syev(char jobz, char uplo, int n, double* A, int lda,
                double* W, double* work, int lwork) {
    int info = 0;
    dsyev_(&jobz, &uplo, &n, A, &lda, W, work, &lwork, &info);
    return info;
}

} // namespace mtl::interface::lapack

#endif // MTL5_HAS_LAPACK
