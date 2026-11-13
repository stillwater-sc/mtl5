#pragma once
// mtl5_bench -- MKL PARDISO, loaded at run time, for the reference entries in
// entries_direct.hpp (#810). Benchmark code only: the library has no PARDISO
// binding and gains no MKL dependency.
//
// WHY dlopen AND NOT A LINK. The entries run in the `blas` variant, which links
// a BLAS (OpenBLAS on most lab machines). libmkl_rt exports dgemm_ and the rest
// of the BLAS as well, so linking it beside OpenBLAS would let the link order
// decide which BLAS every OTHER entry measures. Opened RTLD_LOCAL, MKL's
// symbols stay private to MKL, and the BLAS the build chose is untouched.
//
// WHERE IT LOOKS. MTL5_BENCH_MKL_RT, the path CMake found at configure time,
// then the bare sonames, which the loader resolves through LD_LIBRARY_PATH (a
// sourced oneAPI setvars.sh) and the system paths.
//
// THREADS. MKL_Set_Threading_Layer(GNU) is the first MKL call, so MKL runs on
// libgomp rather than requiring Intel's libiomp5 on the loader path, and
// MKL_Set_Num_Threads takes the run's thread count, so a 1-thread run of the
// entries is a 1-thread run of PARDISO.
//
// Not on Windows: unavailable() says so, and the entries are listed as skipped.

#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <mtl/mat/compressed2D.hpp>

namespace mtl::bench::platform {

class mkl_pardiso {
public:
    // MKL_INT is int in the LP64 interface, libmkl_rt's default.
    using pardiso_fn = void (*)(void*, const int*, const int*, const int*, const int*, const int*,
                                const void*, const int*, const int*, int*, const int*, int*,
                                const int*, void*, void*, int*);
    using pardisoinit_fn = void (*)(void*, const int*, int*);

    /// The library, loaded on first use with the run's thread count. It is
    /// usable when unavailable() is empty.
    static const mkl_pardiso& instance(unsigned threads) {
        static const mkl_pardiso lib(threads);
        return lib;
    }
    const std::string& unavailable() const { return reason_; }

    pardiso_fn pardiso() const { return pardiso_; }
    pardisoinit_fn pardisoinit() const { return init_; }
    const std::string& path() const { return path_; }

private:
    pardiso_fn pardiso_ = nullptr;
    pardisoinit_fn init_ = nullptr;
    std::string path_, reason_;

    explicit mkl_pardiso(unsigned threads) {
#if defined(_WIN32)
        (void)threads;
        reason_ = "MKL PARDISO is loaded only on Linux and macOS";
#else
        std::vector<std::string> candidates;
#ifdef MTL5_BENCH_MKL_RT
        candidates.emplace_back(MTL5_BENCH_MKL_RT);
#endif
        candidates.insert(candidates.end(), {"libmkl_rt.so.2", "libmkl_rt.so", "libmkl_rt.2.dylib",
                                             "libmkl_rt.dylib"});
        void* h = nullptr;
        for (const auto& c : candidates)
            if ((h = dlopen(c.c_str(), RTLD_NOW | RTLD_LOCAL))) { path_ = c; break; }
        if (!h) { reason_ = "libmkl_rt not found"; return; }
        using layer_fn = int (*)(int);
        using threads_fn = void (*)(int);
        auto layer = reinterpret_cast<layer_fn>(dlsym(h, "MKL_Set_Threading_Layer"));
        auto nthreads = reinterpret_cast<threads_fn>(dlsym(h, "MKL_Set_Num_Threads"));
        auto pd = reinterpret_cast<pardiso_fn>(dlsym(h, "pardiso"));
        auto init = reinterpret_cast<pardisoinit_fn>(dlsym(h, "pardisoinit"));
        if (!layer || !nthreads || !pd || !init) { reason_ = path_ + " lacks the PARDISO symbols"; return; }
        constexpr int mkl_threading_gnu = 3;   // mkl_service.h
        layer(mkl_threading_gnu);
        nthreads(static_cast<int>(threads));
        pardiso_ = pd;
        init_ = init;
#endif
    }
};

/// One PARDISO factorization of one matrix: its handle, its CSR arrays and
/// its settings. mtype 2 (real SPD) stores the upper triangle, mtype 11 (real
/// unsymmetric) the whole matrix; both zero-based (iparm[34]) and ordered by
/// minimum degree (iparm[1] = 0), MKL's AMD, to match the ordering the MTL5
/// entries use, rather than MKL's default METIS.
class pardiso_handle {
public:
    pardiso_handle(const mkl_pardiso& lib, const mat::compressed2D<double>& A, int mtype)
        : lib_(lib), mtype_(mtype), n_(static_cast<int>(A.num_rows())) {
        const auto& st = A.ref_major();
        const auto& ix = A.ref_minor();
        const auto& v = A.ref_data();
        ia_.push_back(0);
        for (std::size_t r = 0; r < A.num_rows(); ++r) {
            for (std::size_t q = st[r]; q < st[r + 1]; ++q)
                if (mtype_ != 2 || ix[q] >= r) {
                    ja_.push_back(static_cast<int>(ix[q]));
                    a_.push_back(v[q]);
                }
            ia_.push_back(static_cast<int>(ja_.size()));
        }
        pt_.fill(nullptr);
        lib_.pardisoinit()(pt_.data(), &mtype_, iparm_.data());
        iparm_[1] = 0;    // minimum degree ordering
        iparm_[26] = 0;   // no matrix checker inside the timed phases
        iparm_[34] = 1;   // zero-based ia/ja
    }
    ~pardiso_handle() { try { release(); } catch (...) {} }
    pardiso_handle(const pardiso_handle&) = delete;
    pardiso_handle& operator=(const pardiso_handle&) = delete;

    void analyze() { call(11, nullptr, nullptr); analyzed_ = true; }
    void factor() { call(22, nullptr, nullptr); }
    void solve(double* x, const double* b) { call(33, const_cast<double*>(b), x); }

    /// Frees the factor and the analysis (phase -1); the next analyze() starts over.
    void release() {
        if (!analyzed_) return;
        analyzed_ = false;
        call(-1, nullptr, nullptr);
        pt_.fill(nullptr);
    }

private:
    const mkl_pardiso& lib_;
    int mtype_, n_;
    std::vector<int> ia_, ja_;
    std::vector<double> a_;
    std::array<void*, 64> pt_{};
    std::array<int, 64> iparm_{};
    bool analyzed_ = false;

    void call(int phase, double* b, double* x) {
        const int maxfct = 1, mnum = 1, nrhs = 1, msglvl = 0;
        int error = 0;
        lib_.pardiso()(pt_.data(), &maxfct, &mnum, &mtype_, &phase, &n_, a_.data(), ia_.data(), ja_.data(),
                       nullptr, &nrhs, iparm_.data(), &msglvl, b, x, &error);
        if (error != 0)
            throw std::runtime_error("PARDISO phase " + std::to_string(phase) + ": error " + std::to_string(error));
    }
};

} // namespace mtl::bench::platform
