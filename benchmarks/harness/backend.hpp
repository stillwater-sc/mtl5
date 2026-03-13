#pragma once
// MTL5 Benchmark Harness -- Backend policy tags
// Each tag selects a specific implementation path for benchmarked operations.
// New backends (CUDA, MKL, etc.) are added by defining a new tag here
// and specializing the operation wrappers in the op_*.hpp files.

#include <string_view>
#include <type_traits>

namespace mtl::bench {

// -- Backend tags -----------------------------------------------------------

/// Pure C++ generic implementation (no library dispatch)
struct Native {
    static constexpr std::string_view name = "native";
};

/// BLAS-accelerated path (Level 1/2/3)
struct Blas {
    static constexpr std::string_view name = "blas";
};

/// LAPACK-accelerated path (factorizations, eigensolvers)
struct Lapack {
    static constexpr std::string_view name = "lapack";
};

/// UMFPACK sparse direct solver
struct Umfpack {
    static constexpr std::string_view name = "umfpack";
};

// -- Backend availability traits --------------------------------------------

template <typename Backend>
inline constexpr bool is_available_v = false;

// Native is always available
template <>
inline constexpr bool is_available_v<Native> = true;

#ifdef MTL5_HAS_BLAS
template <>
inline constexpr bool is_available_v<Blas> = true;
#endif

#ifdef MTL5_HAS_LAPACK
template <>
inline constexpr bool is_available_v<Lapack> = true;
#endif

#ifdef MTL5_HAS_UMFPACK
template <>
inline constexpr bool is_available_v<Umfpack> = true;
#endif

/// Concept: a backend that is compiled in
template <typename B>
concept AvailableBackend = is_available_v<B>;

// -- Backend list for compile-time iteration --------------------------------

/// Holds a list of backend tags for fold-expression expansion
template <typename... Backends>
struct backend_list {};

/// All backends that are compiled in
using all_backends = backend_list<
    Native
#ifdef MTL5_HAS_BLAS
    , Blas
#endif
#ifdef MTL5_HAS_LAPACK
    , Lapack
#endif
#ifdef MTL5_HAS_UMFPACK
    , Umfpack
#endif
>;

/// Dense-operation backends (BLAS-level)
using dense_backends = backend_list<
    Native
#ifdef MTL5_HAS_BLAS
    , Blas
#endif
>;

/// Factorization backends (LAPACK-level)
using factor_backends = backend_list<
    Native
#ifdef MTL5_HAS_LAPACK
    , Lapack
#endif
>;

} // namespace mtl::bench
