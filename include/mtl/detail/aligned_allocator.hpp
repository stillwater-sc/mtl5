#pragma once
// MTL5 -- over-aligned heap allocation for SIMD-friendly storage (#84, epic #82).
//
// Dense storage and packing buffers are over-aligned to a cache-line / widest-
// SIMD boundary so the SIMD layer (mtl::simd, #83) can use aligned loads/stores
// and packed panels start on aligned addresses. Uses C++17 over-aligned
// operator new/delete, which is portable across GCC/Clang/MSVC (unlike
// posix_memalign / std::aligned_alloc).

#include <cstddef>
#include <memory>        // uninitialized_value_construct_n, destroy_n
#include <new>           // ::operator new(std::align_val_t), std::align_val_t
#include <type_traits>   // remove_cv_t

namespace mtl::detail {

/// Default over-alignment: 64 bytes covers a typical cache line and the widest
/// common SIMD register (AVX-512). Types with a larger natural alignment keep it.
inline constexpr std::size_t default_alignment = 64;

/// Alignment used for a block of `T`: at least `default_alignment`, but never
/// less than T's natural alignment.
template <typename T>
inline constexpr std::size_t block_alignment =
    alignof(T) > default_alignment ? alignof(T) : default_alignment;

/// Allocate and value-initialize `n` elements of `T` on an over-aligned heap
/// block (`block_alignment<T>`). Returns nullptr for n == 0. Works for cv-
/// qualified `T` (e.g. ndarray's `const double` views) by constructing/freeing
/// through the unqualified type.
template <typename T>
[[nodiscard]] T* allocate_aligned(std::size_t n) {
    using U = std::remove_cv_t<T>;
    if (n == 0) return nullptr;
    void* raw = ::operator new(n * sizeof(U), std::align_val_t(block_alignment<U>));
    U* p = static_cast<U*>(raw);
    try {
        std::uninitialized_value_construct_n(p, n);
    } catch (...) {
        ::operator delete(raw, std::align_val_t(block_alignment<U>));   // no leak if a ctor throws
        throw;
    }
    return p;   // implicit U* -> T* (adds const/volatile back if any)
}

/// Destroy `n` elements and free a block from allocate_aligned. No-op on nullptr.
template <typename T>
void deallocate_aligned(T* p, std::size_t n) noexcept {
    using U = std::remove_cv_t<T>;
    if (p == nullptr) return;
    U* up = const_cast<U*>(p);
    std::destroy_n(up, n);
    ::operator delete(static_cast<void*>(up), std::align_val_t(block_alignment<U>));
}

/// C++ Allocator that hands out `Align`-byte-aligned storage. Usable as
/// `std::vector<T, aligned_allocator<T>>` for SIMD-friendly scratch/packing
/// buffers. Stateless: all instances compare equal.
template <typename T, std::size_t Align = block_alignment<T>>
class aligned_allocator {
public:
    using value_type = T;
    static constexpr std::size_t alignment = Align;

    aligned_allocator() noexcept = default;
    template <typename U>
    aligned_allocator(const aligned_allocator<U, Align>&) noexcept {}

    template <typename U>
    struct rebind { using other = aligned_allocator<U, Align>; };

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(Align)));
    }
    void deallocate(T* p, std::size_t /*n*/) noexcept {
        ::operator delete(static_cast<void*>(p), std::align_val_t(Align));
    }

    template <typename U>
    bool operator==(const aligned_allocator<U, Align>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const aligned_allocator<U, Align>&) const noexcept { return false; }
};

} // namespace mtl::detail
