#pragma once
// MTL5 -- Contiguous memory block (replaces MTL4 contiguous_memory_block)
// Unified stack/heap memory block using std::conditional_t and if constexpr
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <mtl/detail/aligned_allocator.hpp>
#include <mtl/detail/exceptions.hpp>
#include <mtl/tag/storage.hpp>

namespace mtl::detail {

/// Ownership mode for heap-allocated memory
enum class memory_category { own, external, view };

/// Contiguous memory block with compile-time stack/heap selection.
/// Stack path: StaticSize-element array with alignas.
/// Heap path:  new[]/delete[] with three ownership modes (own, external, view).
///
/// Assignment never re-points a non-owning (external or view) block (#829):
///   - from a block of the same size, the elements are copied (or moved) into
///     the caller's buffer, and the block stays non-owning;
///   - copy assignment from a block of another size throws std::domain_error;
///   - move assignment from a block of another size is noexcept, so it
///     asserts; under NDEBUG it falls back to taking over the source's storage.
/// realloc() to another size is an explicit resize, and it still detaches.
///
/// A fixed-size heap block (StaticSize > 0) always holds StaticSize elements:
/// the default constructor allocates them, and a moved-from block gets fresh
/// storage (move assignment swaps instead, so it does not allocate).
template <typename Value, typename Storage, std::size_t StaticSize = 0>
class contiguous_memory_block {
    static constexpr bool on_stack = std::is_same_v<Storage, tag::on_stack>;

    // Stack storage: fixed-size array, over-aligned to the SIMD/cache-line
    // boundary so fixed-size operands also support aligned SIMD loads.
    struct stack_data {
        alignas(block_alignment<Value>) Value data_[StaticSize > 0 ? StaticSize : 1];
    };

    // Heap storage: pointer + size + ownership
    struct heap_data {
        Value*          data_     = nullptr;
        std::size_t     size_     = 0;
        memory_category category_ = memory_category::own;
    };

    std::conditional_t<on_stack, stack_data, heap_data> store_;

    // Heap helpers. Owned blocks are over-aligned (block_alignment<Value>, i.e.
    // >= 64B) so .data() is suitable for aligned SIMD loads/stores.
    void allocate(std::size_t n) {
        assert(!on_stack);
        if constexpr (!on_stack) {
            store_.data_ = allocate_aligned<Value>(n);   // value-initialized, nullptr if n==0
            store_.size_ = n;
            store_.category_ = memory_category::own;
        }
    }

    void deallocate() {
        if constexpr (!on_stack) {
            if (store_.category_ == memory_category::own) {
                deallocate_aligned<Value>(store_.data_, store_.size_);
            }
            store_.data_ = nullptr;
            store_.size_ = 0;
            store_.category_ = memory_category::own;
        }
    }

public:
    using value_type = Value;
    using size_type  = std::size_t;

    static constexpr bool fixed_heap = !on_stack && StaticSize > 0;

    /// Whether assigning a block of other_size writes into this block's
    /// elements in place, rather than replacing its storage.
    bool assigns_in_place(std::size_t other_size) const noexcept {
        if constexpr (on_stack) return true;
        else return store_.category_ != memory_category::own && store_.size_ == other_size;
    }

    // -- Default constructor ----------------------------------------------
    contiguous_memory_block() {
        if constexpr (on_stack) {
            std::fill_n(store_.data_, StaticSize, Value{});
        } else if constexpr (fixed_heap) {
            allocate(StaticSize);
        }
        // dynamic heap: members zero-initialized via heap_data defaults
    }

    // -- Size constructor (heap only) ------------------------------------
    explicit contiguous_memory_block(std::size_t n) {
        if constexpr (on_stack) {
            assert(n == StaticSize && "Stack block size must match StaticSize");
            std::fill_n(store_.data_, StaticSize, Value{});
        } else {
            allocate(n);
        }
    }

    // -- External / view constructor (heap only) -------------------------
    contiguous_memory_block(Value* ptr, std::size_t n, bool is_view) {
        static_assert(!on_stack, "External/view construction requires heap storage");
        if constexpr (!on_stack) {
            store_.data_ = ptr;
            store_.size_ = n;
            store_.category_ = is_view ? memory_category::view : memory_category::external;
        }
    }

    // -- Copy constructor ------------------------------------------------
    contiguous_memory_block(const contiguous_memory_block& other) {
        if constexpr (on_stack) {
            std::copy_n(other.store_.data_, StaticSize, store_.data_);
        } else {
            allocate(other.store_.size_);
            if (other.store_.data_ && store_.data_) {
                std::copy_n(other.store_.data_, store_.size_, store_.data_);
            }
        }
    }

    // -- Move constructor ------------------------------------------------
    // A fixed-size heap block gives the source fresh storage, which allocates.
    contiguous_memory_block(contiguous_memory_block&& other) noexcept(!fixed_heap) {
        if constexpr (on_stack) {
            std::copy_n(other.store_.data_, StaticSize, store_.data_);
        } else {
            store_.data_     = other.store_.data_;
            store_.size_     = other.store_.size_;
            store_.category_ = other.store_.category_;
            other.store_.data_     = nullptr;
            other.store_.size_     = 0;
            other.store_.category_ = memory_category::own;
            if constexpr (fixed_heap) other.allocate(StaticSize);
        }
    }

    // -- Copy assignment -------------------------------------------------
    contiguous_memory_block& operator=(const contiguous_memory_block& other) {
        if (this == &other) return *this;
        if constexpr (on_stack) {
            std::copy_n(other.store_.data_, StaticSize, store_.data_);
        } else {
            if (store_.category_ != memory_category::own) {
                // Write through to the caller's buffer; never re-point it (#829).
                if (store_.size_ != other.store_.size_)
                    throw_or_abort(std::domain_error(
                        "contiguous_memory_block: cannot assign a block of another size to a non-owning block"));
            } else if (store_.size_ != other.store_.size_) {
                // Allocate first, so a failed allocation leaves *this unchanged.
                contiguous_memory_block fresh(other.store_.size_);
                swap(fresh);
            }
            if (other.store_.data_ && store_.data_) {
                std::copy_n(other.store_.data_, store_.size_, store_.data_);
            }
        }
        return *this;
    }

    // -- Move assignment -------------------------------------------------
    contiguous_memory_block& operator=(contiguous_memory_block&& other) noexcept {
        if (this == &other) return *this;
        if constexpr (on_stack) {
            std::copy_n(other.store_.data_, StaticSize, store_.data_);
        } else {
            if (store_.category_ != memory_category::own) {
                // Write through to the caller's buffer (#829). The source keeps
                // its storage and its (moved-from) elements.
                if (store_.size_ == other.store_.size_) {
                    if (other.store_.data_ && store_.data_)
                        std::move(other.store_.data_, other.store_.data_ + store_.size_, store_.data_);
                    return *this;
                }
                assert(false && "contiguous_memory_block: move-assigning a block of another size to a non-owning block");
            }
            if constexpr (fixed_heap) {
                // Both hold StaticSize elements: swap, so neither is left empty.
                swap(other);
                return *this;
            }
            deallocate();
            store_.data_     = other.store_.data_;
            store_.size_     = other.store_.size_;
            store_.category_ = other.store_.category_;
            other.store_.data_     = nullptr;
            other.store_.size_     = 0;
            other.store_.category_ = memory_category::own;
        }
        return *this;
    }

    // -- Destructor ------------------------------------------------------
    ~contiguous_memory_block() {
        if constexpr (!on_stack) {
            deallocate();
        }
    }

    // -- Access ----------------------------------------------------------

    Value*       data()       { if constexpr (on_stack) return store_.data_; else return store_.data_; }
    const Value* data() const { if constexpr (on_stack) return store_.data_; else return store_.data_; }

    Value&       operator[](std::size_t i)       { return data()[i]; }
    const Value& operator[](std::size_t i) const { return data()[i]; }

    std::size_t size() const {
        if constexpr (on_stack) return StaticSize;
        else return store_.size_;
    }

    bool empty() const { return size() == 0; }

    Value*       begin()       { return data(); }
    const Value* begin() const { return data(); }
    Value*       end()         { return data() + size(); }
    const Value* end()   const { return data() + size(); }

    // -- Ownership query (heap only) ------------------------------------
    memory_category category() const {
        if constexpr (on_stack) return memory_category::own;
        else return store_.category_;
    }

    // -- Reallocation (heap only) ---------------------------------------
    void realloc(std::size_t n) {
        if constexpr (on_stack) {
            assert(n == StaticSize && "Cannot realloc stack block to different size");
        } else {
            if (n == store_.size_) return;
            deallocate();
            allocate(n);
        }
    }

    // -- Swap -----------------------------------------------------------
    void swap(contiguous_memory_block& other) noexcept {
        if constexpr (on_stack) {
            for (std::size_t i = 0; i < StaticSize; ++i) {
                using std::swap;
                swap(store_.data_[i], other.store_.data_[i]);
            }
        } else {
            using std::swap;
            swap(store_.data_,     other.store_.data_);
            swap(store_.size_,     other.store_.size_);
            swap(store_.category_, other.store_.category_);
        }
    }
};

} // namespace mtl::detail
