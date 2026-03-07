#pragma once
// MTL5 -- Contiguous memory block (replaces MTL4 contiguous_memory_block)
// Unified stack/heap memory block using std::conditional_t and if constexpr
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <mtl/tag/storage.hpp>

namespace mtl::detail {

/// Ownership mode for heap-allocated memory
enum class memory_category { own, external, view };

/// Contiguous memory block with compile-time stack/heap selection.
/// Stack path: StaticSize-element array with alignas.
/// Heap path:  new[]/delete[] with three ownership modes (own, external, view).
template <typename Value, typename Storage, std::size_t StaticSize = 0>
class contiguous_memory_block {
    static constexpr bool on_stack = std::is_same_v<Storage, tag::on_stack>;

    // Stack storage: fixed-size aligned array
    struct stack_data {
        alignas(alignof(Value)) Value data_[StaticSize > 0 ? StaticSize : 1];
    };

    // Heap storage: pointer + size + ownership
    struct heap_data {
        Value*          data_     = nullptr;
        std::size_t     size_     = 0;
        memory_category category_ = memory_category::own;
    };

    std::conditional_t<on_stack, stack_data, heap_data> store_;

    // Heap helpers
    void allocate(std::size_t n) {
        assert(!on_stack);
        if constexpr (!on_stack) {
            store_.data_ = (n > 0) ? new Value[n]{} : nullptr;
            store_.size_ = n;
            store_.category_ = memory_category::own;
        }
    }

    void deallocate() {
        if constexpr (!on_stack) {
            if (store_.category_ == memory_category::own) {
                delete[] store_.data_;
            }
            store_.data_ = nullptr;
            store_.size_ = 0;
            store_.category_ = memory_category::own;
        }
    }

public:
    using value_type = Value;
    using size_type  = std::size_t;

    // -- Default constructor ----------------------------------------------
    contiguous_memory_block() {
        if constexpr (on_stack) {
            std::fill_n(store_.data_, StaticSize, Value{});
        }
        // heap: members zero-initialized via heap_data defaults
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
    contiguous_memory_block(contiguous_memory_block&& other) noexcept {
        if constexpr (on_stack) {
            std::copy_n(other.store_.data_, StaticSize, store_.data_);
        } else {
            store_.data_     = other.store_.data_;
            store_.size_     = other.store_.size_;
            store_.category_ = other.store_.category_;
            other.store_.data_     = nullptr;
            other.store_.size_     = 0;
            other.store_.category_ = memory_category::own;
        }
    }

    // -- Copy assignment -------------------------------------------------
    contiguous_memory_block& operator=(const contiguous_memory_block& other) {
        if (this == &other) return *this;
        if constexpr (on_stack) {
            std::copy_n(other.store_.data_, StaticSize, store_.data_);
        } else {
            if (store_.size_ != other.store_.size_ || store_.category_ != memory_category::own) {
                deallocate();
                allocate(other.store_.size_);
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
