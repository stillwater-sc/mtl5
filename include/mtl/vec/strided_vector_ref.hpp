#pragma once
// MTL5 — Non-owning strided vector reference
// Primary use: extract columns from row-major dense2D without copy.
// Simplified from MTL4's 275-line CRTP version: no ownership, no clone, no CRTP.
#include <cassert>
#include <cstddef>
#include <iterator>

#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>

namespace mtl::vec {

/// Non-owning reference to strided data: pointer + length + stride.
/// Element i lives at data_[i * stride_].
template <typename Value>
class strided_vector_ref {
public:
    using value_type      = Value;
    using size_type       = std::size_t;
    using reference       = Value&;
    using const_reference = const Value&;
    using pointer         = Value*;
    using const_pointer   = const Value*;

    // ── Strided iterator ────────────────────────────────────────────────
    template <typename Ptr>
    class strided_iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = Value;
        using difference_type   = std::ptrdiff_t;
        using pointer           = Ptr;
        using reference         = decltype(*std::declval<Ptr>());

        strided_iterator() : ptr_(nullptr), stride_(1) {}
        strided_iterator(Ptr p, std::ptrdiff_t s) : ptr_(p), stride_(s) {}

        reference operator*() const { return *ptr_; }
        pointer operator->() const { return ptr_; }
        reference operator[](difference_type n) const { return ptr_[n * stride_]; }

        strided_iterator& operator++() { ptr_ += stride_; return *this; }
        strided_iterator  operator++(int) { auto t = *this; ++(*this); return t; }
        strided_iterator& operator--() { ptr_ -= stride_; return *this; }
        strided_iterator  operator--(int) { auto t = *this; --(*this); return t; }

        strided_iterator& operator+=(difference_type n) { ptr_ += n * stride_; return *this; }
        strided_iterator& operator-=(difference_type n) { ptr_ -= n * stride_; return *this; }

        friend strided_iterator operator+(strided_iterator it, difference_type n) { it += n; return it; }
        friend strided_iterator operator+(difference_type n, strided_iterator it) { it += n; return it; }
        friend strided_iterator operator-(strided_iterator it, difference_type n) { it -= n; return it; }
        friend difference_type operator-(const strided_iterator& a, const strided_iterator& b) {
            return (a.ptr_ - b.ptr_) / a.stride_;
        }

        friend bool operator==(const strided_iterator& a, const strided_iterator& b) { return a.ptr_ == b.ptr_; }
        friend bool operator!=(const strided_iterator& a, const strided_iterator& b) { return a.ptr_ != b.ptr_; }
        friend bool operator< (const strided_iterator& a, const strided_iterator& b) { return a.ptr_ <  b.ptr_; }
        friend bool operator<=(const strided_iterator& a, const strided_iterator& b) { return a.ptr_ <= b.ptr_; }
        friend bool operator> (const strided_iterator& a, const strided_iterator& b) { return a.ptr_ >  b.ptr_; }
        friend bool operator>=(const strided_iterator& a, const strided_iterator& b) { return a.ptr_ >= b.ptr_; }

    private:
        Ptr ptr_;
        std::ptrdiff_t stride_;
    };

    using iterator       = strided_iterator<pointer>;
    using const_iterator = strided_iterator<const_pointer>;

    // ── Constructors ────────────────────────────────────────────────────

    /// Construct from raw pointer, length, and stride.
    strided_vector_ref(pointer data, size_type length, size_type stride)
        : data_(data), size_(length), stride_(stride) {}

    // ── Element access ──────────────────────────────────────────────────

    reference operator()(size_type i) {
        assert(i < size_);
        return data_[i * stride_];
    }

    const_reference operator()(size_type i) const {
        assert(i < size_);
        return data_[i * stride_];
    }

    reference operator[](size_type i) { return operator()(i); }
    const_reference operator[](size_type i) const { return operator()(i); }

    // ── Size / shape ────────────────────────────────────────────────────

    size_type size() const { return size_; }
    size_type stride() const { return stride_; }
    pointer   data() const { return data_; }

    // ── Iterators ───────────────────────────────────────────────────────

    iterator begin() { return {data_, static_cast<std::ptrdiff_t>(stride_)}; }
    iterator end()   { return {data_ + size_ * stride_, static_cast<std::ptrdiff_t>(stride_)}; }

    const_iterator begin() const { return {data_, static_cast<std::ptrdiff_t>(stride_)}; }
    const_iterator end()   const { return {data_ + size_ * stride_, static_cast<std::ptrdiff_t>(stride_)}; }

private:
    pointer   data_;
    size_type size_;
    size_type stride_;
};

// ── Free functions ──────────────────────────────────────────────────────

/// Extract a sub-vector [start, finish) from a strided_vector_ref.
template <typename Value>
strided_vector_ref<Value> sub_vector(strided_vector_ref<Value>& v,
                                     std::size_t start, std::size_t finish) {
    assert(start <= finish && finish <= v.size());
    return strided_vector_ref<Value>(v.data() + start * v.stride(),
                                     finish - start, v.stride());
}

template <typename Value>
strided_vector_ref<const Value> sub_vector(const strided_vector_ref<Value>& v,
                                            std::size_t start, std::size_t finish) {
    assert(start <= finish && finish <= v.size());
    return strided_vector_ref<const Value>(v.data() + start * v.stride(),
                                            finish - start, v.stride());
}

} // namespace mtl::vec

// ── Trait specializations ───────────────────────────────────────────────

namespace mtl::traits {
template <typename Value>
struct category<vec::strided_vector_ref<Value>> { using type = tag::dense; };
} // namespace mtl::traits

namespace mtl::ashape {
template <typename Value>
struct ashape<::mtl::vec::strided_vector_ref<Value>> { using type = cvec<Value>; };
} // namespace mtl::ashape

namespace mtl { using vec::strided_vector_ref; }
