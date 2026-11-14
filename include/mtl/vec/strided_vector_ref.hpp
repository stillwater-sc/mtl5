#pragma once
// MTL5 -- Non-owning strided vector reference
// Primary use: extract columns from row-major dense2D without copy.
// Simplified from MTL4's 275-line CRTP version: no ownership, no clone, no CRTP.
#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>

namespace mtl::vec {

/// Non-owning reference to strided data: pointer + length + stride.
/// Element i lives at data_[i * stride_]. Stride 0 is allowed: every element
/// is data_[0], as for a broadcast scalar.
template <typename Value>
class strided_vector_ref {
public:
    using value_type      = Value;
    using size_type       = std::size_t;
    using reference       = Value&;
    using const_reference = const Value&;
    using pointer         = Value*;
    using const_pointer   = const Value*;

    // -- Strided iterator ------------------------------------------------
    // A base pointer and an element index, not a moving pointer (#830): end()
    // of a column c > 0 of a row-major matrix would otherwise point
    // size * stride past the base, beyond one past the end of the storage,
    // which is undefined even if never dereferenced. A pointer is formed only
    // when an element is read. The distance is the index difference, so it is
    // defined for stride 0 too (every element is the same one).
    template <typename Ptr>
    class strided_iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = std::remove_cv_t<Value>;
        using difference_type   = std::ptrdiff_t;
        using pointer           = Ptr;
        using reference         = decltype(*std::declval<Ptr>());

        strided_iterator() : base_(nullptr), idx_(0), stride_(1) {}
        strided_iterator(Ptr base, difference_type idx, difference_type stride)
            : base_(base), idx_(idx), stride_(stride) {}

        reference operator*() const { return base_[idx_ * stride_]; }
        pointer operator->() const { return base_ + idx_ * stride_; }
        reference operator[](difference_type n) const { return base_[(idx_ + n) * stride_]; }

        strided_iterator& operator++() { ++idx_; return *this; }
        strided_iterator  operator++(int) { auto t = *this; ++idx_; return t; }
        strided_iterator& operator--() { --idx_; return *this; }
        strided_iterator  operator--(int) { auto t = *this; --idx_; return t; }

        strided_iterator& operator+=(difference_type n) { idx_ += n; return *this; }
        strided_iterator& operator-=(difference_type n) { idx_ -= n; return *this; }

        friend strided_iterator operator+(strided_iterator it, difference_type n) { it += n; return it; }
        friend strided_iterator operator+(difference_type n, strided_iterator it) { it += n; return it; }
        friend strided_iterator operator-(strided_iterator it, difference_type n) { it -= n; return it; }
        friend difference_type operator-(const strided_iterator& a, const strided_iterator& b) {
            return a.idx_ - b.idx_;
        }

        friend bool operator==(const strided_iterator& a, const strided_iterator& b) { return a.idx_ == b.idx_; }
        friend bool operator!=(const strided_iterator& a, const strided_iterator& b) { return a.idx_ != b.idx_; }
        friend bool operator< (const strided_iterator& a, const strided_iterator& b) { return a.idx_ <  b.idx_; }
        friend bool operator<=(const strided_iterator& a, const strided_iterator& b) { return a.idx_ <= b.idx_; }
        friend bool operator> (const strided_iterator& a, const strided_iterator& b) { return a.idx_ >  b.idx_; }
        friend bool operator>=(const strided_iterator& a, const strided_iterator& b) { return a.idx_ >= b.idx_; }

    private:
        Ptr base_;
        difference_type idx_;
        difference_type stride_;
    };

    using iterator       = strided_iterator<pointer>;
    using const_iterator = strided_iterator<const_pointer>;

    // -- Constructors ----------------------------------------------------

    /// Construct from raw pointer, length, and stride.
    strided_vector_ref(pointer data, size_type length, size_type stride)
        : data_(data), size_(length), stride_(stride) {}

    // -- Element access --------------------------------------------------

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

    // -- Size / shape ----------------------------------------------------

    size_type size() const { return size_; }
    size_type stride() const { return stride_; }
    pointer   data() const { return data_; }

    // -- Iterators -------------------------------------------------------

    iterator begin() { return {data_, 0, sstride()}; }
    iterator end()   { return {data_, ssize(), sstride()}; }

    const_iterator begin() const { return {data_, 0, sstride()}; }
    const_iterator end()   const { return {data_, ssize(), sstride()}; }

private:
    std::ptrdiff_t ssize()   const { return static_cast<std::ptrdiff_t>(size_); }
    std::ptrdiff_t sstride() const { return static_cast<std::ptrdiff_t>(stride_); }

    pointer   data_;
    size_type size_;
    size_type stride_;
};

// -- Free functions ------------------------------------------------------

/// Extract a sub-vector [start, finish) from a strided_vector_ref.
template <typename Value>
strided_vector_ref<Value> sub_vector(strided_vector_ref<Value>& v,
                                     std::size_t start, std::size_t finish) {
    assert(start <= finish && finish <= v.size());
    // An empty range keeps the base pointer: start * stride may lie beyond
    // one past the end of the storage, and the view never reads it.
    Value* base = start == finish ? v.data() : v.data() + start * v.stride();
    return strided_vector_ref<Value>(base, finish - start, v.stride());
}

template <typename Value>
strided_vector_ref<const Value> sub_vector(const strided_vector_ref<Value>& v,
                                            std::size_t start, std::size_t finish) {
    assert(start <= finish && finish <= v.size());
    const Value* base = start == finish ? v.data() : v.data() + start * v.stride();
    return strided_vector_ref<const Value>(base, finish - start, v.stride());
}

} // namespace mtl::vec

// -- Trait specializations -----------------------------------------------

namespace mtl::traits {
template <typename Value>
struct category<vec::strided_vector_ref<Value>> { using type = tag::dense; };
} // namespace mtl::traits

namespace mtl::ashape {
template <typename Value>
struct ashape<::mtl::vec::strided_vector_ref<Value>> { using type = cvec<Value>; };
} // namespace mtl::ashape

namespace mtl { using vec::strided_vector_ref; }
