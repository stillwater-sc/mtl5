#include <catch2/catch_test_macros.hpp>
#include <mtl/detail/contiguous_memory_block.hpp>
#include <mtl/tag/storage.hpp>
#include <algorithm>
#include <numeric>

using namespace mtl::detail;

// -- Heap tests ----------------------------------------------------------

TEST_CASE("Heap block default construction", "[detail][memory]") {
    contiguous_memory_block<double, mtl::tag::on_heap> block;
    REQUIRE(block.size() == 0);
    REQUIRE(block.empty());
    REQUIRE(block.data() == nullptr);
    REQUIRE(block.category() == memory_category::own);
}

TEST_CASE("Heap block size construction", "[detail][memory]") {
    contiguous_memory_block<double, mtl::tag::on_heap> block(10);
    REQUIRE(block.size() == 10);
    REQUIRE_FALSE(block.empty());
    REQUIRE(block.data() != nullptr);
    REQUIRE(block.category() == memory_category::own);

    // Should be zero-initialized
    for (std::size_t i = 0; i < block.size(); ++i) {
        REQUIRE(block[i] == 0.0);
    }
}

TEST_CASE("Heap block copy construction", "[detail][memory]") {
    contiguous_memory_block<int, mtl::tag::on_heap> original(5);
    for (std::size_t i = 0; i < 5; ++i) original[i] = static_cast<int>(i * 10);

    contiguous_memory_block<int, mtl::tag::on_heap> copy(original);
    REQUIRE(copy.size() == 5);
    REQUIRE(copy.data() != original.data());  // deep copy
    for (std::size_t i = 0; i < 5; ++i) {
        REQUIRE(copy[i] == static_cast<int>(i * 10));
    }
}

TEST_CASE("Heap block move construction", "[detail][memory]") {
    contiguous_memory_block<int, mtl::tag::on_heap> original(5);
    for (std::size_t i = 0; i < 5; ++i) original[i] = static_cast<int>(i);

    auto* ptr = original.data();
    contiguous_memory_block<int, mtl::tag::on_heap> moved(std::move(original));

    REQUIRE(moved.size() == 5);
    REQUIRE(moved.data() == ptr);  // pointer transferred
    REQUIRE(original.size() == 0);
    REQUIRE(original.data() == nullptr);
}

TEST_CASE("Heap block copy assignment", "[detail][memory]") {
    contiguous_memory_block<int, mtl::tag::on_heap> a(3);
    a[0] = 1; a[1] = 2; a[2] = 3;

    contiguous_memory_block<int, mtl::tag::on_heap> b;
    b = a;
    REQUIRE(b.size() == 3);
    REQUIRE(b[0] == 1);
    REQUIRE(b[1] == 2);
    REQUIRE(b[2] == 3);
    REQUIRE(b.data() != a.data());
}

TEST_CASE("Heap block move assignment", "[detail][memory]") {
    contiguous_memory_block<int, mtl::tag::on_heap> a(3);
    a[0] = 10; a[1] = 20; a[2] = 30;
    auto* ptr = a.data();

    contiguous_memory_block<int, mtl::tag::on_heap> b;
    b = std::move(a);
    REQUIRE(b.size() == 3);
    REQUIRE(b.data() == ptr);
    REQUIRE(b[0] == 10);
    REQUIRE(a.size() == 0);
    REQUIRE(a.data() == nullptr);
}

TEST_CASE("Heap block external (non-owning)", "[detail][memory]") {
    int data[4] = {100, 200, 300, 400};
    contiguous_memory_block<int, mtl::tag::on_heap> block(data, 4, false);
    REQUIRE(block.size() == 4);
    REQUIRE(block.data() == data);
    REQUIRE(block.category() == memory_category::external);
    REQUIRE(block[2] == 300);
}

TEST_CASE("Heap block view", "[detail][memory]") {
    int data[3] = {7, 8, 9};
    contiguous_memory_block<int, mtl::tag::on_heap> block(data, 3, true);
    REQUIRE(block.category() == memory_category::view);
    REQUIRE(block.data() == data);
    REQUIRE(block[0] == 7);
}

TEST_CASE("Heap block realloc", "[detail][memory]") {
    contiguous_memory_block<double, mtl::tag::on_heap> block(5);
    REQUIRE(block.size() == 5);

    block.realloc(10);
    REQUIRE(block.size() == 10);

    block.realloc(0);
    REQUIRE(block.size() == 0);
    REQUIRE(block.empty());
}

TEST_CASE("Heap block iteration", "[detail][memory]") {
    contiguous_memory_block<int, mtl::tag::on_heap> block(5);
    std::iota(block.begin(), block.end(), 1);

    REQUIRE(block[0] == 1);
    REQUIRE(block[4] == 5);

    int sum = 0;
    for (auto it = block.begin(); it != block.end(); ++it)
        sum += *it;
    REQUIRE(sum == 15);
}

TEST_CASE("Heap block swap", "[detail][memory]") {
    contiguous_memory_block<int, mtl::tag::on_heap> a(3);
    a[0] = 1; a[1] = 2; a[2] = 3;

    contiguous_memory_block<int, mtl::tag::on_heap> b(2);
    b[0] = 10; b[1] = 20;

    a.swap(b);
    REQUIRE(a.size() == 2);
    REQUIRE(a[0] == 10);
    REQUIRE(b.size() == 3);
    REQUIRE(b[0] == 1);
}

// -- Stack tests ---------------------------------------------------------

TEST_CASE("Stack block default construction", "[detail][memory][stack]") {
    contiguous_memory_block<double, mtl::tag::on_stack, 4> block;
    REQUIRE(block.size() == 4);
    REQUIRE_FALSE(block.empty());
    REQUIRE(block.data() != nullptr);
    REQUIRE(block.category() == memory_category::own);

    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(block[i] == 0.0);
    }
}

TEST_CASE("Stack block copy construction", "[detail][memory][stack]") {
    contiguous_memory_block<int, mtl::tag::on_stack, 3> original;
    original[0] = 10; original[1] = 20; original[2] = 30;

    auto copy = original;
    REQUIRE(copy.size() == 3);
    REQUIRE(copy[0] == 10);
    REQUIRE(copy[1] == 20);
    REQUIRE(copy[2] == 30);
    REQUIRE(copy.data() != original.data());  // separate storage
}

TEST_CASE("Stack block move construction", "[detail][memory][stack]") {
    contiguous_memory_block<int, mtl::tag::on_stack, 3> original;
    original[0] = 5; original[1] = 6; original[2] = 7;

    contiguous_memory_block<int, mtl::tag::on_stack, 3> moved(std::move(original));
    REQUIRE(moved[0] == 5);
    REQUIRE(moved[1] == 6);
    REQUIRE(moved[2] == 7);
}

TEST_CASE("Stack block swap", "[detail][memory][stack]") {
    contiguous_memory_block<int, mtl::tag::on_stack, 3> a;
    a[0] = 1; a[1] = 2; a[2] = 3;

    contiguous_memory_block<int, mtl::tag::on_stack, 3> b;
    b[0] = 10; b[1] = 20; b[2] = 30;

    a.swap(b);
    REQUIRE(a[0] == 10);
    REQUIRE(a[2] == 30);
    REQUIRE(b[0] == 1);
    REQUIRE(b[2] == 3);
}

TEST_CASE("Stack block iteration", "[detail][memory][stack]") {
    contiguous_memory_block<int, mtl::tag::on_stack, 4> block;
    std::iota(block.begin(), block.end(), 100);

    REQUIRE(block[0] == 100);
    REQUIRE(block[3] == 103);
    REQUIRE(std::distance(block.begin(), block.end()) == 4);
}

// -- Assignment to non-owning blocks (#829) -------------------------------
// Assignment writes through to the caller's buffer and never re-points it.
// These fail before #829: the block detached into a private copy.

#include <stdexcept>

namespace {
using heap_block = contiguous_memory_block<double, mtl::tag::on_heap>;
heap_block filled(std::size_t n, double first) {
    heap_block b(n);
    for (std::size_t i = 0; i < n; ++i) b[i] = first + double(i);
    return b;
}
} // namespace

TEST_CASE("Copy assignment writes through an external or view block", "[detail][memory][829]") {
    for (bool is_view : {false, true}) {
        double ext[4] = {0, 0, 0, 0};
        heap_block v(ext, 4, is_view);
        const auto expected = is_view ? memory_category::view : memory_category::external;
        const heap_block s = filled(4, 10.0);
        v = s;
        REQUIRE(v.data() == ext);
        REQUIRE(v.category() == expected);
        REQUIRE(ext[0] == 10.0);
        REQUIRE(ext[3] == 13.0);
        REQUIRE(s[0] == 10.0);   // the source is untouched
    }
}

TEST_CASE("Copy assignment of another size to a non-owning block throws", "[detail][memory][829]") {
    double ext[4] = {1, 2, 3, 4};
    heap_block v(ext, 4, false);
    const heap_block s = filled(3, 10.0);
    REQUIRE_THROWS_AS(v = s, std::domain_error);
    REQUIRE(v.data() == ext);
    REQUIRE(v.size() == 4);
    REQUIRE(ext[0] == 1.0);
}

TEST_CASE("Move assignment writes through an external or view block", "[detail][memory][829]") {
    for (bool is_view : {false, true}) {
        double ext[3] = {0, 0, 0};
        heap_block v(ext, 3, is_view);
        heap_block s = filled(3, 5.0);
        v = std::move(s);
        REQUIRE(v.data() == ext);
        REQUIRE(v.category() != memory_category::own);
        REQUIRE(ext[0] == 5.0);
        REQUIRE(ext[2] == 7.0);
        REQUIRE(s.size() == 3);   // the source keeps its storage
    }
}

TEST_CASE("Assignment and realloc agree on a non-owning block of the same size", "[detail][memory][829]") {
    double ext[2] = {1, 2};
    heap_block v(ext, 2, false);
    v.realloc(2);
    REQUIRE(v.data() == ext);
    v = filled(2, 8.0);
    REQUIRE(v.data() == ext);
    REQUIRE(v.category() == memory_category::external);
    // An explicit resize to another size still detaches.
    v.realloc(5);
    REQUIRE(v.data() != ext);
    REQUIRE(v.category() == memory_category::own);
    REQUIRE(ext[0] == 8.0);
}

TEST_CASE("Assignment to an owning block replaces it", "[detail][memory][829]") {
    heap_block a = filled(2, 1.0);
    const heap_block b = filled(5, 20.0);
    a = b;
    REQUIRE(a.size() == 5);
    REQUIRE(a.data() != b.data());
    REQUIRE(a[4] == 24.0);
    heap_block c;
    c = filled(3, 0.0);
    REQUIRE(c.size() == 3);
    REQUIRE(c.category() == memory_category::own);
}

TEST_CASE("Block self-assignment", "[detail][memory][829]") {
    heap_block a = filled(3, 1.0);
    auto& ra = a;
    a = ra;
    REQUIRE(a[2] == 3.0);
    a = std::move(ra);
    REQUIRE(a.size() == 3);
    REQUIRE(a[2] == 3.0);

    double ext[2] = {4, 5};
    heap_block v(ext, 2, true);
    auto& rv = v;
    v = rv;
    REQUIRE(v.data() == ext);
    REQUIRE(ext[1] == 5.0);
}

TEST_CASE("Copy and move of an external block", "[detail][memory][829]") {
    double ext[2] = {4, 5};
    heap_block v(ext, 2, false);
    heap_block c(v);   // a new object: a deep, owning copy
    REQUIRE(c.data() != ext);
    REQUIRE(c.category() == memory_category::own);
    REQUIRE(c[1] == 5.0);
    heap_block m(std::move(v));   // moves the reference, not the elements
    REQUIRE(m.data() == ext);
    REQUIRE(m.category() == memory_category::external);
    REQUIRE(v.size() == 0);
}

TEST_CASE("Fixed-size heap block always holds its elements", "[detail][memory][829]") {
    using fixed_block = contiguous_memory_block<int, mtl::tag::on_heap, 3>;
    fixed_block a;
    REQUIRE(a.size() == 3);
    REQUIRE(a.data() != nullptr);
    REQUIRE(a[2] == 0);
    a[0] = 7;
    fixed_block b(std::move(a));
    REQUIRE(b[0] == 7);
    REQUIRE(a.size() == 3);      // fresh storage, not a null pointer
    REQUIRE(a.data() != nullptr);
    a[1] = 9;
    fixed_block c;
    c = std::move(a);
    REQUIRE(c[1] == 9);
    REQUIRE(a.size() == 3);
    REQUIRE(a.data() != nullptr);
}

TEST_CASE("Stack block copy and move assignment", "[detail][memory][stack][829]") {
    contiguous_memory_block<int, mtl::tag::on_stack, 3> a, b;
    for (int i = 0; i < 3; ++i) a[i] = i + 1;
    b = a;
    REQUIRE(b[2] == 3);
    a[0] = 42;
    REQUIRE(b[0] == 1);
    b = std::move(a);
    REQUIRE(b[0] == 42);
    auto& rb = b;
    b = rb;
    REQUIRE(b[0] == 42);
}
