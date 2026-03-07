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
