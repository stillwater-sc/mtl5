// zero_copy_rules.cpp - The contract of an MTL5 vector or matrix over your buffer
//
// ode_rhs_callback.cpp and shared_frame_pipeline.cpp show WHY you would lay an
// MTL5 type over memory you own. This program walks through the rules that
// make it safe, one short demonstration each, and checks every claim it makes.
//
//   1. Assignment writes through to your buffer.
//   2. The buffer fixes the size: a source of another size is an error.
//   3. An explicit resize detaches.
//   4. Copying a mutable vector copies; keep the conduit with a reference.
//      (Copying a read-only view gives another view: see 5.)
//   5. Read-only buffers get read-only views; writes don't compile.
//   6. Overlap and aliasing that MTL5 handles, and what it can't.
//   7. Layout: dense2D needs an unpadded buffer; use ndarray strides otherwise.
//   8. Lifetime and synchronization are yours.

#include <mtl/mtl.hpp>
#include <mtl/array/ndarray.hpp>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>

using namespace mtl;

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    std::cout << (ok ? "  [ok]   " : "  [FAIL] ") << what << "\n";
    if (!ok) ++failures;
}
void section(const char* title) { std::cout << "\n" << title << "\n"; }

vec::dense_vector<double> make_vector(std::size_t n, double v) { return vec::dense_vector<double>(n, v); }

template <typename V> concept writable = requires(V v) { v(0) = 1.0; };
template <typename M> concept matrix_writable = requires(M m) { m(0, 0) = 1.0; };

} // namespace

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Zero copy: the rules of an MTL5 type over your buffer\n";
    std::cout << "=============================================================\n";

    section("1. Assignment writes through to the buffer");
    {
        double buf[3] = {0, 0, 0};
        vec::dense_vector<double> v(3, buf);          // v is a conduit to buf
        const vec::dense_vector<double> w{1, 2, 3};
        v = w;                                         // copy assignment
        check(buf[2] == 3.0 && v.data() == buf, "v = w writes buf");
        v = make_vector(3, 7.0);                       // move assignment
        check(buf[0] == 7.0 && v.data() == buf, "v = f() writes buf");
        v = w + w;                                     // expression assignment
        check(buf[1] == 4.0 && v.data() == buf, "v = w + w writes buf");
        v *= 0.5;
        check(buf[1] == 2.0, "compound assignment writes buf");
        check(!v.owns_memory(), "v never owns its elements");
    }

    section("2. The buffer fixes the size");
    {
        double buf[3] = {1, 2, 3};
        vec::dense_vector<double> v(3, buf);
        const vec::dense_vector<double> big(5, 9.0);
        bool threw = false;
        try { v = big; } catch (const std::domain_error&) { threw = true; }
        check(threw, "assigning a 5-vector to a 3-element buffer throws std::domain_error");
        threw = false;
        try { v = big + big; } catch (const std::domain_error&) { threw = true; }
        check(threw, "so does an expression of another size");
        check(v.data() == buf && buf[0] == 1.0, "and the buffer is untouched");
        // Move-assigning a temporary of another size is noexcept, so it asserts
        // in a debug build instead of throwing. Size your sources to the buffer.
    }

    section("3. An explicit resize detaches");
    {
        double buf[3] = {1, 2, 3};
        vec::dense_vector<double> v(3, buf);
        v.change_dim(6);                               // asked for: new, owned storage
        check(v.owns_memory() && v.data() != buf, "change_dim to another size gives v its own storage");
        v(0) = 100.0;
        check(buf[0] == 1.0, "and writes no longer reach buf");
    }

    section("4. Copying a mutable vector copies");
    {
        double buf[2] = {1, 2};
        vec::dense_vector<double> v(2, buf);
        auto copy = v;                                 // a new, owning vector
        copy(0) = 50.0;
        check(copy.owns_memory() && buf[0] == 1.0, "auto copy = v copies; writing it leaves buf alone");
        auto& ref = v;                                 // keep the conduit: a reference...
        ref(0) = 10.0;
        vec::dense_vector<double> second(2, buf);      // ...or a second wrapper
        second(1) = 20.0;
        check(buf[0] == 10.0 && buf[1] == 20.0, "a reference or a second wrapper writes buf");
        // Pass a conduit to functions by reference: f(vec::dense_vector<double>& v).
    }

    section("5. Read-only buffers get read-only views");
    {
        const double cbuf[4] = {1, 2, 3, 4};
        const auto y = vec::const_view(4, cbuf);       // no const_cast
        const auto A = mat::const_view(2, 2, cbuf);
        check(y.data() == cbuf && A.data() == cbuf, "const_view reads the buffer in place");
        check(two_norm(y) > 5.4 && A(1, 0) == 3.0, "and works with the operations");
        static_assert(!writable<vec::const_vector_view<double>>);
        static_assert(!matrix_writable<mat::const_matrix_view<double>>);
        std::cout << "  [ok]   y(0) = 1 and A(0, 0) = 1 do not compile (static_assert)\n";
        const auto y2 = y;                             // the exception to rule 4: a copy of a
        check(y2.data() == cbuf, "copying a read-only view gives another view of the buffer");
        //                                                read-only view is another view, so
        //                                                it too must not outlive the buffer
        const vec::dense_vector<double> owned(y);      // an owning copy when you need one
        check(owned.owns_memory() && owned(3) == 4.0, "an owning copy of a view can be changed");
    }

    section("6. Overlap and aliasing");
    {
        // Two wrappers over overlapping parts of one buffer: handled.
        double buf[5] = {1, 2, 3, 4, 5};
        vec::dense_vector<double> lo(4, buf), hi(4, buf + 1);
        hi = lo;                                       // shift right by one
        check(buf[1] == 1 && buf[2] == 2 && buf[3] == 3 && buf[4] == 4, "overlapping wrappers copy safely");

        // An expression that reads the target elsewhere: evaluated via a temporary.
        double m[4] = {1, 2, 3, 4};
        mat::dense2D<double> E(2, 2, m);
        E = E + trans(E);
        check(m[1] == 5.0 && m[2] == 5.0 && E.data() == m, "E = E + trans(E) is right, and stays in m");
        // What MTL5 cannot see: another thread writing the buffer during an
        // expression. That is a data race; rule 8 is how you avoid it.
    }

    section("7. Layout: dense2D needs an unpadded buffer");
    {
        // A 2 x 3 image whose rows are padded to a pitch of 4 elements, as
        // image and DMA buffers often are. 'x' marks the padding.
        const double x = -1.0;
        double img[8] = {1, 2, 3, x,
                         4, 5, 6, x};
        mat::dense2D<double> wrong(2, 3, img);         // assumes a pitch of 3
        check(wrong(1, 0) == x, "dense2D over a padded buffer reads padding for (1, 0)");
        array::ndarray<double, 2> right(img, {2, 3}, {4, 1});   // strides in elements
        check(right(1, 0) == 4.0 && right(1, 2) == 6.0, "an ndarray with the real pitch reads it right");
        right(1, 1) = 50.0;
        check(img[5] == 50.0, "and writes through to the image");
    }

    section("8. Lifetime and synchronization are yours");
    std::cout << "  MTL5 never frees memory it did not allocate, so the buffer must\n"
                 "  outlive every vector or matrix over it. MTL5 adds no locking: hand\n"
                 "  a buffer between threads as shared_frame_pipeline.cpp does.\n";

    std::cout << "\n" << (failures ? "FAILED" : "All checks passed") << "\n";
    return failures ? 1 : 0;
}
