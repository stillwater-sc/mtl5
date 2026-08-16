#include <mtl/mtl.hpp>
#include <iostream>

int main() {
    std::cout << "MTL5 - Matrix Template Library " << mtl::version_string << '\n';
    std::cout << "C++20 header-only linear algebra for mixed-precision computing\n\n";

    // Demonstrate math identities
    std::cout << "math::zero<double>() = " << mtl::math::zero<double>() << '\n';
    std::cout << "math::one<double>()  = " << mtl::math::one<double>()  << '\n';
    std::cout << "math::zero<int>()    = " << mtl::math::zero<int>()    << '\n';
    std::cout << "math::one<int>()     = " << mtl::math::one<int>()     << '\n';

    // Demonstrate compile-time dimensions
    mtl::mat::fixed::dimensions<3, 4> md;
    std::cout << "\nFixed matrix dimensions: " << md.num_rows() << " x " << md.num_cols() << '\n';
	std::cout << "Total size (elements): " << md.size() << '\n';
	std::cout << "Is fixed size? " << std::boolalpha << md.is_fixed << '\n';

    mtl::vec::fixed::dimension<5> vd;
    std::cout << "\nFixed vector dimension: " << vd.size() << '\n';
	std::cout << "Is fixed size? " << std::boolalpha << vd.is_fixed << '\n';

    // Demonstrate concepts (compile-time checks)
    static_assert(mtl::Scalar<float>, "float satisfies Scalar");
    static_assert(mtl::Field<double>,  "double satisfies Field");
    static_assert(mtl::OrderedField<double>, "double satisfies OrderedField");

    // The integers are a RING, not a field: they add and multiply, but only 1
    // and -1 have inverses, and a / b truncates. These assertions used to read
    // the other way round -- and that is exactly what let lu_factor(dense2D<int>)
    // compile and return a truncated factorization.
    static_assert(mtl::Scalar<int>,        "int is a scalar: it adds and multiplies");
    static_assert(!mtl::Field<unsigned>,   "unsigned is not a field: division truncates");
    static_assert(!mtl::OrderedField<int>, "int is not an ordered field either");

    std::cout << "\nAll concepts verified at compile time.\n";

    // Note: The concepts check syntactic requirements (does a / b compile?) but not semantic axioms (is division exact?
	// is addition associative?). See below for a discussion on the mathematical properties of double and the naming of
	// concepts.

    // linear algebra examples
	mtl::vec::dense_vector<int> v1 = {1, 2, 3};
	mtl::vec::dense_vector<double> v2 = {0.5, 1.5, 2.5};
	auto v3 = v1 + v2; // mixed-type vector addition
	std::cout << "\nMixed-type vector addition (int + double):\n";
	std::cout << v3(0) << ' ' << v3(1) << ' ' << v3(2) << '\n';

    mtl::mat::dense2D<int> A = {{1, 2, 3}, {4, 5, 6}};
	mtl::vec::dense_vector<int> x = {1, 0, -1};
	auto y = A * x; // matrix-vector multiplication
	std::cout << "\nMatrix-vector multiplication (A * x):\n";
	std::cout << y(0) << ' ' << y(1) << '\n';
            




    return EXIT_SUCCESS;
}


/*
  A conceptual problem/inaccuracy

  double is not a mathematical field because:
  - Floating-point addition is not associative: (a + b) + c != a + (b + c) in general
  - There is no true additive inverse for NaN/Inf
  - Multiplication doesn't distribute exactly over addition

  A mathematical field requires exact associativity, commutativity, distributivity, and 
  inverses for both addition and multiplication (except division by zero).

  What actually IS a field?

  - Rational numbers (sw::universal::rational, GMP rationals) -- exact arithmetic, true field
  - Finite fields like GF(2), GF(p) -- used in coding theory, cryptography
  - Exact real arithmetic libraries (symbolic computation)

  What double actually is

  double is an approximation of an ordered field. In numerical linear algebra, we use it as if it were a field and
  accept the rounding errors. MTL4 did the same -- it's a pragmatic engineering concept, not a mathematical proof.

  What the concepts actually test

  - Scalar -- has +, -, *, unary -, and T{0}. This is really "arithmetic type" or "ring-like"
  - Field -- Scalar + division + NOT an integral type
  - OrderedField -- Field + std::totally_ordered

  Where the line is drawn, and why

  A purely syntactic Field admitted the integers: int has operator/ and it returns an int, so `a / b` compiles and the
  requirement is met. That made Field<int> true, and every algorithm that says "requires Field" -- meaning "I will
  divide and expect the quotient back" -- accepted a type that cannot give it one. lu_factor(dense2D<int>) compiled and
  returned a truncated factorization: right shape, right type, wrong numbers, no diagnostic.

  So Field now excludes the integral types. That is not a claim that double is a field in the axiomatic sense -- it is
  not, since floating-point addition is not associative and division is inexact. The concepts remain syntactic where
  they can only be syntactic. The line drawn is the one that decides whether an ALGORITHM works:

    integers          a / b truncates       (a/b)*b != a       factorizations silently wrong
    floating point    a / b rounds          (a/b)*b ~= a       factorizations work, with error bounds
    posit / LNS / rational                  same as above      likewise

  Integers keep Scalar, which is the honest statement: they add and multiply, so dot / gemv / gemm are fine on them
  (a ring is all those need). Only the operations that DIVIDE are refused, at compile time, with a diagnostic that
  names the concept.

*/