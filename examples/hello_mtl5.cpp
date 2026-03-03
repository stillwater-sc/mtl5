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
	static_assert(mtl::Field<unsigned>, "unsigned satisfies Field");
    static_assert(mtl::Field<double>,  "double satisfies Field");
    static_assert(mtl::OrderedField<int>, "int satisfies OrderedField");
	static_assert(mtl::OrderedField<double>, "double satisfies OrderedField");

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

  - Rational numbers (sw::universal::rational, GMP rationals) — exact arithmetic, true field
  - Finite fields like GF(2), GF(p) — used in coding theory, cryptography
  - Exact real arithmetic libraries (symbolic computation)

  What double actually is

  double is an approximation of an ordered field. In numerical linear algebra, we use it as if it were a field and
  accept the rounding errors. MTL4 did the same — it's a pragmatic engineering concept, not a mathematical proof.

  What the concepts actually test

  - Scalar — has +, -, *, unary -, and T{0}. This is really "arithmetic type" or "ring-like"
  - Field — Scalar + has /. This is really "arithmetic type with division" — a division ring at best
  - OrderedField — Field + std::totally_ordered. This is really "comparable arithmetic type with division"

  So what's wrong?

  The names oversell the mathematical guarantees. The concepts only check syntactic requirements (does a / b compile?)
  not semantic axioms (is division exact? is addition associative?). This is fine — C++20 concepts are inherently
  syntactic. But the names Field and OrderedField imply mathematical properties that double doesn't satisfy.

  Options:
  1. Accept the naming convention — MTL4 and most linear algebra libraries do this. It's understood that "Field" means
  "type with division" in the C++ template library context
  2. Rename to be more honest — DivisionType, OrderedDivisionType, or ArithmeticWithDivision
  3. Document the gap — keep the names but document that they're syntactic concepts, not mathematical field axioms

  Most numerical libraries (Eigen, Blaze, MTL4) take option 1. The naming convention is understood by the community,
  even if mathematically imprecise.

*/