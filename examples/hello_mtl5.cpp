#include <mtl/mtl.hpp>
#include <iostream>

int main() {
    std::cout << "MTL5 — Matrix Template Library " << mtl::version_string << '\n';
    std::cout << "C++20 header-only linear algebra for mixed-precision computing\n\n";

    // Demonstrate math identities
    std::cout << "math::zero<double>() = " << mtl::math::zero<double>() << '\n';
    std::cout << "math::one<double>()  = " << mtl::math::one<double>()  << '\n';
    std::cout << "math::zero<int>()    = " << mtl::math::zero<int>()    << '\n';
    std::cout << "math::one<int>()     = " << mtl::math::one<int>()     << '\n';

    // Demonstrate compile-time dimensions
    mtl::mat::fixed::dimensions<3, 4> md;
    std::cout << "\nFixed matrix dimensions: " << md.num_rows() << " x " << md.num_cols() << '\n';

    mtl::vec::fixed::dimension<5> vd;
    std::cout << "Fixed vector dimension: " << vd.size() << '\n';

    // Demonstrate concepts (compile-time checks)
    static_assert(mtl::Scalar<double>, "double satisfies Scalar");
    static_assert(mtl::Field<double>,  "double satisfies Field");
    static_assert(mtl::OrderedField<double>, "double satisfies OrderedField");

    std::cout << "\nAll concepts verified at compile time.\n";
    return 0;
}
