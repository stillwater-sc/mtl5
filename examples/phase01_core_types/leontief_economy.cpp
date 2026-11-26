// leontief_economy.cpp -- Example 1A: Leontief's input-output economy
//
// Wassily Leontief described the American economy of 1919 and 1929 as a table
// of how much each industry buys from every other ("The Structure of American
// Economy", 1941), work that won him the 1973 Nobel prize in economics. The
// table is a matrix A: a_ij is what sector j consumes of sector i's output to
// produce one unit of its own. To deliver a final demand d to consumers, every
// sector must also make what the others use up making it:
//
//     x = d + A x            the Leontief equations, (I - A) x = d
//
// Solving them is Phase 3 and 5 material. Here they are solved the way
// Leontief explained them, as rounds of production: deliver d, then make the
// inputs A d that took, then the inputs A(A d) those took, and so on,
//
//     x = d + A d + A^2 d + A^3 d + ...
//
// which converges because every column of A sums to less than one (each
// sector adds value). The example shows the containers that hold the problem:
//   1. dense2D and dense_vector from initializer lists
//   2. element access A(i, j), num_rows(), num_cols(), size()
//   3. value semantics: a copy is a deep copy
//   4. row-major and column-major storage of the same matrix
//   5. the two operations the rounds need, A * v and +=
//
// The five-sector economy below is illustrative, not Leontief's table.
//
// Run with --trace <file> to write the rounds for the docs-site animation.

#include <mtl/mtl.hpp>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../common/trace.hpp"

using namespace mtl;

namespace {

template <typename Vector>
std::vector<double> to_std(const Vector& v) {
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = v(i);
    return out;
}

} // namespace

int main(int argc, char** argv) {
    example_trace::trace trace(argc, argv, "1A Leontief input-output economy");

    std::cout << "=============================================================\n";
    std::cout << " Phase 1A: Leontief's Input-Output Economy\n";
    std::cout << "=============================================================\n\n";

    // -- The containers ------------------------------------------------------
    // dense2D from a nested initializer list: one inner list per row. Column j
    // is what sector j buys, per unit it produces, from each sector i.
    const std::vector<std::string> sector = {"Agriculture", "Mining", "Manufacturing",
                                             "Energy", "Services"};
    mat::dense2D<double> A = {
        {0.10, 0.02, 0.15, 0.01, 0.03},
        {0.01, 0.05, 0.10, 0.20, 0.01},
        {0.12, 0.15, 0.20, 0.10, 0.08},
        {0.05, 0.15, 0.10, 0.10, 0.05},
        {0.08, 0.10, 0.12, 0.08, 0.15},
    };
    // Final demand: what households, government and exports buy (billions).
    vec::dense_vector<double> d = {50.0, 10.0, 120.0, 40.0, 200.0};

    const std::size_t n = A.num_rows();
    std::cout << "Technical coefficients A (" << A.num_rows() << " x " << A.num_cols()
              << ", " << A.size() << " entries):\n";
    std::cout << std::setw(16) << "";
    for (std::size_t j = 0; j < n; ++j) std::cout << std::setw(8) << sector[j].substr(0, 6);
    std::cout << '\n';
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << std::setw(16) << std::left << sector[i] << std::right;
        for (std::size_t j = 0; j < n; ++j)
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << A(i, j);
        std::cout << '\n';
    }

    // Column sums below one: each sector buys less than a unit of inputs per
    // unit of output. That is what makes the rounds below converge.
    std::cout << "\nInputs per unit of output (column sums of A):\n";
    double max_colsum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += A(i, j);
        max_colsum = std::max(max_colsum, s);
        std::cout << "  " << std::setw(14) << std::left << sector[j] << std::right
                  << std::setprecision(2) << s << '\n';
    }

    // -- Value semantics ---------------------------------------------------
    // A copy owns its own elements: changing it leaves the original alone.
    auto d_copy = d;
    d_copy(0) = 0.0;
    std::cout << "\nValue semantics: after d_copy(0) = 0, d(0) is still "
              << std::setprecision(1) << d(0) << ".\n";

    // -- Storage order -----------------------------------------------------
    // The same matrix in column-major storage. A(i, j) means the same entry;
    // only the order of the elements in memory differs.
    mat::dense2D<double, mat::parameters<tag::col_major>> A_col(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A_col(i, j) = A(i, j);
    std::cout << std::setprecision(2) << "\nFirst six elements in memory:\n  row-major:    ";
    for (std::size_t k = 0; k < 6; ++k) std::cout << A.data()[k] << ' ';
    std::cout << "(row 0, then row 1)\n  column-major: ";
    for (std::size_t k = 0; k < 6; ++k) std::cout << A_col.data()[k] << ' ';
    std::cout << "(column 0, then column 1)\n";

    // -- Rounds of production ----------------------------------------------
    // round_0 = d; round_k = A * round_{k-1}; x = sum of the rounds.
    std::cout << "\n--- Rounds of production: x = d + A d + A^2 d + ... ---\n\n";
    std::cout << std::setw(7) << "round" << std::setw(14) << "this round"
              << std::setw(14) << "total x" << "   ||d-(I-A)x||/||d||\n";

    vec::dense_vector<double> round = d;   // A^k d
    vec::dense_vector<double> x = d;       // running total
    std::vector<std::vector<double>> rounds{to_std(round)};
    std::vector<double> residuals;

    auto residual = [&](const vec::dense_vector<double>& xv) {
        vec::dense_vector<double> r(d - xv + A * xv);   // d - (I - A) x
        return two_norm(r) / two_norm(d);
    };

    auto sum = [](const vec::dense_vector<double>& v) {
        double s = 0.0;
        for (std::size_t i = 0; i < v.size(); ++i) s += v(i);
        return s;
    };

    residuals.push_back(residual(x));
    std::cout << std::setw(7) << 0 << std::setw(14) << std::setprecision(2) << sum(round)
              << std::setw(14) << sum(x) << std::setw(14) << std::scientific
              << std::setprecision(2) << residuals.back() << std::fixed << '\n';

    const int max_rounds = 60;
    const double tol = 1.0e-10;
    int rounds_taken = 0;
    for (int k = 1; k <= max_rounds; ++k) {
        round = vec::dense_vector<double>(A * round);
        x += round;
        rounds.push_back(to_std(round));
        residuals.push_back(residual(x));
        rounds_taken = k;
        if (k <= 10 || residuals.back() < tol)
            std::cout << std::setw(7) << k << std::setw(14) << std::setprecision(2)
                      << sum(round) << std::setw(14) << sum(x) << std::setw(14)
                      << std::scientific << std::setprecision(2) << residuals.back()
                      << std::fixed << '\n';
        if (residuals.back() < tol) break;
    }

    std::cout << "\nGross output needed per sector:\n";
    for (std::size_t i = 0; i < n; ++i)
        std::cout << "  " << std::setw(14) << std::left << sector[i] << std::right
                  << std::setw(9) << std::setprecision(1) << x(i) << "   (demand "
                  << d(i) << ", multiplier " << std::setprecision(2) << x(i) / d(i)
                  << ")\n";
    const double multiplier = sum(x) / sum(d);
    std::cout << "\nEvery dollar of final demand needs " << std::setprecision(2)
              << multiplier << " dollars of gross output.\n";
    std::cout << "Rounds to reach a residual of 1e-10: " << rounds_taken << "\n\n";

    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. dense2D and dense_vector own their elements; copies are deep.\n";
    std::cout << "2. A(i, j) is the same entry in row- and column-major storage;\n";
    std::cout << "   the orientation only decides the order in memory.\n";
    std::cout << "3. Two operations, A * v and +=, are enough to solve (I - A) x = d\n";
    std::cout << "   as a series, because the column sums of A are below one.\n";
    std::cout << "4. Each round shrinks by about the spectral radius of A, so the\n";
    std::cout << "   residual falls geometrically: a first look at Phase 3's iterations.\n";

    if (trace) {
        auto names = example_trace::value::array();
        for (const auto& s : sector) names.push_back(s);
        trace["sectors"] = names;
        auto& rows = trace["A"];
        rows = example_trace::value::array();
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<double> r(n);
            for (std::size_t j = 0; j < n; ++j) r[j] = A(i, j);
            rows.push_back(example_trace::value::numbers(r, 6));
        }
        trace["demand"] = example_trace::value::numbers(to_std(d), 8);
        auto& rv = trace["rounds"];
        rv = example_trace::value::array();
        for (const auto& r : rounds) rv.push_back(example_trace::value::numbers(r, 8));
        trace["residual"] = example_trace::value::numbers(residuals, 6);
        trace["output"] = example_trace::value::numbers(to_std(x), 10);
        trace["multiplier"] = example_trace::value::rounded(multiplier, 6);
        trace["max_column_sum"] = example_trace::value::rounded(max_colsum, 6);
        trace["tolerance"] = tol;
    }
    return EXIT_SUCCESS;
}
