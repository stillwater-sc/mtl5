// durer_magic_square.cpp -- Example 2A: Durer's magic square, through views
//
// In the engraving Melencolia I (1514), Albrecht Durer hung a 4 x 4 magic
// square on the wall behind the angel:
//
//     16  3  2 13
//      5 10 11  8
//      9  6  7 12
//      4 15 14  1
//
// Every row, column and diagonal sums to 34, and so do the four 2 x 2
// quadrants, the centre block and the four corners. The bottom row's middle
// cells give the year, 15 14. Checking those claims is a tour of MTL5's views:
// objects that present part of a matrix, or a rearrangement of it, as a matrix
// in its own right, without copying an element.
//
//   1. trans(M): the transpose, so a column of M is a row of trans(M)
//   2. mapped(M, rows, cols): any sub-matrix, including scattered ones
//      such as the four corners
//   3. permutation_matrix P: P * M reorders the rows, reorder(M, p) applies
//      p to rows and columns together (P M P^T)
//   4. one generic is_magic() that accepts the matrix and every view of it
//
// Which rearrangements keep the square magic? The eight symmetries of the
// square do; swapping two rows does not, because it breaks the diagonals.
//
// Run with --trace <file> to write the views for the docs-site animation.

#include <mtl/mtl.hpp>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../common/trace.hpp"

using namespace mtl;

namespace {

using index_list = std::vector<std::size_t>;

// Any matrix-like object with num_rows(), num_cols() and (i, j): the matrix
// itself, a transposed view, a mapped view.
template <typename M>
double block_sum(const M& m) {
    double s = 0.0;
    for (std::size_t i = 0; i < m.num_rows(); ++i)
        for (std::size_t j = 0; j < m.num_cols(); ++j) s += m(i, j);
    return s;
}

template <typename M>
bool is_magic(const M& m, double target) {
    const std::size_t n = m.num_rows();
    double d = 0.0, a = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double r = 0.0, c = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            r += m(i, j);
            c += m(j, i);
        }
        if (r != target || c != target) return false;
        d += m(i, i);
        a += m(i, n - 1 - i);
    }
    return d == target && a == target;
}

template <typename M>
std::vector<std::vector<double>> to_rows(const M& m) {
    std::vector<std::vector<double>> out(m.num_rows(), std::vector<double>(m.num_cols()));
    for (std::size_t i = 0; i < m.num_rows(); ++i)
        for (std::size_t j = 0; j < m.num_cols(); ++j) out[i][j] = m(i, j);
    return out;
}

// A view refers to its matrix; to keep a rearrangement, copy it into one.
template <typename M>
mat::dense2D<double> copy_of(const M& m) {
    mat::dense2D<double> out(m.num_rows(), m.num_cols());
    for (std::size_t i = 0; i < m.num_rows(); ++i)
        for (std::size_t j = 0; j < m.num_cols(); ++j) out(i, j) = m(i, j);
    return out;
}

template <typename M>
void print(const std::string& title, const M& m) {
    std::cout << title << '\n';
    for (std::size_t i = 0; i < m.num_rows(); ++i) {
        std::cout << "   ";
        for (std::size_t j = 0; j < m.num_cols(); ++j) std::cout << std::setw(4) << m(i, j);
        std::cout << '\n';
    }
}

// One group of cells the square's sums are checked on, and the view that
// presents it as a matrix.
struct group {
    std::string name;
    std::string view;   // the MTL5 expression that selects it
    std::vector<std::pair<std::size_t, std::size_t>> cells;
    double sum;
};

template <typename M>
std::vector<std::pair<std::size_t, std::size_t>> cells_of(const index_list& rows,
                                                          const index_list& cols, const M&) {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    for (auto r : rows)
        for (auto c : cols) out.emplace_back(r, c);
    return out;
}

} // namespace

int main(int argc, char** argv) {
    example_trace::trace trace(argc, argv, "2A Durer's magic square");

    std::cout << "=============================================================\n";
    std::cout << " Phase 2A: Durer's Magic Square, Through Views\n";
    std::cout << "=============================================================\n\n";

    const mat::dense2D<double> M = {
        {16,  3,  2, 13},
        { 5, 10, 11,  8},
        { 9,  6,  7, 12},
        { 4, 15, 14,  1},
    };
    const std::size_t n = M.num_rows();
    const double target = n * (n * n + 1) / 2.0;   // 34 for n = 4
    print("Melencolia I (1514):", M);
    std::cout << "Magic constant n (n^2 + 1) / 2 = " << target << "\n\n";

    std::vector<group> groups;

    // -- Rows of M, and rows of trans(M), which are M's columns ------------
    // trans(M) stores nothing: trans(M)(i, j) reads M(j, i).
    auto T = trans(M);
    for (std::size_t i = 0; i < n; ++i) {
        auto row = mapped(M, index_list{i}, index_list{0, 1, 2, 3});
        groups.push_back({"row " + std::to_string(i), "mapped(M, {" + std::to_string(i) + "}, all)",
                          cells_of({i}, {0, 1, 2, 3}, M), block_sum(row)});
    }
    for (std::size_t j = 0; j < n; ++j) {
        // Column j of M, read as row j of the transposed view.
        double s = 0.0;
        std::vector<std::pair<std::size_t, std::size_t>> cells;
        for (std::size_t k = 0; k < n; ++k) {
            s += T(j, k);
            cells.emplace_back(k, j);
        }
        groups.push_back({"column " + std::to_string(j), "row " + std::to_string(j) + " of trans(M)",
                          cells, s});
    }

    // -- Diagonals: the anti-diagonal is the diagonal of the row-reversed M --
    mat::permutation_matrix<double> J{3, 2, 1, 0};   // (J M)(i, :) = M(3 - i, :)
    mat::dense2D<double> JM(J * M);
    {
        double s = 0.0, a = 0.0;
        std::vector<std::pair<std::size_t, std::size_t>> dc, ac;
        for (std::size_t i = 0; i < n; ++i) {
            s += M(i, i);
            dc.emplace_back(i, i);
            a += JM(i, i);                       // = M(3 - i, i)
            ac.emplace_back(n - 1 - i, i);
        }
        groups.push_back({"diagonal", "M(i, i)", dc, s});
        groups.push_back({"anti-diagonal", "diagonal of J * M, J the reversal", ac, a});
    }

    // -- Blocks: sub-matrices through mapped() ------------------------------
    const std::vector<std::pair<std::string, std::pair<index_list, index_list>>> blocks = {
        {"top-left quadrant",     {{0, 1}, {0, 1}}},
        {"top-right quadrant",    {{0, 1}, {2, 3}}},
        {"bottom-left quadrant",  {{2, 3}, {0, 1}}},
        {"bottom-right quadrant", {{2, 3}, {2, 3}}},
        {"centre block",          {{1, 2}, {1, 2}}},
        {"four corners",          {{0, 3}, {0, 3}}},
    };
    for (const auto& [name, rc] : blocks) {
        auto view = mapped(M, rc.first, rc.second);
        auto braces = [](const index_list& l) {
            std::string s = "{";
            for (std::size_t k = 0; k < l.size(); ++k) s += (k ? ", " : "") + std::to_string(l[k]);
            return s + "}";
        };
        groups.push_back({name, "mapped(M, " + braces(rc.first) + ", " + braces(rc.second) + ")",
                          cells_of(rc.first, rc.second, M), block_sum(view)});
    }

    std::cout << std::left << std::setw(24) << "cells" << std::setw(40) << "view" << "sum\n"
              << std::right;
    bool all34 = true;
    for (const auto& g : groups) {
        std::cout << std::left << std::setw(24) << g.name << std::setw(40) << g.view << std::right
                  << g.sum << '\n';
        all34 = all34 && g.sum == target;
    }
    std::cout << (all34 ? "\nEvery group sums to 34.\n" : "\nA group does not sum to 34.\n");

    auto year = mapped(M, index_list{3}, index_list{1, 2});
    std::cout << "The date, mapped(M, {3}, {1, 2}): " << year(0, 0) << " " << year(0, 1) << "\n\n";

    // -- Views are live -----------------------------------------------------
    // A view refers to its matrix. Change the matrix and the view sees it.
    mat::dense2D<double> W = M;
    auto corners = mapped(W, index_list{0, 3}, index_list{0, 3});
    W(0, 0) = 0.0;
    std::cout << "Views are live: after W(0,0) = 0 the corners view of W sums to "
              << block_sum(corners) << ".\n\n";

    // -- Rearrangements: which ones keep the square magic? -----------------
    struct rearrangement {
        std::string name;
        std::string how;
        mat::dense2D<double> result;
    };
    mat::permutation_matrix<double> swap01{1, 0, 2, 3};
    std::vector<rearrangement> moves;
    moves.push_back({"transpose", "trans(M)", copy_of(T)});
    moves.push_back({"flip rows", "J * M", JM});
    moves.push_back({"flip columns", "reorder_cols(M, {3, 2, 1, 0})",
                     mat::reorder_cols(M, std::vector<std::size_t>{3, 2, 1, 0})});
    moves.push_back({"rotate a quarter turn", "trans(J * M)", copy_of(trans(JM))});
    moves.push_back({"rotate a half turn", "reorder(M, {3, 2, 1, 0})",
                     mat::reorder(M, std::vector<std::size_t>{3, 2, 1, 0})});
    moves.push_back({"swap pairs symmetrically", "reorder(M, {1, 0, 3, 2})",
                     mat::reorder(M, std::vector<std::size_t>{1, 0, 3, 2})});
    moves.push_back({"swap rows 0 and 1", "P * M, P = {1, 0, 2, 3}",
                     mat::dense2D<double>(swap01 * M)});

    std::cout << "Rearrangements:\n";
    std::vector<bool> magic;
    for (const auto& mv : moves) {
        magic.push_back(is_magic(mv.result, target));
        std::cout << "  " << std::left << std::setw(26) << mv.name << std::setw(32) << mv.how
                  << std::right << (magic.back() ? "magic" : "NOT magic") << '\n';
    }
    print("\nswap rows 0 and 1 (the diagonal is now 5 + 3 + 7 + 1 = 16):", moves.back().result);

    std::cout << "\n=== Key Takeaways ===\n";
    std::cout << "1. trans(M) and mapped(M, rows, cols) are matrices that store no\n";
    std::cout << "   elements: they translate (i, j) and read M.\n";
    std::cout << "2. A mapped view can pick any rows and columns, not only a\n";
    std::cout << "   contiguous block: the four corners are one 2 x 2 view.\n";
    std::cout << "3. A permutation_matrix stores only its index vector; P * M\n";
    std::cout << "   reorders rows, and reorder(M, p) is P M P^T.\n";
    std::cout << "4. Generic code (is_magic) takes the matrix and its views alike.\n";

    if (trace) {
        trace["n"] = n;
        trace["magic_constant"] = target;
        auto mrows = example_trace::value::array();
        for (const auto& r : to_rows(M)) mrows.push_back(example_trace::value::numbers(r));
        trace["square"] = mrows;
        auto& gv = trace["groups"];
        gv = example_trace::value::array();
        for (const auto& g : groups) {
            auto o = example_trace::value::object();
            o["name"] = g.name;
            o["view"] = g.view;
            auto cells = example_trace::value::array();
            for (auto [i, j] : g.cells) cells.push_back(example_trace::value::numbers(std::vector<std::size_t>{i, j}));
            o["cells"] = cells;
            o["sum"] = g.sum;
            gv.push_back(o);
        }
        auto& mv = trace["rearrangements"];
        mv = example_trace::value::array();
        for (std::size_t k = 0; k < moves.size(); ++k) {
            auto o = example_trace::value::object();
            o["name"] = moves[k].name;
            o["how"] = moves[k].how;
            auto rows = example_trace::value::array();
            for (const auto& r : to_rows(moves[k].result)) rows.push_back(example_trace::value::numbers(r));
            o["result"] = rows;
            o["magic"] = static_cast<bool>(magic[k]);
            mv.push_back(o);
        }
    }
    return trace.close();   // EXIT_FAILURE if a requested trace could not be written
}
