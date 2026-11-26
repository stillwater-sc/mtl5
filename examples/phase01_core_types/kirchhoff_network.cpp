// kirchhoff_network.cpp -- Example 1B: Kirchhoff's laws and the Wheatstone bridge
//
// Gustav Kirchhoff stated his circuit laws in 1845: the currents into a node
// sum to zero, and the voltage drops around a loop sum to zero. Written for
// node voltages v, the current law is a matrix equation L v = i, where L is
// the network's conductance matrix (today: its weighted graph Laplacian). Each
// resistor of conductance g between nodes a and b contributes a "stamp":
//
//     L(a,a) += g    L(a,b) -= g
//     L(b,a) -= g    L(b,b) += g
//
// so L is assembled by summing stamps, the pattern every circuit simulator
// (SPICE) and finite element code uses. The circuit is Wheatstone's bridge,
// which Samuel Christie invented in 1833 and Charles Wheatstone made famous in
// 1843: four resistors in a diamond with a galvanometer across the middle.
// When R1/R2 = R3/R4 the bridge is balanced and no current flows through the
// galvanometer, which is how an unknown resistance was measured.
//
// The example shows the sparse container and how it is filled:
//   1. compressed2D assembled with an inserter and update_plus, so repeated
//      stamps on one entry add up
//   2. a second inserter amends the finished matrix (it keeps the entries)
//   3. the CRS arrays behind it: row starts, column indices, values
//   4. sparse against dense storage as the network grows
//
// Run with --trace <file> to write the stamps for the docs-site animation.

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

struct resistor {
    std::string name;
    std::size_t a, b;   // the nodes it joins
    double ohms;
};

using conductance = mat::compressed2D<double>;
using stamper = mat::inserter<conductance, mat::update_plus<double>>;

void stamp(stamper& ins, std::size_t a, std::size_t b, double g) {
    ins[a][a] << g;
    ins[b][b] << g;
    ins[a][b] << -g;
    ins[b][a] << -g;
}

conductance assemble(std::size_t nodes, const std::vector<resistor>& rs, std::size_t count) {
    conductance L(nodes, nodes);
    {
        stamper ins(L, 5);   // at most 5 entries per row for this bridge
        for (std::size_t k = 0; k < count; ++k) stamp(ins, rs[k].a, rs[k].b, 1.0 / rs[k].ohms);
    }   // the inserter's destructor compresses the stamps into CRS arrays
    return L;
}

// Node 0 is held at v0 volts by the battery and node 3 is ground. The current
// law at the two middle nodes is a 2 x 2 system in v1 and v2:
//     L11 v1 + L12 v2 = -L10 v0
//     L21 v1 + L22 v2 = -L20 v0
// small enough for Cramer's rule, which needs nothing but element access.
std::pair<double, double> bridge_voltages(const conductance& L, double v0) {
    const double a = L(1, 1), b = L(1, 2), c = L(2, 1), e = L(2, 2);
    const double r1 = -L(1, 0) * v0, r2 = -L(2, 0) * v0;
    const double det = a * e - b * c;
    return {(r1 * e - b * r2) / det, (a * r2 - c * r1) / det};
}

void print_matrix(const conductance& L) {
    for (std::size_t i = 0; i < L.num_rows(); ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < L.num_cols(); ++j)
            std::cout << std::setw(10) << std::fixed << std::setprecision(5) << L(i, j);
        std::cout << " ]\n";
    }
}

template <typename V>
example_trace::value numbers(const std::vector<V>& v, int digits = 17) {
    return example_trace::value::numbers(v, digits);
}

example_trace::value dense_snapshot(const conductance& L) {
    auto rows = example_trace::value::array();
    for (std::size_t i = 0; i < L.num_rows(); ++i) {
        std::vector<double> r(L.num_cols());
        for (std::size_t j = 0; j < L.num_cols(); ++j) r[j] = L(i, j);
        rows.push_back(numbers(r, 8));
    }
    return rows;
}

} // namespace

int main(int argc, char** argv) {
    example_trace::trace trace(argc, argv, "1B Kirchhoff network and the Wheatstone bridge");

    std::cout << "=============================================================\n";
    std::cout << " Phase 1B: Kirchhoff's Laws and the Wheatstone Bridge\n";
    std::cout << "=============================================================\n\n";

    // The bridge, with R5 the galvanometer across the middle:
    //
    //     node 0 (+10 V) --R1-- node 1 --R2-- node 3 (ground)
    //     node 0 (+10 V) --R3-- node 2 --R4-- node 3 (ground)
    //                           node 1 --R5-- node 2
    const std::size_t nodes = 4;
    std::vector<resistor> rs = {
        {"R1", 0, 1, 100.0},
        {"R2", 1, 3, 200.0},
        {"R3", 0, 2, 150.0},
        {"R4", 2, 3, 300.0},
        {"R5", 1, 2, 50.0},
    };
    const double v0 = 10.0;

    std::cout << "Resistors (R1/R2 = " << rs[0].ohms / rs[1].ohms << ", R3/R4 = "
              << rs[2].ohms / rs[3].ohms << "):\n";
    for (const auto& r : rs)
        std::cout << "  " << r.name << ": node " << r.a << " - node " << r.b << ", "
                  << r.ohms << " ohm, g = " << std::setprecision(5) << 1.0 / r.ohms << " S\n";

    // -- Assembly: one inserter, one stamp per resistor ---------------------
    conductance L = assemble(nodes, rs, rs.size());
    std::cout << "\nConductance matrix L (" << L.nnz() << " stored entries of "
              << nodes * nodes << "):\n";
    print_matrix(L);
    std::cout << "\nThe diagonal L(i,i) is the sum of the stamps that met there:\n"
              << "  L(1,1) = g1 + g2 + g5 = " << std::setprecision(5) << L(1, 1) << "\n";

    // Kirchhoff's current law: equal voltages everywhere drive no current, so
    // L times a vector of ones is zero, and every row sums to zero.
    vec::dense_vector<double> ones(nodes, 1.0);
    vec::dense_vector<double> row_sums(L * ones);
    std::cout << "  L * ones = [";
    for (std::size_t i = 0; i < nodes; ++i)
        std::cout << (i ? ", " : "") << std::scientific << std::setprecision(1) << row_sums(i);
    std::cout << "]   (every row sums to zero)\n" << std::fixed;

    // -- What compressed2D actually stores ---------------------------------
    const auto& starts  = L.ref_major();   // row i is entries [starts[i], starts[i+1])
    const auto& columns = L.ref_minor();
    const auto& values  = L.ref_data();
    std::cout << "\nCompressed row storage:\n  row starts: ";
    for (auto s : starts) std::cout << s << ' ';
    std::cout << "\n  columns:    ";
    for (auto c : columns) std::cout << c << ' ';
    std::cout << "\n  values:     ";
    for (auto v : values) std::cout << std::setprecision(4) << v << ' ';
    std::cout << "\n  (0,3) and (3,0) are not stored: no resistor joins nodes 0 and 3.\n";

    // -- The bridge, balanced ----------------------------------------------
    auto [v1, v2] = bridge_voltages(L, v0);
    const double i5_balanced = (v1 - v2) / rs[4].ohms;
    std::cout << "\nBalanced bridge: v1 = " << std::setprecision(6) << v1 << " V, v2 = " << v2
              << " V, galvanometer current = " << std::scientific << std::setprecision(2)
              << i5_balanced << " A\n" << std::fixed;

    // -- Amend: change R4 by stamping the difference ------------------------
    // A second inserter keeps what is stored and, with update_plus, adds to
    // it: replacing R4 = 300 by 250 ohm is one more stamp of g_new - g_old.
    const double r4_new = 250.0;
    const double dg = 1.0 / r4_new - 1.0 / rs[3].ohms;
    conductance L_unbalanced = L;
    {
        stamper ins(L_unbalanced, 5);
        stamp(ins, rs[3].a, rs[3].b, dg);
    }
    auto [u1, u2] = bridge_voltages(L_unbalanced, v0);
    const double i5_unbalanced = (u1 - u2) / rs[4].ohms;
    std::cout << "R4 -> " << r4_new << " ohm: v1 = " << std::setprecision(6) << u1
              << " V, v2 = " << u2 << " V, galvanometer current = " << std::setprecision(3)
              << i5_unbalanced * 1000.0 << " mA\n";

    // -- Scale: a resistor mesh --------------------------------------------
    // An m x m mesh of 1-ohm resistors, as in a model of a resistive sheet.
    std::cout << "\n--- Sparse against dense as the network grows ---\n";
    std::cout << std::setw(8) << "mesh" << std::setw(9) << "nodes" << std::setw(11) << "nnz"
              << std::setw(15) << "dense bytes" << std::setw(13) << "CRS bytes\n";
    auto mesh_rows = example_trace::value::array();
    for (std::size_t m : {4, 8, 16, 32, 64}) {
        const std::size_t N = m * m;
        conductance M(N, N);
        {
            stamper ins(M, 5);
            for (std::size_t r = 0; r < m; ++r)
                for (std::size_t c = 0; c < m; ++c) {
                    const std::size_t p = r * m + c;
                    if (c + 1 < m) stamp(ins, p, p + 1, 1.0);
                    if (r + 1 < m) stamp(ins, p, p + m, 1.0);
                }
        }
        const std::size_t dense_bytes = N * N * sizeof(double);
        const std::size_t crs_bytes = M.nnz() * (sizeof(double) + sizeof(std::size_t))
                                    + (N + 1) * sizeof(std::size_t);
        std::cout << std::setw(5) << m << "x" << std::setw(2) << std::left << m << std::right
                  << std::setw(9) << N << std::setw(11) << M.nnz() << std::setw(15)
                  << dense_bytes << std::setw(12) << crs_bytes << '\n';
        auto row = example_trace::value::object();
        row["m"] = m;
        row["nodes"] = N;
        row["nnz"] = M.nnz();
        row["dense_bytes"] = dense_bytes;
        row["crs_bytes"] = crs_bytes;
        mesh_rows.push_back(row);
    }

    std::cout << "\n=== Key Takeaways ===\n";
    std::cout << "1. An inserter with update_plus sums repeated entries: stamps that\n";
    std::cout << "   land on the same diagonal add up, as Kirchhoff's law requires.\n";
    std::cout << "2. The matrix is finished when the inserter is destroyed; a later\n";
    std::cout << "   inserter amends it instead of starting over.\n";
    std::cout << "3. compressed2D stores three arrays: where each row starts, the\n";
    std::cout << "   columns, and the values. Missing entries cost nothing.\n";
    std::cout << "4. A mesh has about 5 entries per row, so CRS grows like N while\n";
    std::cout << "   dense storage grows like N^2.\n";

    if (trace) {
        trace["nodes"] = nodes;
        trace["source_volts"] = v0;
        auto& rv = trace["resistors"];
        rv = example_trace::value::array();
        for (const auto& r : rs) {
            auto o = example_trace::value::object();
            o["name"] = r.name;
            o["a"] = r.a;
            o["b"] = r.b;
            o["ohms"] = r.ohms;
            o["siemens"] = example_trace::value::rounded(1.0 / r.ohms, 8);
            rv.push_back(o);
        }
        // L after the first k stamps, k = 0 .. 5, each assembled by MTL5.
        auto& snaps = trace["after_stamp"];
        snaps = example_trace::value::array();
        for (std::size_t k = 0; k <= rs.size(); ++k) snaps.push_back(dense_snapshot(assemble(nodes, rs, k)));
        auto& crs = trace["crs"];
        crs["starts"] = numbers(std::vector<std::size_t>(starts.begin(), starts.end()));
        crs["columns"] = numbers(std::vector<std::size_t>(columns.begin(), columns.end()));
        crs["values"] = numbers(std::vector<double>(values.begin(), values.end()), 8);
        auto& bal = trace["balanced"];
        bal["v1"] = example_trace::value::rounded(v1, 10);
        bal["v2"] = example_trace::value::rounded(v2, 10);
        bal["galvanometer_amps"] = example_trace::value::rounded(i5_balanced, 4);
        auto& unb = trace["unbalanced"];
        unb["r4_ohms"] = r4_new;
        unb["matrix"] = dense_snapshot(L_unbalanced);
        unb["v1"] = example_trace::value::rounded(u1, 10);
        unb["v2"] = example_trace::value::rounded(u2, 10);
        unb["galvanometer_amps"] = example_trace::value::rounded(i5_unbalanced, 8);
        trace["mesh"] = mesh_rows;
    }
    return trace.close();   // EXIT_FAILURE if a requested trace could not be written
}
