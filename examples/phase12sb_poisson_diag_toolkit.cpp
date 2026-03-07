// phase12sb_poisson_diag_toolkit.cpp -- Poisson Generator, Diagonal Tools & Solver
//
// This example demonstrates the Poisson generator, diagonal extraction/construction,
// and their interplay with iterative solvers:
//
//   1. Generate a 2D Poisson matrix with Dirichlet BCs (h^2-scaled)
//   2. Compare to the unscaled Laplacian to show the scaling relationship
//   3. Extract the diagonal for condition number estimation
//   4. Build a diagonal matrix using diag() -- the inverse of diagonal()
//   5. Use the diagonal as a Jacobi preconditioner
//   6. Solve the Poisson system and show convergence improvement
//
// The Poisson equation -nabla^2 u = f is the prototypical elliptic PDE.
// Its discretization on a grid with mesh spacing h produces a matrix whose
// entries scale as 1/h^2. The poisson2d_dirichlet() generator handles this
// scaling automatically, unlike the raw laplacian_2d() which has unit entries.
//
// Diagonal preconditioning (Jacobi) is the simplest preconditioner -- it
// scales each equation by the inverse of its diagonal entry. Despite its
// simplicity, it significantly improves convergence for poorly scaled systems.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mtl;

int main() {
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "=== Poisson Generator, Diagonal Tools & Solver ===\n\n";

    // -- 1. Generate Poisson matrix ---------------------------------------
    const std::size_t nx = 5, ny = 5;
    const std::size_t n = nx * ny;  // 25 unknowns

    auto P = generators::poisson2d_dirichlet<double>(nx, ny);
    auto L = generators::laplacian_2d<double>(nx, ny);

    std::cout << "-- 1. Poisson 2D Dirichlet (" << nx << "x" << ny << " grid) --\n";
    std::cout << "Matrix size: " << P.num_rows() << " x " << P.num_cols() << "\n";
    std::cout << "Nonzeros:    " << P.nnz() << "\n\n";

    // -- 2. Show scaling relationship -------------------------------------
    // poisson2d_dirichlet = (nx+1)^2 * laplacian_2d  for uniform grid
    double hx = 1.0 / (nx + 1);
    double hy = 1.0 / (ny + 1);
    double scale = 1.0 / (hx * hx);  // = (nx+1)^2 for uniform grid

    std::cout << "-- 2. Scaling relationship (uniform grid, nx==ny) --\n";
    std::cout << "h = 1/(n+1) = " << hx << "\n";
    std::cout << "Scale factor 1/h^2 = " << scale << "\n";

    // Verify P ~= scale * L at a few entries
    std::cout << "\nCompare Poisson vs. scaled Laplacian:\n";
    std::cout << std::setw(10) << "Entry" << std::setw(18) << "Poisson"
              << std::setw(18) << "scale*Laplacian" << std::setw(14) << "Match?\n";
    for (std::size_t idx : {std::size_t(0), std::size_t(6), std::size_t(12)}) {
        double p_val = P(idx, idx);
        double sl_val = scale * L(idx, idx);
        bool ok = std::abs(p_val - sl_val) < 1e-10;
        std::cout << std::setw(10) << "(" << idx << "," << idx << ")"
                  << std::setw(16) << p_val
                  << std::setw(18) << sl_val
                  << std::setw(12) << (ok ? "yes" : "NO") << "\n";
    }

    // -- 3. Extract diagonal for analysis ---------------------------------
    auto d = diagonal(P);

    std::cout << "\n-- 3. Diagonal extraction --\n";
    std::cout << "Diagonal entries (first 5): ";
    print(std::cout, vec::dense_vector<double>({d(0), d(1), d(2), d(3), d(4)}), 2);
    std::cout << "\n";

    // Condition number estimate from diagonal
    double d_min = d(0), d_max = d(0);
    for (std::size_t i = 1; i < n; ++i) {
        d_min = std::min(d_min, d(i));
        d_max = std::max(d_max, d(i));
    }
    std::cout << "Diagonal range: [" << d_min << ", " << d_max << "]\n";
    std::cout << "Diagonal ratio (d_max/d_min): " << d_max / d_min << "\n";

    // -- 4. Build diagonal matrix with diag() -----------------------------
    auto D = diag(d);
    std::cout << "\n-- 4. Diagonal matrix from diag() --\n";
    std::cout << "D is " << D.num_rows() << "x" << D.num_cols()
              << ", nnz = " << D.nnz() << "\n";

    // Verify roundtrip: diagonal(diag(d)) == d
    auto d2 = diagonal(D);
    double roundtrip_err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        roundtrip_err = std::max(roundtrip_err, std::abs(d(i) - d2(i)));
    std::cout << "Roundtrip error |d - diagonal(diag(d))|: " << roundtrip_err << "\n";

    // -- 5. Solve with CG: unpreconditioned -------------------------------
    vec::dense_vector<double> b(n, 1.0);  // RHS: uniform forcing
    vec::dense_vector<double> x(n, 0.0);

    std::cout << "\n-- 5. CG solve: no preconditioner --\n";
    {
        itl::pc::identity<decltype(P)> pc(P);
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        itl::cg(P, x, b, pc, iter);
        std::cout << "Iterations: " << iter.iterations() << "\n";
        std::cout << "Residual:   " << iter.resid() << "\n";
    }

    // -- 6. Solve with CG: Jacobi (diagonal) preconditioner --------------
    std::cout << "\n-- 6. CG solve: Jacobi (diagonal) preconditioner --\n";
    {
        vec::dense_vector<double> x2(n, 0.0);
        itl::pc::diagonal<decltype(P)> pc(P);
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        itl::cg(P, x2, b, pc, iter);
        std::cout << "Iterations: " << iter.iterations() << "\n";
        std::cout << "Residual:   " << iter.resid() << "\n";
    }

    // -- 7. Solution visualization with pretty-print ----------------------
    std::cout << "\n-- 7. Solution (reshaped to " << ny << "x" << nx << " grid) --\n";
    // Reshape solution vector to 2D grid for display
    mat::dense2D<double> U(ny, nx);
    for (std::size_t iy = 0; iy < ny; ++iy)
        for (std::size_t ix = 0; ix < nx; ++ix)
            U(iy, ix) = x(iy * nx + ix);

    print(std::cout, U, 4);

    // MATLAB export for visualization
    std::cout << "\n-- 8. MATLAB export --\n";
    print_matlab(std::cout, U, "U", 6);

    // -- 8. Sparse structure visualization --------------------------------
    std::cout << "\n-- 9. Poisson matrix structure (first 10 entries) --\n";
    {
        std::ostringstream oss;
        print_sparse(oss, P, 4);
        std::istringstream iss(oss.str());
        std::string line;
        int count = 0;
        while (std::getline(iss, line) && count < 10) {
            std::cout << "  " << line << "\n";
            ++count;
        }
        std::cout << "  ... (" << P.nnz() << " nonzeros total)\n";
    }

    std::cout << "\nAll Poisson and diagonal operations completed successfully.\n";
    return 0;
}
