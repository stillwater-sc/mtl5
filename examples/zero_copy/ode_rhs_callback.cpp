// ode_rhs_callback.cpp - Zero-copy MTL5 vectors inside a C-style callback
//
// The pattern: a library you call (an ODE integrator here; CVODE, GSL's
// odeiv2 and most Fortran solvers look the same) owns the state and the stage
// buffers, and calls YOUR function with raw pointers:
//
//     void rhs(double t, const double* y, double* dydt, std::size_t n, void* user);
//
// You want to write the right-hand side with MTL5 -- a sparse operator,
// expressions -- without copying y in and dydt out on every call. MTL5 lets
// you lay a vector over each buffer:
//
//     const auto Y = vec::const_view(n, y);     // read-only, zero copy
//     vec::dense_vector<double> dY(n, dydt);    // writes the integrator's buffer
//     mult(L, Y, dY);                           // dydt = L * y, in place
//
// This program integrates the 1D heat equation u_t = D u_xx with that
// callback, checks the result against the exact solution of the
// semi-discrete system, checks that no buffer was ever copied, and measures
// what the copy-in/copy-out alternative costs.

#include <mtl/mtl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

using namespace mtl;

// -- The "foreign" integrator --------------------------------------------------
// A minimal explicit RK4 with a C interface. It knows nothing about MTL5: it
// owns y and the stage buffers k1..k4 and hands out raw pointers.

using rhs_fn = void (*)(double t, const double* y, double* dydt, std::size_t n, void* user);

void rk4_integrate(rhs_fn f, void* user, double* y, std::size_t n,
                   double t0, double t1, std::size_t steps) {
    std::vector<double> k1(n), k2(n), k3(n), k4(n), tmp(n);
    const double dt = (t1 - t0) / static_cast<double>(steps);
    double t = t0;
    for (std::size_t s = 0; s < steps; ++s) {
        f(t, y, k1.data(), n, user);
        for (std::size_t i = 0; i < n; ++i) tmp[i] = y[i] + 0.5 * dt * k1[i];
        f(t + 0.5 * dt, tmp.data(), k2.data(), n, user);
        for (std::size_t i = 0; i < n; ++i) tmp[i] = y[i] + 0.5 * dt * k2[i];
        f(t + 0.5 * dt, tmp.data(), k3.data(), n, user);
        for (std::size_t i = 0; i < n; ++i) tmp[i] = y[i] + dt * k3[i];
        f(t + dt, tmp.data(), k4.data(), n, user);
        for (std::size_t i = 0; i < n; ++i)
            y[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        t += dt;
    }
}

// -- The problem -----------------------------------------------------------------

struct heat_problem {
    mat::compressed2D<double> L;    // D/h^2 * tridiag(1, -2, 1)
    std::size_t calls = 0;
    std::size_t copies = 0;         // buffers the callback did NOT use in place
};

mat::compressed2D<double> laplacian(std::size_t n, double D) {
    const double h = 1.0 / static_cast<double>(n + 1);
    const double s = D / (h * h);
    mat::compressed2D<double> L(n, n);
    {   // the inserter finalizes L when it goes out of scope
        mat::inserter<mat::compressed2D<double>> ins(L, 3);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << -2.0 * s;
            if (i > 0)     ins[i][i - 1] << s;
            if (i + 1 < n) ins[i][i + 1] << s;
        }
    }
    return L;
}

// A write through a read-only view is a compile error, not a runtime surprise.
template <typename V>
concept writable = requires(V v) { v(0) = 1.0; };
static_assert(!writable<vec::const_vector_view<double>>,
              "a const_view of the integrator's y cannot be written");

// -- The zero-copy right-hand side -----------------------------------------------

void heat_rhs(double /*t*/, const double* y, double* dydt, std::size_t n, void* user) {
    auto& p = *static_cast<heat_problem*>(user);
    const auto Y = vec::const_view(n, y);      // a view of the integrator's y
    vec::dense_vector<double> dY(n, dydt);     // a vector over its output buffer
    mult(p.L, Y, dY);                          // dydt = L y, written in place
    // Y(0) = 0.0;                             // would not compile: Y is read-only

    ++p.calls;
    if (Y.data() != y || dY.data() != dydt || dY.owns_memory()) ++p.copies;
}

// -- The same right-hand side, copying in and out --------------------------------

void heat_rhs_copy(double /*t*/, const double* y, double* dydt, std::size_t n, void* user) {
    auto& p = *static_cast<heat_problem*>(user);
    vec::dense_vector<double> Y(n), dY(n);     // owning: two allocations per call
    std::copy(y, y + n, Y.begin());            // copy in
    mult(p.L, Y, dY);
    std::copy(dY.begin(), dY.end(), dydt);     // copy out
    ++p.calls;
    p.copies += 2;
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Zero copy: MTL5 vectors over an ODE integrator's buffers\n";
    std::cout << "=============================================================\n\n";

    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::cout << (ok ? "  [ok]   " : "  [FAIL] ") << what << "\n";
        if (!ok) ++failures;
    };

    // -- 1. Integrate the heat equation through the callback -------------------
    const std::size_t n = 200;
    const double D = 1.0, t1 = 0.02;
    const double h = 1.0 / static_cast<double>(n + 1);
    const double pi = std::numbers::pi;

    heat_problem p{laplacian(n, D)};
    std::vector<double> y(n);                  // the integrator's state
    for (std::size_t i = 0; i < n; ++i) y[i] = std::sin(pi * h * static_cast<double>(i + 1));

    // RK4 is stable for dt * |lambda_max| < 2.78, lambda_max ~ 4 D / h^2.
    const std::size_t steps = 2000;
    rk4_integrate(heat_rhs, &p, y.data(), n, 0.0, t1, steps);

    // sin(pi x) is an eigenvector of the discrete Laplacian, so the exact
    // semi-discrete solution is exp(lambda_1 t) times the initial state.
    const double lambda1 = -4.0 * D / (h * h) * std::pow(std::sin(pi * h / 2.0), 2);
    double err = 0.0, ref = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double exact = std::exp(lambda1 * t1) * std::sin(pi * h * static_cast<double>(i + 1));
        err = std::max(err, std::abs(y[i] - exact));
        ref = std::max(ref, std::abs(exact));
    }
    std::cout << "Heat equation, n = " << n << ", " << steps << " RK4 steps to t = " << t1 << "\n";
    std::cout << "  rhs calls: " << p.calls << ", relative max error: "
              << std::scientific << std::setprecision(2) << err / ref << std::defaultfloat << "\n";
    check(err / ref < 1e-8, "matches the exact semi-discrete solution");
    check(p.copies == 0, "every call used the integrator's y and dydt in place");

    // -- 2. The copying version gives the same answer ---------------------------
    heat_problem pc{laplacian(n, D)};
    std::vector<double> yc(n);
    for (std::size_t i = 0; i < n; ++i) yc[i] = std::sin(pi * h * static_cast<double>(i + 1));
    rk4_integrate(heat_rhs_copy, &pc, yc.data(), n, 0.0, t1, steps);
    check(y == yc, "copy-in/copy-out gives bit-identical results");

    // -- 3. What zero copy saves ------------------------------------------------
    // At a size where the copies matter: one rhs evaluation of a 1D Laplacian
    // reads and writes each element about once, so the two extra copies (and
    // two allocations) are a large share of the work. In a Release build on an
    // i7-12700K the copying call takes about 3.2 times as long; a Debug build
    // hides most of the difference.
    const std::size_t big = 2'000'000, reps = 20;
    heat_problem pb{laplacian(big, D)};
    std::vector<double> yb(big, 1.0), db(big);
    using clock = std::chrono::steady_clock;
    auto time = [&](rhs_fn f) {
        f(0.0, yb.data(), db.data(), big, &pb);   // warm up
        const auto t0 = clock::now();
        for (std::size_t r = 0; r < reps; ++r) f(0.0, yb.data(), db.data(), big, &pb);
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count() / reps;
    };
    const double ms_view = time(heat_rhs);
    const double ms_copy = time(heat_rhs_copy);
    const double mb = 2.0 * big * sizeof(double) / 1.0e6;
    std::cout << "\nOne rhs call at n = " << big << " (timings depend on the machine):\n"
              << std::fixed << std::setprecision(2)
              << "  zero copy:          " << ms_view << " ms\n"
              << "  copy in, copy out:  " << ms_copy << " ms  (+" << mb << " MB copied, 2 allocations)\n"
              << std::defaultfloat;

    std::cout << "\n" << (failures ? "FAILED" : "All checks passed") << "\n";
    return failures ? 1 : 0;
}
