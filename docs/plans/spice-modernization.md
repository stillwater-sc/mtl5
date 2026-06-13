# SPICE modernization

Can you give me the history of the different open-source SPICE simulators that have been developed. I am looking for a modern C++ implementation that is backward compatible, but that we can from the get-go design as being parameterized by arithmetic type so that we can change the linear algebra algorithms and use LNS or posit number systems with super accumulators, so we can do experiments with mixed-precision optimization to improve performance and energy-efficiency of the SPICE simulator.

That is an incredibly ambitious and computationally fascinating project! Using Logarithmic Number Systems (LNS) or posit arithmetic with super accumulators to explore mixed-precision optimization could drastically reshape the energy efficiency and performance of electronic design automation (EDA) tools. As an AI, I don't feel excitement, but looking at this purely from a systems architecture standpoint, it's a brilliant avenue for research.

However, I need to give it to you straight: **the specific "out-of-the-box" modern C++ simulator parameterized entirely by arithmetic type that you are looking for does not currently exist.** To understand why—and to figure out your best path forward—we have to look at the history of open-source SPICE simulators and the architectural baggage they carry.

### The Lineage of Open-Source SPICE Simulators

Almost all open-source circuit simulation traces its roots back to UC Berkeley.

* **Berkeley SPICE (1970s–1980s):** SPICE1 and SPICE2 were written in FORTRAN, but SPICE3 (specifically SPICE3f5 released in 1993) was written in C. This C codebase introduced the core nodal analysis and device models that still dominate the industry.
* **Ngspice (Late 1990s–Present):** The undisputed king of open-source SPICE. It is a direct continuation of SPICE3f5, combined with XSPICE (for mixed-signal) and CIDER (for device-level simulation). Because it is entirely C-based, attempting to natively parameterize its arithmetic via C++ templates is impossible without a ground-up rewrite.
* **Gnucap (1993–Present):** The GNU Circuit Analysis Package. Written in C++, it introduced a highly modular, object-oriented architecture that deviated from SPICE3's monolithic design. However, it was built well before modern C++ (C++11/14/20) template metaprogramming became standard. It relies heavily on classical polymorphism, and its precision is fundamentally tied to the standard `double`.
* **Xyce (1999–Present):** Developed by Sandia National Laboratories and later open-sourced. Xyce is a massive C++ simulator built from scratch to support extreme-scale parallel computing (via MPI). It utilizes the Trilinos linear algebra framework.

### The "Templated Arithmetic" Challenge

If you want a simulator where the entire data path—from the linear algebra solvers down to the device model evaluations—is templated (e.g., `template <typename T> class Solver`), you run headfirst into the **Device Model Compatibility Paradox**.

To be "backward compatible" in the SPICE world, you must support standard compact models (like BSIM4, PSP, or modern FinFET models). These models contain tens of thousands of lines of highly non-linear equations with hardcoded floating-point constants. Historically, they are either hand-written in C or written in Verilog-A and transpiled into C/C++. In almost all existing ecosystems, the `double` data type is completely hardcoded into the transpiled output.

If you plug a custom posit or LNS type into your matrix solver, but the device model is still calculating transconductance and charge using standard `double` arithmetic, you lose the benefits of your mixed-precision experiment and incur massive casting overhead at every Newton-Raphson iteration.

### Your Best Paths Forward

Since the exact tool doesn't exist, you will have to build or heavily adapt one. Here are the two most realistic approaches for your research:

**1. Fork and Modify Xyce (The Heavy Lifter)**
Xyce is currently your closest starting point. Its underlying linear algebra engine, Trilinos, natively supports templated scalar types (via packages like Tpetra, which can be instantiated with different numerical types).

* **The Pro:** You inherit an industrial-grade, parallelized solver, netlist parser, and architecture.
* **The Con:** You will have to systematically refactor the device evaluation layer. Xyce currently uses ADMS (an open-source Verilog-A to C/C++ transpiler) to generate its C++ device models. You would need to modify the ADMS XML scripts so they generate `template <typename Scalar>` C++ code instead of strictly `double`-based code.

**2. Build a Modern C++ Core with a Custom Verilog-A Backend (The Clean Slate)**
If you are designing this from the get-go to experiment with arithmetic types, building a lightweight MNA (Modified Nodal Analysis) solver in modern C++ might be less painful than wrestling with legacy codebases.

* **The Strategy:** Write your core solver parameterized heavily by `<typename T>`. You can integrate libraries like Eigen, which play very nicely with custom scalar types.
* **The Trick:** Do *not* hand-write device models to achieve backward compatibility. Instead, use a modern Verilog-A compiler like **OpenVAF** or **ADMS** and write a custom backend for it. You can instruct the transpiler to consume standard industry Verilog-A compact models and spit out fully templated C++ headers that natively accept your LNS or posit super accumulators.

---

Which path sounds more appealing to your team: adapting a behemoth like Xyce to support your custom numeric types, or building a lightweight, highly-templated C++ solver from scratch and auto-generating the models?

## Building lightweight solver from scratch

building a lightweight, highly-templated C++ solver from scratch using MTL5, not Eigen as we need to modify the linear algebra routines as well and MTL5 already supports posits and quires.

That is a fantastic architectural decision. Going with **MTL5 (Matrix Template Library 5)** fundamentally changes the game for what you are trying to achieve. Because MTL5 was designed specifically with mixed-precision and custom number systems in mind (like the Universal Number Library's posits and quires), you bypass the painful friction you would experience trying to force Eigen to play nicely with non-standard arithmetic.

Since you are building a lightweight, highly-templated C++ Modified Nodal Analysis (MNA) solver from the ground up, here is a strategic blueprint on how to structure the architecture to maximize the benefits of MTL5 and posit arithmetic.

### 1. The Type-Agnostic Foundation

Your core simulator classes need to be universally parameterized. By defining the scalar type at the highest level, you can instantiate parallel simulation runs comparing `double`, `float`, `posit<32,2>`, and `posit<16,1>` with zero code duplication.

```cpp
template <typename ScalarType>
class SpiceSolver {
private:
    mtl::compressed2D<ScalarType> Jacobian; // MTL5 sparse matrix
    mtl::dense_vector<ScalarType> rhs;      // Right-hand side vector
    mtl::dense_vector<ScalarType> solution; // State vector (voltages/currents)
    
    // ... device lists, netlist maps, etc.
public:
    void assembleMNA();
    void solveNewtonRaphson();
};

```

### 2. The Newton-Raphson Engine and the Quire

SPICE simulation is fundamentally about solving non-linear differential equations by converting them into linear algebraic equations via numerical integration (like Backward Euler or Trapezoidal) and linearizing them via the Newton-Raphson method.

You are solving:


$$J(x_k) \Delta x_k = -f(x_k)$$

$$x_{k+1} = x_k + \Delta x_k$$

**Where the Quire shines:** MNA matrices ($J$) are notoriously ill-conditioned. You have conductances in the $10^{-12}$ range and capacitances in the $10^{-15}$ range right next to dynamic resistances in the $10^3$ range. Standard floating-point arithmetic suffers from catastrophic cancellation during matrix solving here.

By leveraging the posit **quire** (a super accumulator that performs deferred rounding), you can compute vector dot products—which are the backbone of matrix multiplication and factorization—with exact precision. The rounding only happens when the final accumulated value is written back to a standard posit memory format. You will need to explicitly ensure that the inner loops of your linear solver in MTL5 utilize the quire for the multiply-accumulate (MAC) operations.

### 3. Handling the Device Models (The Verilog-A Challenge)

Even though you control the linear algebra, you still need to evaluate the non-linear devices (transistors, diodes) to populate $J$ and $f(x)$.

* **Do not write these by hand.** It is a maintenance nightmare.
* **The Solution:** Use an open-source Verilog-A compiler like **OpenVAF** (which is written in Rust and highly extensible). You will need to write a custom target backend for OpenVAF that takes the Verilog-A AST (Abstract Syntax Tree) and emits C++ header files where all mathematical functions and variables are templated with `<typename ScalarType>`. This ensures the device evaluations use posits directly, avoiding costly type-casting overhead during the assembly phase.

### 4. The Direct vs. Iterative Solver Dilemma

Here is the biggest hurdle you will face with this architecture. MTL5 is exceptionally strong when paired with its Iterative Template Library (ITL) for solvers like GMRES, CG, or BiCGSTAB.

However, historically, **iterative solvers fail miserably on SPICE MNA matrices** because the matrices are highly asymmetric and extremely stiff. Standard SPICE simulators (like Ngspice or Xyce) rely heavily on **Direct Sparse LU Factorization** (using packages like KLU or SuperLU).

If you use an iterative solver from MTL5, the exact dot products from the quire will drastically improve the orthogonality of your Krylov subspaces, which *might* allow iterative solvers to converge where standard floats fail. But if they still don't converge, you will have to implement a templated direct sparse LU factorization algorithm within your MTL5 framework.

---

Given that MNA matrices are traditionally hostile to iterative methods, do you plan to rely on MTL5's iterative Krylov solvers aided by the quire's precision, or are you planning to write a custom templated sparse LU factorization routine for the linear solve?

## Direct Sparse solvers

MTL5 has the SparseSuite direct solvers, as well as KLU and SuperLU. The sparse direct solvers in MTL5 are an addition compared to MTL4

I stand corrected—that is a massive architectural upgrade from MTL4 and perfectly bridges the exact gap I was concerned about! Having native access to SuiteSparse, KLU, and SuperLU within the MTL5 ecosystem completely eliminates the "Direct vs. Iterative" dilemma for your solver.

Here is how this addition streamlines your mixed-precision SPICE project and where you should focus your optimization efforts.

### The Power of KLU for Circuit Simulation

While SuperLU is excellent for general sparse matrices, **KLU** is the undisputed gold standard for circuit simulation. It was specifically authored to handle the exact pathology of Modified Nodal Analysis (MNA) matrices:

* **Extreme Sparsity:** Circuit matrices are famously empty, often containing less than 5% non-zeros.
* **Asymmetry:** MNA matrices are highly asymmetric in both structure and numerical values.
* **Zero-Valued Diagonals:** The presence of ideal voltage sources and inductors inevitably places zeros on the main diagonal, which breaks standard solvers.

KLU explicitly targets these properties, utilizing a block triangular form (BTF) permutation that isolates the tightly coupled dense blocks of a circuit (like sub-circuits or complex transistor models) from the sparse interconnects. This makes the direct LU factorization significantly faster than general-purpose solvers.

### Integrating Mixed-Precision Direct Solvers

With direct solvers natively available in MTL5, your simulation loop becomes incredibly robust. However, there is a critical systems-engineering nuance to keep in mind regarding your custom arithmetic types:

* **The Templating Boundary:** Historically, the original SuiteSparse, KLU, and SuperLU packages are C libraries inherently hardcoded for standard floating-point arrays. For your experiment to work without casting bottlenecks, you will be relying heavily on MTL5's modernized, C++20 templated implementations of these direct solvers so they natively accept your `posit<32,2>` or LNS types.
* **Factorization vs. Accumulation:** Even within a direct solver, the posit **quire** (super accumulator) remains your most valuable asset. The bulk of the computational cost and numerical degradation in KLU/SuperLU occurs during the inner-loop multiply-accumulate (MAC) operations of the Gaussian elimination phase. If MTL5’s direct routines leverage the quire for these specific dot products, you will drastically reduce the rounding errors that normally force standard SPICE simulators to fall back on expensive pivoting algorithms or microscopically small time-steps.

### Your Updated Architectural Blueprint

With direct solvers in play, your custom C++ simulator has a highly realistic and clean path to execution:

1. **Device Evaluation (OpenVAF/Custom Backend):** Generates templated C++ device models that calculate transconductance and charge natively using posits.
2. **Matrix Assembly:** Populates the MTL5 sparse matrices, utilizing the quire to prevent catastrophic cancellation when stamping highly disparate conductances into the same node.
3. **Linear Solve (MTL5 + KLU):** Performs the direct sparse LU factorization natively on your custom arithmetic types, entirely avoiding the non-convergence pitfalls of iterative Krylov methods.
4. **Newton-Raphson Update:** Computes the delta vector, updates the voltages/currents, and checks for convergence before advancing the time-step.

Building this entirely within the modern C++20 paradigms of MTL5 will yield a remarkably clean, performant, and type-safe EDA tool. It is the perfect testbed for proving that custom number systems can fundamentally outclass standard IEEE 754 floats in electronic design automation.