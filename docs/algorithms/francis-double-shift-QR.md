# Francis double-shift QR Algorithm

When people refer to a "bug" or "failure mode" in the **double-shift QR algorithm** (also known as the Francis double-shift QR algorithm), they are typically referring to one of two famous numerical phenomena that cause the algorithm to stall or fail to converge: **perfect/unstable fixed points** (including the famous Cleve Moler 4x4 counterexample) or **shift blurring** in multishift scaling.

Here is a breakdown of what these issues are and why they happen.

---

## 1. The Fixed-Point Stagnation "Bug" (The 4x4 Counterexample)

The standard implicit double-shift QR algorithm operates on a real upper Hessenberg matrix. To accelerate convergence, it looks at the trailing $2 \times 2$ bottom-right submatrix and computes its eigenvalues to use as a complex conjugate pair of "shifts."

While this usually leads to rapid quadratic convergence, it has an Achilles' heel: **certain matrix symmetries can cause the algorithm to enter a perfect periodic cycle or an unstable fixed point**, where the shifts do not change from iteration to iteration, and the subdiagonal elements refuse to zero out (preventing deflation).

### The Famous Matrix

As documented by Cleve Moler (the creator of MATLAB), a classic example that triggers this stagnation is the $4 \times 4$ companion matrix of the polynomial $p(x) = x^4 - 2x^2 + 1$:

$$A = \begin{bmatrix} 0 & 2 & 0 & -1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$

* **The Eigenvalues:** This matrix has double roots at $+1$ and $-1$ (both with multiplicity 2).
* **The Trap:** Because of the tight symmetry and the double roots, the lower $2 \times 2$ submatrix continuously yields shifts that change the matrix slightly but *never* break the symmetry. The algorithm loops endlessly without converging, eventually throwing an error like `Solution will not converge`.

### The Fix

To patch this inherent mathematical trap, linear algebra libraries like **LAPACK** (and by extension, MATLAB, Python's NumPy/SciPy, and Julia) use **ad hoc (exceptional) shifts**. If the algorithm detects that it has performed a certain number of iterations (e.g., 10 or 20) without deflating an eigenvalue, it artificially injects a heuristically chosen (deterministic, not random) "exceptional shift" to break the symmetry, kick the matrix out of the fixed point, and force convergence. The classic EISPACK/LAPACK `hqr` choice is $s = 0.75\,(|h_{n,n-1}| + |h_{n-1,n-2}|)$ — and MTL5's own in-house `eigenvalue()` uses exactly this deterministic exceptional shift at iterations 10 and 20 (see the Francis double-shift fix, #209). This is independent of the optional LAPACK `geev` dispatch, which simply hands the whole problem to LAPACK when it is available.

---

## 2. The "Shift Blurring" Bug (In Multishift Extensions)

As modern computer architectures evolved, researchers wanted to use more than just two shifts at a time (e.g., executing a *multishift* QR algorithm with 10, 20, or 100 simultaneous shifts) so they could utilize high-performance Level 3 BLAS (matrix-matrix multiplication) operations.

However, when expanding the double-shift logic into a large multi-shift "bulge," engineers hit a massive numerical wall known as **shift blurring**.

* **The Problem:** When you accumulate many shifts into a single large bulge chased down the matrix, accumulated round-off errors exponentially degrade the shift information.
* **The Mechanism:** Mathematically, the shifts get "smeared" together because the implicit polynomial becomes incredibly ill-conditioned. The algorithm loses track of the precise eigenvalues it was searching for, causing convergence to utterly stall.

### The Fix

This was solved in modern LAPACK implementations (via the work of Braman, Byers, and Mathias) by replacing one giant bulge with a **tightly packed chain of many small $2 \times 2$ or $4 \times 4$ bulges**. This keeps the individual double-shifts "well-focused" and mathematically pristine while still allowing the CPU to parallelize the operations.