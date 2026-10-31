# Givens rotations and Householder reflectors: a history

Two orthogonal transformations do most of the work in dense numerical linear
algebra. A Givens rotation zeroes one entry by rotating two rows. A
Householder reflector zeroes a whole column below the diagonal by reflecting
it through a hyperplane. QR, least squares, Hessenberg and tridiagonal
reduction, the QR algorithm for eigenvalues, the SVD and square-root Kalman
filters are all built from one or the other.

This page traces both from their origins to the modern safe-scaling
algorithms, then surveys what is known about them in arithmetic other than
IEEE binary32 and binary64. It is the background for the epic #666 and its
studies in mp-numerics. The last section says where MTL5's own
implementations stand. The geometry is animated in
[givens-householder-pictures.md](givens-householder-pictures.md).

Every reference below was checked against Crossref, OpenAlex, arXiv or the
publisher, and its DOI is given where one exists. A few details could not be
confirmed. They are kept, because they are part of the story as it is usually
told, but each is marked **(unconfirmed)** where it appears and is collected
in [Unconfirmed details](#unconfirmed-details) at the end.

## 1. Plane rotations before computers: Jacobi, 1846

C. G. J. Jacobi diagonalized a symmetric matrix by a sequence of plane
rotations, each chosen to annihilate one off-diagonal pair:

$$
G(i,j,\theta)^T A\, G(i,j,\theta), \qquad
G = \begin{pmatrix} c & s \\ -s & c \end{pmatrix}
\text{ in rows and columns } i, j,\quad c^2 + s^2 = 1 .
$$

A later rotation can refill an entry an earlier one zeroed, so the method
iterates, but the off-diagonal sum of squares falls monotonically and the
iteration converges. Jacobi used it by hand to compute secular perturbations
in astronomy.

- C. G. J. Jacobi, "Über ein leichtes Verfahren, die in der Theorie der
  Säcularstörungen vorkommenden Gleichungen numerisch aufzulösen",
  *J. reine angew. Math.* 30 (1846) 51–94.
  [doi:10.1515/crll.1846.30.51](https://doi.org/10.1515/crll.1846.30.51)

## 2. Givens: rotations as a direct method, 1954–1958

Wallace Givens, at Oak Ridge, turned Jacobi's iteration into a finite
algorithm. Instead of chasing every off-diagonal entry, Givens chose an order
of rotations under which no zero is refilled. A symmetric
matrix then reaches tridiagonal form in a fixed number of steps, after which
its eigenvalues are found by Sturm-sequence bisection. The 1958 paper applied the
same idea to a general matrix, reducing it to triangular form: QR by
rotations.

- W. Givens, *Numerical Computation of the Characteristic Values of a Real
  Symmetric Matrix*, Oak Ridge National Laboratory report ORNL-1574,
  3 March 1954. Available from the UNT Digital Library; no DOI.
- W. Givens, "Computation of plane unitary rotations transforming a general
  matrix to triangular form", *J. SIAM* 6(1) (1958) 26–50.
  [doi:10.1137/0106004](https://doi.org/10.1137/0106004). Crossref records
  the title with the misspelling "Plain".

Two properties made rotations last. Each one touches only two rows, so a
rotation can zero an entry while leaving a sparse or banded structure intact.
And because the transformation is orthogonal, the rounding errors it commits
are not amplified by the conditioning of the matrix being reduced.

The formula for the rotation also contains its weakness. Computing

$$
r = \sqrt{a^2 + b^2}, \qquad c = a/r, \qquad s = b/r
$$

directly squares $a$ and $b$. The squares overflow or underflow long before
$r$ itself would, so a careful implementation scales the inputs first. That
problem took until 2017 to settle (section 6).

## 3. Householder: reflections, 1958

Alston Householder observed that a single elementary reflector

$$
H = I - \tau\, v v^T, \qquad \tau = \frac{2}{v^T v},
$$

maps a vector $x$ onto a multiple of $e_1$ when $v = x \mp \lVert x \rVert e_1$.
One reflector does what a column of rotations does, and for a dense QR it does
it with about two thirds of the arithmetic. A sequence of reflectors triangularizes
a matrix: the QR factorization in its modern form.

- A. S. Householder, "Unitary Triangularization of a Nonsymmetric Matrix",
  *J. ACM* 5(4) (1958) 339–342.
  [doi:10.1145/320941.320947](https://doi.org/10.1145/320941.320947)

The sign in $v = x \mp \lVert x \rVert e_1$ is chosen to avoid cancellation,
so that the first entry of $v$ is a sum of two quantities of the same sign.
LAPACK's `dlarfg` makes this choice, mapping $x$ onto
$-\operatorname{sign}(x_0)\lVert x \rVert e_1$. B. N. Parlett's alternative
keeps the other sign, mapping a vector with $x_0 > 0$ onto
$+\lVert x \rVert e_1$, and avoids the cancellation by rewriting
$x_0 - \lVert x \rVert = -\sigma/(x_0 + \lVert x \rVert)$ with
$\sigma = \sum_{i \ge 1} x_i^2$. MTL5 uses Parlett's form; how it rounds in
narrow formats is taken up in [Where MTL5 sits](#where-mtl5-sits).

## 4. Wilkinson and Golub: the methods become standard, 1960–1965

James Wilkinson applied Householder's reflectors to the symmetric eigenvalue
problem, reducing a symmetric matrix to tridiagonal form with about half the
work of Givens' rotations. Wilkinson's 1965 book analysed both
transformations and established that they are backward stable: the computed
result is exactly orthogonally similar to a matrix close to the original. It
made backward error analysis the standard way to judge such algorithms.

Gene Golub used Householder QR to solve linear least-squares problems
without forming the normal equations $A^TA$, whose condition number is the
square of $A$'s.

- J. H. Wilkinson, "Householder's Method for the Solution of the Algebraic
  Eigenproblem", *Computer J.* 3(1) (1960) 23–27.
  [doi:10.1093/comjnl/3.1.23](https://doi.org/10.1093/comjnl/3.1.23)
- J. H. Wilkinson, *The Algebraic Eigenvalue Problem*, Clarendon Press,
  Oxford, 1965.
- G. H. Golub, "Numerical methods for solving linear least squares problems",
  *Numer. Math.* 7(3) (1965) 206–216.
  [doi:10.1007/BF01436075](https://doi.org/10.1007/BF01436075)

The modern statement of the error analysis is chapter 19 of Higham. For
Householder QR of an $m \times n$ matrix, the computed $\hat R$ is the exact
triangular factor of $A + \Delta A$, with each column of $\Delta A$ bounded by
a small multiple of $mn\,u$ times the corresponding column of $A$. The bound
depends only on the unit roundoff $u$. Whether it carries over to a format
without a fixed $u$, such as a posit, is one of the open questions of
section 10.

- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
  SIAM, 2002, ch. 19 "QR Factorization" (pp. 353–380) and ch. 20 "The Least
  Squares Problem" (pp. 381–406).
  [doi:10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)

## 5. Square-root-free rotations: Gentleman and Hammarling, 1973–1974

On the machines of the early 1970s a square root cost many multiplies, and a
Givens QR needs one per rotation. W. Morven Gentleman showed how to avoid
them. The rows are kept in a factored form $D^{1/2} \bar R$ with a separate
diagonal scaling $D$, and each rotation updates $\bar R$ and $D$ using
multiplies and divides only. Sven Hammarling refined the formulation the
following year. These are the "fast Givens" rotations of the textbooks.

- W. M. Gentleman, "Least Squares Computations by Givens Transformations
  Without Square Roots", *IMA J. Appl. Math.* 12(3) (1973) 329–336.
  [doi:10.1093/imamat/12.3.329](https://doi.org/10.1093/imamat/12.3.329)
- S. Hammarling, "A Note on Modifications to the Givens Plane Rotation",
  *IMA J. Appl. Math.* 13(2) (1974) 215–218.
  [doi:10.1093/imamat/13.2.215](https://doi.org/10.1093/imamat/13.2.215)

The price is range. The entries of $D$ grow or shrink geometrically with the
number of rotations applied, so fast Givens must monitor them and rescale
before they overflow or underflow. In binary64 that is an occasional
bookkeeping step. In a format with an 8-bit or 5-bit exponent it happens
almost immediately, and in a format that saturates instead of overflowing the
monitor never sees an infinity to react to.

## 6. Computing a rotation safely: Bindel et al. 2002, Anderson 2017

Computing $c$, $s$ and $r$ from $a$ and $b$ looks trivial and is not. A
careful routine must avoid overflow and underflow in $a^2 + b^2$, return
consistent results when $a$ or $b$ is zero, infinite or NaN, choose signs so
that $c$ and $s$ vary continuously with the inputs, and handle complex
arguments. LAPACK's original `slartg` and `clartg` fell short of these
requirements in several ways.

David Bindel, James Demmel, William Kahan and Osni Marques set out the
requirements and gave algorithms that meet them, scaling the inputs by powers
of the radix so that no rounding is introduced. Edward Anderson then derived
scaling constants, in the line of J. L. Blue's 1978 algorithm for the
Euclidean norm, that make rotations and norms safe with fewer branches and
less scaling. The resulting routine replaced `dlartg` in Reference LAPACK as `dlartg.f90`, which
cites the 2017 paper. Which LAPACK release first shipped it, possibly 3.10, is
**(unconfirmed)**, as is whether the Bindel et al. algorithms were themselves
adopted into a LAPACK release before Anderson's rework.

- D. Bindel, J. Demmel, W. Kahan, O. Marques, "On computing Givens rotations
  reliably and efficiently", *ACM TOMS* 28(2) (2002) 206–238.
  [doi:10.1145/567806.567809](https://doi.org/10.1145/567806.567809). First
  circulated as LAPACK Working Note 148 (2000).
- E. Anderson, "Algorithm 978: Safe Scaling in the Level 1 BLAS",
  *ACM TOMS* 44(1) (2017) 1–28.
  [doi:10.1145/3061665](https://doi.org/10.1145/3061665)

Both papers derive their thresholds from the exponent range and precision of
IEEE binary32 and binary64. A posit has no fixed precision: it is most precise
near 1 and loses fraction bits toward the ends of its range. A saturating
format has no overflow to guard against but silently clamps instead. Nobody
appears to have re-derived the safe-scaling constants for either (section 10).

## 7. Blocking, hardware and filters

### Blocked reflectors: the WY representations, 1987–1989

A single reflector is a rank-1 update, a Level 2 BLAS operation that runs at
memory speed. Christian Bischof and Charles Van Loan showed that a product of
$k$ reflectors can be written $I + W Y^T$, so that applying all $k$ at once is
a matrix multiply. Robert Schreiber and Van Loan gave the storage-efficient
compact form $I - V T V^T$ with $T$ upper triangular, which LAPACK's
`dlarft` and `dlarfb` use. Blocking changes the order in which roundings
happen, so a blocked and an unblocked QR give different bits.

- C. Bischof, C. Van Loan, "The WY Representation for Products of
  Householder Matrices", *SIAM J. Sci. Stat. Comput.* 8(1) (1987) s2–s13.
  [doi:10.1137/0908009](https://doi.org/10.1137/0908009)
- R. Schreiber, C. Van Loan, "A Storage-Efficient WY Representation for
  Products of Householder Transformations", *SIAM J. Sci. Stat. Comput.*
  10(1) (1989) 53–57.
  [doi:10.1137/0910005](https://doi.org/10.1137/0910005)

### Rotations in hardware: CORDIC and systolic arrays, 1959–1992

Jack Volder's CORDIC computes a rotation by a sequence of shift-and-add
micro-rotations, with no multiplier, and remains the standard way to build
rotations in fixed-point hardware. Gentleman and H. T. Kung mapped Givens QR
onto a triangular systolic array, in which the boundary cells compute
rotations and the internal cells apply them. John McWhirter used
that array for recursive least squares (QRD-RLS), the basis of adaptive
beamformers and equalizers. Yu Hen Hu analysed the quantization error CORDIC
commits. The fixed-point behaviour of rotations is therefore well studied;
section 9 lists the newer formats for which it is not.

- J. E. Volder, "The CORDIC Trigonometric Computing Technique",
  *IRE Trans. Electronic Computers* EC-8(3) (1959) 330–334.
  [doi:10.1109/TEC.1959.5222693](https://doi.org/10.1109/TEC.1959.5222693)
- W. M. Gentleman, H. T. Kung, "Matrix Triangularization by Systolic Arrays",
  *Proc. SPIE* 298 (1981) 19–26.
  [doi:10.1117/12.932507](https://doi.org/10.1117/12.932507)
- J. G. McWhirter, "Recursive Least-Squares Minimization Using a Systolic
  Array", *Proc. SPIE* 431 (1983) 105–113.
  [doi:10.1117/12.936448](https://doi.org/10.1117/12.936448)
- Y. H. Hu, "The quantization effects of the CORDIC algorithm",
  *IEEE Trans. Signal Process.* 40(4) (1992) 834–844.
  [doi:10.1109/78.127956](https://doi.org/10.1109/78.127956)

### Square-root Kalman filters, 1963–1986

The Kalman filter's covariance update loses symmetry and definiteness in
finite precision ([gemmt-origins.md](gemmt-origins.md) traces the Joseph form,
one remedy). The other remedy is to propagate a factor of the covariance
instead of the covariance itself: a square root $S$ with $P = SS^T$, or a
$UDU^T$ factorization. A factor squares back to a positive semi-definite
matrix whatever rounding it carries, and its condition number is the square
root of $P$'s, so it needs about half the digits.

James Potter's square-root measurement update is usually traced to an MIT
Instrumentation Laboratory memo written for Apollo navigation, and to Richard
Battin's 1964 account of it; the memo, its date and Battin's account are
**(unconfirmed)**. The confirmed published version is Potter and Stern's 1963
AIAA paper. Kaminski,
Bryson and Schmidt surveyed the square-root filters in 1971. Gerald Bierman's
1977 book gave the $UDU^T$ measurement update, and C. L. Thornton the
matching time update by modified weighted Gram–Schmidt, in a JPL technical
memorandum in 1976 and with Bierman in 1977. Whether Thornton's 1976 work was
also a UCLA doctoral dissertation is **(unconfirmed)**. Michel Verhaegen
and Paul Van Dooren compared the implementations in 1986 and showed why the
factored forms are more robust.

Rotations and reflectors enter these filters directly. The square-root time
and measurement updates are QR factorizations of a stacked array, the "array
algorithms". For the time update $P^- = F P F^T + Q$ with $P = SS^T$ and
$Q = S_Q S_Q^T$, triangularize
$\begin{bmatrix} S^T F^T \\ S_Q^T \end{bmatrix}$ by Householder or Givens:
the triangular factor $R$ satisfies $R^T R = F S S^T F^T + S_Q S_Q^T = P^-$,
so $R^T$ is the new square root. Their
behaviour in narrow arithmetic is the subject of mp-numerics #15.

- J. E. Potter, R. G. Stern, "Statistical Filtering of Space Navigation
  Measurements", *AIAA Guidance and Control Conference*, 1963.
  [doi:10.2514/6.1963-333](https://doi.org/10.2514/6.1963-333)
- P. Kaminski, A. Bryson, S. Schmidt, "Discrete square root filtering: A
  survey of current techniques", *IEEE Trans. Autom. Control* 16(6) (1971)
  727–736.
  [doi:10.1109/TAC.1971.1099816](https://doi.org/10.1109/TAC.1971.1099816)
- G. J. Bierman, *Factorization Methods for Discrete Sequential Estimation*,
  Academic Press, 1977.
- C. L. Thornton, *Triangular Covariance Factorizations for Kalman
  Filtering*, JPL Technical Memorandum 33-798, 1976 (NASA NTRS).
- C. L. Thornton, G. J. Bierman, "Gram-Schmidt algorithms for covariance
  propagation", *Int. J. Control* 25(2) (1977) 243–260.
  [doi:10.1080/00207177708922225](https://doi.org/10.1080/00207177708922225)
- M. Verhaegen, P. Van Dooren, "Numerical aspects of different Kalman filter
  implementations", *IEEE Trans. Autom. Control* 31(10) (1986) 907–917.
  [doi:10.1109/TAC.1986.1104128](https://doi.org/10.1109/TAC.1986.1104128)

## 8. Which transformation for which job

The two transformations do the same thing, introduce zeros orthogonally, at
different granularities, and the granularity decides where each is used.

- **Dense QR and least squares: Householder.** One reflector clears a whole
  column, at about two thirds of the cost of the rotations that would clear
  it. Through the compact WY form a block of reflectors is applied as matrix
  multiplies, so the factorization runs at Level 3 BLAS speed. LAPACK's
  `dgeqrf` works this way. MTL5's `qr_factor` hands `float` and `double`
  matrices to `geqrf` when LAPACK dispatch is enabled; its generic loop,
  which every other scalar type runs, applies one reflector at a time.
- **Sparse, banded, structured and updating problems: Givens.** A rotation
  touches two rows and zeroes one chosen entry, so it creates no fill outside
  those rows. Adding or deleting a row of a least-squares problem, updating a
  QR factorization after a rank-one change, the Hessenberg least-squares
  problem inside GMRES and the tridiagonal one inside MINRES are all a short
  sequence of rotations. So is each cell of a systolic QR array.
- **Eigenvalues and the SVD: both.** The reduction to Hessenberg,
  tridiagonal or bidiagonal form is done by reflectors. The QR iterations that
  follow chase a bulge down the band: the symmetric tridiagonal QR step and
  the Golub–Kahan SVD step use a rotation per position, and the Francis
  double-shift step on a Hessenberg matrix uses a 3×3 reflector per position.

Both are orthogonal, so both preserve the 2-norm of what they act on. A
rounding error committed at one step is carried forward at its own size
rather than multiplied by the conditioning of the matrix, which is what makes
the backward-stability results of section 4 possible. Gaussian elimination
has no such property: its multipliers can grow, and its stability depends on
pivoting.

## 9. Narrow and non-IEEE arithmetic: what has been studied

**Half precision and bfloat16.** Householder QR in fp16 and bf16 is well
covered. Yang, Fox and Sanders analysed rounding-error bounds for QR in
low precision, Higham and Mary surveyed mixed-precision linear algebra, and
Connolly and Higham studied probabilistic rounding error analysis applied to
Householder QR. Zhang and Wu built QR on fp16 tensor cores. Givens
rotations in these formats have had less attention.

- L. M. Yang, A. Fox, G. Sanders, "Rounding Error Analysis of Mixed
  Precision Block Householder QR Algorithms", *SIAM J. Sci. Comput.* 43(3)
  (2021) A1723–A1753.
  [doi:10.1137/19M1296367](https://doi.org/10.1137/19M1296367)
- N. J. Higham, T. Mary, "Mixed precision algorithms in numerical linear
  algebra", *Acta Numerica* 31 (2022) 347–414.
  [doi:10.1017/S0962492922000022](https://doi.org/10.1017/S0962492922000022)
- M. P. Connolly, N. J. Higham, "Probabilistic Rounding Error Analysis of
  Householder QR Factorization", *SIAM J. Matrix Anal. Appl.* 44(3) (2023)
  1146–1163.
  [doi:10.1137/22M1514817](https://doi.org/10.1137/22M1514817)
- S. Zhang, P. Wu, fp16 tensor-core QR, arXiv:1912.05508.

**Posits and takums.** The only work found that runs Householder QR in posit
or takum arithmetic is Hunhold and Quinlan (ARITH 2025). It compares posit,
takum, bfloat16, float16 and OFP8 E4M3 in sparse LU, sparse Householder QR,
GMRES with ILU, and mixed-precision iterative refinement. For the QR,
SuiteSparseQR runs once in float64 to choose the row and column permutations;
the factorization itself is the authors' own unblocked Householder QR, one
reflector per column, in the target format. At equal width the posits and
takums are generally as accurate as the IEEE-style formats or more, with
exceptions the paper reports: posit16 is less accurate in LU for the lower
quartile of matrices, it occasionally lags in GMRES, and takums are
occasionally marginally less accurate than posits. The paper reports
solution accuracy, not the loss of orthogonality of $Q$, and has no Givens
rotations. Hunhold, Quinlan and Wesner's Arnoldi study (SC25) uses
the same formats, but its dense Schur and QR core runs in IEEE arithmetic
through LAPACK. The ARITH paper does not say whether its OFP8 conversions
saturate or overflow; it records any overflow as a failed run. What the SC25
study does is **(unconfirmed)**. Other posit studies cover Cholesky, LU, conjugate
gradients and iterative refinement only.

- L. Hunhold, J. Quinlan, posit, takum, bfloat16, float16 and OFP8 in sparse
  linear solvers, *ARITH 2025*, 61–68.
  [doi:10.1109/ARITH64983.2025.00019](https://doi.org/10.1109/ARITH64983.2025.00019),
  arXiv:2412.20268.
- L. Hunhold, J. Quinlan, S. Wesner, Arnoldi in OFP8, bfloat16, posit and
  takum arithmetic, *SC25*, 681–694.
  [doi:10.1145/3712285.3759837](https://doi.org/10.1145/3712285.3759837),
  arXiv:2504.21130.
- L. Hunhold, "Beating Posits at Their Own Game: Takum Arithmetic",
  *CoNGA 2024*.
  [doi:10.1007/978-3-031-72709-2_1](https://doi.org/10.1007/978-3-031-72709-2_1)
- J. L. Gustafson, I. Yonemoto, "Beating Floating Point at its Own Game:
  Posit Arithmetic", *Supercomputing Frontiers and Innovations* 4(2) (2017).
  [doi:10.14529/jsfi170206](https://doi.org/10.14529/jsfi170206)
- J. Buoncristiani, S. Shah, D. Donofrio, J. Shalf, *IPDPS 2020*, 612–621
  (posit Cholesky, CG, iterative refinement).
  [doi:10.1109/IPDPS47924.2020.00069](https://doi.org/10.1109/IPDPS47924.2020.00069)
- N. Nakasato et al., *HPC Asia 2024*, 62–72 (posit32 Cholesky and LU on
  FPGA and GPU).
  [doi:10.1145/3635035.3635046](https://doi.org/10.1145/3635035.3635046)
- J. Quinlan, E. T. L. Omtzigt, posit16 LU with iterative refinement,
  arXiv:2408.13400 (2024).
- M. Klöwer, P. D. Düben, T. N. Palmer, "Posits as an alternative to floats
  for weather and climate models", *CoNGA 2019*.
  [doi:10.1145/3316279.3316281](https://doi.org/10.1145/3316279.3316281)

**A posit Kalman filter.** Ilangovan, Rayan and Saxena, "Improving the
Stability of Kalman Filters with Posit Arithmetic" (CoNGA 2023), is the one
paper found on Kalman filtering in posits. Its
[conference slides](https://www.posithub.org/conga/2023/assets/pdf/KalmanFilter.pdf)
show the conventional filter only, with the update $P = (I - KH)P$. Joseph,
Potter, Carlson and $UDU^T$ forms appear only in an operation-count
comparison, as costlier alternatives. Two small filters, a rotating shaft
(2 states) and the relative motion of two shafts (4 states), run in 32-bit
posits with $es$ from 1 to 6 against binary32, with binary64 as the
reference. A posit32 with a suitable $es$ keeps the filter stable where
binary32 diverges, but the right $es$ depends on the range of the data:
posit<32,1> fails when the covariance starts near $10^9$, where posit<32,4>
works. The authors propose choosing $es$ by the average precision over the
range the application uses. The slides were read; the proceedings paper,
which is paywalled, was not.
[doi:10.1007/978-3-031-32180-1_9](https://doi.org/10.1007/978-3-031-32180-1_9)

**Logarithmic number systems.** Givens rotations in LNS appear only in
hardware signal-processing work: an LNS QRD-RLS processor (Sacha and Irwin,
1998), the European Logarithmic Microprocessor and its QR-RLS application
(Coleman et al., 2000–2008), and an LNS complex QR decomposition for MIMO
receivers (Rust, Ludwig and Paul, 2013). These measure energy, area and
accuracy for one application. None analyses the rotation numerically, and no
LNS Householder QR was found.

- J. R. Sacha, M. J. Irwin, *ISLPED 1998*, 256–261.
  [doi:10.1145/280756.280926](https://doi.org/10.1145/280756.280926)
- J. N. Coleman et al., *IEEE Trans. Comput.* 49(7) (2000).
  [doi:10.1109/12.863040](https://doi.org/10.1109/12.863040);
  *Asilomar 2001*, 155–159,
  [doi:10.1109/ACSSC.2001.986897](https://doi.org/10.1109/ACSSC.2001.986897);
  *IEEE Trans. Comput.* 57(4) (2008),
  [doi:10.1109/TC.2007.70791](https://doi.org/10.1109/TC.2007.70791)
- J. Rust, F. Ludwig, S. Paul, *DATE 2013*, 97–102.
  [doi:10.7873/DATE.2013.034](https://doi.org/10.7873/DATE.2013.034)
- Johnson, LNS inner products, *ARITH 2020*, 25–32.
  [doi:10.1109/ARITH48897.2020.00013](https://doi.org/10.1109/ARITH48897.2020.00013)

**Custom-width floating point.** Wang and Leeser built Givens QR on FPGAs
with variable-precision floating point, and Fang and Leeser's VFloat library
generalizes the formats. These papers report throughput and resource use, not
how accuracy varies with width.

- X. Wang, M. Leeser, *FPGA 2008*, 260,
  [doi:10.1145/1344671.1344718](https://doi.org/10.1145/1344671.1344718);
  *ACM TECS* 9(1) (2009),
  [doi:10.1145/1596532.1596535](https://doi.org/10.1145/1596532.1596535)
- X. Fang, M. Leeser, VFloat, *ACM TRETS* 9(3) (2016).
  [doi:10.1145/2851507](https://doi.org/10.1145/2851507)
- Amin-Nejad et al., *EJECE* 3(5) (2019).
  [doi:10.24018/ejece.2019.3.5.127](https://doi.org/10.24018/ejece.2019.3.5.127)

## 10. What the literature does not cover

The searches behind this page used Crossref, OpenAlex, arXiv and the web.
Full-text coverage of IEEE Xplore and Springer is weak, so "nothing found"
below means nothing indexed, not proof that nothing exists. With that caveat,
these appear to be open:

| Topic | Found | Confidence it is open |
|---|---|---|
| Givens rotations in posit or takum | nothing | high |
| Dense Householder QR in posit or takum, with orthogonality loss measured | only sparse QR, solution accuracy only (Hunhold and Quinlan) | medium–high |
| Householder QR in LNS | nothing | medium–high |
| Accuracy against width for odd-width IEEE-like formats, and the effect of subnormals on rotations and reflectors | FPGA throughput only | high |
| Saturating floating point (clamp instead of overflow) in hypot, rotation norms and reflector scaling | nothing | high |
| Square-root and $UDU^T$ Kalman filters in any non-IEEE format | nothing (the one posit Kalman study uses the conventional form) | high |
| Dense eigenvalue and SVD kernels (Hessenberg, tridiagonal and bidiagonal reduction, QR iteration) in posit, takum or LNS | Arnoldi with posit and takum Krylov vectors, but its dense core in IEEE (Hunhold, Quinlan and Wesner) | high |
| Safe-scaling rotation constants (Bindel et al., Anderson) re-derived for tapered or saturating formats | nothing | high |

The studies in mp-numerics #13 (rotations and reflectors as operators), #14
(the decompositions built on them) and #15 (ill-conditioned Kalman filters)
are designed to fill these.

## Where MTL5 sits

MTL5 is generic over the scalar type, so each of its rotations and reflectors
is already an experiment in whatever arithmetic the caller supplies. They do
not all take the same care.

- **Householder reflectors** (`include/mtl/operation/householder.hpp`)
  follow LAPACK's `dlarfg` by default in two respects: the column is scaled
  before its norm is formed, and $v$ is normalized to $v_0 = 1$. They differ
  in sign. MTL5 uses Parlett's formula $v_0 = -\sigma/(x_0 + \lVert x \rVert)$,
  which maps a column with $x_0 > 0$ onto $+\lVert x \rVert e_1$, where
  `dlarfg` maps it onto $-\lVert x \rVert e_1$; for $x = (1, 1)^T$ the two give
  different reflectors. Both avoid cancellation, but they round differently,
  which matters when comparing MTL5 against LAPACK in narrow formats. A textbook form
  is available as an option; [householder-reflector-forms.md](../numerical/householder-reflector-forms.md)
  compares the two. Since 5.21.0, `householder()` checks that the $\tau$ and $v$ it
  computed define a reflector the type can represent, and returns the
  identity when they do not. Without the check, fp8 produced NaN and 16-bit
  fixed point a far from orthogonal $Q$. The rule and its
  measured effect are in [sylvester.md](sylvester.md).
- **Givens rotations** are computed with the unscaled formula
  $r = \sqrt{a^2 + b^2}$ in the GMRES Hessenberg QR
  (`include/mtl/operation/givens.hpp`), in MINRES and in the implicit QR
  step of the symmetric tridiagonal eigensolver
  (`include/mtl/operation/eigenvalue_symmetric.hpp`). None of them scales its
  inputs the way Bindel et al. and Anderson do. In binary64 the squares
  overflow only for inputs near $10^{154}$; in an 8-bit or 16-bit format they
  overflow, saturate or underflow for ordinary inputs.
- **The nonsymmetric eigensolver** (`include/mtl/detail/francis_qr.hpp`)
  uses a generic `lapy2`, LAPACK's scaled $\sqrt{x^2 + y^2}$, written for any
  scalar type because `std::hypot` exists only for the built-in ones.

So MTL5 has a scaled reflector with a fallback for narrow types, a scaled
hypot in one eigensolver, and unscaled rotations elsewhere. Measuring what the
unscaled rotations cost in each format, and what a safe-scaling rotation would
have to look like for posits, takums and saturating formats, is the first
study of mp-numerics #13. Its results, across 65 types, are in
[Givens and Householder across number systems](givens-householder-study.md).

## Unconfirmed details

These are part of the usual account but could not be checked against a
primary source. They are kept above, marked **(unconfirmed)**, and should be
corrected or removed when a source is found.

1. **Potter's MIT memo.** The square-root filter is attributed to an MIT
   Instrumentation Laboratory memo by James Potter, around 1962–1963, written
   for Apollo navigation, and to its description in Richard Battin's 1964
   book *Astronautical Guidance*.
   Neither the memo's number and date nor Battin's account could be confirmed.
   The 1963 AIAA paper by Potter and Stern is confirmed.
2. **Thornton's dissertation.** Whether C. L. Thornton's 1976 work on
   triangular covariance factorizations was also a UCLA PhD dissertation.
   JPL Technical Memorandum 33-798 (1976) is confirmed.
3. **The LAPACK release that first shipped Anderson's `dlartg.f90`.** The
   routine is in Reference LAPACK, is authored by Anderson and cites the 2017
   paper; the first release to include it, possibly 3.10, is not confirmed.
   Nor is whether the Bindel et al. algorithms of 2002 were themselves
   adopted into a LAPACK release before it.
4. **OFP8 overflow semantics in the SC25 Arnoldi study.** Whether its OFP8
   formats saturate on overflow or produce infinity or NaN. The OCP 8-bit
   specification defines both a saturating and a non-saturating conversion,
   and how an OFP8 failure should be read depends on which one was used. The
   ARITH 2025 paper was checked: it does not say, and records an overflow as
   a failed run.
