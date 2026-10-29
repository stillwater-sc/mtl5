# Origins of gemmt: the Joseph form and the triangular-output product

MTL5's `gemmt` and `congruence` ([gemmt.md](gemmt.md)) came from a request by
branes-ai/cortex for the covariance updates of a multi-state constraint Kalman
filter (#656). The operations they compute are older than any BLAS: they are
the computational kernel of the Joseph-stabilized covariance update from the
1960s. This page traces that line from estimation theory to the BLAS
extension, and says where MTL5 sits on it.

## 1. The Joseph-stabilized covariance update

After a measurement, the Kalman filter updates the state-error covariance $P$
with the gain $K_k$, the measurement matrix $H_k$ and the measurement-noise
covariance $R_k$. The textbook update is

$$
P_{k\vert k} = (I - K_k H_k)\, P_{k\vert k-1},
$$

or equivalently, with the innovation covariance
$S_k = H_k P_{k\vert k-1} H_k^T + R_k$,

$$
P_{k\vert k} = P_{k\vert k-1} - K_k S_k K_k^T .
$$

Both forms are cheap. Both are also exact only when $K_k$ is the optimal gain,
and in finite precision they subtract nearly equal quantities. Rounding
errors then accumulate across updates. $P$ drifts away from symmetry and can
lose positive semi-definiteness, and a filter whose covariance has a negative
eigenvalue diverges. The early aerospace implementations of the Kalman filter
ran into exactly this.

Peter D. Joseph's remedy, published in R. S. Bucy and P. D. Joseph,
*Filtering for Stochastic Processes with Applications to Guidance*
(Interscience, 1968), is the form now named after him:

$$
P_{k\vert k} = (I - K_k H_k)\, P_{k\vert k-1}\, (I - K_k H_k)^T + K_k R_k K_k^T .
$$

### Why this structure matters

- **It holds for any gain.** The Joseph form is the covariance of the
  updated error for any $K_k$, not just the optimal one. The short forms are
  that covariance only at the optimal gain $K_k^\star$; with any other gain
  they report a covariance the filter does not have, wrong to first order in
  $K_k - K_k^\star$. The Joseph covariance itself is
  $P_{k\vert k}^\star + (K_k - K_k^\star)\, S_k\, (K_k - K_k^\star)^T$, so near the
  optimum a gain error, such as one from rounding, changes it only to second
  order. Far from the optimum the change is first order, but it is then the
  true covariance of a suboptimal filter.
- **Each term is a congruence.** Both terms have the shape $A M A^T$ with
  $M$ symmetric positive semi-definite: $A = I - K_k H_k$, $M = P_{k\vert k-1}$
  in the first, and $A = K_k$, $M = R_k$ in the second. In exact arithmetic a
  congruence of a positive semi-definite matrix is positive semi-definite,
  and so is the sum. The update never subtracts.
- **Floating point still needs care.** The form removes the cancellation,
  but it does not make the computed result symmetric by itself. A
  product $A M A^T$ formed by two general matrix multiplies has two triangles
  that round differently. Implementations either average the triangles,
  $(C + C^T)/2$, or form one triangle and copy it. For badly conditioned $P$,
  even the Joseph form can lose definiteness in the last bits, which is what
  the square-root (Potter, 1963) and $UDU^T$ (Bierman, 1977) filters address
  by never forming $P$ at all.
- **$R_k$ is often diagonal.** In sensor fusion, robotics and navigation, the
  measurement channels are commonly modelled as independent, which makes
  $R_k$ diagonal, and the second term is then
  $K_k \operatorname{diag}(r) K_k^T$. A block-diagonal $R_k$, whose blocks
  correlate the channels of one sensor, is a general symmetric middle term.

## 2. The gap in classical BLAS

The Level-3 BLAS (Dongarra, Du Croz, Hammarling and Duff, 1990) gave matrix
products their standard interface. Three of its routines come close to the
Joseph form's kernel, and none fits it.

- **`GEMM`** computes $C \leftarrow \alpha\, \mathrm{op}(A)\, \mathrm{op}(B) + \beta C$
  for general matrices. It forms all of $C$, so when the product is known to
  be symmetric about half of its multiply-adds produce entries that are
  already known.
- **`SYRK`** computes $C \leftarrow \alpha A A^T + \beta C$ and forms one
  triangle only. It needs both factors to be the same matrix. Reaching
  $A P A^T$ through it takes a factorization $P = L L^T$ first, and reaching
  $K \operatorname{diag}(r) K^T$ takes the scaled factor
  $K \operatorname{diag}(\sqrt{r})$, which requires $r \ge 0$ and adds a
  square root's rounding to every column.
- **`SYR2K`** computes $C \leftarrow \alpha (A B^T + B A^T) + \beta C$ from two
  distinct matrices, but only that symmetrized sum.

What is missing is the product of two *distinct* general matrices whose
result the caller knows to be symmetric, with one triangle formed. With
$T = A P$, the congruence $A P A^T$ is exactly $T A^T$. Without such a routine,
code either calls `GEMM` and discards half the work, or writes its own
triangular loop and gives up the cache blocking and vectorization of a tuned
kernel.

## 3. GEMMT, the triangular-output product

The routine that fills the gap is `GEMMT`, GEMM with a triangular output:

$$
\text{GEMMT:}\quad C \leftarrow \alpha\, \mathrm{op}(A)\, \mathrm{op}(B) + \beta C,
\quad \text{only the upper or lower triangle of } C \text{ formed.}
$$

$A$ and $B$ are independent general matrices. The routine does not check that
the product is symmetric; it forms the requested triangle and leaves the other
untouched.

It is an extension, not part of the 1990 standard, and arrived library by
library:

- **Intel MKL** shipped `?gemmt` as an extension.
- **BLIS**, the framework of Field G. Van Zee, Robert van de Geijn and
  colleagues, provides `gemmt`, and so does **AMD AOCL-BLAS**, which is built
  on BLIS.
- **OpenBLAS** added `?gemmt` later. Its 0.3.26 release, the one in Ubuntu
  24.04, checks `lda` and `ldb` against each other's dimensions and rejects
  valid calls; 0.3.27 fixed it. MTL5's CMake probe runs a call for this reason
  rather than only linking one ([gemmt.md](gemmt.md#blas-dispatch)).
- **Reference BLAS** standardized it in LAPACK 3.12 under the name `?gemmtr`.

### Diagonal middle terms and packing

A tuned GEMM does not multiply its operands where they lie. It copies panels
of them into contiguous, cache-sized buffers first, and the micro-kernel reads
only those buffers. The BLIS design makes this packing step the place where
cheap transformations of an operand can be folded in, without a separate pass
over memory.

For $K \operatorname{diag}(r) K^T$ that suggests:

1. **No materialized scaling pass.** The scaled operand
   $K \operatorname{diag}(r)$ need not exist as a matrix.
2. **Scaling during packing.** As the panels of $K$ are packed for the
   macro-kernel, each column is multiplied by its $r_j$, so the packed buffer
   holds the scaled operand directly.
3. **Triangle-aware scheduling.** Since only one triangle of $C$ is formed,
   the work per block row grows along the diagonal, and a threaded
   implementation partitions by work rather than by rows. BLIS weights its
   thread partitions this way for its triangular-output operations.

Whether a given library folds a diagonal scaling into packing for `GEMMT`
is a property of that library's implementation, not of the interface: the
BLAS signature has no argument for a diagonal middle term, so a caller who
needs one forms the scaled operand first.

## Where MTL5 sits

MTL5's `congruence(alpha, K, r, beta, C)` does what a BLAS caller does: it
forms $T = K \operatorname{diag}(r)$ explicitly, one rounding per entry, in
$O(mn)$ work, and then calls `gemmt(T, K^T)`. The scaling is a separate pass,
not a packing step. For a dense $A P A^T$, `congruence` forms $T = A P$ with
`mult` and the lower triangle of $T A^T$ with `gemmt`, then mirrors it, which
is the "form one triangle and copy it" route from section 1.

The full Joseph update is two calls:

```cpp
// C = (I - K H) P (I - K H)^T + K diag(r) K^T
mtl::congruence(IKH, P, C);                  // first term
mtl::congruence(1.0, K, r, 1.0, C);          // add the second
```

The second call reads $C$ with $\beta = 1$, which requires $C$ to be
symmetric; the first call leaves it so. A block-diagonal $R$ takes the
dense-middle form, `congruence(1.0, K, R, 1.0, C)`, instead of the second line.
With `MTL5_BLAS_HAS_GEMMT`, dense `float` or `double` operands and an
accumulator that allows BLAS, the second product of each call goes to the
vendor's `?gemmt` (the full conditions are in
[gemmt.md](gemmt.md#blas-dispatch)); otherwise it runs MTL5's generic loop,
which rounds the formed triangle exactly as `mult` does. A
fused, packing-level diagonal scaling would be a kernel of MTL5's own and is
not implemented.
