#pragma once
// MTL5 -- Householder reflector form tags (#559)
//
// householder(), apply_householder_left() and qr_factor() build and apply the
// LAPACK form of the reflector by default: x scaled by max|x|, v(0) == 1,
// tau = 2*v0^2/(sigma + v0^2), and R's diagonal non-negative. The textbook form
// (Golub & Van Loan, sec. 5.1) is just as valid and rounds differently:
//
//   norm  = sqrt(sum x_i^2)              unscaled, ascending i
//   alpha = x_0 >= 0 ? -norm : norm      R's diagonal entry is alpha
//   v     = x;  v_0 = x_0 - alpha        v is not normalized
//   vtv   = sum v_i^2
//   per column: s = sum v_i*A(i,j);  f = (2*s)/vtv;  A(i,j) -= f*v_i
//
// The tag selects the textbook form, so a caller whose results are pinned to
// it can reproduce them bit for bit. See
// docs/numerical/householder-reflector-forms.md.
namespace mtl::tag {

/// The textbook (Golub & Van Loan) Householder reflector: unnormalized v,
/// applied with f = (2*s)/vtv per column.
struct textbook_reflector {};

} // namespace mtl::tag
