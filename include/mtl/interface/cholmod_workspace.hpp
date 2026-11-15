#pragma once
// MTL5 -- An owned, started cholmod_common for the CHOLMOD and SPQR bindings
//
// Included by interface/cholmod.hpp and interface/spqr.hpp, which are guarded
// by MTL5_HAS_CHOLMOD and MTL5_HAS_SPQR; it needs <cholmod.h> on the include
// path and does nothing useful without one of them.

#include <memory>
#include <stdexcept>

#include <cholmod.h>

namespace mtl::interface::detail {

/// Calls cholmod_finish, which frees the common's workspace, then the struct.
struct cholmod_common_deleter {
    void operator()(cholmod_common* c) const noexcept {
        cholmod_finish(c);
        delete c;
    }
};

/// A heap cholmod_common that moves as a pointer and is finished exactly once.
/// Copying the struct itself would leave two owners of its buffers, and both
/// would free them (#774, #839).
using cholmod_workspace = std::unique_ptr<cholmod_common, cholmod_common_deleter>;

/// A started workspace. Throws std::runtime_error if cholmod_start fails, in
/// which case nothing is left to finish.
inline cholmod_workspace make_cholmod_workspace() {
    auto c = std::make_unique<cholmod_common>();
    if (!cholmod_start(c.get()))
        throw std::runtime_error("CHOLMOD: cholmod_start failed");
    return cholmod_workspace(c.release());
}

/// A fresh workspace for one solve, so that concurrent solves on one const
/// solver share no mutable state (#839). The objects the solve reads -- the
/// factor or the sparse matrix -- were allocated with the solver's own
/// workspace; CHOLMOD allocates through SuiteSparse's global allocator, so
/// another workspace may read them, and only the dense vectors the solve
/// allocates are counted against (and freed with) this one. The error
/// reporting and thread settings are carried over from the solver's.
/// A template only so the member checks stay soft across CHOLMOD versions,
/// as in nested_suitesparse_cap.
template <typename Common = cholmod_common>
cholmod_workspace make_solve_workspace(const Common& settings) {
    auto c = make_cholmod_workspace();
    Common& w = *c;
    w.print = settings.print;
    w.try_catch = settings.try_catch;
    w.error_handler = settings.error_handler;
    if constexpr (requires { w.nthreads_max; }) w.nthreads_max = settings.nthreads_max;
    if constexpr (requires { w.SPQR_nthreads; }) w.SPQR_nthreads = settings.SPQR_nthreads;
    return c;
}

} // namespace mtl::interface::detail
