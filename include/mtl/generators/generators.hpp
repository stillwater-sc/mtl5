#pragma once
// MTL5 — Test matrix generation facility
// Umbrella header for all generators (implicit + factory)

// Tier 1: Implicit generators (storage-free, on-the-fly computation)
#include <mtl/generators/hilbert.hpp>
#include <mtl/generators/lehmer.hpp>
#include <mtl/generators/lotkin.hpp>
#include <mtl/generators/ones.hpp>
#include <mtl/generators/minij.hpp>

// Tier 2: Dense factory generators (return dense2D<T>)
#include <mtl/generators/kahan.hpp>
#include <mtl/generators/frank.hpp>
#include <mtl/generators/moler.hpp>
#include <mtl/generators/pascal.hpp>
#include <mtl/generators/clement.hpp>
#include <mtl/generators/companion.hpp>
#include <mtl/generators/vandermonde.hpp>
#include <mtl/generators/forsythe.hpp>

// Tier 2: Sparse factory generators (return compressed2D<T>)
#include <mtl/generators/laplacian.hpp>
