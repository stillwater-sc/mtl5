// Shared-pool test module (#639). Built with MTL5_SHARED_POOL and hidden
// visibility, linked to the pool library: it must use the process's one pool.
#define PLUGIN(name) shared_b_##name
#include "plugin.hpp"
