// Shared-pool test module (#639). Built WITHOUT MTL5_SHARED_POOL, with hidden
// visibility: the case #639 fixes, so this module holds a pool of its own.
#define PLUGIN(name) local_##name
#include "plugin.hpp"
