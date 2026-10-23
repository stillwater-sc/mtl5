// Shared-pool test module (#639). Built with MTL5_SHARED_POOL and DEFAULT
// visibility, so its inline functions and their statics are exported next to
// plugin_local_default's: the two modes must not bind to each other's.
#define PLUGIN(name) shared_default_##name
#include "plugin.hpp"
