// Shared-pool test module (#639). Built WITHOUT MTL5_SHARED_POOL and with
// DEFAULT visibility, and linked ahead of plugin_shared_default, so its
// exported definitions come first in the dynamic linker's search: it must keep
// its own pool and configuration without capturing the shared module's.
#define PLUGIN(name) local_default_##name
#include "plugin.hpp"
