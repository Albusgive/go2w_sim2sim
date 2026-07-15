#pragma once

// MuJoCo <= 3.8 ships mujoco/mjtnum.h.  MuJoCo >= 3.9 moved the same scalar
// definitions into mujoco/mjtype.h.  This include-path shim lets existing
// deployment code compile against either layout without copying ABI types.
#if __has_include(<mujoco/mjtype.h>)
#include <mujoco/mjtype.h>
#else
#include_next <mujoco/mjtnum.h>
#endif
