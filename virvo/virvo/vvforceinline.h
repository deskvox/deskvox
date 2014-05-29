#pragma once

#include "vvcompiler.h"

#if VV_CXX_INTEL
#define VV_FORCE_INLINE __forceinline
#elif VV_CXX_GCC || VV_CXX_CLANG
#define VV_FORCE_INLINE inline __attribute((always_inline))
#elif VV_CXX_MSVC
#define VV_FORCE_INLINE __forceinline
#else
#define VV_FORCE_INLINE inline
#endif

