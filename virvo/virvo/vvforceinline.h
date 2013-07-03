#pragma once

#ifdef __INTEL_COMPILER
#define VV_FORCE_INLINE __forceinline
#elif defined(__GNUC__)
#define VV_FORCE_INLINE __attribute((always_inline))
#elif defined(_MSC_VER)
#define VV_FORCE_INLINE __forceinline
#else
#define VV_FORCE_INLINE inline
#endif

