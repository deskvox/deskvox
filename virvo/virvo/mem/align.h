#pragma once

#if defined(_MSC_VER)
#define VVALIGN(X) __declspec(align(X))
#else
#define VVALIGN(X) __attribute__((aligned(X)))
#endif

#define CACHE_LINE 16
#define CACHE_ALIGN VVALIGN(CACHE_LINE)

