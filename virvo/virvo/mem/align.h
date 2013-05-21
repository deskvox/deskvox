#pragma once

#if defined(_MSC_VER)
#define ALIGN(X) __declspec(align(X))
#else
#define ALIGN(X) __attribute__((aligned(X)))
#endif

#define CACHE_LINE 16
#define CACHE_ALIGN ALIGN(CACHE_LINE)

