// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

//--------------------------------------------------------------------------------------------------
// Detect architecture
//

// SSE2 is always available on 64-bit platforms
#if defined(_M_X64) || defined(_M_AMD64) || defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64)
#define VV_SIMD_ON_64_BIT 1
#endif

//--------------------------------------------------------------------------------------------------
// Detect instruction set
//

#define VV_SIMD_ISA_SSE         10
#define VV_SIMD_ISA_SSE2        20
#define VV_SIMD_ISA_SSE3        30
#define VV_SIMD_ISA_SSSE3       31
#define VV_SIMD_ISA_SSE4_1      41
#define VV_SIMD_ISA_SSE4_2      42
#define VV_SIMD_ISA_AVX         50
#define VV_SIMD_ISA_AVX2        60
#define VV_SIMD_ISA_AVX_512     70

#ifndef VV_SIMD_ISA
#if defined(__AVX2__)
#define VV_SIMD_ISA VV_SIMD_ISA_AVX2
#elif defined(__AVX__)
#define VV_SIMD_ISA VV_SIMD_ISA_AVX
#elif defined(__SSE4_2__)
#define VV_SIMD_ISA VV_SIMD_ISA_SSE4_2
#elif defined(__SSE4_1__)
#define VV_SIMD_ISA VV_SIMD_ISA_SSE4_1
#elif defined(__SSSE3__)
#define VV_SIMD_ISA VV_SIMD_ISA_SSSE3
#elif defined(__SSE3__)
#define VV_SIMD_ISA VV_SIMD_ISA_SSE3
#elif defined(__SSE2__) || defined(VV_SIMD_ON_64_BIT)
#define VV_SIMD_ISA VV_SIMD_ISA_SSE2
#else
#define VV_SIMD_ISA 0
#endif
#endif

// Intel Short Vector Math Library available?
#ifndef VV_SIMD_HAS_SVML
#if defined(__INTEL_COMPILER)
#define VV_SIMD_HAS_SVML 1
#endif
#endif

//--------------------------------------------------------------------------------------------------
// SSE #include's
//

#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE2
#include <emmintrin.h>
#endif
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE3
#include <pmmintrin.h>
#endif
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSSE3
#include <tmmintrin.h>
#endif
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
#include <smmintrin.h>
#endif
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_2
#include <nmmintrin.h>
#endif
#if VV_SIMD_ISA >= VV_SIMD_ISA_AVX
#include <immintrin.h>
#endif
#if VV_SIMD_ISA >= VV_SIMD_ISA_AVX_512
#include <zmmintrin.h>
#endif
