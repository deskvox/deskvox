#pragma once

#include "vec.h"
#include "vec3.h"
#include "veci.h"

#include "../vvforceinline.h"

namespace virvo
{
namespace simd
{
template <class T, class U>
VV_FORCE_INLINE T simd_cast(U u);

template <>
VV_FORCE_INLINE sse_veci simd_cast(sse_vec v)
{
  return floor(v);
}

template <>
VV_FORCE_INLINE sse_vec simd_cast(sse_veci v)
{
  return _mm_cvtepi32_ps(v);
}

} // simd
} // virvo

