#pragma once

#include "vec.h"
#include "vec3.h"
#include "veci.h"

namespace virvo
{
namespace sse
{
template <class T, class U>
inline T sse_cast(U u);

template <>
inline Veci sse_cast(Vec v)
{
  _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
  return _mm_cvtps_epi32(v);
}

template <>
inline Vec sse_cast(Veci v)
{
  return _mm_cvtepi32_ps(v);
}

} // sse
} // virvo

