#pragma once

#include "intrinsics.h"

#include "../vvforceinline.h"
#include "../vvmacros.h"

#include "../mem/align.h"

#include <ostream>
#include <stdexcept>

namespace virvo
{

namespace simd
{

template < typename T >
class CACHE_ALIGN base_veci;


template < >
class CACHE_ALIGN base_veci< __m128i >
{
public:
  typedef __m128i value_type;
  value_type value;

  VV_FORCE_INLINE base_veci()
    : value(_mm_setzero_si128())
  {
  }

  /*! \brief  value[i] = mask[i] == 0xFF ? u[i] : v[i];
   */
  VV_FORCE_INLINE base_veci(base_veci const& u, base_veci const& v, base_veci const& mask)
    : value(_mm_add_epi32(_mm_and_si128(mask, u), _mm_andnot_si128(mask, v)))
  {
  }

  VV_FORCE_INLINE base_veci(int x, int y, int z, int w)
    : value(_mm_set_epi32(w, z, y, x))
  {
  }

  VV_FORCE_INLINE base_veci(int s)
    : value(_mm_set1_epi32(s))
  {
  }

  VV_FORCE_INLINE base_veci(value_type const& v)
    : value(v)
  {
  }

  VV_FORCE_INLINE operator value_type() const
  {
    return value;
  }
};


typedef base_veci< __m128i > sse_veci;
typedef sse_veci Veci;


VV_FORCE_INLINE void store(sse_veci const& v, int dst[4])
{
  _mm_store_si128(reinterpret_cast<__m128i*>(dst), v);
}

/* operators */

VV_FORCE_INLINE sse_veci operator-(sse_veci const& v)
{
  return _mm_sub_epi32(_mm_setzero_si128(), v);
}

VV_FORCE_INLINE sse_veci operator+(sse_veci const& u, sse_veci const& v)
{
  return _mm_add_epi32(u, v);
}

VV_FORCE_INLINE sse_veci operator-(sse_veci const& u, sse_veci const& v)
{
  return _mm_sub_epi32(u, v);
}

VV_FORCE_INLINE sse_veci operator*(sse_veci const& u, sse_veci const& v)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
  return _mm_mullo_epi32(u, v);
#else
  VV_UNUSED(u);
  VV_UNUSED(v);
  throw std::runtime_error("not implemented yet");
#endif
}

VV_FORCE_INLINE sse_veci& operator+=(sse_veci& u, sse_veci const& v)
{
  u = u + v;
  return u;
}

VV_FORCE_INLINE sse_veci& operator-=(sse_veci& u, sse_veci const& v)
{
  u = u - v;
  return u;
}

VV_FORCE_INLINE sse_veci& operator*=(sse_veci& u, sse_veci const& v)
{
  u = u * v;
  return u;
}

VV_FORCE_INLINE sse_veci operator<(sse_veci const& u, sse_veci const& v)
{
  return _mm_cmplt_epi32(u, v);
}

VV_FORCE_INLINE sse_veci operator>(sse_veci const& u, sse_veci const& v)
{
  return _mm_cmpgt_epi32(u, v);
}

VV_FORCE_INLINE sse_veci operator<=(sse_veci const& u, sse_veci const& v)
{
  return _mm_or_si128(_mm_cmplt_epi32(u, v), _mm_cmpeq_epi32(u, v));
}

VV_FORCE_INLINE sse_veci operator>=(sse_veci const& u, sse_veci const& v)
{
  return _mm_or_si128(_mm_cmpgt_epi32(u, v), _mm_cmpeq_epi32(u, v));
}

VV_FORCE_INLINE sse_veci operator==(sse_veci const& u, sse_veci const& v)
{
  return _mm_cmpeq_epi32(u, v);
}

VV_FORCE_INLINE sse_veci operator&&(sse_veci const& u, sse_veci const& v)
{
  return _mm_and_si128(u, v);
}

VV_FORCE_INLINE std::ostream& operator<<(std::ostream& out, sse_veci const& v)
{
  CACHE_ALIGN int vals[4];
  store(v, vals);
  out << vals[0] << " " << vals[1] << " " << vals[2] << " " << vals[3];
  return out;
}


/* function analogs for cstdlib */

VV_FORCE_INLINE sse_veci min(sse_veci const& u, sse_veci const& v)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
  return _mm_min_epi32(u, v);
#else
  VV_UNUSED(u);
  VV_UNUSED(v);
  throw std::runtime_error("not implemented yet");
#endif
}

VV_FORCE_INLINE sse_veci max(sse_veci const& u, sse_veci const& v)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
  return _mm_max_epi32(u, v);
#else
  VV_UNUSED(u);
  VV_UNUSED(v);
  throw std::runtime_error("not implemented yet");
#endif
}


/* function analogs for virvo::toolshed */

template <typename T>
VV_FORCE_INLINE T clamp(T const& v, T const& a, T const& b);

template <>
VV_FORCE_INLINE sse_veci clamp(sse_veci const& v, sse_veci const& a, sse_veci const& b)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
  return _mm_max_epi32(a, _mm_min_epi32(v, b));
#else
  sse_veci maska = v < a;
  sse_veci tmp(a, v, maska);
  sse_veci maskb = tmp > b;
  return sse_veci(b, tmp, maskb);
#endif
}

} // simd
} // virvo

