#pragma once

#include "intrinsics.h"

#include <virvo/vvmacros.h>

#include <virvo/mem/align.h>

#include <ostream>
#include <stdexcept>

namespace virvo
{

namespace math
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

  VV_FORCE_INLINE base_veci(base_vec< __m128 > const& f)
    : value(_mm_cvtps_epi32(f))
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

VV_FORCE_INLINE sse_veci select(sse_veci const& mask, sse_veci const& a, sse_veci const& b)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
  __m128 k = _mm_castsi128_ps(mask);
  __m128 x = _mm_castsi128_ps(a);
  __m128 y = _mm_castsi128_ps(b);

  return _mm_castps_si128(_mm_blendv_ps(y, x, k));
#else
  return _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b));
#endif
}

template <int A0, int A1, int A2, int A3>
VV_FORCE_INLINE sse_veci shuffle(sse_veci const& a)
{
  return _mm_shuffle_epi32(a, _MM_SHUFFLE(A3, A2, A1, A0));
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
  __m128i t0 = shuffle<1,0,3,0>(u);             // a1  ... a3  ...
  __m128i t1 = shuffle<1,0,3,0>(v);             // b1  ... b3  ...
  __m128i t2 = _mm_mul_epu32(u, v);             // ab0 ... ab2 ...
  __m128i t3 = _mm_mul_epu32(t0, t1);           // ab1 ... ab3 ...
  __m128i t4 = _mm_unpacklo_epi32(t2, t3);      // ab0 ab1 ... ...
  __m128i t5 = _mm_unpackhi_epi32(t2, t3);      // ab2 ab3 ... ...
  __m128i t6 = _mm_unpacklo_epi64(t4, t5);      // ab0 ab1 ab2 ab3

  return t6;
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
  return select(u < v, u, v);
#endif
}

VV_FORCE_INLINE sse_veci max(sse_veci const& u, sse_veci const& v)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
  return _mm_max_epi32(u, v);
#else
  return select(u > v, u, v);
#endif
}


} // math


} // virvo

