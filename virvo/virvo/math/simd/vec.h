#pragma once

#include "intrinsics.h"

#include "../vector.h"

#include <virvo/vvcompiler.h>
#include <virvo/vvforceinline.h>
#include <virvo/vvmacros.h>

#include <virvo/mem/align.h>

#include <ostream>
#include <stdexcept>

namespace virvo
{

namespace math
{

template < typename T >
class CACHE_ALIGN base_vec;


template < >
class CACHE_ALIGN base_vec< __m128 >
{
public:
  typedef __m128 value_type;
  value_type value;

  VV_FORCE_INLINE base_vec()
  {
  }

  VV_FORCE_INLINE base_vec(float x, float y, float z, float w)
    : value(_mm_set_ps(w, z, y, x))
  {
  }

  VV_FORCE_INLINE base_vec(float const v[4])
    : value(_mm_load_ps(v))
  {
  }

  VV_FORCE_INLINE base_vec(float s)
    : value(_mm_set1_ps(s))
  {
  }

  VV_FORCE_INLINE base_vec(__m128i const& i)
    : value(_mm_cvtepi32_ps(i))
  {
  }

  VV_FORCE_INLINE base_vec(value_type const& v)
    : value(v)
  {
  }

  VV_FORCE_INLINE operator value_type() const
  {
    return value;
  }
};


typedef base_vec< __m128 > sse_vec;


VV_FORCE_INLINE void store(sse_vec const& v, float dst[4])
{
  _mm_store_ps(dst, v);
}

VV_FORCE_INLINE sse_vec select(sse_vec const& mask, sse_vec const& a, sse_vec const& b)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
  return _mm_blendv_ps(b, a, mask);
#else
  return _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b));
#endif
}

template <int U0, int U1, int V2, int V3>
VV_FORCE_INLINE sse_vec shuffle(sse_vec const& u, sse_vec const& v)
{
  return _mm_shuffle_ps(u, v, _MM_SHUFFLE(V3, V2, U1, U0));
}

template <int V0, int V1, int V2, int V3>
VV_FORCE_INLINE sse_vec shuffle(sse_vec const& v)
{
  return _mm_shuffle_ps(v, v, _MM_SHUFFLE(V3, V2, V1, V0));
}


/* operators */

VV_FORCE_INLINE sse_vec operator-(sse_vec const& v)
{
  return _mm_sub_ps(_mm_setzero_ps(), v);
}

VV_FORCE_INLINE sse_vec operator+(sse_vec const& u, sse_vec const& v)
{
  return _mm_add_ps(u, v);
}

VV_FORCE_INLINE sse_vec operator-(sse_vec const& u, sse_vec const& v)
{
  return _mm_sub_ps(u, v);
}

VV_FORCE_INLINE sse_vec operator*(sse_vec const& u, sse_vec const& v)
{
  return _mm_mul_ps(u, v);
}

VV_FORCE_INLINE sse_vec operator/(sse_vec const& u, sse_vec const& v)
{
  return _mm_div_ps(u, v);
}

VV_FORCE_INLINE sse_vec& operator+=(sse_vec& u, sse_vec const& v)
{
  u = u + v;
  return u;
}

VV_FORCE_INLINE sse_vec& operator-=(sse_vec& u, sse_vec const& v)
{
  u = u - v;
  return u;
}

VV_FORCE_INLINE sse_vec& operator*=(sse_vec& u, sse_vec const& v)
{
  u = u * v;
  return u;
}

VV_FORCE_INLINE sse_vec& operator/=(sse_vec& u, sse_vec const& v)
{
  u = u / v;
  return u;
}


VV_FORCE_INLINE std::ostream& operator<<(std::ostream& out, sse_vec const& v)
{
  CACHE_ALIGN float vals[4];
  store(v, vals);
  out << vals[0] << " " << vals[1] << " " << vals[2] << " " << vals[3];
  return out;
}


/* function analogs for cstdlib */

VV_FORCE_INLINE sse_vec min(sse_vec const& u, sse_vec const& v)
{
  return _mm_min_ps(u, v);
}

VV_FORCE_INLINE sse_vec max(sse_vec const& u, sse_vec const& v)
{
  return _mm_max_ps(u, v);
}

VV_FORCE_INLINE sse_vec powf(sse_vec const& v, sse_vec const& exp)
{
#if VV_SIMD_HAS_SVML
  return _mm_pow_ps(v, exp);
#else
  // TODO: not implemented yet
  VV_UNUSED(exp);
  return v;
#endif
}

VV_FORCE_INLINE sse_vec round(sse_vec const& v)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
  return _mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT);
#else
  // Mask out the signbits of v
  __m128 s = _mm_and_ps(v, _mm_castsi128_ps(_mm_set1_epi32(0x80000000)));
  // Magic number: 2^23 with the signbits of v
  __m128 m = _mm_or_ps(s, _mm_castsi128_ps(_mm_set1_epi32(0x4B000000)));
  __m128 x = _mm_add_ps(v, m);
  __m128 y = _mm_sub_ps(x, m);

  return y;
#endif
}

VV_FORCE_INLINE sse_vec ceil(sse_vec const& v)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
  return _mm_ceil_ps(v);
#else
  throw std::runtime_error("not implemented yet");
#endif
}

VV_FORCE_INLINE sse_vec floor(sse_vec const& v)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
  return _mm_floor_ps(v);
#else
  // i = trunc(v)
  __m128 i = _mm_cvtepi32_ps(_mm_cvttps_epi32(v));
  // r = i > v ? i - 1 : i
  __m128 t = _mm_cmpgt_ps(i, v);
  __m128 d = _mm_cvtepi32_ps(_mm_castps_si128(t)); // mask to float: 0 -> 0.0f, 0xFFFFFFFF -> -1.0f
  __m128 r = _mm_add_ps(i, d);

  return r;
#endif
}

VV_FORCE_INLINE sse_vec sqrt(sse_vec const& v)
{
  return _mm_sqrt_ps(v);
}

VV_FORCE_INLINE sse_vec approx_rsqrt(sse_vec const& v)
{
  return _mm_rsqrt_ps(v);
}


/* vector math functions */

/*! \brief  returns a vector with each element {x|y|z|w} containing
 the result of the dot product
 */
VV_FORCE_INLINE sse_vec dot(sse_vec const& u, sse_vec const& v)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
  return _mm_dp_ps(u, v, 0xFF);
#else
  __m128 t1 = _mm_mul_ps(u, v);
  __m128 t2 = _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(2,3,0,1));
  __m128 t3 = _mm_add_ps(t1, t2);
  __m128 t4 = _mm_shuffle_ps(t3, t3, _MM_SHUFFLE(0,1,2,3));
  __m128 t5 = _mm_add_ps(t3, t4);

  return t5;
#endif
}


/*! \brief  Newton Raphson refinement
 */

template <unsigned N>
VV_FORCE_INLINE sse_vec rcp_step(sse_vec const& v)
{
  sse_vec t = v;

  for (unsigned i = 0; i < N; ++i)
  {
    t = (t + t) - (v * t * t);
  }

  return t;
}

template <unsigned N>
VV_FORCE_INLINE sse_vec rcp(sse_vec const& v)
{
  sse_vec x0 = _mm_rcp_ps(v);
  rcp_step<N>(x0);
  return x0;
}

VV_FORCE_INLINE sse_vec rcp(sse_vec const& v)
{
  sse_vec x0 = _mm_rcp_ps(v);
  rcp_step<1>(x0);
  return x0;
}

template <unsigned N>
VV_FORCE_INLINE sse_vec rsqrt_step(sse_vec const& v)
{
  sse_vec threehalf(1.5f);
  sse_vec vhalf = v * sse_vec(0.5f);
  sse_vec t = v;

  for (unsigned i = 0; i < N; ++i)
  {
    t = t * (threehalf - vhalf * t * t);
  }

  return t;
}

template <unsigned N>
VV_FORCE_INLINE sse_vec rsqrt(sse_vec const& v)
{
  sse_vec x0 = _mm_rsqrt_ps(v);
  rsqrt_step<N>(x0);
  return x0;
}

VV_FORCE_INLINE sse_vec rsqrt(sse_vec const& v)
{
  sse_vec x0 = _mm_rsqrt_ps(v);
  rsqrt_step<1>(x0);
  return x0;
}

// TODO: find a better place for this
VV_FORCE_INLINE vector< 4, sse_vec > transpose(vector< 4, sse_vec > const& v)
{
  vector< 4, sse_vec > result = v;

  sse_vec tmp1 = _mm_unpacklo_ps(result.x, result.y);
  sse_vec tmp2 = _mm_unpacklo_ps(result.z, result.w);
  sse_vec tmp3 = _mm_unpackhi_ps(result.x, result.y);
  sse_vec tmp4 = _mm_unpackhi_ps(result.z, result.w);

  result.x = _mm_movelh_ps(tmp1, tmp2);
  result.y = _mm_movehl_ps(tmp2, tmp1);
  result.z = _mm_movelh_ps(tmp3, tmp4);
  result.w = _mm_movehl_ps(tmp4, tmp3);

  return result;
}


} // math


} // virvo

