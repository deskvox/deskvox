#pragma once

#include "intrinsics.h"

#include "../vector.h"

#include <virvo/vvmacros.h>

#include <stdexcept>

namespace virvo
{


namespace math
{


namespace simd
{


class float4
{
public:

    typedef __m128 value_type;
    __m128 value;

    VV_FORCE_INLINE float4()
    {
    }

    VV_FORCE_INLINE float4(float x, float y, float z, float w)
        : value(_mm_set_ps(w, z, y, x))
    {
    }

    VV_FORCE_INLINE float4(float const v[4])
        : value(_mm_load_ps(v))
    {
    }

    VV_FORCE_INLINE float4(float s)
        : value(_mm_set1_ps(s))
    {
    }

    VV_FORCE_INLINE float4(__m128i const& i)
        : value(_mm_cvtepi32_ps(i))
    {
    }

    VV_FORCE_INLINE float4(__m128 const& v)
        : value(v)
    {
    }

    VV_FORCE_INLINE operator __m128() const
    {
        return value;
    }
};


VV_FORCE_INLINE void store(float4 const& v, float dst[4])
{
  _mm_store_ps(dst, v);
}

VV_FORCE_INLINE float4 select(float4 const& mask, float4 const& a, float4 const& b)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
  return _mm_blendv_ps(b, a, mask);
#else
  return _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b));
#endif
}

template <int U0, int U1, int V2, int V3>
VV_FORCE_INLINE float4 shuffle(float4 const& u, float4 const& v)
{
  return _mm_shuffle_ps(u, v, _MM_SHUFFLE(V3, V2, U1, U0));
}

template <int V0, int V1, int V2, int V3>
VV_FORCE_INLINE float4 shuffle(float4 const& v)
{
  return _mm_shuffle_ps(v, v, _MM_SHUFFLE(V3, V2, V1, V0));
}


/* operators */

VV_FORCE_INLINE float4 operator-(float4 const& v)
{
  return _mm_sub_ps(_mm_setzero_ps(), v);
}

VV_FORCE_INLINE float4 operator+(float4 const& u, float4 const& v)
{
  return _mm_add_ps(u, v);
}

VV_FORCE_INLINE float4 operator-(float4 const& u, float4 const& v)
{
  return _mm_sub_ps(u, v);
}

VV_FORCE_INLINE float4 operator*(float4 const& u, float4 const& v)
{
  return _mm_mul_ps(u, v);
}

VV_FORCE_INLINE float4 operator/(float4 const& u, float4 const& v)
{
  return _mm_div_ps(u, v);
}

VV_FORCE_INLINE float4& operator+=(float4& u, float4 const& v)
{
  u = u + v;
  return u;
}

VV_FORCE_INLINE float4& operator-=(float4& u, float4 const& v)
{
  u = u - v;
  return u;
}

VV_FORCE_INLINE float4& operator*=(float4& u, float4 const& v)
{
  u = u * v;
  return u;
}

VV_FORCE_INLINE float4& operator/=(float4& u, float4 const& v)
{
  u = u / v;
  return u;
}


/* function analogs for cstdlib */

VV_FORCE_INLINE float4 min(float4 const& u, float4 const& v)
{
  return _mm_min_ps(u, v);
}

VV_FORCE_INLINE float4 max(float4 const& u, float4 const& v)
{
  return _mm_max_ps(u, v);
}

VV_FORCE_INLINE float4 powf(float4 const& v, float4 const& exp)
{
#if VV_SIMD_HAS_SVML
  return _mm_pow_ps(v, exp);
#else
  // TODO: not implemented yet
  VV_UNUSED(exp);
  return v;
#endif
}

VV_FORCE_INLINE float4 round(float4 const& v)
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

VV_FORCE_INLINE float4 ceil(float4 const& v)
{
#if VV_SIMD_ISA >= VV_SIMD_ISA_SSE4_1
  return _mm_ceil_ps(v);
#else
  // i = trunc(v)
  __m128 i = _mm_cvtepi32_ps(_mm_cvttps_epi32(v));
  // r = i < v ? i i + 1 : i
  __m128 t = _mm_cmplt_ps(i, v);
  __m128 d = _mm_cvtepi32_ps(_mm_castps_si128(t)); // mask to float: 0 -> 0.0f, 0xFFFFFFFF -> -1.0f
  __m128 r = _mm_sub_ps(i, d);

  return r;
#endif
}

VV_FORCE_INLINE float4 floor(float4 const& v)
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

VV_FORCE_INLINE float4 sqrt(float4 const& v)
{
  return _mm_sqrt_ps(v);
}

VV_FORCE_INLINE float4 approx_rsqrt(float4 const& v)
{
  return _mm_rsqrt_ps(v);
}


/* vector math functions */

/*! \brief  returns a vector with each element {x|y|z|w} containing
 the result of the dot product
 */
VV_FORCE_INLINE float4 dot(float4 const& u, float4 const& v)
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
VV_FORCE_INLINE float4 rcp_step(float4 const& v)
{
  float4 t = v;

  for (unsigned i = 0; i < N; ++i)
  {
    t = (t + t) - (v * t * t);
  }

  return t;
}

template <unsigned N>
VV_FORCE_INLINE float4 rcp(float4 const& v)
{
  float4 x0 = _mm_rcp_ps(v);
  rcp_step<N>(x0);
  return x0;
}

VV_FORCE_INLINE float4 rcp(float4 const& v)
{
  float4 x0 = _mm_rcp_ps(v);
  rcp_step<1>(x0);
  return x0;
}

template <unsigned N>
VV_FORCE_INLINE float4 rsqrt_step(float4 const& v)
{
  float4 threehalf(1.5f);
  float4 vhalf = v * float4(0.5f);
  float4 t = v;

  for (unsigned i = 0; i < N; ++i)
  {
    t = t * (threehalf - vhalf * t * t);
  }

  return t;
}

template <unsigned N>
VV_FORCE_INLINE float4 rsqrt(float4 const& v)
{
  float4 x0 = _mm_rsqrt_ps(v);
  rsqrt_step<N>(x0);
  return x0;
}

VV_FORCE_INLINE float4 rsqrt(float4 const& v)
{
  float4 x0 = _mm_rsqrt_ps(v);
  rsqrt_step<1>(x0);
  return x0;
}

// TODO: find a better place for this
VV_FORCE_INLINE vector< 4, float4 > transpose(vector< 4, float4 > const& v)
{
  vector< 4, float4 > result = v;

  float4 tmp1 = _mm_unpacklo_ps(result.x, result.y);
  float4 tmp2 = _mm_unpacklo_ps(result.z, result.w);
  float4 tmp3 = _mm_unpackhi_ps(result.x, result.y);
  float4 tmp4 = _mm_unpackhi_ps(result.z, result.w);

  result.x = _mm_movelh_ps(tmp1, tmp2);
  result.y = _mm_movehl_ps(tmp2, tmp1);
  result.z = _mm_movelh_ps(tmp3, tmp4);
  result.w = _mm_movehl_ps(tmp4, tmp3);

  return result;
}


} // simd


} // math


} // virvo


