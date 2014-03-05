#pragma once

#include "../vvcompiler.h"
#include "../vvforceinline.h"
#include "../vvmacros.h"

#include "../mem/align.h"

#include <xmmintrin.h>
#include <emmintrin.h>
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#include <ostream>
#include <stdexcept>

namespace virvo
{
namespace simd
{
class CACHE_ALIGN Vec
{
public:
  typedef __m128 value_type;
  value_type value;

  VV_FORCE_INLINE Vec()
    : value(_mm_setzero_ps())
  {
  }

  /*! \brief  value[i] = mask[i] == 0xFF ? u[i] : v[i];
   */
  VV_FORCE_INLINE Vec(Vec const& u, Vec const& v, Vec const& mask)
    : value(_mm_add_ps(_mm_and_ps(mask, u), _mm_andnot_ps(mask, v)))
  {
  }

  VV_FORCE_INLINE Vec(float x, float y, float z, float w)
    : value(_mm_set_ps(w, z, y, x))
  {
  }

  VV_FORCE_INLINE Vec(float const v[4])
    : value(_mm_load_ps(v))
  {
  }

  VV_FORCE_INLINE Vec(float s)
    : value(_mm_set1_ps(s))
  {
  }

  VV_FORCE_INLINE Vec(value_type const& v)
    : value(v)
  {
  }

  VV_FORCE_INLINE operator value_type() const
  {
    return value;
  }
};


/* operators */

VV_FORCE_INLINE void store(Vec const& v, float dst[4])
{
  _mm_store_ps(dst, v);
}

VV_FORCE_INLINE Vec operator-(Vec const& v)
{
  return _mm_sub_ps(_mm_setzero_ps(), v);
}

VV_FORCE_INLINE Vec operator+(Vec const& u, Vec const& v)
{
  return _mm_add_ps(u, v);
}

VV_FORCE_INLINE Vec operator-(Vec const& u, Vec const& v)
{
  return _mm_sub_ps(u, v);
}

VV_FORCE_INLINE Vec operator*(Vec const& u, Vec const& v)
{
  return _mm_mul_ps(u, v);
}

VV_FORCE_INLINE Vec operator/(Vec const& u, Vec const& v)
{
  return _mm_div_ps(u, v);
}

VV_FORCE_INLINE Vec& operator+=(Vec& u, Vec const& v)
{
  u = u + v;
  return u;
}

VV_FORCE_INLINE Vec& operator-=(Vec& u, Vec const& v)
{
  u = u - v;
  return u;
}

VV_FORCE_INLINE Vec& operator*=(Vec& u, Vec const& v)
{
  u = u * v;
  return u;
}

VV_FORCE_INLINE Vec& operator/=(Vec& u, Vec const& v)
{
  u = u / v;
  return u;
}


VV_FORCE_INLINE std::ostream& operator<<(std::ostream& out, Vec const& v)
{
  CACHE_ALIGN float vals[4];
  store(v, vals);
  out << vals[0] << " " << vals[1] << " " << vals[2] << " " << vals[3];
  return out;
}

template <unsigned const X, unsigned const Y, unsigned const Z, unsigned const W>
VV_FORCE_INLINE Vec shuffle(Vec const& u, Vec const& v)
{
  return _mm_shuffle_ps(u, v, _MM_SHUFFLE(W, Z, Y, X));
}

template <unsigned const X, unsigned const Y, unsigned const Z, unsigned const W>
VV_FORCE_INLINE Vec shuffle(Vec const& v)
{
  return _mm_shuffle_ps(v, v, _MM_SHUFFLE(W, Z, Y, X));
}


/* function analogs for cstdlib */

VV_FORCE_INLINE Vec floor(Vec const& v)
{
#ifdef __SSE4_1__
  return _mm_floor_ps(v);
#else
  VV_UNUSED(v);
  throw std::runtime_error("not implemented yet");
#endif
}
 
VV_FORCE_INLINE Vec min(Vec const& u, Vec const& v)
{
  return _mm_min_ps(u, v);
}

VV_FORCE_INLINE Vec max(Vec const& u, Vec const& v)
{
  return _mm_max_ps(u, v);
}

VV_FORCE_INLINE Vec powf(Vec const& v, Vec const& exp)
{
#if VV_CXX_INTEL
  return _mm_pow_ps(v, exp);
#else
  // TODO: not implemented yet
  return v;
#endif
}

VV_FORCE_INLINE Vec round(Vec const& v)
{
#ifdef __SSE4_1__
  return _mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT);
#else
  return _mm_cvtepi32_ps(_mm_cvttps_epi32(v));
#endif
}

VV_FORCE_INLINE Vec sqrt(Vec const& v)
{
  return _mm_sqrt_ps(v);
}

VV_FORCE_INLINE Vec rsqrt(Vec const& v)
{
  return _mm_rsqrt_ps(v);
}

/* masked function analogs for cstdlib */

template <typename M>
VV_FORCE_INLINE Vec floor(Vec const& v, M const& mask)
{
  return if_else(floor(v), 0.0f, mask);
}

template <typename M>
VV_FORCE_INLINE Vec min(Vec const& u, Vec const& v, M const& mask)
{
  return if_else(min(u, v), 0.0f, mask);
}

template <typename M>
VV_FORCE_INLINE Vec max(Vec const& u, Vec const& v, M const& mask)
{
  return if_else(max(u, v), 0.0f, mask);
}

template <typename M>
VV_FORCE_INLINE Vec sqrt(Vec const& v, M const& mask)
{
  return if_else(sqrt(v), 0.0f, mask);
}

template <typename M>
VV_FORCE_INLINE Vec rsqrt(Vec const& v, M const& mask)
{
  return if_else(rsqrt(v), 0.0f, mask);
}

/* function analogs for virvo::toolshed */

template <typename T>
VV_FORCE_INLINE T clamp(T const& v, T const& a, T const& b);

template <>
VV_FORCE_INLINE Vec clamp(Vec const& v, Vec const& a, Vec const& b)
{
  return _mm_max_ps(a, _mm_min_ps(v, b));
}

/* vector math functions */

/*! \brief  returns a vector with each element {x|y|z|w} containing
 the result of the dot product
 */
VV_FORCE_INLINE Vec dot(Vec const& u, Vec const& v)
{
#ifdef __SSE4_1__
  return _mm_dp_ps(u, v, 0xFF);
#else
  VV_UNUSED(u);
  VV_UNUSED(v);
  throw std::runtime_error("not implemented yet");
#endif
}

namespace fast
{

/*! \brief  Newton Raphson refinement
 */

template <unsigned refinements>
VV_FORCE_INLINE Vec refine(Vec const& v)
{
  Vec x0 = v;
  for (unsigned i = 0; i < refinements; ++i)
  {
    Vec tmp = (x0 + x0) - (v * x0 * x0);
    x0 = tmp;
  }
  return x0;
}

template <unsigned refinements>
VV_FORCE_INLINE Vec rcp(Vec const& v)
{
  Vec x0 = _mm_rcp_ps(v);
  refine<refinements>(x0);
  return x0;
}

VV_FORCE_INLINE Vec rcp(Vec const& v)
{
  Vec x0 = _mm_rcp_ps(v);
  refine<1>(x0);
  return x0;
}

template <unsigned refinements>
VV_FORCE_INLINE Vec rsqrt(Vec const& v)
{
  Vec x0 = _mm_rsqrt_ps(v);
  refine<refinements>(x0);
  return x0;
}

VV_FORCE_INLINE Vec rsqrt(Vec const& v)
{
  Vec x0 = _mm_rsqrt_ps(v);
  refine<1>(x0);
  return x0;
}

} // fast

} // simd
} // virvo

