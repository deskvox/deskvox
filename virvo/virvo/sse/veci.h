#pragma once

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
namespace sse
{
class CACHE_ALIGN Veci
{
public:
  typedef __m128i value_type;
  value_type value;

  VV_FORCE_INLINE Veci()
    : value(_mm_setzero_si128())
  {
  }

  /*! \brief  value[i] = mask[i] == 0xFF ? u[i] : v[i];
   */
  VV_FORCE_INLINE Veci(Veci const& u, Veci const& v, Veci const& mask)
    : value(_mm_add_epi32(_mm_and_si128(mask, u), _mm_andnot_si128(mask, v)))
  {
  }

  VV_FORCE_INLINE Veci(int x, int y, int z, int w)
    : value(_mm_set_epi32(w, z, y, x))
  {
  }

  VV_FORCE_INLINE Veci(int s)
    : value(_mm_set1_epi32(s))
  {
  }

  VV_FORCE_INLINE Veci(value_type const& v)
    : value(v)
  {
  }

  VV_FORCE_INLINE operator value_type() const
  {
    return value;
  }
};

VV_FORCE_INLINE void store(Veci const& v, int dst[4])
{
  _mm_store_si128(reinterpret_cast<__m128i*>(dst), v);
}

/* operators */

VV_FORCE_INLINE Veci operator-(Veci const& v)
{
  Veci const& zero = _mm_setzero_si128();
  Veci tmp_2_0 = _mm_sub_epi32(zero, v);
  Veci tmp_3_1 = _mm_sub_epi32(_mm_srli_si128(zero, 4), _mm_srli_si128(v, 4));
  return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp_2_0, _MM_SHUFFLE(0, 0, 2, 0)),
    _mm_shuffle_epi32(tmp_3_1, _MM_SHUFFLE(0, 0, 2, 0)));
}

VV_FORCE_INLINE Veci operator+(Veci const& u, Veci const& v)
{
  Veci tmp_2_0 = _mm_add_epi32(u, v);
  Veci tmp_3_1 = _mm_add_epi32(_mm_srli_si128(u, 4), _mm_srli_si128(v, 4));
  return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp_2_0, _MM_SHUFFLE(0, 0, 2, 0)),
    _mm_shuffle_epi32(tmp_3_1, _MM_SHUFFLE(0, 0, 2, 0)));
}

VV_FORCE_INLINE Veci operator-(Veci const& u, Veci const& v)
{
  Veci tmp_2_0 = _mm_sub_epi32(u, v);
  Veci tmp_3_1 = _mm_sub_epi32(_mm_srli_si128(u, 4), _mm_srli_si128(v, 4));
  return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp_2_0, _MM_SHUFFLE(0, 0, 2, 0)),
    _mm_shuffle_epi32(tmp_3_1, _MM_SHUFFLE(0, 0, 2, 0)));
}

VV_FORCE_INLINE Veci operator*(Veci const& u, Veci const& v)
{
#ifdef __SSE4_1__
  Veci tmp_2_0 = _mm_mul_epi32(u, v);
  Veci tmp_3_1 = _mm_mul_epi32(_mm_srli_si128(u, 4), _mm_srli_si128(v, 4));
  return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp_2_0, _MM_SHUFFLE(0, 0, 2, 0)),
    _mm_shuffle_epi32(tmp_3_1, _MM_SHUFFLE(0, 0, 2, 0)));
#else
  VV_UNUSED(u);
  VV_UNUSED(v);
  throw std::runtime_error("not implemented yet");
#endif
}

VV_FORCE_INLINE Veci& operator+=(Veci& u, Veci const& v)
{
  u = u + v;
  return u;
}

VV_FORCE_INLINE Veci& operator-=(Veci& u, Veci const& v)
{
  u = u - v;
  return u;
}

VV_FORCE_INLINE Veci& operator*=(Veci& u, Veci const& v)
{
  u = u * v;
  return u;
}

VV_FORCE_INLINE Veci operator<(Veci const& u, Veci const& v)
{
  return _mm_cmplt_epi32(u, v);
}

VV_FORCE_INLINE Veci operator>(Veci const& u, Veci const& v)
{
  return _mm_cmpgt_epi32(u, v);
}

VV_FORCE_INLINE Veci operator<=(Veci const& u, Veci const& v)
{
  return _mm_or_si128(_mm_cmplt_epi32(u, v), _mm_cmpeq_epi32(u, v));
}

VV_FORCE_INLINE Veci operator>=(Veci const& u, Veci const& v)
{
  return _mm_or_si128(_mm_cmpgt_epi32(u, v), _mm_cmpeq_epi32(u, v));
}

VV_FORCE_INLINE Veci operator==(Veci const& u, Veci const& v)
{
  return _mm_cmpeq_epi32(u, v);
}

VV_FORCE_INLINE Veci operator&&(Veci const& u, Veci const& v)
{
  return _mm_and_si128(u, v);
}

VV_FORCE_INLINE std::ostream& operator<<(std::ostream& out, Veci const& v)
{
  CACHE_ALIGN int vals[4];
  store(v, vals);
  out << vals[0] << " " << vals[1] << " " << vals[2] << " " << vals[3];
  return out;
}


/* function analogs for cstdlib */

VV_FORCE_INLINE Veci min(Veci const& u, Veci const& v)
{
#ifdef __SSE4_1__
  return _mm_min_epi32(u, v);
#else
  VV_UNUSED(u);
  VV_UNUSED(v);
  throw std::runtime_error("not implemented yet");
#endif
}

VV_FORCE_INLINE Veci max(Veci const& u, Veci const& v)
{
#ifdef __SSE4_1__
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
VV_FORCE_INLINE Veci clamp(Veci const& v, Veci const& a, Veci const& b)
{
#ifdef __SSE4_1__
  return _mm_max_epi32(a, _mm_min_epi32(v, b));
#else
  Veci maska = v < a;
  Veci tmp(a, v, maska);
  Veci maskb = tmp > b;
  return Veci(b, tmp, maskb);
#endif
}

} // sse
} // virvo

