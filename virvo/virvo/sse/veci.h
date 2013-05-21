#pragma once

#include "../mem/align.h"

#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include <ostream>

namespace virvo
{
namespace sse
{
class CACHE_ALIGN Veci
{
public:
  typedef __m128i value_type;
  value_type value;

  inline Veci()
    : value(_mm_setzero_si128())
  {
  }

  /*! \brief  value[i] = mask[i] == 0xFF ? u[i] : v[i];
   */
  inline Veci(Veci const& u, Veci const& v, Veci const& mask)
    : value(_mm_add_epi32(_mm_and_si128(mask, u), _mm_andnot_si128(mask, v)))
  {
  }

  inline Veci(int x, int y, int z, int w)
    : value(_mm_set_epi32(w, z, y, x))
  {
  }

  inline Veci(int s)
    : value(_mm_set1_epi32(s))
  {
  }

  inline Veci(value_type const& v)
    : value(v)
  {
  }

  inline operator value_type() const
  {
    return value;
  }
};

inline void store(Veci const& v, int dst[4])
{
  _mm_store_si128(reinterpret_cast<__m128i*>(dst), v);
}

/* operators */

inline Veci operator-(Veci const& v)
{
  Veci const& zero = _mm_setzero_si128();
  Veci tmp_2_0 = _mm_sub_epi32(zero, v);
  Veci tmp_3_1 = _mm_sub_epi32(_mm_srli_si128(zero, 4), _mm_srli_si128(v, 4));
  return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp_2_0, _MM_SHUFFLE(0, 0, 2, 0)),
    _mm_shuffle_epi32(tmp_3_1, _MM_SHUFFLE(0, 0, 2, 0)));
}

inline Veci operator+(Veci const& u, Veci const& v)
{
  Veci tmp_2_0 = _mm_add_epi32(u, v);
  Veci tmp_3_1 = _mm_add_epi32(_mm_srli_si128(u, 4), _mm_srli_si128(v, 4));
  return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp_2_0, _MM_SHUFFLE(0, 0, 2, 0)),
    _mm_shuffle_epi32(tmp_3_1, _MM_SHUFFLE(0, 0, 2, 0)));
}

inline Veci operator-(Veci const& u, Veci const& v)
{
  Veci tmp_2_0 = _mm_sub_epi32(u, v);
  Veci tmp_3_1 = _mm_sub_epi32(_mm_srli_si128(u, 4), _mm_srli_si128(v, 4));
  return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp_2_0, _MM_SHUFFLE(0, 0, 2, 0)),
    _mm_shuffle_epi32(tmp_3_1, _MM_SHUFFLE(0, 0, 2, 0)));
}

inline Veci operator*(Veci const& u, Veci const& v)
{
  Veci tmp_2_0 = _mm_mul_epi32(u, v);
  Veci tmp_3_1 = _mm_mul_epi32(_mm_srli_si128(u, 4), _mm_srli_si128(v, 4));
  return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp_2_0, _MM_SHUFFLE(0, 0, 2, 0)),
    _mm_shuffle_epi32(tmp_3_1, _MM_SHUFFLE(0, 0, 2, 0)));
}

inline Veci& operator+=(Veci& u, Veci const& v)
{
  u = u + v;
  return u;
}

inline Veci& operator-=(Veci& u, Veci const& v)
{
  u = u - v;
  return u;
}

inline Veci& operator*=(Veci& u, Veci const& v)
{
  u = u * v;
  return u;
}

inline Veci operator<(Veci const& u, Veci const& v)
{
  return _mm_cmplt_epi32(u, v);
}

inline Veci operator>(Veci const& u, Veci const& v)
{
  return _mm_cmpgt_epi32(u, v);
}

inline Veci operator<=(Veci const& u, Veci const& v)
{
  return _mm_or_si128(_mm_cmplt_epi32(u, v), _mm_cmpeq_epi32(u, v));
}

inline Veci operator>=(Veci const& u, Veci const& v)
{
  return _mm_or_si128(_mm_cmpgt_epi32(u, v), _mm_cmpeq_epi32(u, v));
}

inline Veci operator==(Veci const& u, Veci const& v)
{
  return _mm_cmpeq_epi32(u, v);
}

inline Veci operator&&(Veci const& u, Veci const& v)
{
  return _mm_and_si128(u, v);
}

inline std::ostream& operator<<(std::ostream& out, Veci const& v)
{
  CACHE_ALIGN int vals[4];
  store(v, vals);
  out << vals[0] << " " << vals[1] << " " << vals[2] << " " << vals[3];
  return out;
}

/* function analogs for virvo::toolshed */

template <typename T>
inline T clamp(T const& v, T const& a, T const& b);

template <>
inline Veci clamp(Veci const& v, Veci const& a, Veci const& b)
{
  Veci maska = v < a;
  Veci tmp(a, v, maska);
  Veci maskb = tmp > b;
  return Veci(b, tmp, maskb);
}

} // sse
} // virvo

