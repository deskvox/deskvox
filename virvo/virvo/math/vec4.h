#pragma once

#include "simd/vec.h"
#include "simd/veci.h"


namespace virvo
{
namespace math
{
template <typename T>
class CACHE_ALIGN base_vec4
{
public:
  T x;
  T y;
  T z;
  T w;

  VV_FORCE_INLINE base_vec4()
  {
  }

  VV_FORCE_INLINE base_vec4(T const& s)
    : x(s)
    , y(s)
    , z(s)
    , w(s)
  {
  }

  VV_FORCE_INLINE base_vec4(T const& x, T const& y, T const& z, T const& w)
    : x(x)
    , y(y)
    , z(z)
    , w(w)
  {
  }

  /*! \brief  construct from aligned float[4]
   */
  VV_FORCE_INLINE base_vec4(T const v[4])
    : x(v[0])
    , y(v[1])
    , z(v[2])
    , w(v[3])
  {
  }

  template < typename S >
  VV_FORCE_INLINE base_vec4(base_vec4< S > const& rhs)
    : x(rhs.x)
    , y(rhs.y)
    , z(rhs.z)
    , w(rhs.w)
  {
  }

  template < typename S >
  VV_FORCE_INLINE base_vec4& operator=(base_vec4< S > const& rhs)
  {
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    w = rhs.w;
    return *this;
  }

  VV_FORCE_INLINE T& operator[](size_t i)
  {
    return reinterpret_cast<T*>(this)[i];
  }

  VV_FORCE_INLINE const T& operator[](size_t i) const
  {
    return reinterpret_cast<T const*>(this)[i];
  }
};


VV_FORCE_INLINE base_vec4< sse_vec > transpose(base_vec4< sse_vec > const& v)
{
  base_vec4< sse_vec > result = v;

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

/* operators */

template <typename T>
VV_FORCE_INLINE base_vec4<T> operator-(base_vec4<T> const& v)
{
  return base_vec4<T>(-v.x, -v.y, -v.z, -v.w);
}

template <typename T>
VV_FORCE_INLINE base_vec4<T> operator+(base_vec4<T> const& u, base_vec4<T> const& v)
{
  return base_vec4<T>(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
}

template <typename T>
VV_FORCE_INLINE base_vec4<T> operator-(base_vec4<T> const& u, base_vec4<T> const& v)
{
  return base_vec4<T>(u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w);
}

template <typename T>
VV_FORCE_INLINE base_vec4<T> operator*(base_vec4<T> const& u, base_vec4<T> const& v)
{
  return base_vec4<T>(u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w);
}

template <typename T>
VV_FORCE_INLINE base_vec4<T> operator/(base_vec4<T> const& u, base_vec4<T> const& v)
{
  return base_vec4<T>(u.x / v.x, u.y / v.y, u.z / v.z, u.w / v.w);
}

template <typename T>
VV_FORCE_INLINE base_vec4<T> operator+(base_vec4<T> const& v, T const& s)
{
  return base_vec4<T>(v.x + s, v.y + s, v.z + s, v.w + s);
}

template <typename T>
VV_FORCE_INLINE base_vec4<T> operator-(base_vec4<T> const& v, T const& s)
{
  return base_vec4<T>(v.x - s, v.y - s, v.z - s, v.w - s);
}

template <typename T>
VV_FORCE_INLINE base_vec4<T> operator*(base_vec4<T> const& v, T const& s)
{
  return base_vec4<T>(v.x * s, v.y * s, v.z * s, v.w * s);
}

template <typename T>
VV_FORCE_INLINE base_vec4<T> operator/(base_vec4<T> const& v, T const& s)
{
  return base_vec4<T>(v.x / s, v.y / s, v.z / s, v.w / s);
}

template <typename T>
VV_FORCE_INLINE base_vec4<T> operator+(T const& s, base_vec4<T> const& v)
{
  return base_vec4<T>(s + v.x, s + v.y, s + v.z, s + v.w);
}

template <typename T>
VV_FORCE_INLINE base_vec4<T> operator-(T const& s, base_vec4<T> const& v)
{
  return base_vec4<T>(s - v.x, s - v.y, s - v.z, s - v.w);
}

template <typename T>
VV_FORCE_INLINE base_vec4<T> operator*(T const& s, base_vec4<T> const& v)
{
  return base_vec4<T>(s * v.x, s * v.y, s * v.z, s * v.w);
}

template <typename T>
VV_FORCE_INLINE base_vec4<T> operator/(T const& s, base_vec4<T> const& v)
{
  return base_vec4<T>(s / v.x, s / v.y, s / v.z, s / v.w);
}


/* vector math functions */

template <typename T>
VV_FORCE_INLINE T dot(base_vec4<T> const& u, base_vec4<T> const& v)
{
  return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
}

} // math
} // virvo

