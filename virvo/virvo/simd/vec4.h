#pragma once

#include "vec.h"
#include "veci.h"

#include "../vvvecmath.h"

#include <ostream>

namespace virvo
{
namespace simd
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

  /*! \brief  construct from virvo vector
   */
  template <typename U>
  VV_FORCE_INLINE base_vec4(vvBaseVector4<U> const& v)
    : x(v[0])
    , y(v[1])
    , z(v[2])
    , w(v[3])
  {
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

typedef base_vec4<Veci> Vec4i;
typedef base_vec4<Vec>  Vec4;

template <typename T>
VV_FORCE_INLINE base_vec4<T> transpose(base_vec4<T> const& v)
{
  base_vec4<T> result = v;

  Vec tmp1 = _mm_unpacklo_ps(result.x, result.y);
  Vec tmp2 = _mm_unpacklo_ps(result.z, result.w);
  Vec tmp3 = _mm_unpackhi_ps(result.x, result.y);
  Vec tmp4 = _mm_unpackhi_ps(result.z, result.w);

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

VV_FORCE_INLINE Vec4 operator/(Vec4 const& u, Vec4 const& v)
{
  return Vec4(u.x / v.x, u.y / v.y, u.z / v.z, u.w / v.w);
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

VV_FORCE_INLINE Vec4 operator/(Vec4 const& v, Vec const& s)
{
  return Vec4(v.x / s, v.y / s, v.z / s, v.w / s);
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

VV_FORCE_INLINE Vec4 operator/(Vec const& s, Vec4 const& v)
{
  return Vec4(s / v.x, s / v.y, s / v.z, s / v.w);
}

template <typename T>
VV_FORCE_INLINE std::ostream& operator<<(std::ostream& out, base_vec4<T> const& v)
{
  out << "x: " << v.x << "\n";
  out << "y: " << v.y << "\n";
  out << "z: " << v.z << "\n";
  out << "w: " << v.w << "\n";
  return out;
}


/* vector math functions */

template <typename T>
VV_FORCE_INLINE T dot(base_vec4<T> const& u, base_vec4<T> const& v)
{
  return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
}

} // simd
} // virvo

