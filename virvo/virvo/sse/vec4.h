#pragma once

#include "vec.h"
#include "veci.h"

#include "../vvvecmath.h"

#include <ostream>

namespace virvo
{
namespace sse
{
template <typename T>
class CACHE_ALIGN base_vec4
{
public:
  T x;
  T y;
  T z;
  T w;

  inline base_vec4()
  {
  }

  inline base_vec4(T const& s)
    : x(s)
    , y(s)
    , z(s)
    , w(s)
  {
  }

  inline base_vec4(T const& x, T const& y, T const& z, T const& w)
    : x(x)
    , y(y)
    , z(z)
    , w(w)
  {
  }

  /*! \brief  construct from aligned float[4]
   */
  inline base_vec4(T const v[4])
    : x(v[0])
    , y(v[1])
    , z(v[2])
    , w(v[3])
  {
  }

  /*! \brief  construct from virvo vector
   */
  template <typename U>
  inline base_vec4(vvBaseVector4<U> const& v)
    : x(v[0])
    , y(v[1])
    , z(v[2])
    , w(v[3])
  {
  }

  inline T& operator[](size_t i)
  {
    if (i == 0)
    {
      return x;
    }
    else if (i == 1)
    {
      return y;
    }
    else if (i == 2)
    {
      return z;
    }
    else if (i == 3)
    {
      return w;
    }
    assert(0);
  }

  inline const T& operator[](size_t i) const
  {
    if (i == 0)
    {
      return x;
    }
    else if (i == 1)
    {
      return y;
    }
    else if (i == 2)
    {
      return z;
    }
    else if (i == 3)
    {
      return w;
    }
    assert(0);
  }
};

typedef base_vec4<Veci> Vec4i;
typedef base_vec4<Vec>  Vec4;

template <typename T>
inline base_vec4<T> transpose(base_vec4<T> const& v)
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
inline base_vec4<T> operator-(base_vec4<T> const& v)
{
  return base_vec4<T>(-v.x, -v.y, -v.z, -v.w);
}

template <typename T>
inline base_vec4<T> operator+(base_vec4<T> const& u, base_vec4<T> const& v)
{
  return base_vec4<T>(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
}

template <typename T>
inline base_vec4<T> operator-(base_vec4<T> const& u, base_vec4<T> const& v)
{
  return base_vec4<T>(u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w);
}

template <typename T>
inline base_vec4<T> operator*(base_vec4<T> const& u, base_vec4<T> const& v)
{
  return base_vec4<T>(u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w);
}

inline Vec4 operator/(Vec4 const& u, Vec4 const& v)
{
  return Vec4(u.x / v.x, u.y / v.y, u.z / v.z, u.w / v.w);
}

template <typename T>
inline base_vec4<T> operator+(base_vec4<T> const& v, T const& s)
{
  return base_vec4<T>(v.x + s, v.y + s, v.z + s, v.w + s);
}

template <typename T>
inline base_vec4<T> operator-(base_vec4<T> const& v, T const& s)
{
  return base_vec4<T>(v.x - s, v.y - s, v.z - s, v.w - s);
}

template <typename T>
inline base_vec4<T> operator*(base_vec4<T> const& v, T const& s)
{
  return base_vec4<T>(v.x * s, v.y * s, v.z * s, v.w * s);
}

inline Vec4 operator/(Vec4 const& v, Vec const& s)
{
  return Vec4(v.x / s, v.y / s, v.z / s, v.w / s);
}

template <typename T>
inline base_vec4<T> operator+(T const& s, base_vec4<T> const& v)
{
  return base_vec4<T>(s + v.x, s + v.y, s + v.z, s + v.w);
}

template <typename T>
inline base_vec4<T> operator-(T const& s, base_vec4<T> const& v)
{
  return base_vec4<T>(s - v.x, s - v.y, s - v.z, s - v.w);
}

template <typename T>
inline base_vec4<T> operator*(T const& s, base_vec4<T> const& v)
{
  return base_vec4<T>(s * v.x, s * v.y, s * v.z, s * v.w);
}

inline Vec4 operator/(Vec const& s, Vec4 const& v)
{
  return Vec4(s / v.x, s / v.y, s / v.z, s / v.w);
}

template <typename T>
inline std::ostream& operator<<(std::ostream& out, base_vec4<T> const& v)
{
  out << "x: " << v.x << "\n";
  out << "y: " << v.y << "\n";
  out << "z: " << v.z << "\n";
  out << "w: " << v.w << "\n";
  return out;
}

/* masked operators */

template <typename T, typename M>
inline base_vec4<T> neg(base_vec4<T> const& v, M const& mask)
{
  return base_vec4<T>(neg(v.x, mask), neg(v.y, mask), neg(v.z, mask), neg(v.w, mask));
}

template <typename T, typename M>
inline base_vec4<T> add(base_vec4<T> const& u, base_vec4<T> const& v, M const& mask)
{
  return base_vec4<T>(add(u.x, v.x, mask), add(u.y, v.y, mask), add(u.z, v.z, mask), add(u.w, v.w, mask));
}

template <typename T, typename M>
inline base_vec4<T> sub(base_vec4<T> const& u, base_vec4<T> const& v, M const& mask)
{
  return base_vec4<T>(sub(u.x, v.x, mask), sub(u.y, v.y, mask), sub(u.z, v.z, mask), sub(u.w, v.w, mask));
}

template <typename T, typename M>
inline base_vec4<T> mul(base_vec4<T> const& u, base_vec4<T> const& v, M const& mask)
{
  return base_vec4<T>(mul(u.x, v.x, mask), mul(u.y, v.y, mask), mul(u.z, v.z, mask), mul(u.w, v.w, mask));
}

template <typename T, typename M>
inline base_vec4<T> div(base_vec4<T> const& u, base_vec4<T> const& v, M const& mask)
{
  return base_vec4<T>(div(u.x, v.x, mask), div(u.y, v.y, mask), div(u.z, v.z, mask), div(u.w, v.w, mask));
}

template <typename T, typename M>
inline base_vec4<T> add(base_vec4<T> const& v, T const& s, M const& mask)
{
  return base_vec4<T>(add(v.x, s, mask), add(v.y, s, mask), add(v.z, s, mask), add(v.w, s, mask));
}

template <typename T, typename M>
inline base_vec4<T> sub(base_vec4<T> const& v, T const& s, M const& mask)
{
  return base_vec4<T>(sub(v.x, s, mask), sub(v.y, s, mask), sub(v.z, s, mask), sub(v.w, s, mask));
}

template <typename T, typename M>
inline base_vec4<T> mul(base_vec4<T> const& v, T const& s, M const& mask)
{
  return base_vec4<T>(mul(v.x, s, mask), mul(v.y, s, mask), mul(v.z, s, mask), mul(v.w, s, mask));
}

template <typename T, typename M>
inline base_vec4<T> div(base_vec4<T> const& v, T const& s, M const& mask)
{
  return base_vec4<T>(div(v.x, s, mask), div(v.y, s, mask), div(v.z, s, mask), div(v.w, s, mask));
}

template <typename T, typename M>
inline base_vec4<T> add(T const& s, base_vec4<T> const& v, M const& mask)
{
  return base_vec4<T>(add(s, v.x, mask), add(s, v.y, mask), add(s, v.z, mask), add(s, v.w, mask));
}

template <typename T, typename M>
inline base_vec4<T> sub(T const& s, base_vec4<T> const& v, M const& mask)
{
  return base_vec4<T>(sub(s, v.x, mask), sub(s, v.y, mask), sub(s, v.z, mask), sub(s, v.w, mask));
}

template <typename T, typename M>
inline base_vec4<T> mul(T const& s, base_vec4<T> const& v, M const& mask)
{
  return base_vec4<T>(mul(s, v.x, mask), mul(s, v.y, mask), mul(s, v.z, mask), mul(s, v.w, mask));
}

template <typename T, typename M>
inline base_vec4<T> div(T const& s, base_vec4<T> const& v, M const& mask)
{
  return base_vec4<T>(div(s, v.x, mask), div(s, v.y, mask), div(s, v.z, mask), div(s, v.w, mask));
}

/* vector math functions */

template <typename T>
inline T dot(base_vec4<T> const& u, base_vec4<T> const& v)
{
  return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
}

} // sse
} // virvo

