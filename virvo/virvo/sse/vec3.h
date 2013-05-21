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
class CACHE_ALIGN base_vec3
{
public:
  T x;
  T y;
  T z;

  inline base_vec3()
  {
  }

  inline base_vec3(T const& s)
    : x(s)
    , y(s)
    , z(s)
  {
  }

  inline base_vec3(T const& x, T const& y, T const& z)
    : x(x)
    , y(y)
    , z(z)
  {
  }

  /*! \brief  construct from virvo vector
   */
  template <typename U>
  inline base_vec3(vvBaseVector3<U> const& v)
    : x(v[0])
    , y(v[1])
    , z(v[2])
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
    assert(0);
  }
};

typedef base_vec3<Veci> Vec3i;
typedef base_vec3<Vec>  Vec3;

/* operators */

template <typename T>
inline base_vec3<T> operator-(base_vec3<T> const& v)
{
  return base_vec3<T>(-v.x, -v.y, -v.z);
}

template <typename T>
inline base_vec3<T> operator+(base_vec3<T> const& u, base_vec3<T> const& v)
{
  return base_vec3<T>(u.x + v.x, u.y + v.y, u.z + v.z);
}

template <typename T>
inline base_vec3<T> operator-(base_vec3<T> const& u, base_vec3<T> const& v)
{
  return base_vec3<T>(u.x - v.x, u.y - v.y, u.z - v.z);
}

template <typename T>
inline base_vec3<T> operator*(base_vec3<T> const& u, base_vec3<T> const& v)
{
  return base_vec3<T>(u.x * v.x, u.y * v.y, u.z * v.z);
}

inline Vec3 operator/(Vec3 const& u, Vec3 const& v)
{
  return Vec3(u.x / v.x, u.y / v.y, u.z / v.z);
}

template <typename T>
inline base_vec3<T>& operator+=(base_vec3<T>& u, base_vec3<T> const& v)
{
  u.x += v.x;
  u.y += v.y;
  u.z += v.z;
  return u;
}

template <typename T>
inline base_vec3<T> operator+(base_vec3<T> const& v, T const& s)
{
  return base_vec3<T>(v.x + s, v.y + s, v.z + s);
}

template <typename T>
inline base_vec3<T> operator-(base_vec3<T> const& v, T const& s)
{
  return base_vec3<T>(v.x - s, v.y - s, v.z - s);
}

template <typename T>
inline base_vec3<T> operator*(base_vec3<T> const& v, T const& s)
{
  return base_vec3<T>(v.x * s, v.y * s, v.z * s);
}

inline Vec3 operator/(Vec3 const& v, Vec const& s)
{
  return Vec3(v.x / s, v.y / s, v.z / s);
}

template <typename T>
inline base_vec3<T> operator+(T const& s, base_vec3<T> const& v)
{
  return base_vec3<T>(s + v.x, s + v.y, s + v.z);
}

template <typename T>
inline base_vec3<T> operator-(T const& s, base_vec3<T> const& v)
{
  return base_vec3<T>(s - v.x, s - v.y, s - v.z);
}

template <typename T>
inline base_vec3<T> operator*(T const& s, base_vec3<T> const& v)
{
  return base_vec3<T>(s * v.x, s * v.y, s * v.z);
}

inline Vec3 operator/(Vec const& s, Vec3 const& v)
{
  return Vec3(s / v.x, s / v.y, s / v.z);
}

template <typename T>
inline std::ostream& operator<<(std::ostream& out, base_vec3<T> const& v)
{
  out << "x: " << v.x << "\n";
  out << "y: " << v.y << "\n";
  out << "z: " << v.z << "\n";
  return out;
}

/* masked operators */

template <typename T, typename M>
inline base_vec3<T> neg(base_vec3<T> const& v, M const& mask)
{
  return base_vec3<T>(neg(v.x, mask), neg(v.y, mask), neg(v.z, mask));
}

template <typename T, typename M>
inline base_vec3<T> add(base_vec3<T> const& u, base_vec3<T> const& v, M const& mask)
{
  return base_vec3<T>(add(u.x, v.x, mask), add(u.y, v.y, mask), add(u.z, v.z, mask));
}

template <typename T, typename M>
inline base_vec3<T> sub(base_vec3<T> const& u, base_vec3<T> const& v, M const& mask)
{
  return base_vec3<T>(sub(u.x, v.x, mask), sub(u.y, v.y, mask), sub(u.z, v.z, mask));
}

template <typename T, typename M>
inline base_vec3<T> mul(base_vec3<T> const& u, base_vec3<T> const& v, M const& mask)
{
  return base_vec3<T>(mul(u.x, v.x, mask), mul(u.y, v.y, mask), mul(u.z, v.z, mask));
}

template <typename T, typename M>
inline base_vec3<T> div(base_vec3<T> const& u, base_vec3<T> const& v, M const& mask)
{
  return base_vec3<T>(div(u.x, v.x, mask), div(u.y, v.y, mask), div(u.z, v.z, mask));
}

template <typename T, typename M>
inline base_vec3<T> add(base_vec3<T> const& v, T const& s, M const& mask)
{
  return base_vec3<T>(add(v.x, s, mask), add(v.y, s, mask), add(v.z, s, mask));
}

template <typename T, typename M>
inline base_vec3<T> sub(base_vec3<T> const& v, T const& s, M const& mask)
{
  return base_vec3<T>(sub(v.x, s, mask), sub(v.y, s, mask), sub(v.z, s, mask));
}

template <typename T, typename M>
inline base_vec3<T> mul(base_vec3<T> const& v, T const& s, M const& mask)
{
  return base_vec3<T>(mul(v.x, s, mask), mul(v.y, s, mask), mul(v.z, s, mask));
}

template <typename T, typename M>
inline base_vec3<T> div(base_vec3<T> const& v, T const& s, M const& mask)
{
  return base_vec3<T>(div(v.x, s, mask), div(v.y, s, mask), div(v.z, s, mask));
}

template <typename T, typename M>
inline base_vec3<T> add(T const& s, base_vec3<T> const& v, M const& mask)
{
  return base_vec3<T>(add(s, v.x, mask), add(s, v.y, mask), add(s, v.z, mask));
}

template <typename T, typename M>
inline base_vec3<T> sub(T const& s, base_vec3<T> const& v, M const& mask)
{
  return base_vec3<T>(sub(s, v.x, mask), sub(s, v.y, mask), sub(s, v.z, mask));
}

template <typename T, typename M>
inline base_vec3<T> mul(T const& s, base_vec3<T> const& v, M const& mask)
{
  return base_vec3<T>(mul(s, v.x, mask), mul(s, v.y, mask), sub(s, v.z, mask));
}

template <typename T, typename M>
inline base_vec3<T> div(T const& s, base_vec3<T> const& v, M const& mask)
{
  return base_vec3<T>(div(s, v.x, mask), div(s, v.y, mask), div(s, v.z, mask));
}

/* vector math functions */

template <typename T>
inline T dot(base_vec3<T> const& u, base_vec3<T> const& v)
{
  return u.x * v.x + u.y * v.y + u.z * v.z;
}

inline Vec length(Vec3 const& v)
{
  return sqrt(dot(v, v));
}

inline Vec3 normalize(Vec3 const& v)
{
  return v / length(v);
}

/* masked vector math functions */

template <typename T, typename M>
inline T dot(base_vec3<T> const& u, base_vec3<T> const& v, M const& mask)
{
  return add(add(mul(u.x, v.x, mask), mul(u.y, v.y, mask), mask), mul(u.z, v.z, mask), mask);
}

template <typename M>
inline Vec length(Vec3 const& v, M const& mask)
{
  return sqrt(dot(v, v, mask), mask);
}

namespace fast
{

template <unsigned refinements>
inline Vec3 rcp(Vec3 const& v)
{
  return Vec3(rcp<refinements>(v.x), rcp<refinements>(v.y), rcp<refinements>(v.z));
}

inline Vec3 rcp(Vec3 const& v)
{
  return Vec3(rcp<1>(v.x), rcp<1>(v.y), rcp<1>(v.z));
}

template <unsigned refinements>
inline Vec3 normalize(Vec3 const& v)
{
  return v * rsqrt<refinements>(dot(v, v));
}

inline Vec3 normalize(Vec3 const& v)
{
  return v * rsqrt<1>(dot(v, v));
}

}

} // sse
} // virvo

