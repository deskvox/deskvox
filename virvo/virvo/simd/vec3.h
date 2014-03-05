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
class CACHE_ALIGN base_vec3
{
public:
  T x;
  T y;
  T z;

  VV_FORCE_INLINE base_vec3()
  {
  }

  VV_FORCE_INLINE base_vec3(T const& s)
    : x(s)
    , y(s)
    , z(s)
  {
  }

  VV_FORCE_INLINE base_vec3(T const& x, T const& y, T const& z)
    : x(x)
    , y(y)
    , z(z)
  {
  }

  /*! \brief  construct from virvo vector
   */
  template <typename U>
  VV_FORCE_INLINE base_vec3(vvBaseVector3<U> const& v)
    : x(v[0])
    , y(v[1])
    , z(v[2])
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

typedef base_vec3<Veci> Vec3i;
typedef base_vec3<Vec>  Vec3;

/* operators */

template <typename T>
VV_FORCE_INLINE base_vec3<T> operator-(base_vec3<T> const& v)
{
  return base_vec3<T>(-v.x, -v.y, -v.z);
}

template <typename T>
VV_FORCE_INLINE base_vec3<T> operator+(base_vec3<T> const& u, base_vec3<T> const& v)
{
  return base_vec3<T>(u.x + v.x, u.y + v.y, u.z + v.z);
}

template <typename T>
VV_FORCE_INLINE base_vec3<T> operator-(base_vec3<T> const& u, base_vec3<T> const& v)
{
  return base_vec3<T>(u.x - v.x, u.y - v.y, u.z - v.z);
}

template <typename T>
VV_FORCE_INLINE base_vec3<T> operator*(base_vec3<T> const& u, base_vec3<T> const& v)
{
  return base_vec3<T>(u.x * v.x, u.y * v.y, u.z * v.z);
}

VV_FORCE_INLINE Vec3 operator/(Vec3 const& u, Vec3 const& v)
{
  return Vec3(u.x / v.x, u.y / v.y, u.z / v.z);
}

template <typename T>
VV_FORCE_INLINE base_vec3<T>& operator+=(base_vec3<T>& u, base_vec3<T> const& v)
{
  u.x += v.x;
  u.y += v.y;
  u.z += v.z;
  return u;
}

template <typename T>
VV_FORCE_INLINE base_vec3<T> operator+(base_vec3<T> const& v, T const& s)
{
  return base_vec3<T>(v.x + s, v.y + s, v.z + s);
}

template <typename T>
VV_FORCE_INLINE base_vec3<T> operator-(base_vec3<T> const& v, T const& s)
{
  return base_vec3<T>(v.x - s, v.y - s, v.z - s);
}

template <typename T>
VV_FORCE_INLINE base_vec3<T> operator*(base_vec3<T> const& v, T const& s)
{
  return base_vec3<T>(v.x * s, v.y * s, v.z * s);
}

VV_FORCE_INLINE Vec3 operator/(Vec3 const& v, Vec const& s)
{
  return Vec3(v.x / s, v.y / s, v.z / s);
}

template <typename T>
VV_FORCE_INLINE base_vec3<T> operator+(T const& s, base_vec3<T> const& v)
{
  return base_vec3<T>(s + v.x, s + v.y, s + v.z);
}

template <typename T>
VV_FORCE_INLINE base_vec3<T> operator-(T const& s, base_vec3<T> const& v)
{
  return base_vec3<T>(s - v.x, s - v.y, s - v.z);
}

template <typename T>
VV_FORCE_INLINE base_vec3<T> operator*(T const& s, base_vec3<T> const& v)
{
  return base_vec3<T>(s * v.x, s * v.y, s * v.z);
}

VV_FORCE_INLINE Vec3 operator/(Vec const& s, Vec3 const& v)
{
  return Vec3(s / v.x, s / v.y, s / v.z);
}

template <typename T>
VV_FORCE_INLINE std::ostream& operator<<(std::ostream& out, base_vec3<T> const& v)
{
  out << "x: " << v.x << "\n";
  out << "y: " << v.y << "\n";
  out << "z: " << v.z << "\n";
  return out;
}


/* vector math functions */

template <typename T>
VV_FORCE_INLINE T dot(base_vec3<T> const& u, base_vec3<T> const& v)
{
  return u.x * v.x + u.y * v.y + u.z * v.z;
}

VV_FORCE_INLINE Vec length(Vec3 const& v)
{
  return sqrt(dot(v, v));
}

VV_FORCE_INLINE Vec3 normalize(Vec3 const& v)
{
  return v / length(v);
}

/* masked vector math functions */

template <typename T, typename M>
VV_FORCE_INLINE T dot(base_vec3<T> const& u, base_vec3<T> const& v, M const& mask)
{
  return add(add(mul(u.x, v.x, mask), mul(u.y, v.y, mask), mask), mul(u.z, v.z, mask), mask);
}

template <typename M>
VV_FORCE_INLINE Vec length(Vec3 const& v, M const& mask)
{
  return sqrt(dot(v, v, mask), mask);
}

namespace fast
{

template <unsigned refinements>
VV_FORCE_INLINE Vec3 rcp(Vec3 const& v)
{
  return Vec3(rcp<refinements>(v.x), rcp<refinements>(v.y), rcp<refinements>(v.z));
}

VV_FORCE_INLINE Vec3 rcp(Vec3 const& v)
{
  return Vec3(rcp<1>(v.x), rcp<1>(v.y), rcp<1>(v.z));
}

template <unsigned refinements>
VV_FORCE_INLINE Vec3 normalize(Vec3 const& v)
{
  return v * rsqrt<refinements>(dot(v, v));
}

VV_FORCE_INLINE Vec3 normalize(Vec3 const& v)
{
  return v * rsqrt<1>(dot(v, v));
}

}

} // simd
} // virvo

