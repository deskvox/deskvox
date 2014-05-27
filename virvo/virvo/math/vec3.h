#pragma once

#include "simd/vec.h"
#include "simd/veci.h"

#include <ostream>

namespace virvo
{
namespace math
{
template <typename T>
class CACHE_ALIGN base_vec3
{
public:

  typedef T value_type;

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

  template < typename S >
  VV_FORCE_INLINE base_vec3(base_vec3< S > const& rhs)
    : x(rhs.x)
    , y(rhs.y)
    , z(rhs.z)
  {
  }

  template < typename S >
  VV_FORCE_INLINE base_vec3& operator=(base_vec3< S > const& rhs)
  {
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
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

template <typename T>
VV_FORCE_INLINE base_vec3<T> operator/(base_vec3<T> const& u, base_vec3<T> const& v)
{
  return base_vec3<T>(u.x / v.x, u.y / v.y, u.z / v.z);
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

template <typename T>
VV_FORCE_INLINE base_vec3<T> operator/(base_vec3<T> const& v, T const& s)
{
  return base_vec3<T>(v.x / s, v.y / s, v.z / s);
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

template <typename T>
VV_FORCE_INLINE base_vec3<T> operator/(T const& s, base_vec3<T> const& v)
{
  return base_vec3<T>(s / v.x, s / v.y, s / v.z);
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
VV_FORCE_INLINE base_vec3< sse_vec > normalize(base_vec3< sse_vec > const& v)
{
  return v * rsqrt<refinements>(dot(v, v));
}

VV_FORCE_INLINE Vec3 normalize(Vec3 const& v)
{
  return v * rsqrt<1>(dot(v, v));
}

VV_FORCE_INLINE base_vec3< float > normalize(base_vec3< float > const& v)
{
  return v * (1.0f / std::sqrt(dot(v, v)));
}

}

} // math
} // virvo

