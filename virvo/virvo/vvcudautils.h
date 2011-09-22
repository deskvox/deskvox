// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef _VV_CUDAUTILS_H_
#define _VV_CUDAUTILS_H_

#include <cuda.h>

#ifndef __CUDACC__

#include <cmath>

inline float rsqrtf(const float x)
{
  return 1.0f / sqrtf(x);
}

#endif

#include <ostream>

inline std::ostream& operator<<(std::ostream& out, const uchar2& v)
{
  out << v.x << " " << v.y;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const uchar3& v)
{
  out << v.x << " " << v.y << " " << v.z;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const uchar4& v)
{
  out << v.x << " " << v.y << " " << v.z << " " << v.w;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const int2& v)
{
  out << v.x << " " << v.y;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const int3& v)
{
  out << v.x << " " << v.y << " " << v.z;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const int4& v)
{
  out << v.x << " " << v.y << " " << v.z << " " << v.w;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const float2& v)
{
  out << v.x << " " << v.y;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const float3& v)
{
  out << v.x << " " << v.y << " " << v.z;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const float4& v)
{
  out << v.x << " " << v.y << " " << v.z << " " << v.w;
  return out;
}

inline __device__ __host__ float3 operator-(const float3& v)
{
  return make_float3(-v.x, -v.y, -v.z);
}

inline __device__ __host__ float3 operator+(const float3& v1, const float3& v2)
{
  return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

inline __device__ __host__ float4 operator+(const float4& v1, const float4& v2)
{
  return make_float4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}

inline __device__ __host__ uchar4 operator+(const uchar4& v1, const uchar4& v2)
{
  return make_uchar4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}

inline __device__ __host__ float3 operator-(const float3& v1, const float3& v2)
{
  return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

inline __device__ __host__ float4 operator-(const float4& v1, const float4& v2)
{
  return make_float4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
}

inline __device__ __host__ float3 operator*(const float3& v1, const float3& v2)
{
  return make_float3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

inline __device__ __host__ float3 operator/(const float3& v1, const float3& v2)
{
  return make_float3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

inline __device__ __host__ float3 operator+(const float3& v, const float f)
{
  return make_float3(v.x + f, v.y + f, v.z + f);
}

inline __device__ __host__ float3 operator-(const float3& v, const float f)
{
  return make_float3(v.x - f, v.y - f, v.z - f);
}

inline __device__ __host__ float3 operator*(const float3& v, const float f)
{
  return make_float3(v.x * f, v.y * f, v.z * f);
}

inline __device__ __host__ float4 operator*(const float4& v, const float f)
{
  return make_float4(v.x * f, v.y * f, v.z * f, v.w * f);
}

inline __device__ __host__ float3 operator/(const float3& v, const float f)
{
  return v * (1.0f / f);
}

inline __device__ __host__ float3 operator*(const float3& v, const float m[16])
{
  float3 result = make_float3(m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * 1.0f,
                              m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7] * 1.0f,
                              m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11] * 1.0f);
  const float w = m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15] * 1.0f;

  if (w != 1.0f)
  {
    const float wInv = 1.0f / w;
    result.x *= wInv;
    result.y *= wInv;
    result.z *= wInv;
  }
  return result;
}

inline __device__ __host__ float3 make_float3(const float v)
{
  return make_float3(v, v, v);
}

inline __device__ __host__ float3 make_float3(const float4& v)
{
  return make_float3(v.x, v.y, v.z);
}

inline __device__ __host__ float4 make_float4(const float v)
{
  return make_float4(v, v, v, v);
}

inline __device__ __host__ float4 make_float4(const float3& v)
{
  return make_float4(v.x, v.y, v.z, 1.0f);
}

inline __device__ __host__ uchar4 make_uchar4(const unsigned char v)
{
  return make_uchar4(v, v, v, v);
}

inline __device__ __host__ void operator+=(float3& v, const float f)
{
  v.x += f;
  v.y += f;
  v.z += f;
}

inline __device__ __host__ void operator-=(float3& v, const float f)
{
  v.x -= f;
  v.y -= f;
  v.z -= f;
}

inline __device__ __host__ void operator*=(float3& v, const float f)
{
  v.x *= f;
  v.y *= f;
  v.z *= f;
}

inline __device__ __host__ void operator*=(float4& v, const float f)
{
  v.x *= f;
  v.y *= f;
  v.z *= f;
  v.w *= f;
}

inline __device__ __host__ void operator/=(float3& v, const float f)
{
  const float fInv = 1.0f / f;
  v.x *= fInv;
  v.y *= fInv;
  v.z *= fInv;
}

inline __device__ __host__ void operator+=(float3& v1, const float3& v2)
{
  v1.x += v2.x;
  v1.y += v2.y;
  v1.z += v2.z;
}

inline __device__ __host__ void operator-=(float3& v1, const float3& v2)
{
  v1.x -= v2.x;
  v1.y -= v2.y;
  v1.z -= v2.z;
}

inline __device__ __host__ void operator*=(float3& v1, const float3& v2)
{
  v1.x *= v2.x;
  v1.y *= v2.y;
  v1.z *= v2.z;
}

inline __device__ __host__ void operator/=(float3& v1, const float3& v2)
{
  v1.x /= v2.x;
  v1.y /= v2.y;
  v1.z /= v2.z;
}

inline __device__ __host__ void operator*=(float3& v, const float m[16])
{
  v = v * m;
}

inline __device__ __host__ void clamp(float& f, const float a = 0.0f, const float b = 1.0f)
{
  f = fmaxf(a, fminf(f, b));
}

inline __device__ __host__ void clamp(float3& v, const float a = 0.0f, const float b = 1.0f)
{
  clamp(v.x, a, b);
  clamp(v.y, a, b);
  clamp(v.z, a, b);
}

inline __device__ __host__ float3 cross(const float3& v1, const float3& v2)
{
  return make_float3(v1.y * v2.z - v1.z * v2.y,
                     v1.z * v2.x - v1.x * v2.z,
                     v1.x * v2.y - v1.y * v2.x);
}

inline __device__ __host__ float dot(const float3& v1, const float3& v2)
{
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __device__ __host__ float dot(const float4& v1, const float4& v2)
{
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

inline __device__ __host__ float norm(const float3& v)
{
  const float f = v.x * v.x + v.y * v.y + v.z * v.z;
  if (fabs(f - 1.0f) < (1e-4 * 1e-4))
  {
    return 1.0f;
  }
  else
  {
    return sqrtf(f);
  }
}

inline __device__ __host__ float3 normalize(const float3& v)
{
  const float lengthInv = rsqrtf(dot(v, v));
  return v * lengthInv;
}

inline __device__ __host__ void invert(float3& v)
{
  v = make_float3(1.0f / v.x, 1.0f / v.y, 1.0f / v.z);
}

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
