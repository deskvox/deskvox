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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_CUDA

#include "vvglew.h"

#include "vvcuda.h"
#include "vvcudaimg.h"
#include "vvcudautils.h"
#include "vvdebugmsg.h"
#include "vvgltools.h"
#include "vvrayrend.h"

#include <iostream>
#include <limits>

using std::cerr;
using std::endl;

texture<uchar, 3, cudaReadModeNormalizedFloat> volTexture8;
texture<ushort, 3, cudaReadModeNormalizedFloat> volTexture16;
texture<float4, 1, cudaReadModeElementType> tfTexture;
texture<float4, 1, cudaReadModeElementType> randTexture;

const int NUM_RAND_VECS = 8192;

#ifndef ALPHA_PASS_IBR
#define ALPHA_PASS_IBR 1
#endif

#ifndef ENTRANCE_IBR
#define ENTRANCE_IBR 0
#endif

#ifndef EXIT_IBR
#define EXIT_IBR 0
#endif

int iDivUp(const int a, const int b)
{
  return (a + b - 1) / b;
}

typedef struct
{
  float m[4][4];
} matrix4x4;

__constant__ matrix4x4 c_invViewMatrix;
__constant__ matrix4x4 c_MvPrMatrix;

struct Ray
{
  float3 o;
  float3 d;
};

template<int t_bpc>
__device__ float volume(const float x, const float y, const float z)
{
  if (t_bpc == 1)
  {
    return tex3D(volTexture8, x, y, z);
  }
  else if (t_bpc == 2)
  {
    return tex3D(volTexture16, x, y, z);
  }
  else
  {
    return -1.0f;
  }
}

template<int t_bpc>
__device__ float volume(const float3& pos)
{
  if (t_bpc == 1)
  {
    return tex3D(volTexture8, pos.x, pos.y, pos.z);
  }
  else if (t_bpc == 2)
  {
    return tex3D(volTexture16, pos.x, pos.y, pos.z);
  }
  else
  {
    return -1.0f;
  }
}

__device__ bool skipSpace(const float3& pos)
{
  //return (tex3D(spaceSkippingTexture, pos.x, pos.y, pos.z) == 0.0f);
  return false;
}

__device__ float3 calcTexCoord(const float3& pos, const float3& volPos, const float3& volSizeHalf)
{
  return make_float3((pos.x - volPos.x + volSizeHalf.x) / (volSizeHalf.x * 2.0f),
                     (pos.y - volPos.y + volSizeHalf.y) / (volSizeHalf.y * 2.0f),
                     (pos.z - volPos.z + volSizeHalf.z) / (volSizeHalf.z * 2.0f));
}

__device__ bool solveQuadraticEquation(const float A, const float B, const float C,
                                       float* tnear, float* tfar)
{
  const float discrim = B * B - 4.0f * A * C;
  if (discrim < 0.0f)
  {
    *tnear = -1.0f;
    *tfar = -1.0f;
  }
  const float rootDiscrim = __fsqrt_rn(discrim);
  float q;
  if (B < 0)
  {
    q = -0.5f * (B - rootDiscrim);
  }
  else
  {
    q = -0.5f * (B + rootDiscrim);
  }
  *tnear = q / A;
  *tfar = C / q;
  if (*tnear > *tfar)
  {
    float tmp = *tnear;
    *tnear = *tfar;
    *tfar = tmp;
    return true;
  }
  return false;
}

__device__ bool intersectBox(const Ray& ray, const float3& boxmin, const float3& boxmax,
                             float* tnear, float* tfar)
{
  // compute intersection of ray with all six bbox planes
  float3 invR = make_float3(1.0f, 1.0f, 1.0f) / ray.d;
  float t1 = (boxmin.x - ray.o.x) * invR.x;
  float t2 = (boxmax.x - ray.o.x) * invR.x;
  float tmin = fminf(t1, t2);
  float tmax = fmaxf(t1, t2);

  t1 = (boxmin.y - ray.o.y) * invR.y;
  t2 = (boxmax.y - ray.o.y) * invR.y;
  tmin = fmaxf(fminf(t1, t2), tmin);
  tmax = fminf(fmaxf(t1, t2), tmax);

  t1 = (boxmin.z - ray.o.z) * invR.z;
  t2 = (boxmax.z - ray.o.z) * invR.z;
  tmin = fmaxf(fminf(t1, t2), tmin);
  tmax = fminf(fmaxf(t1, t2), tmax);

  *tnear = tmin;
  *tfar = tmax;

  return ((tmax >= tmin) && (tmax >= 0.0f));
}

__device__ bool intersectSphere(const Ray& ray, const float3& center, const float radiusSqr,
                                float* tnear, float* tfar)
{
  Ray r = ray;
  r.o -= center;
  float A = r.d.x * r.d.x + r.d.y * r.d.y
          + r.d.z * r.d.z;
  float B = 2 * (r.d.x * r.o.x + r.d.y * r.o.y
               + r.d.z * r.o.z);
  float C = r.o.x * r.o.x + r.o.y * r.o.y
          + r.o.z * r.o.z - radiusSqr;
  return solveQuadraticEquation(A, B, C, tnear, tfar);
}

__device__ void intersectPlane(const Ray& ray, const float3& normal, const float& dist,
                               float* nddot, float* tnear)
{
  *nddot = dot(normal, ray.d);
  const float vOrigin = dist - dot(normal, ray.o);
  *tnear = vOrigin / *nddot;
}


__device__ float4 mulPost(const matrix4x4& M, const float4& v)
{
  float4 result;
  result.x = M.m[0][0] * v.x + M.m[0][1] * v.y + M.m[0][2] * v.z + M.m[0][3] * v.w;
  result.y = M.m[1][0] * v.x + M.m[1][1] * v.y + M.m[1][2] * v.z + M.m[1][3] * v.w;
  result.z = M.m[2][0] * v.x + M.m[2][1] * v.y + M.m[2][2] * v.z + M.m[2][3] * v.w;
  result.w = M.m[3][0] * v.x + M.m[3][1] * v.y + M.m[3][2] * v.z + M.m[3][3] * v.w;
  return result;
}

__device__ float4 mulPre(const matrix4x4& M, const float4& v)
{
  float4 result;
  result.x = M.m[0][0] * v.x + M.m[1][0] * v.y + M.m[2][0] * v.z + M.m[3][0] * v.w;
  result.y = M.m[0][1] * v.x + M.m[1][1] * v.y + M.m[2][1] * v.z + M.m[3][1] * v.w;
  result.z = M.m[0][2] * v.x + M.m[1][2] * v.y + M.m[2][2] * v.z + M.m[3][2] * v.w;
  result.w = M.m[0][3] * v.x + M.m[1][3] * v.y + M.m[2][3] * v.z + M.m[3][3] * v.w;
  return result;
}

__device__ float3 perspectiveDivide(const float4& v)
{
  const float wInv = 1.0f / v.w;
  return make_float3(v.x * wInv, v.y * wInv, v.z * wInv);
}

__device__ uchar4 rgbaFloatToInt(float4 rgba)
{
  clamp(rgba.x);
  clamp(rgba.y);
  clamp(rgba.z);
  clamp(rgba.w);
  return make_uchar4(rgba.x * 255, rgba.y * 255,rgba.z * 255, rgba.w * 255);
}

template<int t_bpc>
__device__ float3 gradient(const float3& pos)
{
  const float DELTA = 0.01f;

  float3 sample1;
  float3 sample2;

  sample1.x = volume<t_bpc>(pos - make_float3(DELTA, 0.0f, 0.0f));
  sample2.x = volume<t_bpc>(pos + make_float3(DELTA, 0.0f, 0.0f));
  sample1.y = volume<t_bpc>(pos - make_float3(0.0f, DELTA, 0.0f));
  sample2.y = volume<t_bpc>(pos + make_float3(0.0f, DELTA, 0.0f));
  sample1.z = volume<t_bpc>(pos - make_float3(0.0f, 0.0f, DELTA));
  sample2.z = volume<t_bpc>(pos + make_float3(0.0f, 0.0f, DELTA));

  return sample2 - sample1;
}

template<int t_bpc>
__device__ float4 blinnPhong(const float4& classification, const float3& pos,
                             const float3& L, const float3& H,
                             const float3& Ka, const float3& Kd, const float3& Ks,
                             const float shininess,
                             const float3* normal = NULL)
{
  float3 N = normalize(gradient<t_bpc>(pos));

  if (normal != NULL)
  {
    // Interpolate gradient with normal from clip object (based on opacity).
    N = (*normal * classification.w) + (N * (1.0f - classification.w));
    N = normalize(N);
  }

  const float ldot = dot(L, N);

  const float3 c = make_float3(classification);

  // Ambient term.
  float3 tmp = Ka * c;
  if (ldot > 0.0f)
  {
    // Diffuse term.
    tmp += Kd * ldot * c;

    // Specular term.
    const float spec = powf(dot(H, N), shininess);
    if (spec > 0.0f)
    {
      tmp += Ks * spec * c;
    }
  }
  return make_float4(tmp.x, tmp.y, tmp.z, classification.w);
}

template<
         bool t_earlyRayTermination,
         int t_bpc,
         int t_mipMode,
         bool t_lighting,
         bool t_opacityCorrection,
         bool t_jittering,
         bool t_clipPlane,
         bool t_clipSphere,
         bool t_useSphereAsProbe,
         bool t_useIbr,
         int t_ibrMode
        >
__global__ void render(uchar4* d_output, const uint width, const uint height,
                       const float4 backgroundColor,
                       const uint texwidth, const float dist,
                       const float3 volPos, const float3 volSizeHalf,
                       const float3 probePos, const float3 probeSizeHalf,
                       const float3 L, const float3 H,
                       const float3 sphereCenter, const float sphereRadius,
                       const float3 planeNormal, const float planeDist,
                       void* d_depth, int dp,
                       const float2 ibrPlanes, const bool alphaPassIbr,
                       const bool gatherPass, float* d_gatheredAlpha)
{
  const float opacityThreshold = 0.95f;

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= width) || (y >= height))
  {
    return;
  }

  const float u = (x / static_cast<float>(width)) * 2.0f - 1.0f;
  const float v = (y / static_cast<float>(height)) * 2.0f - 1.0f;

  /*
   * Rays like if the view were orthographic with origins over each pixel.
   * These are multiplied with the inverse modelview projection matrix.
   * First of all, the rays will be transformed to fit to the frustum.
   * Then the rays will be oriented so that they can hit the volume.
   */
  const float4 o = mulPost(c_invViewMatrix, make_float4(u, v, -1.0f, 1.0f));
  const float4 d = mulPost(c_invViewMatrix, make_float4(u, v, 1.0f, 1.0f));

  Ray ray;
  ray.o = perspectiveDivide(o);
  ray.d = perspectiveDivide(d);
  ray.d = ray.d - ray.o;
  ray.d = normalize(ray.d);

  float tnear;
  float tfar;
  const bool hit = intersectBox(ray, probePos - probeSizeHalf, probePos + probeSizeHalf, &tnear, &tfar);
  if (!hit)
  {
    d_output[y * texwidth + x] = make_uchar4(0);
    if(t_useIbr)
    {
      switch(dp)
      {
      case 8:
        ((unsigned char*)(d_depth))[y * texwidth + x] = 0;
        break;
      case 16:
        ((unsigned short*)(d_depth))[y * texwidth + x] = 0;
        break;
      case 32:
        ((float*)(d_depth))[y * texwidth + x] = 0.f;
        break;
      }
    }
    return;
  }

  if (fmodf(tnear, dist) != 0.0f)
  {
    int tmp = (tnear / dist);
    tnear = dist * tmp;
  }

  if (tnear < 0.0f)
  {
    tnear = 0.0f;
  }

  // Calc hits with clip sphere.
  float tsnear;
  float tsfar;
  if (t_clipSphere)
  {
    // In probe mode, rays that don't hit the sphere simply aren't rendered.
    // In ordinary sphere mode, the intersection data is memorized.
    if (!intersectSphere(ray, sphereCenter, sphereRadius, &tsnear, &tsfar) && t_useSphereAsProbe)
    {
      d_output[y * texwidth + x] = make_uchar4(0);
      return;
    }
  }

  // Calc hits with clip plane.
  float tpnear;
  float nddot;
  if (t_clipPlane)
  {
    intersectPlane(ray, planeNormal, planeDist, &nddot, &tpnear);
  }

  float4 dst;

  if (t_mipMode > 0)
  {
    dst = backgroundColor;
  }
  else
  {
    dst = make_float4(0.0f);
  }

  float t = tnear;
  float3 pos = ray.o + ray.d * tnear;

  if (t_jittering)
  {
    const float4 randOffset = tex1D(randTexture, (y * width + x) % NUM_RAND_VECS);
    pos += make_float3(randOffset);
  }
  const float3 step = ray.d * dist;

  // If just clipped, shade with the normal of the clipping surface.
  bool justClippedPlane = false;
  bool justClippedSphere = false;

  float3 ibrDepth    = make_float3(0.0f, 0.0f, 0.0f);
  bool ibrDepthFound = false;
  float ibrOpacityWeight = 0.8f;
#if !ALPHA_PASS_IBR
  float maxDiff      = 0.0f;
  float lastAlpha    = 0.0f;
#endif

  // Ensure that dist is big enough
  bool infinite = false;
  if(tnear+dist != tnear && tfar+dist != tfar)
    infinite = true;

  while(infinite)
  {
    // Test for clipping.
    const bool clippedPlane = (t_clipPlane && (((t <= tpnear) && (nddot >= 0.0f))
                                            || ((t >= tpnear) && (nddot < 0.0f))));
    const bool clippedSphere = t_useSphereAsProbe ? (t_clipSphere && ((t < tsnear) || (t > tsfar)))
                                                  : (t_clipSphere && (t >= tsnear) && (t <= tsfar));

    if (clippedPlane || clippedSphere)
    {
      justClippedPlane = clippedPlane;
      justClippedSphere = clippedSphere;

      t += dist;
      if (t > tfar)
      {
        break;
      }
      pos += step;
      continue;
    }

    const float3 texCoord = calcTexCoord(pos, volPos, volSizeHalf);

    const float sample = volume<t_bpc>(texCoord);

    // Post-classification transfer-function lookup.
    float4 src = tex1D(tfTexture, sample);

    if (t_mipMode == 1)
    {
      dst.x = fmaxf(src.x, dst.x);
      dst.y = fmaxf(src.y, dst.y);
      dst.z = fmaxf(src.z, dst.z);
      dst.w = 1;
    }
    else if (t_mipMode == 2)
    {
      dst.x = fminf(src.x, dst.x);
      dst.y = fminf(src.y, dst.y);
      dst.z = fminf(src.z, dst.z);
      dst.w = 1;
    }

    // Local illumination.
    if (t_lighting && (src.w > 0.1f))
    {
      const float3 Ka = make_float3(0.0f, 0.0f, 0.0f);
      const float3 Kd = make_float3(0.8f, 0.8f, 0.8f);
      const float3 Ks = make_float3(0.8f, 0.8f, 0.8f);
      const float shininess = 1000.0f;
      if (justClippedPlane)
      {
        src = blinnPhong<t_bpc>(src, texCoord, L, H, Ka, Kd, Ks, shininess, &planeNormal);
        justClippedPlane = false;
      }
      else if (justClippedSphere)
      {
        float3 sphereNormal = normalize(pos - sphereCenter);
        src = blinnPhong<t_bpc>(src, texCoord, L, H, Ka, Kd, Ks, shininess, &sphereNormal);
        justClippedSphere = false;
      }
      else
      {
        src = blinnPhong<t_bpc>(src, texCoord, L, H, Ka, Kd, Ks, shininess);
      }
    }
    justClippedPlane = false;
    justClippedSphere = false;

    if (t_opacityCorrection)
    {
      src.w = 1 - powf(1 - src.w, dist);
    }

    if (t_mipMode == 0)
    {
      // pre-multiply alpha
      src.x *= src.w;
      src.y *= src.w;
      src.z *= src.w;
    }

    if (t_mipMode == 0)
    {
      dst = dst + src * (1.0f - dst.w);
    }

    if (t_earlyRayTermination && (dst.w > opacityThreshold))
    {
      break;
    }

    t += dist;
    if (t > tfar)
    {
      break;
    }

    pos += step;

    if(t_useIbr)
    {
#if !ALPHA_PASS_IBR
      if(dst.w - lastAlpha > maxDiff)
      {
        maxDiff  = dst.w - lastAlpha;
        ibrDepth = pos;
      }
      lastAlpha = dst.w;
#else
      if (alphaPassIbr && !gatherPass && !ibrDepthFound)
      {
        if (dst.w > (d_gatheredAlpha[y * texwidth + x] * ibrOpacityWeight))
        {
          ibrDepth = pos;
          ibrDepthFound = true;
        }
      }
#endif
    }
  }

#if ALPHA_PASS_IBR
  if(!ibrDepthFound)
    ibrDepth = pos;
#endif

  if(t_useIbr)
  {
    float3 depth = make_float3(0.0f, 0.0f, 0.0f);
    // convert position to window-coordinates
    const float4 depthWin = mulPost(c_MvPrMatrix, make_float4(ibrDepth  .x, ibrDepth  .y, ibrDepth  .z, 1.0f));
    depth = perspectiveDivide(depthWin);

    depth.z++;
    depth.z = depth.z/2.0f;

    if(ibrPlanes.x != 0.0f && ibrPlanes.y != 0.0f) // SCALED depth mode
    {
      depth.z = (depth.z - ibrPlanes.x) / (ibrPlanes.y - ibrPlanes.x);
    }

    if     (depth.z > 1.0f) depth.z = 1.0f;
    else if(depth.z < 0.0f) depth.z = 0.0f;

    switch(dp)
    {
    case 8:
      ((uchar*)(d_depth))[y * texwidth + x] = (uchar)(depth.z*float(UCHAR_MAX));
      break;
    case 16:
      ((ushort*)(d_depth))[y * texwidth + x] = (ushort)(depth.z*float(USHRT_MAX));
      break;
    case 32:
      ((float*)(d_depth))[y * texwidth + x] = depth.z;
      break;
    }
  }
  d_output[y * texwidth + x] = rgbaFloatToInt(dst);

  if (alphaPassIbr && gatherPass)
  {
    d_gatheredAlpha[y * texwidth + x] = dst.w;
  }
}

typedef void(*renderKernel)(uchar4*, const uint, const uint, const float4,
                            const uint, const float, const float3, const float3,
                            const float3, const float3, const float3, const float3, const float3,
                            const float, const float3, const float, void*, int dp,
                            const float2, const bool alphaPassIbr, const bool gatherPass,
                            float* d_gatheredAlpha);

#ifdef FAST_COMPILE
template<
         int t_bpc,
         bool t_illumination,
         bool t_opacityCorrection,
         bool t_earlyRayTermination,
         bool t_clipPlane,
         bool t_clipSphere,
         bool t_useSphereAsProbe,
         int t_mipMode
        >
renderKernel getKernel(vvRayRend*)
{
  return &render<t_earlyRayTermination, // Early ray termination.
                 t_bpc, // Bytes per channel.
                 t_mipMode, // Mip mode.
                 false, // Local illumination.
                 t_opacityCorrection, // Opacity correction.
                 false, // Jittering.
                 t_clipPlane, // Clip plane.
                 t_clipSphere, // Clip sphere.
                 t_useSphereAsProbe, // Show what's inside the clip sphere.
                 true, // image based rendering,
                 vvRenderState::VV_MAX_GRADIENT
                >;
}
#else

template<
        int t_bpc,
        bool t_illumination,
        bool t_opacityCorrection,
        bool t_earlyRayTermination,
        bool t_clipPlane,
        bool t_clipSphere,
        bool t_useSphereAsProbe,
        int t_mipMode,
        bool t_useIbr,
        vvRayRend::IbrMode t_ibrMode
       >
renderKernel getKernelWithIbrMode(vvRayRend*)
{
  return &render<t_earlyRayTermination, // Early ray termination.
                 t_bpc, // Bytes per channel.
                 t_mipMode, // Mip mode.
                 t_illumination, // Local illumination.
                 t_opacityCorrection, // Opacity correction.
                 false, // Jittering.
                 t_clipPlane, // Clip plane.
                 t_clipSphere, // Clip sphere.
                 t_useSphereAsProbe, // Show what's inside the clip sphere.
                 t_useIbr, // image based rendering
                 t_ibrMode
                >;
}

template<
         int t_bpc,
         bool t_illumination,
         bool t_opacityCorrection,
         bool t_earlyRayTermination,
         bool t_clipPlane,
         bool t_clipSphere,
         bool t_useSphereAsProbe,
         int t_mipMode,
         bool t_useIbr
        >
renderKernel getKernelWithIbr(vvRayRend* rayRend)
{
  switch ((int)rayRend->getParameter(vvRenderState::VV_IBR_MODE))
  {
  case vvRenderState::VV_MAX_GRADIENT:
    return getKernelWithIbrMode<
                                t_bpc,
                                t_illumination,
                                t_opacityCorrection,
                                t_earlyRayTermination,
                                t_clipPlane,
                                t_clipSphere,
                                t_useSphereAsProbe,
                                t_mipMode,
                                t_useIbr,
                                vvRenderState::VV_MAX_GRADIENT
                               >(rayRend);
  case vvRenderState::VV_MIDDLE:
    return getKernelWithIbrMode<
                                t_bpc,
                                t_illumination,
                                t_opacityCorrection,
                                t_earlyRayTermination,
                                t_clipPlane,
                                t_clipSphere,
                                t_useSphereAsProbe,
                                t_mipMode,
                                t_useIbr,
                                vvRenderState::VV_MIDDLE
                               >(rayRend);
  case vvRenderState::VV_SURFACE:
    return getKernelWithIbrMode<
                                t_bpc,
                                t_illumination,
                                t_opacityCorrection,
                                t_earlyRayTermination,
                                t_clipPlane,
                                t_clipSphere,
                                t_useSphereAsProbe,
                                t_mipMode,
                                t_useIbr,
                                vvRenderState::VV_SURFACE
                               >(rayRend);
  default:
    return getKernelWithIbrMode<
                                t_bpc,
                                t_illumination,
                                t_opacityCorrection,
                                t_earlyRayTermination,
                                t_clipPlane,
                                t_clipSphere,
                                t_useSphereAsProbe,
                                t_mipMode,
                                t_useIbr,
                                vvRenderState::VV_MAX_GRADIENT
                               >(rayRend);

  }
}

template<
         int t_bpc,
         bool t_illumination,
         bool t_opacityCorrection,
         bool t_earlyRayTermination,
         bool t_clipPlane,
         bool t_clipSphere,
         bool t_useSphereAsProbe,
         int t_mipMode
        >
renderKernel getKernelWithMip(vvRayRend* rayRend)
{
  if(rayRend->getParameter(vvRenderState::VV_USE_IBR))
  {
    return getKernelWithIbr<
                            t_bpc,
                            t_illumination,
                            t_opacityCorrection,
                            t_earlyRayTermination,
                            t_clipPlane,
                            t_clipSphere,
                            t_useSphereAsProbe,
                            t_mipMode,
                            true
                           >(rayRend);
  }
  else
  {
    return getKernelWithIbr<
                            t_bpc,
                            t_illumination,
                            t_opacityCorrection,
                            t_earlyRayTermination,
                            t_clipPlane,
                            t_clipSphere,
                            t_useSphereAsProbe,
                            t_mipMode,
                            false
                           >(rayRend);
  }
}
template<
         int t_bpc,
         bool t_illumination,
         bool t_opacityCorrection,
         bool t_earlyRayTermination,
         bool t_clipPlane,
         bool t_clipSphere,
         bool t_useSphereAsProbe
        >
renderKernel getKernelWithSphereAsProbe(vvRayRend* rayRend)
{
  switch ((int)rayRend->getParameter(vvRenderState::VV_MIP_MODE))
  {
  case 0:
    return getKernelWithMip<
                            t_bpc,
                            t_illumination,
                            t_opacityCorrection,
                            t_earlyRayTermination,
                            t_clipPlane,
                            t_clipSphere,
                            t_useSphereAsProbe,
                            0
                           >(rayRend);
  case 1:
    // No early ray termination possible with max intensity projection.
    return getKernelWithMip<
                            t_bpc,
                            t_illumination,
                            t_opacityCorrection,
                            false,
                            t_clipPlane,
                            t_clipSphere,
                            t_useSphereAsProbe,
                            1
                           >(rayRend);
  case 2:
    // No early ray termination possible with min intensity projection.
    return getKernelWithMip<
                            t_bpc,
                            t_illumination,
                            t_opacityCorrection,
                            false,
                            t_clipPlane,
                            t_clipSphere,
                            t_useSphereAsProbe,
                            2
                           >(rayRend);
  default:
    return getKernelWithMip<
                            t_bpc,
                            t_illumination,
                            t_opacityCorrection,
                            t_earlyRayTermination,
                            t_clipPlane,
                            t_clipSphere,
                            t_useSphereAsProbe,
                            0
                           >(rayRend);
  }
}

template<
         int t_bpc,
         bool t_illumination,
         bool t_opacityCorrection,
         bool t_earlyRayTermination,
         bool t_clipPlane
        >
renderKernel getKernelWithClipPlane(vvRayRend* rayRend)
{
  if (((rayRend->getParameter(vvRenderState::VV_IS_ROI_USED)!=0.0)
     && (rayRend->getParameter(vvRenderState::VV_SPHERICAL_ROI)!=0.0)))
  {
    return getKernelWithSphereAsProbe<
                                      t_bpc,
                                      t_illumination,
                                      t_opacityCorrection,
                                      t_earlyRayTermination,
                                      t_clipPlane,
                                      true,
                                      true
                                     >(rayRend);
  }
  else
  {
    return getKernelWithSphereAsProbe<
                                      t_bpc,
                                      t_illumination,
                                      t_opacityCorrection,
                                      t_earlyRayTermination,
                                      t_clipPlane,
                                      false,
                                      false
                                     >(rayRend);
  }
}

template<
         int t_bpc,
         bool t_illumination,
         bool t_opacityCorrection,
         bool t_earlyRayTermination
        >
renderKernel getKernelWithEarlyRayTermination(vvRayRend* rayRend)
{
  if (rayRend->getParameter(vvRenderState::VV_CLIP_MODE))
  {
    return getKernelWithClipPlane<
                                  t_bpc,
                                  t_illumination,
                                  t_opacityCorrection,
                                  t_earlyRayTermination,
                                  true
                                 >(rayRend);
  }
  else
  {
    {
      return getKernelWithClipPlane<
                                    t_bpc,
                                    t_illumination,
                                    t_opacityCorrection,
                                    t_earlyRayTermination,
                                    false
                                   >(rayRend);
    }
  }
}

template<
         int t_bpc,
         bool t_illumination,
         bool t_opacityCorrection
        >
renderKernel getKernelWithOpacityCorrection(vvRayRend* rayRend)
{
  if (rayRend->getEarlyRayTermination())
  {
    return getKernelWithEarlyRayTermination<
                                            t_bpc,
                                            t_illumination,
                                            t_opacityCorrection,
                                            true
                                           >(rayRend);
  }
  else
  {
    return getKernelWithEarlyRayTermination<
                                            t_bpc,
                                            t_illumination,
                                            t_opacityCorrection,
                                            false
                                           >(rayRend);
  }
}

template<
         int t_bpc,
         bool t_illumination
        >
renderKernel getKernelWithIllumination(vvRayRend* rayRend)
{
  if (rayRend->getOpacityCorrection())
  {
    return getKernelWithOpacityCorrection<t_bpc, t_illumination, true>(rayRend);
  }
  else
  {
    return getKernelWithOpacityCorrection<t_bpc, t_illumination, false>(rayRend);
  }
}

template<
         int t_bpc
        >
renderKernel getKernelWithBpc(vvRayRend* rayRend)
{
  if (rayRend->getIllumination())
  {
    return getKernelWithIllumination<t_bpc, true>(rayRend);
  }
  else
  {
    return getKernelWithIllumination<t_bpc, false>(rayRend);
  }
}

renderKernel getKernel(vvRayRend* rayRend)
{
  if (rayRend->getVolDesc()->bpc == 1)
  {
    return getKernelWithBpc<1>(rayRend);
  }
  else if (rayRend->getVolDesc()->bpc == 2)
  {
    return getKernelWithBpc<2>(rayRend);
  }
  else
  {
    return getKernelWithBpc<1>(rayRend);
  }
}
#endif

vvRayRend::vvRayRend(vvVolDesc* vd, vvRenderState renderState)
  : vvSoftVR(vd, renderState)
{
  glewInit();

  bool ok;

  // Free "cuda error cache".
  vvCuda::checkError(&ok, cudaGetLastError(), "vvRayRend::vvRayRend() - free cuda error cache");

  _volumeCopyToGpuOk = true;

  _earlyRayTermination = true;
  _illumination = false;
  _interpolation = true;
  _opacityCorrection = true;
  _gatherAlphaForIbr = ALPHA_PASS_IBR;

  _depthRange[0] = 0.0f;
  _depthRange[1] = 0.0f;

  _depthPrecision = vvIbrImage::VV_UCHAR;

  _rgbaTF = NULL;

  d_depth = NULL;
  intImg = new vvCudaImg(0, 0);
  allocIbrDepth(0, 0);

  const vvCudaImg::Mode mode = dynamic_cast<vvCudaImg*>(intImg)->getMode();
  if (mode == vvCudaImg::TEXTURE)
  {
    setWarpMode(CUDATEXTURE);
  }

  factorViewMatrix();

  d_randArray = NULL;
  initRandTexture();

  initVolumeTexture();

  d_transferFuncArray = NULL;
  updateTransferFunction();
}

vvRayRend::~vvRayRend()
{
  bool ok;
  for (int f=0; f<d_volumeArrays.size(); ++f)
  {
    vvCuda::checkError(&ok, cudaFreeArray(d_volumeArrays[f]),
                       "vvRayRend::~vvRayRend() - free volume frame");
  }

  vvCuda::checkError(&ok, cudaFreeArray(d_transferFuncArray),
                     "vvRayRend::~vvRayRend() - free tf");
  vvCuda::checkError(&ok, cudaFreeArray(d_randArray),
                     "vvRayRend::~vvRayRend() - free rand array");
  vvCuda::checkError(&ok, cudaFree(d_depth),
                     "vvRayRend::~vvRayRend() - free depth");

  delete[] _rgbaTF;
}

int vvRayRend::getLUTSize() const
{
   vvDebugMsg::msg(2, "vvRayRend::getLUTSize()");
   return (vd->getBPV()==2) ? 4096 : 256;
}

void vvRayRend::updateTransferFunction()
{
  bool ok;

  int lutEntries = getLUTSize();
  delete[] _rgbaTF;
  _rgbaTF = new float[4 * lutEntries];

  vd->computeTFTexture(lutEntries, 1, 1, _rgbaTF);

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

  vvCuda::checkError(&ok, cudaFreeArray(d_transferFuncArray),
                     "vvRayRend::updateTransferFunction() - free tf texture");
  vvCuda::checkError(&ok, cudaMallocArray(&d_transferFuncArray, &channelDesc, lutEntries, 1),
                     "vvRayRend::updateTransferFunction() - malloc tf texture");
  vvCuda::checkError(&ok, cudaMemcpyToArray(d_transferFuncArray, 0, 0, _rgbaTF, lutEntries * 4 * sizeof(float),
                                            cudaMemcpyHostToDevice),
                     "vvRayRend::updateTransferFunction() - copy tf texture to device");


  tfTexture.filterMode = cudaFilterModeLinear;
  tfTexture.normalized = true;    // access with normalized texture coordinates
  tfTexture.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

  vvCuda::checkError(&ok, cudaBindTextureToArray(tfTexture, d_transferFuncArray, channelDesc),
                     "vvRayRend::updateTransferFunction() - bind tf texture");
}

void vvRayRend::compositeVolume(int, int)
{
  vvDebugMsg::msg(1, "vvRayRend::compositeVolume()");

  bool ok;

  if(!_volumeCopyToGpuOk)
  {
    std::cerr << "vvRayRend::compositeVolume() aborted because of previous CUDA-Error" << std::endl;
    return;
  }
  vvDebugMsg::msg(1, "vvRayRend::compositeVolume()");

  const vvGLTools::Viewport vp = vvGLTools::getViewport();

  allocIbrDepth(vp[2], vp[3]);
  intImg->setSize(vp[2], vp[3]);

<<<<<<< HEAD
  vvCuda::checkError(&ok, cudaMalloc(&d_depth, vp[2]*vp[3]*_depthPrecision/8),
          "vvRayRend::compositeVolume() - malloc d_depth");
  vvCuda::checkError(&ok, cudaMemset(d_depth, 0, vp[2]*vp[3]*_depthPrecision/8),
          "vvRayRend::compositeVolume() - memset d_depth");

=======
>>>>>>> Make ibr specific variables in rayrend private
  dynamic_cast<vvCudaImg*>(intImg)->map();

  dim3 blockSize(16, 16);
  dim3 gridSize = dim3(iDivUp(vp[2], blockSize.x), iDivUp(vp[3], blockSize.y));
  const vvVector3 size(vd->getSize());

  vvVector3 probePosObj;
  vvVector3 probeSizeObj;
  vvVector3 probeMin;
  vvVector3 probeMax;
  calcProbeDims(probePosObj, probeSizeObj, probeMin, probeMax);

  vvVector3 clippedProbeSizeObj;
  clippedProbeSizeObj.copy(&probeSizeObj);
  for (int i=0; i<3; ++i)
  {
    if (clippedProbeSizeObj[i] < vd->getSize()[i])
    {
      clippedProbeSizeObj[i] = vd->getSize()[i];
    }
  }

  if (_isROIUsed && !_sphericalROI)
  {
    drawBoundingBox(&probeSizeObj, &_roiPos, &_probeColor);
  }

  const float diagonalVoxels = sqrtf(float(vd->vox[0] * vd->vox[0] +
                                           vd->vox[1] * vd->vox[1] +
                                           vd->vox[2] * vd->vox[2]));
  int numSlices = max(1, static_cast<int>(_quality * diagonalVoxels));

  vvMatrix Mv, MvPr;
  vvGLTools::getModelviewMatrix(&Mv);
  vvGLTools::getProjectionMatrix(&MvPr);
  MvPr.multiplyPre(&Mv);

  float* mvprM = new float[16];
  MvPr.get(mvprM);
  cudaMemcpyToSymbol(c_MvPrMatrix, mvprM, sizeof(float4) * 4);

  vvMatrix invMv;
  invMv.copy(&Mv);
  invMv.invert();

  vvMatrix pr;
  vvGLTools::getProjectionMatrix(&pr);

  vvMatrix invMvpr;
  vvGLTools::getModelviewMatrix(&invMvpr);
  invMvpr.multiplyPost(&pr);
  invMvpr.invert();

  float* viewM = new float[16];
  invMvpr.get(viewM);
  cudaMemcpyToSymbol(c_invViewMatrix, viewM, sizeof(float4) * 4);
  delete[] viewM;

  const float3 volPos = make_float3(vd->pos[0], vd->pos[1], vd->pos[2]);
  float3 probePos = volPos;
  if (_isROIUsed && !_sphericalROI)
  {
    probePos = make_float3(probePosObj[0],  probePosObj[1], probePosObj[2]);
  }
  vvVector3 sz = vd->getSize();
  const float3 volSize = make_float3(sz[0], sz[1], sz[2]);
  float3 probeSize = make_float3(probeSizeObj[0], probeSizeObj[1], probeSizeObj[2]);
  if (_sphericalROI)
  {
    probeSize = make_float3((float)vd->vox[0], (float)vd->vox[1], (float)vd->vox[2]);
  }

  const bool isOrtho = pr.isProjOrtho();

  vvVector3 eye;
  getEyePosition(&eye);
  eye.multiply(&invMv);

  vvVector3 origin;

  vvVector3 normal;
  getShadingNormal(normal, origin, eye, invMv, isOrtho);

  const float3 N = make_float3(normal[0], normal[1], normal[2]);

  const float3 L(-N);

  // Viewing direction.
  const float3 V(-N);

  // Half way vector.
  const float3 H = normalize(L + V);

  // Clip sphere.
  const float3 center = make_float3(_roiPos[0], _roiPos[1], _roiPos[2]);
  const float radius = _roiSize[0] * vd->getSize()[0];

  // Clip plane.
  const float3 pnormal = normalize(make_float3(_clipNormal[0], _clipNormal[1], _clipNormal[2]));
  const float pdist = _clipNormal.dot(&_clipPoint);

  if (_clipMode && _clipPerimeter)
  {
    drawPlanePerimeter(&size, &vd->pos, &_clipPoint, &_clipNormal, &_clipColor);
  }

  GLfloat bgcolor[4];
  glGetFloatv(GL_COLOR_CLEAR_VALUE, bgcolor);
  float4 backgroundColor = make_float4(bgcolor[0], bgcolor[1], bgcolor[2], bgcolor[3]);

#ifdef FAST_COMPILE
  renderKernel kernel = getKernel<
                        1, // bpc.
                        false, // Local illumination.
                        true, // Opacity correction
                        false, // Early ray termination.
                        false, // Use clip plane.
                        false, // Use clip sphere.
                        false, // Use clip sphere as probe (inverted sphere).
                        0 // Mip mode.
                       >(this);
#else
  renderKernel kernel = getKernel(this);
#endif

  if (kernel != NULL)
  {
    if (vd->bpc == 1)
    {
      cudaBindTextureToArray(volTexture8, d_volumeArrays[vd->getCurrentFrame()], _channelDesc);
    }
    else if (vd->bpc == 2)
    {
      cudaBindTextureToArray(volTexture16, d_volumeArrays[vd->getCurrentFrame()], _channelDesc);
    }

    float* d_gatheredAlpha = NULL;
    if (_gatherAlphaForIbr)
    {
      const size_t size = vp[2] * vp[2] * sizeof(float);
      vvCuda::checkError(&ok, cudaMalloc(&d_gatheredAlpha, size),
                         "vvRayRend::compositeVolume() - malloc gathered alpha");
      vvCuda::checkError(&ok, cudaMemset(d_gatheredAlpha, 0, size),
                         "vvRayRend::compositeVolume() - memset gathered alpha");

      (kernel)<<<gridSize, blockSize>>>(static_cast<vvCudaImg*>(intImg)->getDeviceImg(), vp[2], vp[3],
                                        backgroundColor, intImg->width,diagonalVoxels / (float)numSlices,
                                        volPos, volSize * 0.5f,
                                        probePos, probeSize * 0.5f,
                                        L, H,
                                        center, radius * radius,
                                        pnormal, pdist, d_depth, _depthPrecision,
                                        make_float2(_depthRange[0], _depthRange[1]),
                                        true, true, d_gatheredAlpha);
    }
    const bool twoPassIbr = _gatherAlphaForIbr;
    (kernel)<<<gridSize, blockSize>>>(static_cast<vvCudaImg*>(intImg)->getDeviceImg(), vp[2], vp[3],
                                      backgroundColor, intImg->width,diagonalVoxels / (float)numSlices,
                                      volPos, volSize * 0.5f,
                                      probePos, probeSize * 0.5f,
                                      L, H,
                                      center, radius * radius,
                                      pnormal, pdist, d_depth, _depthPrecision,
                                      make_float2(_depthRange[0], _depthRange[1]),
                                      twoPassIbr, false, d_gatheredAlpha);
    cudaFree(d_gatheredAlpha);
  }
  static_cast<vvCudaImg*>(intImg)->unmap();

  // For bounding box, tf palette display, etc.
  vvRenderer::renderVolumeGL();
}
//----------------------------------------------------------------------------
// see parent
void vvRayRend::setParameter(const ParameterType param, const float newValue)
{
  vvDebugMsg::msg(3, "vvTexRend::setParameter()");

  switch (param)
  {
  case vvRenderer::VV_SLICEINT:
    {
      const bool newInterpol = (newValue!=0.0);
      if (_interpolation != newInterpol)
      {
        _interpolation = newInterpol;
        initVolumeTexture();
        updateTransferFunction();
      }
    }
    break;
  case vvRenderer::VV_LIGHTING:
    _illumination = (newValue!=0.0);
    break;
  case vvRenderer::VV_OPCORR:
    _opacityCorrection = (newValue!=0.0);
    break;
  case vvRenderer::VV_TERMINATEEARLY:
    _earlyRayTermination = (newValue!=0.0);
    break;
  case vvRenderer::VV_IBR_DEPTH_PREC:
    _depthPrecision = (int)newValue;
    break;
  default:
    vvRenderer::setParameter(param, newValue);
    break;
  }
}

//----------------------------------------------------------------------------
// see parent
float vvRayRend::getParameter(const ParameterType param) const
{
  vvDebugMsg::msg(3, "vvTexRend::getParameter()");

  switch (param)
  {
  case vvRenderer::VV_SLICEINT:
    return _interpolation ? 1.f : 0.f;
  case vvRenderer::VV_LIGHTING:
    return _illumination ? 1.f : 0.f;
  case vvRenderer::VV_OPCORR:
    return _opacityCorrection ? 1.f : 0.f;
  case vvRenderer::VV_TERMINATEEARLY:
    return _earlyRayTermination ? 1.f : 0.f;
  case vvRenderer::VV_IBR_DEPTH_PREC:
    return _depthPrecision;
  default:
    return vvRenderer::getParameter(param);
  }
}

bool vvRayRend::getEarlyRayTermination() const
{
  return _earlyRayTermination;
}
bool vvRayRend::getIllumination() const
{
  return _illumination;
}

bool vvRayRend::getInterpolation() const
{
  return _interpolation;
}

bool vvRayRend::getOpacityCorrection() const
{
  return _opacityCorrection;
}

void* vvRayRend::getDeviceDepth() const
{
  return d_depth;
}

void vvRayRend::setDepthRange(const float min, const float max)
{
  _depthRange[0] = min;
  _depthRange[1] = max;
}

const float* vvRayRend::getDepthRange() const
{
  return _depthRange;
}

void vvRayRend::initRandTexture()
{
  bool ok;

  const float scale = 2.0f;

  //srand(time(NULL));

  float4* randVecs = new float4[NUM_RAND_VECS];
  for (int i=0; i<NUM_RAND_VECS; ++i)
  {
    randVecs[i].x = (static_cast<float>(rand()) / static_cast<float>(INT_MAX)) * scale;
    randVecs[i].y = (static_cast<float>(rand()) / static_cast<float>(INT_MAX)) * scale;
    randVecs[i].z = (static_cast<float>(rand()) / static_cast<float>(INT_MAX)) * scale;
  }

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

  vvCuda::checkError(&ok, cudaFreeArray(d_randArray),
                     "vvRayRend::initRandTexture()");
  vvCuda::checkError(&ok, cudaMallocArray(&d_randArray, &channelDesc, NUM_RAND_VECS, 1),
                     "vvRayRend::initRandTexture()");
  vvCuda::checkError(&ok, cudaMemcpyToArray(d_randArray, 0, 0, randVecs, NUM_RAND_VECS * sizeof(float4),
                                            cudaMemcpyHostToDevice), "vvRayRend::initRandTexture()");

  randTexture.filterMode = cudaFilterModeLinear;
  randTexture.addressMode[0] = cudaAddressModeClamp;

  vvCuda::checkError(&ok, cudaBindTextureToArray(randTexture, d_randArray, channelDesc),
                     "vvRayRend::initRandTexture()");

  delete[] randVecs;
}

void vvRayRend::initVolumeTexture()
{
  bool ok;

  cudaExtent volumeSize = make_cudaExtent(vd->vox[0], vd->vox[1], vd->vox[2]);
  if (vd->bpc == 1)
  {
    _channelDesc = cudaCreateChannelDesc<uchar>();
  }
  else if (vd->bpc == 2)
  {
    _channelDesc = cudaCreateChannelDesc<ushort>();
  }
  d_volumeArrays.resize(vd->frames);

  int outOfMemFrame = -1;
  for (int f=0; f<vd->frames; ++f)
  {
    vvCuda::checkError(&_volumeCopyToGpuOk, cudaMalloc3DArray(&d_volumeArrays[f],
                                            &_channelDesc,
                                            volumeSize),
                       "vvRayRend::initVolumeTexture() - try to alloc 3D array");
    size_t availableMem;
    size_t totalMem;
    vvCuda::checkError(&ok, cudaMemGetInfo(&availableMem, &totalMem),
                       "vvRayRend::initVolumeTexture() - get mem info from device");

    if(!_volumeCopyToGpuOk)
    {
      outOfMemFrame = f;
      break;
    }

    vvDebugMsg::msg(1, "Total CUDA memory:     ", (int)totalMem);
    vvDebugMsg::msg(1, "Available CUDA memory: ", (int)availableMem);

    cudaMemcpy3DParms copyParams = { 0 };

    if (vd->bpc == 1)
    {
      copyParams.srcPtr = make_cudaPitchedPtr(vd->getRaw(f), volumeSize.width*vd->bpc, volumeSize.width, volumeSize.height);
    }
    else if (vd->bpc == 2)
    {
      const int size = vd->vox[0] * vd->vox[1] * vd->vox[2] * vd->bpc;
      uchar* raw = vd->getRaw(0);
      uchar* data = new uchar[size];

      for (int i=0; i<size; i+=2)
      {
        int val = ((int) raw[i] << 8) | (int) raw[i + 1];
        val >>= 4;
        data[i] = raw[i];
        data[i + 1] = val;
      }
      copyParams.srcPtr = make_cudaPitchedPtr(data, volumeSize.width*vd->bpc, volumeSize.width, volumeSize.height);
    }
    copyParams.dstArray = d_volumeArrays[f];
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    vvCuda::checkError(&ok, cudaMemcpy3D(&copyParams),
                       "vvRayRend::initVolumeTexture() - copy volume frame to 3D array");
  }

  if (outOfMemFrame >= 0)
  {
    cerr << "Couldn't accomodate the volume" << endl;
    for (int f=0; f<=outOfMemFrame; ++f)
    {
      vvCuda::checkError(&ok, cudaFree(d_volumeArrays[f]),
                         "vvRayRend::initVolumeTexture() - free memory after failure");
      d_volumeArrays[f] = NULL;
    }
  }

  if (vd->bpc == 1)
  {
    for (int f=0; f<outOfMemFrame; ++f)
    {
      vvCuda::checkError(&ok, cudaFreeArray(d_volumeArrays[f]),
                         "vvRayRend::initVolumeTexture() - why do we do this right here?");
      d_volumeArrays[f] = NULL;
    }
  }

  if (_volumeCopyToGpuOk)
  {
    if (vd->bpc == 1)
    {
        volTexture8.normalized = true;
        if (_interpolation)
        {
          volTexture8.filterMode = cudaFilterModeLinear;
        }
        else
        {
          volTexture8.filterMode = cudaFilterModePoint;
        }
        volTexture8.addressMode[0] = cudaAddressModeClamp;
        volTexture8.addressMode[1] = cudaAddressModeClamp;
        vvCuda::checkError(&ok, cudaBindTextureToArray(volTexture8, d_volumeArrays[0], _channelDesc),
                           "vvRayRend::initVolumeTexture() - bind volume texture (bpc == 1)");
    }
    else if (vd->bpc == 2)
    {
        volTexture16.normalized = true;
        if (_interpolation)
        {
          volTexture16.filterMode = cudaFilterModeLinear;
        }
        else
        {
          volTexture16.filterMode = cudaFilterModePoint;
        }
        volTexture16.addressMode[0] = cudaAddressModeClamp;
        volTexture16.addressMode[1] = cudaAddressModeClamp;
        vvCuda::checkError(&ok, cudaBindTextureToArray(volTexture16, d_volumeArrays[0], _channelDesc),
                           "vvRayRend::initVolumeTexture() - bind volume texture (bpc == 2)");
    }
  }
}

void vvRayRend::factorViewMatrix()
{
  vvDebugMsg::msg(1, "vvRayRend::factorViewMatrix()");

  vvGLTools::Viewport vp = vvGLTools::getViewport();
  const int w = vvToolshed::getTextureSize(vp[2]);
  const int h = vvToolshed::getTextureSize(vp[3]);

  if ((intImg->width != w) || (intImg->height != h))
  {
    intImg->setSize(w, h);
    allocIbrDepth(w, h);
  }

  iwWarp.identity();
  iwWarp.translate(-1.0f, -1.0f, 0.0f);
  iwWarp.scale(1.0f / (static_cast<float>(vp[2]) * 0.5f), 1.0f / (static_cast<float>(vp[3]) * 0.5f), 0.0f);
}

void vvRayRend::findAxisRepresentations()
{
  // Overwrite default behavior.
}

#endif
