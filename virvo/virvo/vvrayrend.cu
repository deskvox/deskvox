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


#include "vvrayrend.h"
#include "vvrayrend-common.h"
#include "vvcudatools.h"
#include "vvcudautils.h"
#include "vvvoldesc.h"

#include "Cuda/Symbol.h"
#include "Cuda/Texture.h"


namespace cu = virvo::cuda;


struct Ray
{
  float3 o;
  float3 d;
};


static texture<uchar, 3, cudaReadModeNormalizedFloat> volTexture8;
static texture<ushort, 3, cudaReadModeNormalizedFloat> volTexture16;
static texture<float4, 1, cudaReadModeElementType> tfTexture;

static __constant__ matrix4x4 c_invViewMatrix;
static __constant__ matrix4x4 c_MvPrMatrix;


// referenced in vvrayrend.cpp
cu::Texture cVolTexture8 = &volTexture8;
cu::Texture cVolTexture16 = &volTexture16;
cu::Texture cTFTexture = &tfTexture;

// referenced in vvrayrend.cpp
cu::Symbol<matrix4x4> cInvViewMatrix = &c_invViewMatrix;
cu::Symbol<matrix4x4> cMvPrMatrix = &c_MvPrMatrix;


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
                     (-pos.y - volPos.y + volSizeHalf.y) / (volSizeHalf.y * 2.0f),
                     (-pos.z - volPos.z + volSizeHalf.z) / (volSizeHalf.z * 2.0f));
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
                             const float3* normal = NULL,
                             float constAtt = 1.0f, float linearAtt = 0.0f, float quadAtt = 0.0f)
{
  float3 N = normalize(gradient<t_bpc>(pos));

  if (normal != NULL)
  {
    // Interpolate gradient with normal from clip object (based on opacity).
    N = (*normal * classification.w) + (N * (1.0f - classification.w));
    N = normalize(N);
  }

  float dist = norm(L);
  float att = 1.0f / (constAtt + linearAtt * dist + quadAtt * dist * dist);
  const float ldot = dot(L, N);

  const float3 c = make_float3(classification);

  // Ambient term.
  float3 tmp = Ka * c;
  if (ldot > 0.0f)
  {
    // Diffuse term.
    tmp += Kd * ldot * c * att;

    // Specular term.
    const float spec = powf(dot(H, N), shininess);
    if (spec > 0.0f)
    {
      tmp += Ks * spec * c * att;
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
         bool t_clipping,
         bool t_useIbr
        >
__global__ void render(uchar4* d_output, const uint width, const uint height,
                       const float4 backgroundColor,
                       const uint texwidth, const float dist,
                       const float3 volPos, const float3 volSizeHalf,
                       const float3 probePos, const float3 probeSizeHalf,
                       const float3 L, const float3 H,
                       float constAtt, float linearAtt, float quadAtt,
                       const bool clipPlane,
                       const bool clipSphere,
                       const bool useSphereAsProbe,
                       const float3 sphereCenter, const float sphereRadius,
                       const float3 planeNormal, const float planeDist,
                       void* d_depth, int dp,
                       const float2 ibrPlanes,
                       const IbrMode ibrMode,
                       const bool gatherPass, float* d_firstIbrPass)
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

  // Dist to far-clipping plane from opengl.
  const float tfar = norm(perspectiveDivide(d) - perspectiveDivide(o));

  Ray ray;
  ray.o = perspectiveDivide(o);
  ray.d = perspectiveDivide(d);
  ray.d = ray.d - ray.o;
  ray.d = normalize(ray.d);

  float tbnear = 0.0f;
  float tbfar = 0.0f;
  const bool hit = intersectBox(ray, probePos - probeSizeHalf, probePos + probeSizeHalf, &tbnear, &tbfar);
  if (!hit)
  {
    d_output[y * texwidth + x] = make_uchar4(0);
    if(t_useIbr)
    {
      switch(dp)
      {
      case 8:
        ((uchar*)(d_depth))[y * texwidth + x] = 0;
        break;
      case 16:
        ((ushort*)(d_depth))[y * texwidth + x] = 0;
        break;
      case 32:
        ((float*)(d_depth))[y * texwidth + x] = 0.f;
        break;
      }
    }
    return;
  }

  if (fmodf(tbnear, dist) != 0.0f)
  {
    int tmp = (tbnear / dist);
    tbnear = dist * tmp;
  }

  if (tbnear < 0.0f)
  {
    tbnear = 0.0f;
  }

  if (tbfar > tfar)
  {
    tbfar = tfar;
  }

  // Calc hits with clip sphere.
  float tsnear = 0.0f;
  float tsfar = 0.0f;
  if (t_clipping && clipSphere)
  {
    // In probe mode, rays that don't hit the sphere simply aren't rendered.
    // In ordinary sphere mode, the intersection data is memorized.
    if (!intersectSphere(ray, sphereCenter, sphereRadius, &tsnear, &tsfar) && useSphereAsProbe)
    {
      d_output[y * texwidth + x] = make_uchar4(0);
      return;
    }
  }

  // Calc hits with clip plane.
  float tpnear = 0.0f;
  float nddot = 0.0f;
  if (t_clipping && clipPlane)
  {
    intersectPlane(ray, planeNormal, planeDist, &nddot, &tpnear);
  }

  float4 dst = make_float4(0.0f);

  if (t_mipMode > 0)
  {
    dst = backgroundColor;
  }
  else
  {
    dst = make_float4(0.0f);
  }

  float t = tbnear;
  float3 pos = ray.o + ray.d * tbnear;
  const float3 step = ray.d * dist;

  // If just clipped, shade with the normal of the clipping surface.
  bool justClippedPlane = false;
  bool justClippedSphere = false;

  // ibr
  float3 ibrDepth        = make_float3(0.0f, 0.0f, 0.0f);
  bool ibrDepthFound     = false;
  float ibrOpacityWeight = 0.8f;
  float maxDiff          = 0.0f;
  float maxAlpha         = 0.0f;
  float lastAlpha        = 0.0f;
  float3 ibrEntrance     = make_float3(0.0f, 0.0f, 0.0f);
  float3 ibrExit         = make_float3(0.0f, 0.0f, 0.0f);
  bool ibrEntranceFound  = false;
  float entranceOpacity  = 0.0f;
  float exitOpacity      = 0.0f;

  // Ensure that dist is big enough
  const bool infinite = (tbnear+dist != tbnear && tbfar+dist != tbfar);

  while(infinite)
  {
    if (t_clipping)
    {
      // Test for clipping.
      const bool clippedPlane = (clipPlane && (((t <= tpnear) && (nddot >= 0.0f))
                                              || ((t >= tpnear) && (nddot < 0.0f))));
      const bool clippedSphere = useSphereAsProbe ? (clipSphere && ((t < tsnear) || (t > tsfar)))
                                                  : (clipSphere && (t >= tsnear) && (t <= tsfar));

      if (clippedPlane || clippedSphere)
      {
        justClippedPlane = clippedPlane;
        justClippedSphere = clippedSphere;

        t += dist;
        if (t > tbfar)
        {
          break;
        }
        pos += step;
        continue;
      }
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
        src = blinnPhong<t_bpc>(src, texCoord, L, H, Ka, Kd, Ks, shininess, &planeNormal, constAtt, linearAtt, quadAtt);
        justClippedPlane = false;
      }
      else if (justClippedSphere)
      {
        float3 sphereNormal = normalize(pos - sphereCenter);
        src = blinnPhong<t_bpc>(src, texCoord, L, H, Ka, Kd, Ks, shininess, &sphereNormal, constAtt, linearAtt, quadAtt);
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
    if (t > tbfar)
    {
      break;
    }

    pos += step;

    if(t_useIbr)
    {
      switch (ibrMode)
      {
      // single-pass heuristics
      case VV_ENTRANCE:
        if (!ibrDepthFound && src.w > 0.0f)
        {
          ibrDepth = pos;
          ibrDepthFound = true;
        }
        break;
      case VV_EXIT:
        if (src.w > 0.0f)
        {
          ibrDepth = pos;
        }
        break;
      case VV_MIDPOINT:
        if (!ibrEntranceFound && src.w > 0.0f)
        {
          ibrEntrance = pos;
          ibrEntranceFound  = true;
        }

        if (src.w > 0.0f)
        {
          ibrExit = pos;
        }
        break;
      case VV_THRESHOLD:
        if (!ibrDepthFound && dst.w > ibrOpacityWeight)
        {
          ibrDepth = pos;
          ibrDepthFound = true;
        }
        break;
      case VV_PEAK:
        if (src.w > maxAlpha)
        {
          maxAlpha = src.w;
          ibrDepth = pos;
        }
        break;
      case VV_GRADIENT:
        if(dst.w - lastAlpha > maxDiff)
        {
          maxDiff  = dst.w - lastAlpha;
          ibrDepth = pos;
        }
        lastAlpha = dst.w;
        break;

      // two-pass heuristics
      case VV_REL_THRESHOLD:
        if (!gatherPass && !ibrDepthFound)
        {
          // second pass
          if (dst.w > (d_firstIbrPass[y * texwidth + x] * ibrOpacityWeight))
          {
            ibrDepth = pos;
            ibrDepthFound = true;
          }
        }
        break;
      case VV_EN_EX_MEAN:
        if (gatherPass)
        {
          // first pass
          if (!ibrEntranceFound && src.w > 0.0f)
          {
            ibrEntrance = pos;
            entranceOpacity = src.w;
            ibrEntranceFound  = true;
          }

          if (src.w > 0.0f)
          {
            ibrExit = pos;
            exitOpacity = src.w;
          }
        }
        else
        {
          // second pass
          if (!ibrDepthFound && dst.w > d_firstIbrPass[y * texwidth + x])
          {
            ibrDepth = pos;
            ibrDepthFound = true;
          }
        }
        break;
      default:
        break;
      }
    }
  }

  if (t_useIbr)
  {
    const bool twoPassIbr = (ibrMode == VV_REL_THRESHOLD || ibrMode == VV_EN_EX_MEAN);
    if (twoPassIbr && !ibrDepthFound)
    {
      ibrDepth = pos;
    }

    if (ibrMode == VV_MIDPOINT)
    {
      ibrDepth = (ibrExit - ibrEntrance) * 0.5f;
    }

    // convert position to window-coordinates
    const float4 depthWin = mulPost(c_MvPrMatrix, make_float4(ibrDepth.x, ibrDepth.y, ibrDepth.z, 1.0f));
    float3 depth = perspectiveDivide(depthWin);

    // Scale to 0.0 - 1.0
    depth.z += 1.0f;
    depth.z *= 0.5f;

    depth.z = (depth.z - ibrPlanes.x) / (ibrPlanes.y - ibrPlanes.x);
    clamp(depth.z);

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

    if (ibrMode == VV_REL_THRESHOLD && gatherPass)
    {
      d_firstIbrPass[y * texwidth + x] = dst.w;
    }
    else if (ibrMode == VV_EN_EX_MEAN && gatherPass)
    {
      d_firstIbrPass[y * texwidth + x] = (entranceOpacity + exitOpacity) * 0.5f;
    }
  }
  d_output[y * texwidth + x] = rgbaFloatToInt(dst);
}

typedef void(*renderKernel)(uchar4* d_output, const uint width, const uint height,
                            const float4 backgroundColor,
                            const uint texwidth, const float dist,
                            const float3 volPos, const float3 volSizeHalf,
                            const float3 probePos, const float3 probeSizeHalf,
                            const float3 L, const float3 H,
                            float constAtt, float linearAtt, float quadAtt,
                            const bool clipPlane,
                            const bool clipSphere,
                            const bool useSphereAsProbe,
                            const float3 sphereCenter,
                            const float sphereRadius,
                            const float3 planeNormal, const float planeDist,
                            void* d_depth, int dp,
                            const float2 ibrPlanes,
                            const IbrMode ibrMode,
                            const bool gatherPass, float* d_firstIbrPass);

template<
         int t_bpc,
         bool t_illumination,
         bool t_opacityCorrection,
         bool t_earlyRayTermination,
         bool t_clipping,
         int t_mipMode,
         bool t_useIbr
        >
renderKernel getKernelWithIbr(vvRayRend*)
{
  return &render<t_earlyRayTermination, // Early ray termination.
                 t_bpc, // Bytes per channel.
                 t_mipMode, // Mip mode.
                 t_illumination, // Local illumination.
                 t_opacityCorrection, // Opacity correction.
                 t_clipping,
                 t_useIbr // image based rendering
                >;
}

template<
         int t_bpc,
         bool t_illumination,
         bool t_opacityCorrection,
         bool t_earlyRayTermination,
         bool t_clipping,
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
                            t_clipping,
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
                            t_clipping,
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
         bool t_clipping
        >
renderKernel getKernelWithClipping(vvRayRend* rayRend)
{

  switch ((int)rayRend->getParameter(vvRenderState::VV_MIP_MODE))
  {
  case 0:
    return getKernelWithMip<
                            t_bpc,
                            t_illumination,
                            t_opacityCorrection,
                            t_earlyRayTermination,
                            t_clipping,
                            0
                           >(rayRend);
  case 1:
    // No early ray termination possible with max intensity projection.
    return getKernelWithMip<
                            t_bpc,
                            t_illumination,
                            t_opacityCorrection,
                            false,
                            t_clipping,
                            1
                           >(rayRend);
  case 2:
    // No early ray termination possible with min intensity projection.
    return getKernelWithMip<
                            t_bpc,
                            t_illumination,
                            t_opacityCorrection,
                            false,
                            t_clipping,
                            2
                           >(rayRend);
  default:
    return getKernelWithMip<
                            t_bpc,
                            t_illumination,
                            t_opacityCorrection,
                            t_earlyRayTermination,
                            t_clipping,
                            0
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
    return getKernelWithClipping<
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
      return getKernelWithClipping<
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

extern "C" void CallRayRendKernel(vvRayRend* rayrend,
                                  uchar4* d_output, const uint width, const uint height,
                                  const float4 backgroundColor,
                                  const uint texwidth, const float dist,
                                  const float3 volPos, const float3 volSizeHalf,
                                  const float3 probePos, const float3 probeSizeHalf,
                                  const float3 L, const float3 H,
                                  float constAtt, float linearAtt, float quadAtt,
                                  const bool clipPlane,
                                  const bool clipSphere,
                                  const bool useSphereAsProbe,
                                  const float3 sphereCenter, const float sphereRadius,
                                  const float3 planeNormal, const float planeDist,
                                  void* d_depth, int dp,
                                  const float2 ibrPlanes,
                                  const IbrMode ibrMode,
                                  bool twoPassIbr)
{
  renderKernel kernel = getKernel(rayrend);

  dim3 blockSize(16, 16);
  dim3 gridSize = dim3(vvToolshed::iDivUp(width, blockSize.x), vvToolshed::iDivUp(height, blockSize.y));

  float* d_firstIbrPass = NULL;

  bool ok = true;

  if (twoPassIbr)
  {
    const size_t size = width * height * sizeof(float);
    vvCudaTools::checkError(&ok, cudaMalloc(&d_firstIbrPass, size),
                        "vvRayRend::compositeVolume() - malloc first ibr pass array");
    vvCudaTools::checkError(&ok, cudaMemset(d_firstIbrPass, 0, size),
                        "vvRayRend::compositeVolume() - memset first ibr pass array");

    (kernel)<<<gridSize, blockSize>>>(d_output, width, height,
                                      backgroundColor,
                                      texwidth, dist,
                                      volPos, volSizeHalf,
                                      probePos, probeSizeHalf,
                                      L, H,
                                      constAtt, linearAtt, quadAtt,
                                      clipPlane,
                                      clipSphere,
                                      useSphereAsProbe,
                                      sphereCenter, sphereRadius,
                                      planeNormal, planeDist,
                                      d_depth, dp,
                                      ibrPlanes,
                                      ibrMode, true, d_firstIbrPass);
  }

  (kernel)<<<gridSize, blockSize>>>(d_output, width, height,
                                    backgroundColor,
                                    texwidth, dist,
                                    volPos, volSizeHalf,
                                    probePos, probeSizeHalf,
                                    L, H,
                                    constAtt, linearAtt, quadAtt,
                                    clipPlane,
                                    clipSphere,
                                    useSphereAsProbe,
                                    sphereCenter, sphereRadius,
                                    planeNormal, planeDist,
                                    d_depth, dp,
                                    ibrPlanes,
                                    ibrMode, false, d_firstIbrPass);

  vvCudaTools::checkError(&ok, cudaFree(d_firstIbrPass),
                          "vvRayRend::compositeVolume() - free first ibr pass array");
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
