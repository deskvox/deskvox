// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 2010 University of Cologne
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

#include <iostream>
using std::cerr;
using std::endl;

#include <cmath>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "vvdebugmsg.h"
#include "vvvecmath.h"
#include "vvstopwatch.h"
#include "vvcuda.h"
#include "vvcudaimg.h"
#include "vvcudasw.h"

const int MAX_SLICES = 1600;
const int SliceStack = 32;

__constant__ int   c_vox[5];
__constant__ int2 c_start[MAX_SLICES];
static int2 h_start[MAX_SLICES];
__constant__ int2 c_stop[MAX_SLICES];
static int2 h_stop[MAX_SLICES];
__constant__ float2 c_tcStart[MAX_SLICES];
static float2 h_tcStart[MAX_SLICES];
__constant__ float2 c_tcStep[MAX_SLICES];
static float2 h_tcStep[MAX_SLICES];
__constant__ float c_tc3[MAX_SLICES];
static float h_tc3[MAX_SLICES];
__constant__ float c_zStep;


const int MaxCompositeSlices = MAX_SLICES;

const int nthreads = 128;
const dim3 Patch(16, 16);

#define VARSTEP
#define PATCHES
//#define UNROLL
//#define NOSHMEM
//#define SHMCLASS
//#define NOOP
//#define NOLOAD
//#define NODISPLAY
//#define SHMLOAD
#define VOLTEX3D 3 // undef, 1 or 3
//#define PITCHED
//#define FLOATDATA
//#define CONSTLOAD
//#define THREADPERVOXEL
//#define CONSTDATA

#ifdef PATCHES
#define NOSHMEM
#endif

#ifdef VOLTEX3D
#undef PITCHED
#undef CONSTLOAD
#undef CONSTDATA
#undef NOLOAD
#undef SHMLOAD
#undef SHMCLASS
#endif

#ifdef CONSTDATA
#undef SHMLOAD
#endif

#ifdef CONSTLOAD
#undef NOLOAD
#endif

#ifdef NOLOAD
#define SHMLOAD
#endif

typedef uchar4 LutEntry;

#ifdef SHMCLASS
#define SHMLOAD
texture<LutEntry, 1, cudaReadModeElementType> tex_tf;
#else
texture<LutEntry, 1, cudaReadModeNormalizedFloat> tex_tf;
#endif

texture<LutEntry, 2, cudaReadModeNormalizedFloat> tex_preint;
texture<uchar, 2, cudaReadModeNormalizedFloat> tex_minmaxTable;

#ifdef VOLTEX3D
#ifdef FLOATDATA
texture<float, 3, cudaReadModeElementType> tex_raw;
#else
texture<uchar, 3, cudaReadModeNormalizedFloat> tex_raw;
#endif
texture<uchar, 3, cudaReadModeNormalizedFloat> tex_min;
texture<uchar, 3, cudaReadModeNormalizedFloat> tex_max;
#endif

#ifdef FLOATDATA
typedef float Scalar;
#else
typedef uchar Scalar;
#endif

typedef void (*CompositionFunction)(
      uchar4 * __restrict__ img, int width, int height,
#ifdef PITCHED
      const cudaPitchedPtr pvoxels,
#else
      const Scalar * __restrict__ voxels,
#endif
      int firstSlice, int lastSlice,
      int2 from, int2 to, int nslice, float scale);

//----------------------------------------------------------------------------
// device code (CUDA)
//----------------------------------------------------------------------------

__device__ int2 coord(int2 from)
{
    return make_int2(threadIdx.x + blockDim.x*blockIdx.x + from.x,
            threadIdx.y + blockDim.y*blockIdx.y + from.y);
}

__global__ void clearImage(uchar4 * __restrict__ img, int width, int height,
      int fromY, int toY)
{
    const int2 p = coord(make_int2(0, fromY));
    if (p.y >= toY)
        return;
    if (p.x >= width)
        return;

    uchar4 *dest = img + p.y*width+p.x;
    *dest = make_uchar4(0, 0, 0, 0);
}

__device__ void blend(uchar4 *dst, float4 src)
{
    uchar4 c = *dst;
    *dst = make_uchar4(src.x * 255.f + c.x*src.w,
            src.y * 255.f + c.y*src.w,
            src.z * 255.f + c.z*src.w,
            (1.f-src.w) * (255-c.w) + c.w);
}

__device__ void blend(uchar4 *dst, uchar4 src)
{
    uchar4 c = *dst;
    const float w = src.w/255.f;
    *dst = make_uchar4(src.x + c.x*w,
            src.y + c.y*w,
            src.z + c.z*w,
            (1.f-w) * (255-c.w) + c.w);
}

__device__ void initPixel(float4 *pix)
{
    *pix = make_float4(0, 0, 0, 1.f);
}

__device__ void initPixel(uchar4 *pix)
{
    *pix = make_uchar4(0, 0, 0, 255);
}

__device__ bool isOpaque(float4 pix)
{
    return (pix.w < 0.05f);
}

__device__ bool isOpaque(uchar4 pix)
{
    return (pix.w < 13);
}

template<typename Pixel>
__device__ Pixel classify(float s)
{
    return tex1Dfetch(tex_tf, s*255.f);
}

template<typename Pixel>
__device__ Pixel classify(uchar s)
{
    return tex1Dfetch(tex_tf, s);
}

#ifdef VOLTEX3D
__device__ float2 minmax(float x, float y, float z, int principal)
{
    switch(principal)
    {
        case 0:
            return make_float2(tex3D(tex_min, z, x, y),
                    tex3D(tex_max, z, x, y));
        case 1:
            return make_float2(tex3D(tex_min, y, z, x),
                    tex3D(tex_max, y, z, x));
        case 2:
            return make_float2(tex3D(tex_min, x, y, z),
                    tex3D(tex_max, x, y, z));
    }
    return make_float2(-1.f, -1.f);
}

__device__ float volume(float x, float y, float z, int principal)
{
#if VOLTEX3D==3
    return tex3D(tex_raw, x, y, z);
#else
    switch(principal)
    {
        case 0:
            return tex3D(tex_raw, z, x, y);
        case 1:
            return tex3D(tex_raw, y, z, x);
        case 2:
            return tex3D(tex_raw, x, y, z);
    }
    return -1.f;
#endif
}

__device__ float volume(int px, int py, int slice, int principal)
{
    const float x = c_tcStart[slice].x + c_tcStep[slice].x*px;
    const float y = c_tcStart[slice].y + c_tcStep[slice].y*py;
    const float z = c_tc3[slice];
#if VOLTEX3D==3
    return tex3D(tex_raw, x, y, z);
#else
    switch(principal)
    {
        case 0:
            return tex3D(tex_raw, z, x, y);
        case 1:
            return tex3D(tex_raw, y, z, x);
        case 2:
            return tex3D(tex_raw, x, y, z);
    }
    return -1.f;
#endif
}
#endif

template<typename Scalar, int BPV, typename Pixel, int sliceStep, int principal, bool earlyRayTerm>
__global__ void compositeSlicesNearest(
      uchar4 * __restrict__ img, int width, int height,
#ifdef PITCHED
      const cudaPitchedPtr pvoxels,
#else
      const Scalar * __restrict__ voxels,
#endif
      int firstSlice, int lastSlice,
      int2 from, int2 to, int nslice, float sclae)
{
#ifndef VOLTEX3D
#ifdef PITCHED
    const Scalar *voxels = (Scalar *)pvoxels.ptr;
    const int pitch = pvoxels.pitch;
#else
    const int pitch = c_vox[principal] * BPV * sizeof(Scalar);
#endif
#endif
    const int line = blockIdx.x+from.y;
    if (line >= to.y)
        return;

    // initialise intermediate image line
    extern __shared__ char smem[];
    Pixel *imgLine = (Pixel *)smem;
#ifdef SHMLOAD
#ifdef SHMCLASS
    uchar4 *voxel = (uchar4 *)(smem+width*sizeof(Pixel));
#else
    Scalar *voxel = (Scalar *)(smem+width*sizeof(Pixel));
#endif
#endif

    for (int ix=threadIdx.x+from.x; ix<to.x; ix+=blockDim.x)
    {
        initPixel(&imgLine[ix]);
    }

    // composite slices for this image line
    for (int slice=firstSlice; slice!=lastSlice; slice += sliceStep)
    {
#ifdef CONSTLOAD
        const Scalar *voxLine = (Scalar *)(((uchar *)voxels) + pitch * c_vox[principal+1]);
#endif
#ifdef NOOP
        const int iPosY = line;
#else
        // compute upper left image corner
        const int iPosY = c_start[slice].y;

        if(line < iPosY)
            continue;
        if(line >= iPosY+c_vox[principal+1])
            continue;

        const int iPosX = c_start[slice].x;
#endif

        // the voxel row of the current slice corresponding to this image line
#ifndef NOLOAD
#ifndef CONSTLOAD
#ifndef VOLTEX3D
        const Scalar *voxLine = (Scalar *)(((uchar *)voxels) + pitch * ((slice+1)*c_vox[principal+1] + (iPosY-line-1)));
#endif
#endif

#ifdef SHMLOAD
        for (int ix=threadIdx.x; ix<c_vox[principal+0]; ix+=blockDim.x)
        {
#ifdef SHMCLASS
            voxel[ix] = classify<uchar4>(voxLine[ix]);
#else
            voxel[ix] = voxLine[ix];
#endif
        }
        __syncthreads();
#endif
#endif

#ifndef NOOP
        // Traverse intermediate image pixels which correspond to the current slice.
        // 1 is subtracted from each loop counter to remain inside of the volume boundaries:
#ifdef THREADPERVOXEL
        for (int ix=threadIdx.x; ix<c_vox[principal+0]; ix+=blockDim.x)
#else
        for (int ix=threadIdx.x+from.x; ix<c_vox[principal+0]+iPosX; ix+=blockDim.x)
#endif
        {
#ifndef THREADPERVOXEL
            if(ix<iPosX)
                continue;
#endif
#ifdef THREADPERVOXEL
            const int vidx = ix;
            const int iidx = ix + iPosX;
#else
            const int vidx = ix - iPosX;
            const int iidx = ix;
#endif

            // pointer to destination pixel
            Pixel *pix = imgLine + iidx;
            Pixel d = *pix;
            if(earlyRayTerm && isOpaque(d))
                continue;

#ifdef VOLTEX3D
#if VOLTEX3D == 3
            const float v = tex3D(tex_raw, vidx, c_vox[principal+1]+iPosY-line-1, slice);
#else
            float v;
            switch(principal)
            {
                case 0:
                    v = tex3D(tex_raw, c_vox[2]-slice-1, c_vox[0]-vidx-1, c_vox[1]+iPosY-line-1);
                    break;
                case 1:
                    v = tex3D(tex_raw, line-iPosY, slice, c_vox[1]-vidx-1);
                    break;
                case 2:
                    v = tex3D(tex_raw, vidx, c_vox[0]+iPosY-line-1, slice);
                    break;
            }
#endif
            const float4 c = classify<float4>(v);
#else
#ifdef CONSTDATA
            const float4 c = classify<float4>(uchar(ix));
#else
#ifdef SHMCLASS
            const uchar4 v = *(voxel + BPV * vidx);
            const float4 c = make_float4(v.x/255.f, v.y/255.f, v.z/255.f, v.w/255.f);
#else
            // fetch scalar voxel value
#ifdef SHMLOAD
            const Scalar *v = voxel + BPV * vidx;
#else
            const Scalar *v = voxLine + BPV * vidx;
#endif
            // apply transfer function
            const float4 c = classify<float4>(*v);
#endif
#endif
#endif

            // blend
            const float w = d.w*c.w;
            d.x += w*c.x;
            d.y += w*c.y;
            d.z += w*c.z;
            d.w -= w;

            // store into shmem
            *pix = d;
#ifdef THREADPERVOXEL
            __syncthreads();
#endif
        }
#endif
    }

#ifndef NODISPLAY
    // copy line to intermediate image
    for (int ix=threadIdx.x+from.x; ix<to.x; ix+=blockDim.x)
    {
        uchar4 *dest = img + line*width+ix;
        blend(dest, imgLine[ix]);
    }
#endif
}


template<typename Pixel, int principal, int sliceStep, bool preInt>
struct Ray
{
};

template<typename Pixel, int principal, int sliceStep>
struct Ray<Pixel, principal, sliceStep, false>
{
    Pixel d;

    __device__ Ray()
    {
        initPixel(&d);
    }

    __device__ void accumulate(float x, float y, float z)
    {
        const float v = volume(x, y, z, principal);
        const float4 c = classify<float4>(v);

        // blend
        const float w = d.w*c.w;
        d.x += w*c.x;
        d.y += w*c.y;
        d.z += w*c.z;
        d.w -= w;
    }
};

template<typename Pixel, int principal, int sliceStep>
struct Ray<Pixel, principal, sliceStep, true>
{
    Pixel d;
    float sf;

    __device__ Ray()
        : sf(-1.f)
    {
        initPixel(&d);
    }

    __device__ void accumulate(float x, float y, float z)
    {
        const float sb = volume(x, y, z, principal);
        if(sf >= 0.f)
        {
            const float4 c = tex2D(tex_preint, sf, sb);
            // blend
            const float w = d.w*c.w;
            d.x += w*c.x;
            d.y += w*c.y;
            d.z += w*c.z;
            d.w -= w;
        }
        sf = sb;
    }
};

__device__ bool outsideBounds(int2 p, int2 from1, int2 to1, int2 from2, int2 to2)
{
    return (p.x < from1.x && p.x < from2.x)
        || (p.y < from1.y && p.y < from2.y)
        || (p.x >= to1.x && p.x >= to2.x)
        || (p.y >= to1.y && p.y >= to2.y);
}

__device__ bool fullyInsideIntersection(int2 p1, int2 p2, int2 from1, int2 to1, int2 from2, int2 to2)
{
    return p1.x >= from1.x && p1.x >= from2.x
        && p1.y >= from1.y && p1.y >= from2.y
        && p2.x < to1.x && p2.x < to2.x
        && p2.y < to1.y && p2.y < to2.y;
}

#ifdef VOLTEX3D
template<typename Scalar, int BPV, typename Pixel, int sliceStep, int principal, bool earlyRayTerm, bool preInt>
__global__ void compositeSlicesBilinear(
      uchar4 * __restrict__ img, int width, int height,
#ifdef PITCHED
      const cudaPitchedPtr pvoxels,
#else
      const Scalar * __restrict__ voxels,
#endif
      int firstSlice, int lastSlice,
      int2 from, int2 to, int nslice, float scale)
{
#ifdef PATCHES
    Ray<Pixel, principal, sliceStep, preInt> ray;
    int2 p(coord(from));

    float2 tc = make_float2((p.x-c_start[firstSlice].x)*c_tcStep[firstSlice].x+c_tcStart[firstSlice].x,
            (p.y-c_start[firstSlice].y)*c_tcStep[firstSlice].y+c_tcStart[firstSlice].y);
    float2 tc_inc = make_float2((p.x-c_start[firstSlice+sliceStep].x)*c_tcStep[firstSlice+sliceStep].x+c_tcStart[firstSlice+sliceStep].x,
            (p.y-c_start[firstSlice+sliceStep].y)*c_tcStep[firstSlice+sliceStep].y+c_tcStart[firstSlice+sliceStep].y);
    tc_inc.x -= tc.x;
    tc_inc.y -= tc.y;

    // composite slices for this image line
    for (int sliceb=firstSlice; sliceStep>0 ? sliceb<lastSlice : sliceb>lastSlice; sliceb += sliceStep*SliceStack)
    {
        if(earlyRayTerm && isOpaque(ray.d))
            break;

        int last = lastSlice;
        if(sliceStep > 0 && last>sliceb+SliceStack)
            last = sliceb+SliceStack;
        if(sliceStep < 0 && last<sliceb-SliceStack)
            last = sliceb-SliceStack;

        if(outsideBounds(p, c_start[sliceb], c_stop[sliceb],
                    c_start[last-sliceStep], c_stop[last-sliceStep]))
        {
            tc.x += SliceStack*tc_inc.x;
            tc.y += SliceStack*tc_inc.y;
            continue;
        }

#ifdef UNROLL
        const int2 p1 = make_int2(blockDim.x*blockIdx.x+from.x,
                blockDim.y*blockIdx.y+from.y);
        const int2 p2 = make_int2(blockDim.x*blockIdx.x+from.x+blockDim.x-1,
                blockDim.y*blockIdx.y+from.y+blockDim.y-1);
        if(fullyInsideIntersection(p1, p2,
                    c_start[sliceb], c_stop[sliceb],
                    c_start[last-sliceStep], c_stop[last-sliceStep]))
        {
#pragma unroll 4
            for (int slice=sliceb; sliceStep>0 ? slice<last : slice>last; slice += sliceStep)
            {
                ray.accumulate(tc.x, tc.y, c_tc3[slice]);
                tc.x += tc_inc.x;
                tc.y += tc_inc.y;
            }
        }
        else
#endif
        {
            for (int slice=sliceb; sliceStep>0 ? slice<last : slice>last; slice += sliceStep)
            {
                if(earlyRayTerm && isOpaque(ray.d))
                    break;

                if(p.y < c_start[slice].y
                        || p.y >= c_stop[slice].y
                        || p.x<c_start[slice].x
                        || p.x>=c_stop[slice].x)
                {
                    tc.x += tc_inc.x;
                    tc.y += tc_inc.y;
                    continue;
                }

                ray.accumulate(tc.x, tc.y, c_tc3[slice]);
                tc.x += tc_inc.x;
                tc.y += tc_inc.y;
            }
        }
    }

    // copy pixel to intermediate image
    if(p.x >= from.x && p.x < to.x && p.y >= from.y && p.y < to.y)
    {
        uchar4 *dest = img + p.y*width+p.x;
        blend(dest, ray.d);
    }
#else
    // this block's line from the intermediate image
    const int line = blockIdx.x+from.y;
    if (line >= to.y)
        return;

#ifdef NOSHMEM
    for (int ix=threadIdx.x+from.x; ix<to.x; ix+=blockDim.x)
    {
        Ray<Pixel, principal, preInt> ray;

        // composite slices for this image line
        for (int slice=firstSlice; slice!=lastSlice; slice += sliceStep)
        {
            if(earlyRayTerm && isOpaque(ray.d))
                break;

            // compute upper left image corner
            const int iPosY = c_start[slice].y;

            const int iPosX = c_start[slice].x;
            const int endX = c_stop[slice].x;

            if(line < iPosY)
                continue;
            if(line >= c_stop[slice].y)
                continue;

            // Traverse intermediate image pixels which correspond to the current slice.
            // 1 is subtracted from each loop counter to remain inside of the volume boundaries:
            if(ix<iPosX)
                continue;
            if(ix>=endX)
                continue;

            const int vidx = ix - iPosX;
            ray.accumulate(vidx, line-iPosY, slice);
        }

        // copy pixel to intermediate image
        uchar4 *dest = img + line*width+ix;
        blend(dest, ray.d);
    }
#else
    // initialise intermediate image line
    extern __shared__ char smem[];
    Pixel *imgLine = (Pixel *)smem;
    float *sf = (float *)&imgLine[width*preInt];

    for (int ix=threadIdx.x+from.x; ix<to.x; ix+=blockDim.x)
    {
        initPixel(&imgLine[ix]);
        if(preInt)
            sf[ix] = -1.f;
    }

    // composite slices for this image line
    for (int slice=firstSlice; slice!=lastSlice; slice += sliceStep)
    {
        // compute upper left image corner
        const int iPosY = c_start[slice].y;

        if(line < iPosY)
            continue;
        if(line >= c_stop[slice].y)
            continue;

        const int iPosX = c_start[slice].x;
        const int endX = c_stop[slice].x;

        // Traverse intermediate image pixels which correspond to the current slice.
        // 1 is subtracted from each loop counter to remain inside of the volume boundaries:
        for (int ix=threadIdx.x+from.x; ix<endX; ix+=blockDim.x)
        {
            if(ix<iPosX)
                continue;
            const int vidx = ix - iPosX;
            const int iidx = ix;

            // pointer to destination pixel
            Pixel *pix = imgLine + iidx;
            Pixel d = *pix;
            if(earlyRayTerm && isOpaque(d))
                continue;

            if(preInt)
            {
                const float sb = volume(vidx, line-iPosY, slice, principal);
                if(sf[ix] >= 0.f)
                {
                    const float4 c = tex2D(tex_preint, sf[ix], sb);

                    // blend
                    const float w = d.w*c.w;
                    d.x += w*c.x;
                    d.y += w*c.y;
                    d.z += w*c.z;
                    d.w -= w;

                    // store into shmem
                    *pix = d;
                }
                sf[ix] = sb;
            }
            else
            {
                const float v = volume(vidx, line-iPosY, slice, principal);
                const float4 c = classify<float4>(v);

                // blend
                const float w = d.w*c.w;
                d.x += w*c.x;
                d.y += w*c.y;
                d.z += w*c.z;
                d.w -= w;

                // store into shmem
                *pix = d;
            }
        }
    }

    // copy line to intermediate image
    for (int ix=threadIdx.x+from.x; ix<to.x; ix+=blockDim.x)
    {
        uchar4 *dest = img + line*width+ix;
        blend(dest, imgLine[ix]);
    }
#endif
#endif
}

template<typename Scalar, int BPV, typename Pixel, int sliceStep, int principal, bool earlyRayTerm, bool preInt>
__global__ void compositeSORC(
      uchar4 * __restrict__ img, int width, int height,
#ifdef PITCHED
      const cudaPitchedPtr pvoxels,
#else
      const Scalar * __restrict__ voxels,
#endif
      int firstSlice, int lastSlice,
      int2 from, int2 to, int nslice, float scale)
{
    Ray<Pixel, principal, sliceStep, preInt> ray;
    int2 p(coord(from));

    float2 tc = make_float2((p.x-c_start[firstSlice].x)*c_tcStep[firstSlice].x+c_tcStart[firstSlice].x,
            (p.y-c_start[firstSlice].y)*c_tcStep[firstSlice].y+c_tcStart[firstSlice].y);
    float2 tc_inc = make_float2((p.x-c_start[firstSlice+sliceStep].x)*c_tcStep[firstSlice+sliceStep].x+c_tcStart[firstSlice+sliceStep].x,
            (p.y-c_start[firstSlice+sliceStep].y)*c_tcStep[firstSlice+sliceStep].y+c_tcStart[firstSlice+sliceStep].y);
    float z = sliceStep == -1 ? 1.f : 0.f;
#if 1
    if(principal == 0)
    {
        z = sliceStep == -1 ? 0.f : 1.f;
    }
#endif
    tc_inc.x -= tc.x;
    tc_inc.y -= tc.y;
    tc_inc.x *= scale;
    tc_inc.y *= scale;

    // composite slices for this image line
    for(int sl =0; sl<nslice; ++sl)
    {
        if(earlyRayTerm && isOpaque(ray.d))
            break;;

        if(tc.x >= 0.f && tc.x < 1.f
                && tc.y >= 0.f && tc.y < 1.f)
        {
#if 0
            float2 mm = minmax(tc.x, tc.y, z, principal);
            float op = tex2D(tex_minmaxTable, mm.x, mm.y);
            if(op == 0.f)
            {
                tc.x += 16*tc_inc.x;
                tc.y += 16*tc_inc.y;
                if(principal == 0)
                    z += 16*c_zStep;
                else
                    z -= 16*c_zStep;
            }
#endif
                ray.accumulate(tc.x, tc.y, z);
        }
        tc.x += tc_inc.x;
        tc.y += tc_inc.y;
#if 1
        if(principal == 0)
            z += c_zStep;
        else
#endif
            z -= c_zStep;
        if(sliceStep==1 && z > 1.f)
            break;
        if(sliceStep==-1 && z < 0.f)
            break;
    }

    // copy pixel to intermediate image
    if(p.x >= from.x && p.x < to.x && p.y >= from.y && p.y < to.y)
    {
        uchar4 *dest = img + p.y*width+p.x;
        blend(dest, ray.d);
    }
}
#endif

//----------------------------------------------------------------------------
// host code
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
/** Constructor.
  @param vd volume description of volume to display
  @see vvRenderer
*/
template<class Base>
vvCudaSW<Base>::vvCudaSW(vvVolDesc* vd, vvRenderState rs) : Base(vd, rs)
{
   vvDebugMsg::msg(1, "vvCudaSW::vvCudaSW()");

   if(Base::rendererType == Base::SOFTPAR)
       Base::rendererType = Base::CUDAPAR;
   else if(Base::rendererType == Base::SOFTPER)
       Base::rendererType = Base::CUDAPER;

   bool ok = true;

   interSliceInt = false;

   uchar *minArr = NULL, *maxArr = NULL;
   if(Base::vd->bpc == 1)
   {
       const int ds = 16; // downsampling factor
       int vox[3];
       for(int i=0; i<3; ++i)
           vox[i] = (Base::vd->vox[i]+ds-1)/ds;

       minArr = new uchar[vox[0]*vox[1]*vox[2]];
       maxArr = new uchar[vox[0]*vox[1]*vox[2]];

       vd->computeMinMaxArrays(minArr, maxArr, ds);
       cudaExtent extent = make_cudaExtent(vox[0], vox[1], vox[2]);
       cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar>();
       vvCuda::checkError(&ok,
               cudaMalloc3DArray(&d_minarr, &desc, extent, 0), "cudaMalloc3DArray min");
       vvCuda::checkError(&ok,
               cudaMalloc3DArray(&d_maxarr, &desc, extent, 0), "cudaMalloc3DArray max");
       cudaMemcpy3DParms parms = {0};
       parms.kind = cudaMemcpyHostToDevice;
       parms.extent = make_cudaExtent(vox[0], vox[1], vox[2]);
       parms.srcPtr = make_cudaPitchedPtr(minArr, vox[0], vox[0], vox[1]);
       parms.dstArray = d_minarr;
       vvCuda::checkError(&ok, cudaMemcpy3D(&parms), "cudaMemcpy3D min");
       parms.srcPtr = make_cudaPitchedPtr(maxArr, vox[0], vox[0], vox[1]);
       parms.dstArray = d_maxarr;
       vvCuda::checkError(&ok, cudaMemcpy3D(&parms), "cudaMemcpy3D max");

       delete[] minArr;
       delete[] maxArr;
       minArr = NULL;
       maxArr = NULL;
   }

   oldLutDist = -1.f;
   imagePrecision = 8;
   earlyRayTerm = true;

   delete Base::intImg; // already allocated as vvSoftImg by vvSoftPer/vvSoftPar
   // we need a power-of-2 image size for glTexImage2D
   int imgSize = vvToolshed::getTextureSize(2 * ts_max(vd->vox[0], vd->vox[1], vd->vox[2]));
   Base::intImg = new vvCudaImg(imgSize, imgSize);
   if(static_cast<vvCudaImg*>(Base::intImg)->getMode() == vvCudaImg::TEXTURE)
       Base::setWarpMode(Base::CUDATEXTURE);

   setQuality(Base::_renderState._quality);

#ifdef FLOATDATA
   for(int i=0; i<3; ++i)
   {
       size_t vox = vd->vox[0]*vd->vox[1]*vd->vox[2];
       fraw[i] = new float[vox];
       for(size_t j=0; j<vox; ++j)
       {
           fraw[i][j] = raw[i][j] / 255.f;
       }
   }
#endif

#if defined(PITCHED) || defined(VOLTEX3D)
#ifdef VOLTEX3D
   for (int i=0; i<VOLTEX3D; ++i)
#else
   for (int i=0; i<3; ++i)
#endif
   {
#ifdef PITCHED
       cudaExtent extent = make_cudaExtent(vd->vox[(i+1)%3]*sizeof(Scalar), vd->vox[(i+2)%3], vd->vox[(i+3)%3]);
       if(!vvCuda::checkError(&ok, cudaMalloc3D(&d_voxptr[i], extent), "cudaMalloc3D vox"))
           break;
#else
#if defined(VOLTEX3D) && VOLTEX3D == 1
       cudaExtent extent = make_cudaExtent(vd->vox[0], vd->vox[1], vd->vox[2]);
#else
       cudaExtent extent = make_cudaExtent(vd->vox[(i+1)%3], vd->vox[(i+2)%3], vd->vox[(i+3)%3]);
#endif
       cudaChannelFormatDesc desc = cudaCreateChannelDesc<Scalar>();
       if(!vvCuda::checkError(&ok, cudaMalloc3DArray(&d_voxarr[i], &desc, extent, 0), "cudaMalloc3DArray vox"))
           break;
#endif
       cudaMemcpy3DParms parms = {0};
#if defined(VOLTEX3D) && VOLTEX3D == 1
#ifdef FLOATDATA
       parms.srcPtr = make_cudaPitchedPtr(fraw[2], sizeof(Scalar)*vd->vox[0], vd->vox[0], vd->vox[1]);
#else
       parms.srcPtr = make_cudaPitchedPtr(vd->getRaw(), sizeof(Scalar)*vd->vox[0], vd->vox[0], vd->vox[1]);
#endif
#else
#ifdef FLOATDATA
       parms.srcPtr = make_cudaPitchedPtr(fraw[i], sizeof(Scalar)*vd->vox[(i+1)%3], vd->vox[(i+1)%3], vd->vox[(i+2)%3]);
#else
       parms.srcPtr = make_cudaPitchedPtr(Base::raw[i], sizeof(Scalar)*vd->vox[(i+1)%3], vd->vox[(i+1)%3], vd->vox[(i+2)%3]);
#endif
#endif

#ifdef PITCHED
       parms.dstPtr = d_voxptr[i];
       parms.extent = make_cudaExtent(vd->vox[(i+1)%3]*sizeof(Scalar), vd->vox[(i+2)%3], vd->vox[(i+3)%3]);
#else
       parms.dstArray = d_voxarr[i];
#if defined(VOLTEX3D) && VOLTEX3D == 1
       parms.extent = make_cudaExtent(vd->vox[0], vd->vox[1], vd->vox[2]);
#else
       parms.extent = make_cudaExtent(vd->vox[(i+1)%3], vd->vox[(i+2)%3], vd->vox[(i+3)%3]);
#endif
#endif
       parms.kind = cudaMemcpyHostToDevice;
       if(!vvCuda::checkError(&ok, cudaMemcpy3D(&parms), "cudaMemcpy3D vox"))
           break;
   }
#else
   // alloc memory for voxel arrays (for each principal viewing direction)
   vvCuda::checkError(&ok, cudaMalloc(&d_voxels, sizeof(Scalar)*vd->vox[0]*vd->vox[1]*vd->vox[2]*3), "cudaMalloc vox");
   for (int i=0; i<3; ++i)
   {
#ifdef FLOATDATA
       if (!vvCuda::checkError(&ok, cudaMemcpy(d_voxels+i*sizeof(Scalar)*vd->vox[0]*vd->vox[1]*vd->vox[2],
                   fraw[i], sizeof(Scalar)*vd->getFrameBytes(), cudaMemcpyHostToDevice), "cudaMemcpy vox"))
#else
       if (!vvCuda::checkError(&ok, cudaMemcpy(d_voxels+i*sizeof(Scalar)*vd->vox[0]*vd->vox[1]*vd->vox[2],
                   raw[i], vd->getFrameBytes(), cudaMemcpyHostToDevice), "cudaMemcpy vox"))
#endif
          break;
   }
#endif

   // transfer function is stored as a texture
   vvCuda::checkError(&ok, cudaMalloc(&d_tf, Base::getLUTSize()*sizeof(LutEntry)), "cudaMalloc tf");
   vvCuda::checkError(&ok, cudaBindTexture(NULL, tex_tf, d_tf, Base::getLUTSize()*sizeof(LutEntry)), "bind tf tex");

   // pre-integration table
   cudaChannelFormatDesc desc = cudaCreateChannelDesc<LutEntry>();
   vvCuda::checkError(&ok, cudaMallocArray(&d_preint, &desc, Base::PRE_INT_TABLE_SIZE, Base::PRE_INT_TABLE_SIZE), "cudaMalloc preint");
   tex_preint.normalized = true;
   tex_preint.filterMode = Base::bilinLookup ? cudaFilterModeLinear : cudaFilterModePoint;
   tex_preint.addressMode[0] = cudaAddressModeClamp;
   tex_preint.addressMode[1] = cudaAddressModeClamp;
   vvCuda::checkError(&ok, cudaBindTextureToArray(tex_preint, d_preint, desc), "bind preint tex");

   // min-max-table
   cudaChannelFormatDesc descMinMaxTable = cudaCreateChannelDesc<uchar>();
   vvCuda::checkError(&ok, cudaMallocArray(&d_minmaxTable, &descMinMaxTable, Base::getLUTSize(), Base::getLUTSize()), "cudaMalloc minmax");
   tex_minmaxTable.normalized = true;
   tex_minmaxTable.filterMode = cudaFilterModePoint;
   tex_minmaxTable.addressMode[0] = cudaAddressModeClamp;
   tex_minmaxTable.addressMode[1] = cudaAddressModeClamp;
   vvCuda::checkError(&ok, cudaBindTextureToArray(tex_minmaxTable, d_minmaxTable, descMinMaxTable), "bind minmax tex");

   // copy volume size (in voxels)
   int h_vox[5];
   for (int i=0; i<5; ++i)
       h_vox[i] = vd->vox[(i+1)%3];
   vvCuda::checkError(&ok, cudaMemcpyToSymbol(c_vox, h_vox, sizeof(int)*5), "cudaMemcpy vox");

   Base::updateTransferFunction();
}


//----------------------------------------------------------------------------
/// Destructor.
template<class Base>
vvCudaSW<Base>::~vvCudaSW()
{
   vvDebugMsg::msg(1, "vvCudaSW::~vvCudaSW()");

#ifdef FLOATDATA
   for(int i=0; i<3; ++i)
     delete[] fraw[i];
#endif

   cudaUnbindTexture(tex_raw);
   cudaUnbindTexture(tex_tf);
   cudaFree(d_tf);

   cudaUnbindTexture(tex_preint);
   cudaFree(d_preint);
#ifdef VOLTEX3D
   for(int i=0; i<VOLTEX3D; ++i)
     cudaFree(d_voxarr[i]);
#else
#ifdef PITCHED
   for(int i=0; i<3; ++i)
       cudaFree(d_voxptr[i].ptr);
#else
   cudaFree(d_voxels);
#endif
#endif
}

template<class Base>
void vvCudaSW<Base>::findAxisRepresentations()
{
#if !defined(VOLTEX3D) || VOLTEX3D!=1
    Base::findAxisRepresentations();
#endif
}

template<class Base>
void vvCudaSW<Base>::updateLUT(float dist)
{
    vvDebugMsg::msg(3, "vvCudaSW::updateLUT()", dist);

    float corr[4];                                  // gamma/alpha corrected RGBA values [0..1]

    const int lutEntries = Base::getLUTSize();
    for (int i=0; i<lutEntries; ++i)
    {
        // Gamma correction:
        if (Base::_renderState._gammaCorrection)
        {
            corr[0] = gammaCorrect(Base::rgbaTF[i * 4],     Base::VV_RED);
            corr[1] = gammaCorrect(Base::rgbaTF[i * 4 + 1], Base::VV_GREEN);
            corr[2] = gammaCorrect(Base::rgbaTF[i * 4 + 2], Base::VV_BLUE);
            corr[3] = gammaCorrect(Base::rgbaTF[i * 4 + 3], Base::VV_ALPHA);
        }
        else
        {
            corr[0] = Base::rgbaTF[i * 4];
            corr[1] = Base::rgbaTF[i * 4 + 1];
            corr[2] = Base::rgbaTF[i * 4 + 2];
            corr[3] = Base::rgbaTF[i * 4 + 3];
        }

        // Opacity correction:
        // for 0 distance draw opaque slices
        if (dist<=0.0 || (Base::_renderState._clipMode && Base::_renderState._clipOpaque)) corr[3] = 1.0f;
        else if (Base::opCorr) corr[3] = 1.0f - powf(1.0f - corr[3], dist);

        // Convert float to uchar and copy to rgbaLUT array:
        for (int c=0; c<4; ++c)
        {
            Base::rgbaConv[i][c] = uchar(corr[c] * 255.0f);
        }
    }

    // update min-max-table
    uchar *minmax = new uchar[lutEntries*lutEntries];
    Base::vd->tf.makeMinMaxTable(lutEntries, minmax);
    vvCuda::checkError(NULL, cudaMemcpyToArray(d_minmaxTable, 0, 0, minmax,
                lutEntries*lutEntries, cudaMemcpyHostToDevice), "cudaMemcpy minmax");
    delete[] minmax;

    // Make pre-integrated LUT:
    if (Base::preIntegration)
    {
        //Base::makeLookupTextureOptimized(dist);           // use this line for fast pre-integration LUT
        Base::makeLookupTextureCorrect(dist);   // use this line for slow but more correct pre-integration LUT
    }

    vvCuda::checkError(NULL, cudaMemcpy(d_tf, Base::rgbaConv, Base::getLUTSize()*sizeof(LutEntry), cudaMemcpyHostToDevice), "cudaMemcpy tf");
    if(Base::preIntegration)
    {
        vvCuda::checkError(NULL, cudaMemcpyToArray(d_preint, 0, 0, &Base::preIntTable[0][0][0],
                    Base::PRE_INT_TABLE_SIZE*Base::PRE_INT_TABLE_SIZE*sizeof(LutEntry), cudaMemcpyHostToDevice), "cudaMemcpy preint");
    }
}

template<class Base, typename Pixel, int principal, int sliceStep, bool earlyRayTerm>
CompositionFunction selectComposition(vvCudaSW<Base> *rend)
{
#ifdef VOLTEX3D
    if(rend->getSliceInterpol() || rend->getRendererType() == vvRenderer::CUDAPER)
    {
        if(rend->getInterSliceInterpol() || 1)
        {
            if(rend->getPreIntegration())
                return compositeSORC<Scalar, 1, Pixel, sliceStep, principal, earlyRayTerm, true>;
            else
                return compositeSORC<Scalar, 1, Pixel, sliceStep, principal, earlyRayTerm, false>;
        }
        else
        {
            if(rend->getPreIntegration())
                return compositeSlicesBilinear<Scalar, 1, Pixel, sliceStep, principal, earlyRayTerm, true>;
            else
                return compositeSlicesBilinear<Scalar, 1, Pixel, sliceStep, principal, earlyRayTerm, false>;
        }
    }
    else
#endif
    {
        if(rend->getRendererType() == vvRenderer::CUDAPAR)
            return compositeSlicesNearest<Scalar, 1, Pixel, sliceStep, principal, earlyRayTerm>;
    }

    return NULL;
}

template<class Base, typename Pixel, int principal, int sliceStep>
CompositionFunction selectCompositionWithEarlyTermination(vvCudaSW<Base> *rend)
{
    if(rend->getEarlyRayTerm())
        return selectComposition<Base, Pixel, principal, sliceStep, true>(rend);
    else
        return selectComposition<Base, Pixel, principal, sliceStep, false>(rend);
}

template<class Base, typename Pixel, int principal>
CompositionFunction selectCompositionWithSliceStep(vvCudaSW<Base> *rend, int sliceStep)
{
    switch(sliceStep)
    {
        case 1:
            return selectCompositionWithEarlyTermination<Base, Pixel, principal,1>(rend);
        case -1:
            return selectCompositionWithEarlyTermination<Base, Pixel, principal,-1>(rend);
        default:
            assert("slice step out of range" == NULL);
    }

    return NULL;
}

template<class Base, typename Pixel>
CompositionFunction selectCompositionWithPrincipal(vvCudaSW<Base> *rend, int sliceStep)
{
    switch(rend->getPrincipal())
    {
        case 0:
            return selectCompositionWithSliceStep<Base, Pixel, 0>(rend, sliceStep);
        case 1:
            return selectCompositionWithSliceStep<Base, Pixel, 1>(rend, sliceStep);
        case 2:
            return selectCompositionWithSliceStep<Base, Pixel, 2>(rend, sliceStep);
        default:
            assert("principal axis out of range" == NULL);

    }

    return NULL;
}

template<class Base>
CompositionFunction selectCompositionWithPrecision(vvCudaSW<Base> *rend, int sliceStep)
{
    switch(rend->getPrecision())
    {
        case 8:
            return selectCompositionWithPrincipal<Base, uchar4>(rend, sliceStep);
        case 32:
            return selectCompositionWithPrincipal<Base, float4>(rend, sliceStep);
        default:
            assert("invalid precision" == NULL);
    }

    return NULL;
}

//----------------------------------------------------------------------------
/** Composite the volume slices to the intermediate image.
  The function prepareRendering() must be called before this method.
  The shear transformation matrices have to be computed before calling this method.
  The volume slices are processed from front to back.
  @param from,to optional arguments to define first and last intermediate image line to render.
                 if not passed, the entire intermediate image will be rendered
*/
template<class Base>
void vvCudaSW<Base>::compositeVolume(int fromY, int toY)
{
   vvDebugMsg::msg(3, "vvCudaSW::compositeVolume(): ", fromY, toY);

   // If stacking==true then draw front to back, else draw back to front:
   int firstSlice = (Base::stacking) ? 0 : (Base::len[2]-1);  // first slice to process
   int lastSlice  = (Base::stacking) ? (Base::len[2]-1) : 0;  // last slice to process
   int sliceStep  = (Base::stacking) ? 1 : -1;          // step size to get to next slice

   Base::earlyRayTermination = 0;

   if (fromY == -1)
       fromY = 0;
   if (toY == -1)
       toY = Base::intImg->height;

   // compute data for determining upper left image corner of each slice and copy it to device
   vvVector4 start, end;
   Base::findSlicePosition(firstSlice, &start, &end);
   vvVector4 sinc, einc;
   Base::findSlicePosition(firstSlice+sliceStep, &sinc, &einc);
   sinc.sub(&start);
   einc.sub(&end);

   float dist = sqrtf(1.0f + sinc.e[0] * sinc.e[0] + sinc.e[1] * sinc.e[1]);

   float q = Base::_renderState._quality;
   float s = 1.f/dist;
   int nslice = Base::len[2]/s;
#if 0
   for(int i=0; i<4; ++i)
   {
       sinc.e[i] *= s;
       einc.e[i] *= s;
   }
#endif

   float zstep = -s / Base::vd->vox[Base::principal];
   switch(Base::principal)
   {
       case 0:
           zstep = -(float)sliceStep / Base::vd->vox[Base::principal];
           break;
       case 1:
           zstep = -(float)sliceStep / Base::vd->vox[Base::principal];
           break;
       case 2:
           zstep = -(float)sliceStep / Base::vd->vox[Base::principal];
           break;
   }
   zstep *= s;
   //fprintf(stderr, "nslice=%d, step=%f, tot=%f\n", nslice, zstep, nslice*zstep);
#if 0
   fprintf(stderr, "step=%d, zstep=%f, princ=%d\n",
           sliceStep, zstep, Base::principal);
#endif

   cudaMemcpyToSymbol(c_zStep, &zstep, sizeof(float));

   dist = 1.f/Base::_renderState._quality;

   if(oldLutDist/dist < 0.9f || dist/oldLutDist < 0.9f)
   {
       updateLUT(dist);
       oldLutDist = dist;
   }

   int2 from = make_int2(0, fromY);
   int2 to = make_int2(Base::intImg->width, toY);

   if(Base::sliceInterpol)
   {
       from = make_int2(Base::intImg->width, Base::intImg->height);
       to = make_int2(0, 0);
   }

   vvVector4 scur = start;
   vvVector4 ecur = end;
#if defined(VOLTEX3D) && VOLTEX3D==1
   const int p = Base::principal;
#else
   const int p = 2;
#endif
   for(int slice=firstSlice; slice != lastSlice; slice += sliceStep)
   {
#ifdef VOLTEX3D
       if(Base::sliceInterpol)
       {
           const float sx = scur.e[0]/scur.e[3];
           const float sy = scur.e[1]/scur.e[3];
           const float ex = ecur.e[0]/ecur.e[3];
           const float ey = ecur.e[1]/ecur.e[3];

           h_start[slice].x = max(0,int(floor(sx)));
           h_start[slice].y = max(0,int(floor(sy)));

           from.x = min(from.x, h_start[slice].x);
           from.y = min(from.y, h_start[slice].y);

           h_stop[slice].x = min(Base::intImg->width-1,int(ceil(ex)));
           h_stop[slice].y = min(Base::intImg->height-1,int(ceil(ey)));

           to.x = max(to.x, h_stop[slice].x);
           to.y = max(to.y, h_stop[slice].y);

           switch(p)
           {
               case 0:
                   h_tcStep[slice].x = -1.f/(ex-sx);
                   h_tcStep[slice].y = -1.f/(ey-sy);

                   h_tcStart[slice].x = 1.f + (h_start[slice].x - sx + 0.5f)*h_tcStep[slice].x;
                   h_tcStart[slice].y = 1.f + (h_start[slice].y - sy + 0.5f)*h_tcStep[slice].y;

                   h_tc3[slice] = 1.f-(slice+0.5f)*1.f/Base::vd->vox[Base::principal];
                   break;
                case 1:
                   h_tcStep[slice].x = -1.f/(ex-sx);
                   h_tcStep[slice].y = 1.f/(ey-sy);

                   h_tcStart[slice].x = 1.f + (h_start[slice].x - sx + 0.5f)*h_tcStep[slice].x;
                   h_tcStart[slice].y = (h_start[slice].y - sy + 0.5f)*h_tcStep[slice].y;

                   h_tc3[slice] = (slice+0.5f)*1.f/Base::vd->vox[Base::principal];
                   break;
                case 2:
                   h_tcStep[slice].x = 1.f/(ex-sx);
                   h_tcStep[slice].y = -1.f/(ey-sy);

                   h_tcStart[slice].x = (h_start[slice].x - sx + 0.5f)*h_tcStep[slice].x;
                   h_tcStart[slice].y = 1.f + (h_start[slice].y - sy + 0.5f)*h_tcStep[slice].y;

                   h_tc3[slice] = (slice+0.5f)*1.f/Base::vd->vox[Base::principal];
                   break;
           }

           ecur.add(&einc);
       }
       else
#endif
       {
           h_start[slice].x = int(scur.e[0] / scur.e[3] + 0.5f);
           h_start[slice].y = int(scur.e[1] / scur.e[3] + 0.5f);
       }
       scur.add(&sinc);
   }
   from.y = max(from.y, fromY);
   to.y = min(to.y, toY);

   //fprintf(stderr, "p=%d: (%d,%d) - (%d,%d)\n", Base::principal, from.x, from.y, to.x, to.y);

   bool ok = true;
   vvCuda::checkError(&ok, cudaMemcpyToSymbol(c_start, h_start, sizeof(h_start)), "cudaMemcpy start");
#ifdef VOLTEX3D
   if(Base::sliceInterpol)
   {
       vvCuda::checkError(&ok, cudaMemcpyToSymbol(c_stop, h_stop, sizeof(h_stop)), "cudaMemcpy stop");
       vvCuda::checkError(&ok, cudaMemcpyToSymbol(c_tcStart, h_tcStart, sizeof(h_tcStart)), "cudaMemcpy tcStart");
       vvCuda::checkError(&ok, cudaMemcpyToSymbol(c_tcStep, h_tcStep, sizeof(h_tcStep)), "cudaMemcpy tcStep");
       vvCuda::checkError(&ok, cudaMemcpyToSymbol(c_tc3, h_tc3, sizeof(h_tc3)), "cudaMemcpy tc3");
   }
#endif

#ifdef VOLTEX3D
   tex_raw.normalized = Base::sliceInterpol;
   tex_raw.filterMode = Base::sliceInterpol ? cudaFilterModeLinear : cudaFilterModePoint;
   tex_raw.addressMode[0] = cudaAddressModeClamp;
   tex_raw.addressMode[1] = cudaAddressModeClamp;
   tex_raw.addressMode[2] = cudaAddressModeClamp;
   cudaChannelFormatDesc desc = cudaCreateChannelDesc<Scalar>();
#if VOLTEX3D == 1
   cudaBindTextureToArray(tex_raw, d_voxarr[0], desc);
#else
   cudaBindTextureToArray(tex_raw, d_voxarr[Base::principal], desc);
#endif
#endif

   cudaChannelFormatDesc minmaxDesc = cudaCreateChannelDesc<uchar>();
   tex_min.normalized = true;
   tex_min.filterMode = cudaFilterModePoint;
   tex_min.addressMode[0] = cudaAddressModeClamp;
   tex_min.addressMode[1] = cudaAddressModeClamp;
   tex_min.addressMode[2] = cudaAddressModeClamp;
   cudaBindTextureToArray(tex_min, d_minarr, minmaxDesc);
   tex_max.normalized = true;
   tex_max.filterMode = cudaFilterModePoint;
   tex_max.addressMode[0] = cudaAddressModeClamp;
   tex_max.addressMode[1] = cudaAddressModeClamp;
   tex_max.addressMode[2] = cudaAddressModeClamp;
   cudaBindTextureToArray(tex_max, d_maxarr, minmaxDesc);

   static_cast<vvCudaImg*>(Base::intImg)->map();

   int shmsize = Base::intImg->width*imagePrecision/8*4;
#ifdef SHMLOAD
   shmsize += Base::vd->vox[Base::principal]*Base::vd->getBPV()*sizeof(Scalar);
#endif
   if(Base::preIntegration)
   {
       shmsize += Base::intImg->width*sizeof(float);
   }
#ifdef NOSHMEM
   if(Base::sliceInterpol)
       shmsize = 0;
#endif

   uchar4 *d_img = static_cast<vvCudaImg*>(Base::intImg)->getDImg();
   dim3 grid((Base::intImg->width+Patch.x-1)/Patch.x, (toY-fromY+Patch.y-1)/Patch.y);
   dim3 block = Patch;
   clearImage <<<grid, Patch>>>(d_img, Base::intImg->width, Base::intImg->height, fromY, toY);

   if(CompositionFunction compose = selectCompositionWithPrecision(this, sliceStep))
   {
#ifdef PATCHES
       if(Base::sliceInterpol)
       {
           grid = dim3((to.x-from.x+Patch.x-1)/Patch.x, (to.y-from.y+Patch.y-1)/Patch.y);
       }
       else
#endif
       {
           grid = dim3(to.y-from.y);
           block = dim3(nthreads);
       }

       // do the computation on the device
       for(int i=lastSlice; i*sliceStep>firstSlice*sliceStep; i-=sliceStep*MaxCompositeSlices)
       {
           cudaThreadSynchronize();
#ifdef PITCHED
           compose <<<grid, block, shmsize>>>(
                   d_img, Base::intImg->width, Base::intImg->height,
                   d_voxptr[Base::principal],
                   sliceStep*max(sliceStep*i-MaxCompositeSlices, sliceStep*firstSlice), i,
                   from, to, nslice, s);
#else
           compose <<<grid, block, shmsize>>>(
                   d_img, Base::intImg->width, Base::intImg->height,
                   (Scalar *)(d_voxels+sizeof(Scalar)*Base::vd->getBPV()*Base::principal*(Base::vd->vox[0]*Base::vd->vox[1]*Base::vd->vox[2])),
                   sliceStep*max(sliceStep*i-MaxCompositeSlices, sliceStep*firstSlice), i,
                   from, to, nslice, s);
#endif
       }
   }

#ifdef VOLTEX3D
   cudaUnbindTexture(tex_raw);
#endif

   // copy back or unmap for using as PBO
   static_cast<vvCudaImg*>(Base::intImg)->unmap();
}

template<class Base>
void vvCudaSW<Base>::setParameter(typename Base::ParameterType param, float val, char *cval)
{
    vvDebugMsg::msg(3, "vvCudaSW::setParameter()");
    switch(param)
    {
        case Base::VV_IMG_PRECISION:
            if(val == 8)
                imagePrecision = 8;
            else
                imagePrecision = 32;
            break;
        case Base::VV_TERMINATEEARLY:
            earlyRayTerm = (val != 0.f);
            break;
        case Base::VV_INTERSLICEINT:
            interSliceInt = (val != 0.f);
            break;
        default:
            Base::setParameter(param, val, cval);
            break;
    }
}

template<class Base>
float vvCudaSW<Base>::getParameter(typename Base::ParameterType param, char *cval) const
{
    vvDebugMsg::msg(3, "vvCudaSW::getParameter()");
    switch(param)
    {
        case Base::VV_IMG_PRECISION:
            return imagePrecision;
        case Base::VV_TERMINATEEARLY:
            return (earlyRayTerm ? 1.f : 0.f);
        case Base::VV_INTERSLICEINT:
            return (interSliceInt ? 1.f : 0.f);
        default:
            return Base::getParameter(param, cval);
    }
}

//----------------------------------------------------------------------------
/** Set rendering quality.
  When quality changes, the intermediate image must be resized and the shear
  matrix has to be recomputed.
  @see vvRenderer#setQuality
*/
template<>
void vvCudaSW<vvSoftPar>::setQuality(float q)
{
   typedef vvSoftPar Base;

   vvDebugMsg::msg(3, "vvCudaSW::setQuality()", q);

#ifdef VV_XVID
   q = 1.0f;
#endif

   Base::_renderState._quality = q;

   if(!Base::sliceInterpol)
       q = 1.f;

   quality = q;

   // edge size of intermediate image [pixels]
   int intImgSize = (int)((2.0f * q) * ts_max(Base::vd->vox[0], Base::vd->vox[1], Base::vd->vox[2]));
   if (intImgSize<1)
   {
      intImgSize = 1;
      quality = 1.0f / (2.0f * ts_max(Base::vd->vox[0], Base::vd->vox[1], Base::vd->vox[2]));
   }

   intImgSize = ts_clamp(intImgSize, 16, 4096);
   intImgSize = vvToolshed::getTextureSize(intImgSize);

   Base::intImg->setSize(intImgSize, intImgSize);
   vvSoftPar::findShearMatrix();
   vvDebugMsg::msg(3, "Intermediate image edge length: ", intImgSize);
}

template<>
void vvCudaSW<vvSoftPer>::setQuality(float q)
{
   vvDebugMsg::msg(3, "vvCudaSW::setQuality()", q);
   vvSoftPer::setQuality(q);
}

vvCudaPer::vvCudaPer(vvVolDesc *vd, vvRenderState rs)
: vvCudaSW<vvSoftPer>(vd, rs)
{
}

vvCudaPar::vvCudaPar(vvVolDesc *vd, vvRenderState rs)
: vvCudaSW<vvSoftPar>(vd, rs)
{
}
//============================================================================
// End of File
//============================================================================
