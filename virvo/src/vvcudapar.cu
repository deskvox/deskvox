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
#include "vvcudapar.h"

const int MAX_SLICES = 1600;

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


const int MaxCompositeSlices = MAX_SLICES;

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

#ifdef SHMCLASS
#define SHMLOAD
texture<uchar4, 1, cudaReadModeElementType> tex_tf;
#else
texture<uchar4, 1, cudaReadModeNormalizedFloat> tex_tf;
#endif

#ifdef VOLTEX3D
texture<uchar4, 2, cudaReadModeNormalizedFloat> tex_preint;
#ifdef FLOATDATA
texture<float, 3, cudaReadModeElementType> tex_raw;
#else
texture<uchar, 3, cudaReadModeNormalizedFloat> tex_raw;
#endif
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
      int from, int to);

//----------------------------------------------------------------------------
// device code (CUDA)
//----------------------------------------------------------------------------

__global__ void clearImage(uchar4 * __restrict__ img, int width, int height,
      int from, int to)
{
    const int line = blockIdx.x+from;
    if (line >= to)
        return;

    for (int ix=threadIdx.x; ix<width; ix+=blockDim.x)
    {
        uchar4 *dest = img + line*width+ix;
        *dest = make_uchar4(0, 0, 0, 0);
    }
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
    //src.w = 255 - src.w;
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
    return (pix.w < 0.003f);
}

__device__ bool isOpaque(uchar4 pix)
{
    return (pix.w < 1);
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
      int from, int to)
{
#ifndef VOLTEX3D
#ifdef PITCHED
    const Scalar *voxels = (Scalar *)pvoxels.ptr;
    const int pitch = pvoxels.pitch;
#else
    const int pitch = c_vox[principal] * BPV * sizeof(Scalar);
#endif
#endif
    const int line = blockIdx.x+from;
    if (line >= to)
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

    for (int ix=threadIdx.x; ix<width; ix+=blockDim.x)
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
        for (int ix=threadIdx.x; ix<c_vox[principal+0]+iPosX; ix+=blockDim.x)
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
    for (int ix=threadIdx.x; ix<width; ix+=blockDim.x)
    {
        uchar4 *dest = img + line*width+ix;
        blend(dest, imgLine[ix]);
    }
#endif
}


#ifdef VOLTEX3D
template<typename Scalar, int BPV, typename Pixel, int sliceStep, int principal, bool earlyRayTerm>
__global__ void compositeSlicesBilinear(
      uchar4 * __restrict__ img, int width, int height,
#ifdef PITCHED
      const cudaPitchedPtr pvoxels,
#else
      const Scalar * __restrict__ voxels,
#endif
      int firstSlice, int lastSlice,
      int from, int to)
{
    // this block's line from the intermediate image
    const int line = blockIdx.x+from;
    if (line >= to)
        return;

    // initialise intermediate image line
    extern __shared__ char smem[];
    // store intermediate image line in shmem
    Pixel *imgLine = (Pixel *)smem;

    // clear image line
    for (int ix=threadIdx.x; ix<width; ix+=blockDim.x)
    {
        initPixel(&imgLine[ix]);
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
        for (int ix=threadIdx.x; ix<endX; ix+=blockDim.x)
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

    // copy line to intermediate image
    for (int ix=threadIdx.x; ix<width; ix+=blockDim.x)
    {
        uchar4 *dest = img + line*width+ix;
        blend(dest, imgLine[ix]);
    }
}

template<typename Scalar, int BPV, typename Pixel, int sliceStep, int principal, bool earlyRayTerm>
__global__ void compositeSlicesPreIntegrated(
      uchar4 * __restrict__ img, int width, int height,
#ifdef PITCHED
      const cudaPitchedPtr pvoxels,
#else
      const Scalar * __restrict__ voxels,
#endif
      int firstSlice, int lastSlice,
      int from, int to)
{
    const int line = blockIdx.x+from;
    if (line >= to)
        return;

    // initialise intermediate image line
    extern __shared__ char smem[];
    Pixel *imgLine = (Pixel *)smem;
    Scalar *sf = (Scalar *)&imgLine[width];

    for (int ix=threadIdx.x; ix<width; ix+=blockDim.x)
    {
        initPixel(&imgLine[ix]);
        sf[ix] = 0.f;
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
        for (int ix=threadIdx.x; ix<endX; ix+=blockDim.x)
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

            const float sb = volume(vidx, line-iPosY, slice, principal);
            if(slice != firstSlice)
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
    }

    // copy line to intermediate image
    for (int ix=threadIdx.x; ix<width; ix+=blockDim.x)
    {
        uchar4 *dest = img + line*width+ix;
        blend(dest, imgLine[ix]);
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

   bool ok = true;
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
       parms.srcPtr = make_cudaPitchedPtr(raw[2], sizeof(Scalar)*vd->vox[0], vd->vox[0], vd->vox[1]);
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
   vvCuda::checkError(&ok, cudaMalloc(&d_tf, 4096*4), "cudaMalloc tf");
   vvCuda::checkError(&ok, cudaBindTexture(NULL, tex_tf, d_tf, 4096), "bind tf tex");

   // pre-integration table
   cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
   vvCuda::checkError(&ok, cudaMallocArray(&d_preint, &desc, Base::PRE_INT_TABLE_SIZE, Base::PRE_INT_TABLE_SIZE), "cudaMalloc preint");
   tex_preint.normalized = true;
   tex_preint.filterMode = Base::bilinLookup ? cudaFilterModeLinear : cudaFilterModePoint;
   tex_preint.addressMode[0] = cudaAddressModeClamp;
   tex_preint.addressMode[1] = cudaAddressModeClamp;
   vvCuda::checkError(&ok, cudaBindTextureToArray(tex_preint, d_preint, desc), "bind preint tex");

   // copy volume size (in voxels)
   int h_vox[5];
   for (int i=0; i<5; ++i)
       h_vox[i] = vd->vox[(i+1)%3];
   vvCuda::checkError(&ok, cudaMemcpyToSymbol(c_vox, h_vox, sizeof(int)*5), "cudaMemcpy vox");

   updateTransferFunction();
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
void vvCudaSW<Base>::updateTransferFunction()
{
   vvDebugMsg::msg(2, "vvCudaSW::updateTransferFunction()");

   Base::updateTransferFunction();

   vvCuda::checkError(NULL, cudaMemcpy(d_tf, Base::rgbaConv, sizeof(Base::rgbaConv), cudaMemcpyHostToDevice), "cudaMemcpy tf");
   if(Base::preIntegration)
   {
       vvCuda::checkError(NULL, cudaMemcpyToArray(d_preint, 0, 0, &Base::preIntTable[0][0][0],
                   Base::PRE_INT_TABLE_SIZE*Base::PRE_INT_TABLE_SIZE*4, cudaMemcpyHostToDevice), "cudaMemcpy preint");
   }
}

template<class Base, typename Pixel, int principal, int sliceStep, bool earlyRayTerm>
CompositionFunction selectComposition(vvCudaSW<Base> *rend)
{
#ifdef VOLTEX3D
    if(rend->getSliceInterpol() || rend->getRendererType() == vvRenderer::CUDAPER)
    {
        if(rend->getPreIntegration())
            return compositeSlicesPreIntegrated<Scalar, 1, Pixel, sliceStep, principal, earlyRayTerm>;
        else
            return compositeSlicesBilinear<Scalar, 1, Pixel, sliceStep, principal, earlyRayTerm>;
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
void vvCudaSW<Base>::compositeVolume(int from, int to)
{
   vvDebugMsg::msg(3, "vvCudaSW::compositeVolume(): ", from, to);

   // If stacking==true then draw front to back, else draw back to front:
   int firstSlice = (Base::stacking) ? 0 : (Base::len[2]-1);  // first slice to process
   int lastSlice  = (Base::stacking) ? (Base::len[2]-1) : 0;  // last slice to process
   int sliceStep  = (Base::stacking) ? 1 : -1;          // step size to get to next slice

   Base::earlyRayTermination = 0;

   if (from == -1)
       from = 0;
   if (to == -1)
       to = Base::intImg->height;

   // compute data for determining upper left image corner of each slice and copy it to device
   vvVector4 start, end;
   Base::findSlicePosition(firstSlice, &start, &end);
   vvVector4 sinc, einc;
   Base::findSlicePosition(firstSlice+sliceStep, &sinc, &einc);
   sinc.sub(&start);
   einc.sub(&end);
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

           h_stop[slice].x = min(Base::intImg->width-1,int(ceil(ex)));
           h_stop[slice].y = min(Base::intImg->height-1,int(ceil(ey)));

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

           ecur.add(&sinc);
       }
       else
#endif
       {
           h_start[slice].x = int(scur.e[0] / scur.e[3] + 0.5f);
           h_start[slice].y = int(scur.e[1] / scur.e[3] + 0.5f);
       }
       scur.add(&sinc);
   }

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

   static_cast<vvCudaImg*>(Base::intImg)->map();

   int shmsize = Base::intImg->width*imagePrecision/8*4;
#ifdef SHMLOAD
   shmsize += Base::vd->vox[Base::principal]*Base::vd->getBPV()*sizeof(Scalar);
#endif
   if(Base::preIntegration)
   {
       shmsize += Base::intImg->width*sizeof(Scalar);
   }

   uchar4 *d_img = static_cast<vvCudaImg*>(Base::intImg)->getDImg();
   clearImage <<<to-from, 128, shmsize>>>(d_img, Base::intImg->width, Base::intImg->height, from, to);

   if(CompositionFunction compose = selectCompositionWithPrecision(this, sliceStep))
   {
       // do the computation on the device
       for(int i=lastSlice; i*sliceStep>firstSlice*sliceStep; i-=sliceStep*MaxCompositeSlices)
       {
           cudaThreadSynchronize();
#ifdef PITCHED
           compose <<<to-from, 128, shmsize>>>(
                   d_img, Base::intImg->width, Base::intImg->height,
                   d_voxptr[principal],
                   sliceStep*max(sliceStep*i-MaxCompositeSlices, sliceStep*firstSlice), i,
                   from, to);
#else
           compose <<<to-from, 128, shmsize>>>(
                   d_img, Base::intImg->width, Base::intImg->height,
                   (Scalar *)(d_voxels+sizeof(Scalar)*Base::vd->getBPV()*Base::principal*(Base::vd->vox[0]*Base::vd->vox[1]*Base::vd->vox[2])),
                   sliceStep*max(sliceStep*i-MaxCompositeSlices, sliceStep*firstSlice), i,
                   from, to);
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
