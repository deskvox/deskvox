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
#include "vvcudapar.h"

const int MAX_SLICES = 512;

__constant__ int   c_vox[5];
__constant__ float2 c_start[MAX_SLICES];
static float2 h_start[MAX_SLICES];

const int MaxCompositeSlices = 1024;

//#define SHMCLASS
//#define NOOP
//#define NOLOAD
//#define NODISPLAY
//#define SHMLOAD
#define VOL3TEX3D
//#define PITCHED
#define FLOATDATA
//#define CONSTLOAD
//#define THREADPERVOXEL
//#define CONSTDATA
//#define FLOATIMG
#define EARLYRAYTERM

const int Repetitions = 1;

#ifdef VOL3TEX3D
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

#ifdef VOL3TEX3D
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

template<typename Scalar, int BPV, int sliceStep, int principal>
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
#ifdef PITCHED
    const Scalar *voxels = (Scalar *)pvoxels.ptr;
    const int pitch = pvoxels.pitch;
#else
    const int pitch = c_vox[principal] * BPV * sizeof(Scalar);
#endif
    const int line = blockIdx.x+from;
    if (line >= to)
        return;

    // initialise intermediate image line
    extern __shared__ char smem[];
#ifdef FLOATIMG
    float4 *imgLine = (float4 *)smem;
    const int pixsize = sizeof(float4);
#else
    uchar4 *imgLine = (uchar4 *)smem;
    const int pixsize = sizeof(uchar4);
#endif
#ifdef SHMLOAD
#ifdef SHMCLASS
    uchar4 *voxel = (uchar4 *)(smem+width*pixsize);
#else
    Scalar *voxel = (Scalar *)(smem+width*pixsize);
#endif
#endif

    for (int ix=threadIdx.x; ix<width; ix+=blockDim.x)
    {
#ifdef FLOATIMG
        imgLine[ix] = make_float4(0.f,0.f,0.f,1.f);
#else
        imgLine[ix] = make_uchar4(0,0,0,255);
#endif
    }

    for(int i=0; i<Repetitions; ++i)
    {

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
        const int iPosY = float2int(c_start[slice].y+0.5f);

        if(line < iPosY)
            continue;
        if(line >= iPosY+c_vox[principal+1])
            continue;

        const int iPosX = float2int(c_start[slice].x+0.5f);
#endif

        // the voxel row of the current slice corresponding to this image line
#ifndef NOLOAD
#ifndef CONSTLOAD
        const Scalar *voxLine = (Scalar *)(((uchar *)voxels) + pitch * ((slice+1)*c_vox[principal+1] + (iPosY-line-1)));
#endif

#ifdef SHMLOAD
        for (int ix=threadIdx.x; ix<c_vox[principal+0]; ix+=blockDim.x)
        {
#ifdef SHMCLASS
#ifdef FLOATDATA
            voxel[ix] = tex1Dfetch(tex_tf, voxLine[ix]*255.f);
#else
            voxel[ix] = tex1Dfetch(tex_tf, voxLine[ix]);
#endif
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
#ifdef FLOATIMG
            float4 *pix = imgLine + iidx;
            float4 d = *pix;
#ifdef EARLYRAYTERM
            if(d.w < 0.001f)
                continue;
#endif
#else
            uchar4 *pix = imgLine + iidx;
            uchar4 d = *pix;
#ifdef EARLYRAYTERM
            if(d.w < 1)
                continue;
#endif
#endif

#ifdef VOL3TEX3D
            const float v = tex3D(tex_raw, float(vidx), float(c_vox[principal+1]+iPosY-line-1), float(slice));
            const float4 c = tex1Dfetch(tex_tf, v*255.f);
#else
#ifdef CONSTDATA
            const float4 c = tex1Dfetch(tex_tf, ix);
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
#ifdef FLOATDATA
            const float4 c = tex1Dfetch(tex_tf, *v*255.f);
#else
            const float4 c = tex1Dfetch(tex_tf, *v);
#endif
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
    }

#ifndef NODISPLAY
    // copy line to intermediate image
    for (int ix=threadIdx.x; ix<width; ix+=blockDim.x)
    {
        uchar4 *dest = img + line*width+ix;
        uchar4 c = *dest;
#ifdef FLOATIMG
        *dest = make_uchar4(imgLine[ix].x * 255.f + c.x*imgLine[ix].w,
                imgLine[ix].y * 255.f + c.y*imgLine[ix].w,
                imgLine[ix].z * 255.f + c.z*imgLine[ix].w,
                (1.f-imgLine[ix].w) * (255-c.w) + c.w);
#else
        const float w= imgLine[ix].w/255.f;
        imgLine[ix].w = 255 - imgLine[ix].w;
        *dest = make_uchar4(imgLine[ix].x + c.x*w,
                imgLine[ix].y + c.y*w,
                imgLine[ix].z + c.z*w,
                (1.f-w) * (255-c.w) + c.w);
#endif
    }
#endif
}


//----------------------------------------------------------------------------
// host code
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
/** Constructor.
  @param vd volume description of volume to display
  @see vvRenderer
*/
vvCudaPar::vvCudaPar(vvVolDesc* vd, vvRenderState rs) : vvSoftPar(vd, rs)
{
   vvDebugMsg::msg(1, "vvCudaPar::vvCudaPar()");

   rendererType = CUDAPAR;
   mappedImage = false;
   if (warpMode==TEXTURE)
   {
       // we need a power-of-2 image size for glTexImage2D
       int imgSize = vvToolshed::getTextureSize(2 * ts_max(vd->vox[0], vd->vox[1], vd->vox[2]));

#ifndef NODISPLAY
       if (vvCuda::initGlInterop())
       {
           vvDebugMsg::msg(1, "using CUDA/GL interop");
           // avoid image copy from GPU to CPU and back
           setWarpMode(CUDATEXTURE);
           intImg->setSize(imgSize, imgSize, NULL, true);
       }
       else
       {
         vvDebugMsg::msg(1, "can't use CUDA/GL interop");
         intImg->setSize(imgSize, imgSize);
       }
#endif
   }

   wViewDir.set(0.0f, 0.0f, 1.0f);

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
#if defined(PITCHED) || defined(VOL3TEX3D)
   for (int i=0; i<3; ++i)
   {
#ifdef PITCHED
       cudaExtent extent = make_cudaExtent(vd->vox[(i+1)%3]*sizeof(Scalar), vd->vox[(i+2)%3], vd->vox[(i+3)%3]);
       if(!vvCuda::checkError(&ok, cudaMalloc3D(&d_voxptr[i], extent), "cudaMalloc3D vox"))
           break;
#else
       cudaExtent extent = make_cudaExtent(vd->vox[(i+1)%3], vd->vox[(i+2)%3], vd->vox[(i+3)%3]);
       cudaChannelFormatDesc desc = cudaCreateChannelDesc<Scalar>();
       if(!vvCuda::checkError(&ok, cudaMalloc3DArray(&d_voxarr[i], &desc, extent, 0), "cudaMalloc3DArray vox"))
           break;
#endif
       cudaMemcpy3DParms parms = {0};
#ifdef FLOATDATA
       parms.srcPtr = make_cudaPitchedPtr(fraw[i], sizeof(Scalar)*vd->vox[(i+1)%3], vd->vox[(i+1)%3], vd->vox[(i+2)%3]);
#else
       parms.srcPtr = make_cudaPitchedPtr(raw[i], sizeof(Scalar)*vd->vox[(i+1)%3], vd->vox[(i+1)%3], vd->vox[(i+2)%3]);
#endif

#ifdef PITCHED
       parms.dstPtr = d_voxptr[i];
       parms.extent = make_cudaExtent(vd->vox[(i+1)%3]*sizeof(Scalar), vd->vox[(i+2)%3], vd->vox[(i+3)%3]);
#else
       parms.dstArray = d_voxarr[i];
       parms.extent = make_cudaExtent(vd->vox[(i+1)%3], vd->vox[(i+2)%3], vd->vox[(i+3)%3]);
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

#ifndef NODISPLAY
   // allocate output image (intermediate image)
   if (warpMode==CUDATEXTURE)
   {
       vvCuda::checkError(&ok, cudaGraphicsGLRegisterBuffer(&intImgRes, intImg->getPboName(), cudaGraphicsMapFlagsWriteDiscard), "map PBO to CUDA");
   }
   else if (mappedImage)
   {
       vvCuda::checkError(&ok, cudaHostAlloc(&h_img, intImg->width*intImg->height*vvSoftImg::PIXEL_SIZE, cudaHostAllocMapped), "img alloc");;
       intImg->setSize(intImg->width, intImg->height, h_img, false);
       vvCuda::checkError(&ok, cudaHostGetDevicePointer(&d_img, h_img, 0), "get dev ptr img");
   }
   else
#endif
   {
       vvCuda::checkError(&ok, cudaMalloc(&d_img, intImg->width*intImg->height*vvSoftImg::PIXEL_SIZE), "cudaMalloc img");
   }

   // copy volume size (in voxels)
   int h_vox[5];
   for (int i=0; i<5; ++i)
       h_vox[i] = vd->vox[(i+1)%3];
   vvCuda::checkError(&ok, cudaMemcpyToSymbol(c_vox, h_vox, sizeof(int)*5), "cudaMemcpy vox");

   updateTransferFunction();
}


//----------------------------------------------------------------------------
/// Destructor.
vvCudaPar::~vvCudaPar()
{
   vvDebugMsg::msg(1, "vvCudaPar::~vvCudaPar()");

#ifdef FLOATDATA
   for(int i=0; i<3; ++i)
     delete[] fraw[i];
#endif

#ifndef NODISPLAY
   if (warpMode==CUDATEXTURE)
      cudaGraphicsUnregisterResource(intImgRes);
   else if (mappedImage)
       cudaFreeHost(h_img);
   else
#endif
       cudaFree(d_img);

   cudaUnbindTexture(tex_tf);

   cudaFree(d_tf);
#ifdef VOL3TEX3D
   for(int i=0; i<3; ++i)
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

void vvCudaPar::updateTransferFunction()
{
   vvDebugMsg::msg(2, "vvCudaPar::updateTransferFunction()");

   vvSoftPar::updateTransferFunction();
   vvCuda::checkError(NULL, cudaMemcpy(d_tf, rgbaConv, sizeof(rgbaConv), cudaMemcpyHostToDevice), "cudaMemcpy tf");
}

template<int principal, int sliceStep>
CompositionFunction selectComposition()
{
   return compositeSlicesNearest<Scalar, 1, sliceStep, principal>;
}

template<int principal>
CompositionFunction selectCompositionWithSliceStep(int sliceStep)
{
    switch(sliceStep)
    {
        case 1:
            return selectComposition<principal,1>();
        case -1:
            return selectComposition<principal,-1>();
        default:
            assert("slice step out of range" == NULL);
    }

    return NULL;
}

CompositionFunction selectCompositionWithPrincipalAndSliceStep(int principal, int sliceStep)
{
    switch(principal)
    {
        case 0:
            return selectCompositionWithSliceStep<0>(sliceStep);
        case 1:
            return selectCompositionWithSliceStep<1>(sliceStep);
        case 2:
            return selectCompositionWithSliceStep<2>(sliceStep);
        default:
            assert("principal axis out of range" == NULL);

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
void vvCudaPar::compositeVolume(int from, int to)
{
   vvDebugMsg::msg(3, "vvCudaPar::compositeVolume(): ", from, to);

   // If stacking==true then draw front to back, else draw back to front:
   int firstSlice = (stacking) ? 0 : (len[2]-1);  // first slice to process
   int lastSlice  = (stacking) ? (len[2]-1) : 0;  // last slice to process
   int sliceStep  = (stacking) ? 1 : -1;          // step size to get to next slice

   earlyRayTermination = 0;

   if (from == -1)
       from = 0;
   if (to == -1)
       to = intImg->height;

   // compute data for determining upper left image corner of each slice and copy it to device
   vvVector4 start, inc;
   findSlicePosition(firstSlice, &start, NULL);
   findSlicePosition(firstSlice+sliceStep, &inc, NULL);
   inc.sub(&start);
   vvVector4 cur = start;
   for(int slice=firstSlice; slice != lastSlice; slice += sliceStep)
   {
       h_start[slice].x = cur.e[0] / cur.e[3];
       h_start[slice].y = cur.e[1] / cur.e[3];
       cur.add(&inc);
   }

   bool ok = true;
   vvCuda::checkError(&ok, cudaMemcpyToSymbol(c_start, h_start, sizeof(h_start)), "cudaMemcpy start");

#ifdef VOL3TEX3D
   tex_raw.normalized = false;
   tex_raw.filterMode = cudaFilterModePoint;
   tex_raw.addressMode[0] = cudaAddressModeClamp;
   tex_raw.addressMode[1] = cudaAddressModeClamp;
   tex_raw.addressMode[2] = cudaAddressModeClamp;
   cudaChannelFormatDesc desc = cudaCreateChannelDesc<Scalar>();
   cudaBindTextureToArray(tex_raw, d_voxarr[principal], desc);
#endif

#ifndef NODISPLAY
   // prepare intermediate image
   if (warpMode==CUDATEXTURE)
   {
       vvCuda::checkError(&ok, cudaGraphicsMapResources(1, &intImgRes, NULL), "map CUDA resource");
       size_t size;
       vvCuda::checkError(&ok, cudaGraphicsResourceGetMappedPointer((void**)&d_img, &size, intImgRes), "get PBO mapping");
       assert(size == intImg->width*intImg->height*vvSoftImg::PIXEL_SIZE);
   }
   else
   {
       intImg->clear();
   }
#endif

#ifdef FLOATIMG
   int shmsize = intImg->width*sizeof(float4);
#else
   int shmsize = intImg->width*vvSoftImg::PIXEL_SIZE;
#endif
#ifdef SHMLOAD
   shmsize += vd->vox[principal]*vd->getBPV()*sizeof(Scalar);
#endif

   clearImage <<<to-from, 128, shmsize>>>(d_img, intImg->width, intImg->height, from, to);

   CompositionFunction compose = selectCompositionWithPrincipalAndSliceStep(principal, sliceStep);
   // do the computation on the device
   for(int i=lastSlice; i*sliceStep>firstSlice*sliceStep; i-=sliceStep*MaxCompositeSlices)
   {
       cudaThreadSynchronize();
#ifdef PITCHED
       compose <<<to-from, 128, shmsize>>>(
               d_img, intImg->width, intImg->height,
               d_voxptr[principal],
               sliceStep*max(sliceStep*i-MaxCompositeSlices, sliceStep*firstSlice), i,
               from, to);
#else
       compose <<<to-from, 128, shmsize>>>(
               d_img, intImg->width, intImg->height,
               (Scalar *)(d_voxels+sizeof(Scalar)*vd->getBPV()*principal*(vd->vox[0]*vd->vox[1]*vd->vox[2])),
               sliceStep*max(sliceStep*i-MaxCompositeSlices, sliceStep*firstSlice), i,
               from, to);
#endif
   }

#ifdef VOL3TEX3D
   cudaUnbindTexture(tex_raw);
#endif

   // copy back or unmap for using as PBO
   ok = vvCuda::checkError(&ok, cudaGetLastError(), "start kernel");
#ifndef NODISPLAY
   if (warpMode==CUDATEXTURE)
   {
       vvCuda::checkError(&ok, cudaGraphicsUnmapResources(1, &intImgRes, NULL), "unmap CUDA resource");
   }
   else if (mappedImage)
   {
       cudaThreadSynchronize();
   }
   else
   {
       cudaMemcpy(intImg->data, d_img, intImg->width*intImg->height*vvSoftImg::PIXEL_SIZE, cudaMemcpyDeviceToHost);
       ok = vvCuda::checkError(&ok, cudaGetLastError(), "cpy to host");
   }
#endif
}
//============================================================================
// End of File
//============================================================================
