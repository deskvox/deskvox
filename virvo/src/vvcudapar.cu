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

texture<uchar4, 1, cudaReadModeNormalizedFloat> tex_tf;


#define SM20
#ifdef SM20
#define MUL24(a,b) ((a)*(b))
#else
#define MUL24(a,b) __umul24((a),(b))
#endif


//----------------------------------------------------------------------------
// device code (CUDA)
//----------------------------------------------------------------------------

template<int BPV, int sliceStep, int principal>
__global__ void compositeSlicesNearest(
      uchar4 * __restrict__ img, int width, int height,
      const uchar * __restrict__ voxels,
      int firstSlice, int lastSlice,
      int from, int to)
{
    const int line = blockIdx.x+from;
    if (line >= to)
        return;

    // initialise intermediate image line
    __shared__ extern uchar4 imgLine[];
    for (int ix=threadIdx.x; ix<width; ix+=blockDim.x)
    {
        imgLine[ix] = make_uchar4(0,0,0,0);
    }

    // composite slices for this image line
    for (int slice=firstSlice; slice!=lastSlice; slice += sliceStep)
    {
        // compute upper left image corner
        const int iPosX = int(c_start[slice].x+0.5f);
        const int iPosY = int(c_start[slice].y+0.5f);

        if(line < iPosY)
            continue;
        if(line >= iPosY+c_vox[principal+1])
            continue;

        // the voxel row of the current slice corresponding to this image line
        const uchar *voxLine = voxels +BPV * (MUL24(slice+1,MUL24(c_vox[principal+1],c_vox[principal+0]))
                + MUL24((iPosY-line-1),c_vox[principal+0]));

        // Traverse intermediate image pixels which correspond to the current slice.
        // 1 is subtracted from each loop counter to remain inside of the volume boundaries:
        for (int ix=threadIdx.x; ix<c_vox[principal+0]+iPosX; ix+=blockDim.x)
        {
            if(ix<iPosX)
                continue;
            // fetch scalar voxel value
            const uchar *v = voxLine + BPV * (ix-iPosX);
            // pointer to destination pixel
            uchar4 *pix = imgLine + ix;
            // apply transfer function
            const float4 c = tex1Dfetch(tex_tf, *v);
            uchar4 d = *pix;

            // blend
            const float w = (255-d.w)*c.w;
            d.x += w*c.x;
            d.y += w*c.y;
            d.z += w*c.z;
            d.w += w;

            // store into shmem
            *pix = d;
        }
    }

    // copy line to intermediate image
    for (int ix=threadIdx.x; ix<width; ix+=blockDim.x)
    {
        uchar4 *dest = img + MUL24(line,width)+ix;
        *dest = imgLine[ix];
    }
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
   }

   wViewDir.set(0.0f, 0.0f, 1.0f);

   bool ok = true;
   // alloc memory for voxel arrays (for each principal viewing direction)
   vvCuda::checkError(&ok, cudaMalloc(&d_voxels, vd->vox[0]*vd->vox[1]*vd->vox[2]*3), "cudaMalloc vox");
   for (int i=0; i<3; ++i)
   {
       if (!vvCuda::checkError(&ok, cudaMemcpy(d_voxels+i*vd->vox[0]*vd->vox[1]*vd->vox[2],
                   raw[i], vd->getFrameBytes(), cudaMemcpyHostToDevice), "cudaMemcpy vox"))
          break;
   }

   // transfer function is stored as a texture
   vvCuda::checkError(&ok, cudaMalloc(&d_tf, 4096*4), "cudaMalloc tf");
   vvCuda::checkError(&ok, cudaBindTexture(NULL, tex_tf, d_tf, 4096), "bind tf tex");

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

   if (warpMode==CUDATEXTURE)
      cudaGraphicsUnregisterResource(intImgRes);
   else if (mappedImage)
       cudaFreeHost(h_img);
   else
       cudaFree(d_img);

   cudaUnbindTexture(tex_tf);

   cudaFree(d_tf);
   cudaFree(d_voxels);
}

void vvCudaPar::updateTransferFunction()
{
   vvDebugMsg::msg(2, "vvCudaPar::updateTransferFunction()");

   vvSoftPar::updateTransferFunction();
   vvCuda::checkError(NULL, cudaMemcpy(d_tf, rgbaConv, sizeof(rgbaConv), cudaMemcpyHostToDevice), "cudaMemcpy tf");
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

   // do the computation on the device
#define compositeNearest(sliceStep, principal) \
   compositeSlicesNearest<1, sliceStep, principal> <<<to-from, 128, intImg->width*vvSoftImg::PIXEL_SIZE>>>( \
         d_img, intImg->width, intImg->height, \
         d_voxels+vd->getBPV()*principal*(vd->vox[0]*vd->vox[1]*vd->vox[2]), \
         firstSlice, lastSlice, \
         from, to)

   if (principal==0)
   {
       if(sliceStep == 1)
           compositeNearest(1, 0);
       else
           compositeNearest(-1, 0);
   }
   else if (principal==1)
   {
       if (sliceStep == 1)
           compositeNearest(1, 1);
       else
           compositeNearest(-1, 1);
   }
   else
   {
       if (sliceStep == 1)
           compositeNearest(1, 2);
       else
           compositeNearest(-1, 2);
   }

   // copy back or unmap for using as PBO
   ok = vvCuda::checkError(&ok, cudaGetLastError(), "start kernel");
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
}
//============================================================================
// End of File
//============================================================================
