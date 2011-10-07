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

#ifndef VV_CUDASW_H
#define VV_CUDASW_H

#include "vvrenderer.h"
#include "vvsoftpar.h"
#include "vvsoftper.h"
#include "vvexport.h"
#include "vvswitchrenderer.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

struct cudaGraphicsResource;

/** Parallel implementation of shear-warp algorithm using CUDA.
 Only vvSoftPar and vvSoftPer are supported for Base.

@author Martin Aumueller <aumueller@uni-koeln.de>
@see vvRenderer
@see vvSoftVR
@see vvSoftPar
@see vvSoftPer
@see vvSoftImg
*/
template<class Base>
class VIRVOEXPORT vvCudaSW : public Base
{
   private:
      cudaArray *d_voxarr[3]; ///< device storage for voxel data (3d texture)
      cudaArray *d_minarr; ///< device storage for min data (3d texture)
      cudaArray *d_maxarr; ///< device storage for max data (3d texture)
      cudaArray *d_oparr; ///< device storage for max data (3d texture)
      cudaArray *d_minmaxTable; ///< device storage for min/max table (2d texture)
      cudaPitchedPtr d_voxptr[3]; ///< device storage for voxel data (pitched)
      uchar *d_voxels; ///< device storage for voxel data (linear array)
      uchar4 *d_tf; ///< device storage for transfer function
      cudaArray *d_preint; ///< device storage for pre-integration table
      bool earlyRayTerm; ///< true = approximate early ray termination
      bool emptySpaceSkip; ///< true = try to skip empty voxels
      bool interSliceInt; ///< true = tri-linear inter-slice interpolation
      int imagePrecision; ///< number of bits per pixel component used during compositing
      float oldLutDist;
      uchar *h_minarr, *h_maxarr;
      uchar *h_minmaxTable;

      float *fraw[3]; ///< pointer to voxel data converted to floating point

      bool initTF();
      void freeTF();
      bool initPreInt();
      void freePreInt();
      bool initMinMax();
      void freeMinMax();
      bool initFloatData();
      void freeFloatData();
      bool initVolData();
      void freeVolData();
      bool initVolDataPitched();
      void freeVolDataPitched();
      bool initVolTex();
      void freeVolTex();

      bool compositeNearest(int fromY, int toY, int firstSlice, int lastSlice, int sliceStep);
      bool compositeBilinear(int fromY, int toY, int firstSlice, int lastSlice, int sliceStep);
      bool compositeRaycast(int fromY, int toY, int firstSlice, int lastSlice, int sliceStep);

      bool updateOpacityMap();

   protected:
      virtual void updateLUT(float dist);
      virtual void findAxisRepresentations();
      using Base::factorViewMatrix;

   public:
      vvCudaSW(vvVolDesc*, vvRenderState);
      virtual ~vvCudaSW();
      void compositeVolume(int = -1, int = -1);
      virtual void setParameter(typename Base::ParameterType, float newValue);
      virtual float getParameter(typename Base::ParameterType) const;
      virtual void setQuality(float q);

      int getPrincipal() const { return Base::principal; }
      bool getPreIntegration() const { return Base::preIntegration; }
      bool getSliceInterpol() const { return Base::sliceInterpol; }
      bool getInterSliceInterpol() const { return interSliceInt; }
      bool getEarlyRayTerm() const { return earlyRayTerm; }
      bool getEmptySpaceLeaping() const { return emptySpaceSkip; }
      int getPrecision() const { return imagePrecision; }
};

/** Parallel implementation of perspective shear-warp algorithm using CUDA.

@author Martin Aumueller <aumueller@uni-koeln.de>
@see vvSoftPer
@see vvCudaSW
*/
class VIRVOEXPORT vvCudaPer: public vvCudaSW<vvSoftPer>
{
    public:
    vvCudaPer(vvVolDesc *vd, vvRenderState rs);
};

/** Parallel implementation of parallel projection shear-warp algorithm using CUDA.

@author Martin Aumueller <aumueller@uni-koeln.de>
@see vvSoftPar
@see vvCudaSW
*/
class VIRVOEXPORT vvCudaPar: public vvCudaSW<vvSoftPar>
{
    public:
    vvCudaPar(vvVolDesc *vd, vvRenderState rs);
};

class VIRVOEXPORT vvCudaShearWarp: public vvSwitchRenderer<vvCudaPar, vvCudaPer>
{
public:
  vvCudaShearWarp(vvVolDesc *vd, vvRenderState rs);
};
#endif /* HAVE_CUDA */
#endif
//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
