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

#ifndef VV_CUDAPAR_H
#define VV_CUDAPAR_H

#include "vvrenderer.h"
#include "vvsoftimg.h"
#include "vvsoftpar.h"
#include "vvexport.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

struct cudaGraphicsResource;

/** Parallel implementation of parallel projection shear-warp algorithm using CUDA.

@author Martin Aumueller <aumueller@uni-koeln.de>
@see vvRenderer
@see vvSoftVR
@see vvSoftPar
@see vvSoftImg
*/
class VIRVOEXPORT vvCudaPar : public vvSoftPar
{
   private:
      cudaArray *d_voxarr[3];
      cudaPitchedPtr d_voxptr[3];
      uchar *d_voxels;
      uchar4 *d_img;
      uchar4 *d_tf;
      uchar *h_img;
      cudaGraphicsResource *intImgRes;              ///< CUDA resource mapped to PBO
      bool mappedImage;
      bool earlyRayTerm;
      int imagePrecision;

      float *fraw[3];

   protected:
      virtual void updateTransferFunction();

   public:
      vvCudaPar(vvVolDesc*, vvRenderState);
      virtual ~vvCudaPar();
      void compositeVolume(int = -1, int = -1);
      virtual void setParameter(ParameterType, float, char *);
      virtual float getParameter(ParameterType, char *) const;

      int getPrincipal() const { return principal; }
      bool getPreIntegration() const { return preIntegration; }
      bool getSliceInterpol() const { return sliceInterpol; }
      bool getEarlyRayTerm() const { return earlyRayTerm; }
      int getPrecision() const { return imagePrecision; }
};
#endif /* HAVE_CUDA */

#endif
//============================================================================
// End of File
//============================================================================
