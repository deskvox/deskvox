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

#ifndef _VV_RAYREND_H_
#define _VV_RAYREND_H_

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_CUDA

#include "vvexport.h"
#include "vvsoftvr.h"
#include "vvcuda.h"
#include <vector>
#include <cuda.h>
#include <cuda_gl_interop.h>


class vvVolDesc;

/** Ray casting renderer. Based on the volume
  rendering implementation from the NVIDIA CUDA SDK,
  as of November 25th, 2010 could be downloaded from
  the following location:
  http://developer.download.nvidia.com/compute/cuda/sdk/website/samples.html
 */
class VIRVOEXPORT vvRayRend : public vvSoftVR
{
public:
  vvRayRend(vvVolDesc* vd, vvRenderState renderState);
  ~vvRayRend();

  int getLUTSize() const;
  void updateTransferFunction();
  void compositeVolume(int w = -1, int h = -1);
  virtual void setParameter(ParameterType param, const vvParam& newValue);
  virtual vvParam getParameter(ParameterType param) const;

  bool getEarlyRayTermination() const;
  bool getIllumination() const;
  bool getInterpolation() const;
  bool getOpacityCorrection() const;

  uchar4* getDeviceImg() const;
  void* getDeviceDepth() const;
  void* getDeviceUncertainty() const;
  void setDepthRange(float min, float max);
  const float* getDepthRange() const;

private:
  cudaChannelFormatDesc _channelDesc;
  std::vector<cudaArray*> d_volumeArrays;
  cudaArray* d_transferFuncArray;

  float* _rgbaTF;

  bool _earlyRayTermination;        ///< Terminate ray marching when enough alpha was gathered
  bool _illumination;               ///< Use local illumination
  bool _interpolation;              ///< interpolation mode: true=linear interpolation (default), false=nearest neighbor
  bool _opacityCorrection;          ///< true = opacity correction on
  bool _volumeCopyToGpuOk;          ///< must be true for memCopy to be run
  bool _twoPassIbr;                 ///< Perform an alpha-gathering pass before the actual render pass

  int _depthPrecision;              ///< number of bits in depth buffer for image based rendering
  int _uncertaintyPrecision;        ///< number of bits in uncertainty array for image based rendering
  void* d_depth;
  void* d_uncertainty;
  float _depthRange[2];

  void initVolumeTexture();
  void factorViewMatrix();
  void findAxisRepresentations();
  bool allocIbrArrays(int w, int h);
};

#endif // HAVE_CUDA
#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
