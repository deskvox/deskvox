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

#include "vvexport.h"
#include "vvopengl.h"
#include "vvsoftvr.h"
#include "vvtransfunc.h"
#include "vvvoldesc.h"
#include "vvimage.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_CUDA

#include "vvcuda.h"

#endif

/** Ray casting renderer. Based on the volume
  rendering implementation from the NVIDIA CUDA SDK,
  as of November 25th, 2010 could be downloaded from
  the following location:
  http://developer.download.nvidia.com/compute/cuda/sdk/website/samples.html
 */
class VIRVOEXPORT vvRayRend : public vvSoftVR
{
public:
  enum IbrMode
  {
    VV_MAX_GRADIENT,
    VV_MIDDLE,
    VV_SURFACE
  };

#ifdef HAVE_CUDA
  vvRayRend(vvVolDesc* vd, vvRenderState renderState);
  ~vvRayRend();

  int getLUTSize() const;
  void updateTransferFunction();
  void compositeVolume(int w = -1, int h = -1);
  void setParameter(ParameterType param, float newValue);
  void setNumSpaceSkippingCells(const int numSpaceSkippingCells[3]);
  void setDepthPrecision(const vvImage2_5d::DepthPrecision dp);

  bool getEarlyRayTermination() const;
  bool getIllumination() const;
  bool getInterpolation() const;
  bool getOpacityCorrection() const;
  bool getSpaceSkipping() const;

  float* getIbrPlanes() const;

  uchar*  _depthUchar;
  ushort* _depthUshort;
  uint*   _depthUint;

  float _ibrPlanes[2];                         ///< ibr clipping planes, updated every frame

private:
  cudaChannelFormatDesc _channelDesc;
  std::vector<cudaArray*> d_volumeArrays;
  cudaArray* d_transferFuncArray;
  cudaArray* d_randArray;
  cudaArray* d_spaceSkippingArray;

  bool* h_spaceSkippingArray;
  int* h_cellMinValues;
  int* h_cellMaxValues;
  int h_numCells[3];

  float* _rgbaTF;

  bool _earlyRayTermination;        ///< Terminate ray marching when enough alpha was gathered
  bool _illumination;               ///< Use local illumination
  bool _interpolation;              ///< interpolation mode: true=linear interpolation (default), false=nearest neighbor
  bool _opacityCorrection;          ///< true = opacity correction on
  bool _spaceSkipping;              ///< true = skip over homogeneous regions
  bool _volumeCopyToGpuOk;          ///< must be true for memCopy to be run

  vvImage2_5d::DepthPrecision _depthPrecision; ///< enum indicating precision of depth buffer for image based rendering

  void initRandTexture();
  void initSpaceSkippingTexture();
  void initVolumeTexture();
  void factorViewMatrix();
  void findAxisRepresentations();

  void calcSpaceSkippingGrid();
  void computeSpaceSkippingTexture();
#endif // HAVE_CUDA
};

#endif
