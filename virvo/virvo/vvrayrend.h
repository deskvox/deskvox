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

#ifndef VV_RAYREND_H
#define VV_RAYREND_H

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_CUDA

#include "vvibrrenderer.h"

#include <memory>


class vvVolDesc;

/** Ray casting renderer. Based on the volume
  rendering implementation from the NVIDIA CUDA SDK,
  as of November 25th, 2010 could be downloaded from
  the following location:
  http://developer.download.nvidia.com/compute/cuda/sdk/website/samples.html
 */
class VIRVOEXPORT vvRayRend : public vvIbrRenderer
{
public:
  vvRayRend(vvVolDesc* vd, vvRenderState renderState);
  ~vvRayRend();

  size_t getLUTSize() const;
  void updateTransferFunction();
  void compositeVolume(int w = -1, int h = -1);
  void getColorBuffer(uchar** colors) const;
  void getDepthBuffer(uchar** depths) const;
  virtual void setParameter(ParameterType param, const vvParam& newValue);
  virtual vvParam getParameter(ParameterType param) const;

  bool getEarlyRayTermination() const;
private:
  float* _rgbaTF;

  bool _volumeCopyToGpuOk;          ///< must be true for memCopy to be run
  bool _twoPassIbr;                 ///< Perform an alpha-gathering pass before the actual render pass

  void initVolumeTexture();
  void factorViewMatrix();
  void findAxisRepresentations();
  bool allocIbrArrays(size_t w, size_t h);

  struct Impl;
  std::auto_ptr<Impl> impl;

private:
  vvRayRend(vvRayRend const&); // = delete;
  vvRayRend& operator =(vvRayRend const&); // = delete;
};

#include "vvrayrendfactory.h"

#endif // HAVE_CUDA

#endif // VV_RAYREND_H

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
