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

#include "vvcudarendertarget.h"
#include "vvrenderer.h"

#include <boost/shared_ptr.hpp>


class vvVolDesc;

/** Ray casting renderer. Based on the volume
  rendering implementation from the NVIDIA CUDA SDK,
  as of November 25th, 2010 could be downloaded from
  the following location:
  http://developer.download.nvidia.com/compute/cuda/sdk/website/samples.html
 */
class vvRayRend : public vvRenderer
{
  typedef vvRenderer BaseType;

public:
  VVAPI vvRayRend(vvVolDesc* vd, vvRenderState renderState);
  VVAPI ~vvRayRend();

  VVAPI size_t getLUTSize() const;
  VVAPI bool instantClassification() const;
  VVAPI void setVolDesc(vvVolDesc* vd) VV_OVERRIDE;
  VVAPI virtual void updateTransferFunction() VV_OVERRIDE;
  VVAPI virtual void renderVolumeGL() VV_OVERRIDE;
  VVAPI bool checkParameter(ParameterType param, vvParam const& value) const VV_OVERRIDE;
  VVAPI virtual void setParameter(ParameterType param, const vvParam& newValue) VV_OVERRIDE;
  VVAPI virtual vvParam getParameter(ParameterType param) const VV_OVERRIDE;

private:
  float* _rgbaTF;

  bool _volumeCopyToGpuOk;          ///< must be true for memCopy to be run
  bool _twoPassIbr;                 ///< Perform an alpha-gathering pass before the actual render pass

  void initVolumeTexture();

  struct Impl;
  boost::shared_ptr<Impl> impl;
};

#include "vvrayrendfactory.h"

#endif // HAVE_CUDA

#endif // VV_RAYREND_H

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
