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

#ifndef _VV_IBRRENDERER_H_
#define _VV_IBRRENDERER_H_

#include "vvsoftvr.h"

class vvVolDesc;

class VIRVOEXPORT vvIbrRenderer : public vvSoftVR
{
public:
  vvIbrRenderer(vvVolDesc* vd, vvRenderState renderState);
  virtual ~vvIbrRenderer();
  virtual void compositeVolume(int w = -1, int h = -1) = 0;
  virtual void getColorBuffer(uchar** colors) const = 0;
  virtual void getDepthBuffer(uchar** depths) const = 0;
  virtual void setParameter(ParameterType param, const vvParam& newValue);
  virtual vvParam getParameter(ParameterType param) const;
protected:
  int _depthPrecision;             ///< number of bits in depth buffer for image based rendering
  vvVector2 _depthRange;
};

#endif

