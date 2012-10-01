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

#include "vvibrrenderer.h"
#include "vvvoldesc.h"

vvIbrRenderer::vvIbrRenderer(vvVolDesc* vd, vvRenderState renderState)
  : vvSoftVR(vd, renderState)
  , _depthPrecision(8)
{

}

vvIbrRenderer::~vvIbrRenderer()
{

}

void vvIbrRenderer::setParameter(ParameterType param, const vvParam& newValue)
{
  switch (param)
  {
  case vvRenderer::VV_IBR_DEPTH_PREC:
    _depthPrecision = newValue;
    break;
  case vvRenderer::VV_IBR_DEPTH_RANGE:
    _depthRange = newValue;
    break;
  default:
    vvSoftVR::setParameter(param, newValue);
    break;
  }
}

vvParam vvIbrRenderer::getParameter(ParameterType param) const
{
  switch (param)
  {
  case vvRenderer::VV_IBR_DEPTH_PREC:
    return _depthPrecision;
  case vvRenderer::VV_IBR_DEPTH_RANGE:
    return _depthRange;
  default:
    return vvSoftVR::getParameter(param);
  }
}

