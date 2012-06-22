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

#include "vvibr.h"

void vvIbr::calcDepthRange(const vvMatrix& pr, const vvMatrix& mv,
                           const vvAABB& aabb, float& minval, float& maxval)
{
  vvVector4 center4(aabb.getCenter()[0], aabb.getCenter()[1], aabb.getCenter()[2], 1.0f);
  vvVector4 min4(aabb.getMin()[0], aabb.getMin()[1], aabb.getMin()[2], 1.0f);
  vvVector4 max4(aabb.getMax()[0], aabb.getMax()[1], aabb.getMax()[2], 1.0f);

  center4.multiply(mv);
  min4.multiply(mv);
  max4.multiply(mv);

  vvVector3 center(center4[0], center4[1], center4[2]);
  vvVector3 min3(min4[0], min4[1], min4[2]);
  vvVector3 max3(max4[0], max4[1], max4[2]);

  float radius = (max3 - min3).length() * 0.5f;

  // Depth buffer of ibrPlanes
  vvVector3 scal(center);
  scal.normalize();
  scal.scale(radius);
  min3 = center - scal;
  max3 = center + scal;

  min4 = vvVector4(min3, 1.f);
  max4 = vvVector4(max3, 1.f);
  min4.multiply(pr);
  max4.multiply(pr);
  min4.perspectiveDivide();
  max4.perspectiveDivide();

  minval = (min4[2]+1.f) * 0.5f;
  maxval = (max4[2]+1.f) * 0.5f;
}

vvMatrix vvIbr::calcImgMatrix(const vvMatrix& pr, const vvMatrix& mv,
                              const vvGLTools::Viewport& vp,
                              const float depthRangeMin, const float depthRangeMax)
{
  vvMatrix invModelviewProjection = pr * mv;
  invModelviewProjection.invert();

  return invModelviewProjection
    * calcViewportMatrix(vp)
    * calcDepthScaleMatrix(depthRangeMin, depthRangeMax);
}

vvMatrix vvIbr::calcViewportMatrix(const vvGLTools::Viewport& vp)
{
  vvMatrix result;
  result.identity();
  result.scaleLocal(1.0f / (0.5f * vp[2]),
                    1.0f / (0.5f * vp[3]),
                    2.0f);
  result.translate((vp[0] / (0.5f * vp[2])) - 1.0f,
                   (vp[1] / (0.5f * vp[3])) - 1.0f,
                   -1.0f);
  return result;
}

vvMatrix vvIbr::calcDepthScaleMatrix(const float depthRangeMin, const float depthRangeMax)
{
  vvMatrix result;
  result.identity();
  result.scaleLocal(1.0f, 1.0f, (depthRangeMax - depthRangeMin));
  result.translate(0.0f, 0.0f, depthRangeMin);
  return result;
}
