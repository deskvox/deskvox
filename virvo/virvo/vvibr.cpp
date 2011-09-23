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
                           const vvAABB& aabb, float& min, float& max)
{
  vvVector4 center4(aabb.getCenter()[0], aabb.getCenter()[1], aabb.getCenter()[2], 1.0f);
  vvVector4 min4(aabb.min()[0], aabb.min()[1], aabb.min()[2], 1.0f);
  vvVector4 max4(aabb.max()[0], aabb.max()[1], aabb.max()[2], 1.0f);

  center4.multiply(&mv);
  min4.multiply(&mv);
  max4.multiply(&mv);

  vvVector3 center(center4[0], center4[1], center4[2]);
  vvVector3 min3(min4.e[0], min4.e[1], min4.e[2]);
  vvVector3 max3(max4.e[0], max4.e[1], max4.e[2]);

  float radius = (max3 - min3).length() * 0.5f;

  // Depth buffer of ibrPlanes
  vvVector3 scal(center);
  scal.normalize();
  scal.scale(radius);
  min3 = center - scal;
  max3 = center + scal;

  min4 = vvVector4(&min3, 1.f);
  max4 = vvVector4(&max3, 1.f);
  min4.multiply(&pr);
  max4.multiply(&pr);
  min4.perspectiveDivide();
  max4.perspectiveDivide();

  min = (min4[2]+1.f) * 0.5f;
  max = (max4[2]+1.f) * 0.5f;
}

vvMatrix vvIbr::calcImgMatrix(const vvMatrix& pr, const vvMatrix& mv,
                              const vvGLTools::Viewport& vp,
                              const float depthRangeMin, const float depthRangeMax)
{
  vvMatrix vpMatrix;
  vpMatrix.identity();
  vpMatrix.scale(1.0f / (0.5f * vp[2]),
                 1.0f / (0.5f * vp[3]),
                 2.0f);
  vpMatrix.translate((vp[0] / (0.5f * vp[2])) - 1.0f,
                     (vp[1] / (0.5f * vp[3])) - 1.0f,
                     -1.0f);

  vvMatrix invModelviewProjection = mv * pr;
  invModelviewProjection.invert();

  vvMatrix depthScaleMatrix;
  depthScaleMatrix.identity();
  depthScaleMatrix.scale(1.0f, 1.0f, (depthRangeMax - depthRangeMin));
  depthScaleMatrix.translate(0.0f, 0.0f, depthRangeMin);

  return depthScaleMatrix * vpMatrix * invModelviewProjection;
}
