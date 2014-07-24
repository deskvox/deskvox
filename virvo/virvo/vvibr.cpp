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

#include "math/math.h"
#include "vvibr.h"


void virvo::ibr::calcDepthRange(const vvMatrix& pr, const vvMatrix& mv,
                           const vvAABB& aabb, float& minval, float& maxval)
{
  vec4f center4(aabb.getCenter()[0], aabb.getCenter()[1], aabb.getCenter()[2], 1.0f);
  vec4f min4(aabb.getMin()[0], aabb.getMin()[1], aabb.getMin()[2], 1.0f);
  vec4f max4(aabb.getMax()[0], aabb.getMax()[1], aabb.getMax()[2], 1.0f);

  vvVector4 tmp( center4 );     // TODO
  tmp.multiply(mv);
  center4 = tmp;                // TODO
  tmp = vvVector4( min4 );      // TODO
  tmp.multiply(mv);
  min4 = tmp;                   // TODO
  tmp = vvVector4( max4 );      // TODO
  tmp.multiply(mv);
  max4 = tmp;                   // TODO

  vec3f center(center4[0], center4[1], center4[2]);
  vec3f min3(min4[0], min4[1], min4[2]);
  vec3f max3(max4[0], max4[1], max4[2]);

  float radius = length(max3 - min3) * 0.5f;

  // Depth buffer of ibrPlanes
  vec3f scal(center);
  scal = normalize(scal) * radius;
  min3 = center - scal;
  max3 = center + scal;

  min4 = vec4f(min3, 1.f);
  max4 = vec4f(max3, 1.f);

  tmp = vvVector4( min4 );      // TODO
  tmp.multiply(pr);
  min4 = tmp;                   // TODO
  tmp = vvVector4( max4 );      // TODO
  tmp.multiply(pr);
  max4 = tmp;                   // TODO
  min3 = min4.xyz() / min4.w;
  max3 = max4.xyz() / max4.w;

  minval = (min3[2]+1.f) * 0.5f;
  maxval = (max3[2]+1.f) * 0.5f;
}

vvMatrix virvo::ibr::calcImgMatrix(const vvMatrix& pr, const vvMatrix& mv,
                                  recti const& vp,
                                  const float depthRangeMin, const float depthRangeMax)
{
  vvMatrix invModelviewProjection = pr * mv;
  invModelviewProjection.invert();

  return invModelviewProjection
    * calcViewportMatrix(vp)
    * calcDepthScaleMatrix(depthRangeMin, depthRangeMax);
}

vvMatrix virvo::ibr::calcViewportMatrix(recti const& vp)
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

vvMatrix virvo::ibr::calcDepthScaleMatrix(const float depthRangeMin, const float depthRangeMax)
{
  vvMatrix result;
  result.identity();
  result.scaleLocal(1.0f, 1.0f, (depthRangeMax - depthRangeMin));
  result.translate(0.0f, 0.0f, depthRangeMin);
  return result;
}
