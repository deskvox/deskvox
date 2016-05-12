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

#ifndef VV_CLIP_OBJ_H
#define VV_CLIP_OBJ_H

#include <cstddef>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "math/math.h"
#include "mem/allocator.h"
#include "vvexport.h"

//============================================================================
// Clip object base
//============================================================================

class VIRVOEXPORT vvClipObj
{
public:

  enum Type
  {
    VV_PLANE,
    VV_SPHERE,
    VV_TRIANGLE_LIST
  };

  static boost::shared_ptr<vvClipObj> create(Type t);

public:

  virtual ~vvClipObj() {}

};


//============================================================================
// Clip plane
//============================================================================

class VIRVOEXPORT vvClipPlane
    : public vvClipObj
    , public virvo::basic_plane<3, float>
{
};


//============================================================================
// Clip sphere
//============================================================================

class VIRVOEXPORT vvClipSphere : public vvClipObj
{
public:
  virvo::vec3 center;
  float radius;
};


//============================================================================
// Clip triangle list
//============================================================================

class VIRVOEXPORT vvClipTriangleList : public vvClipObj
{
public:
  typedef struct { virvo::vec3 v1, v2, v3; } Triangle;
  typedef std::vector<Triangle, virvo::mem::aligned_allocator<Triangle, 32> > Triangles;
  typedef virvo::mat4 Matrix;

  void resize(size_t size);

  const Triangle* data() const;
        Triangle* data();

  const Triangle& operator[](size_t i) const;
        Triangle& operator[](size_t i);

  const Matrix& transform() const;
        Matrix& transform();

private:

  Triangles triangles_;
  virvo::mat4 transform_;

};

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
