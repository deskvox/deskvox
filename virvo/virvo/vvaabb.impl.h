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

#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>

using std::cerr;
using std::endl;

//============================================================================
// vvBaseAABB<T> Method Definitions
//============================================================================

template <typename T>
vvBaseAABB<T>::vvBaseAABB(const vvBaseVector3<T>& minval, const vvBaseVector3<T>& maxval)
  : _min(minval)
  , _max(maxval)
{
  _center = vvBaseVector3<T>((_min[0] + _max[0]) * 0.5f,
                             (_min[1] + _max[1]) * 0.5f,
                             (_min[2] + _max[2]) * 0.5f);
  calcVertices();
}

template <typename T>
const vvBaseVector3<T>& vvBaseAABB<T>::getMin() const
{
  return _min;
}

template <typename T>
const vvBaseVector3<T>& vvBaseAABB<T>::getMax() const
{
  return _max;
}

template <typename T>
T vvBaseAABB<T>::calcWidth() const
{
  return _max[0] - _min[0];
}

template <typename T>
T vvBaseAABB<T>::calcHeight() const
{
  return _max[1] - _min[1];
}

template <typename T>
T vvBaseAABB<T>::calcDepth() const
{
  return _max[2] - _min[2];
}

template <typename T>
const vvBaseVector3<T>& vvBaseAABB<T>::getCenter() const
{
  return _center;
}

template <typename T>
void vvBaseAABB<T>::intersect(const vvBaseAABB<T>& rhs)
{
  for (int i = 0; i < 3; ++i)
  {
    if (rhs._min[i] > _min[i])
    {
      _min[i] = rhs._min[i];
    }

    if (rhs._max[i] < _max[i])
    {
      _max[i] = rhs._max[i];
    }
  }
  calcVertices();
}

template <typename T>
void vvBaseAABB<T>::calcVertices()
{
  for (int i=0; i<8; ++i)
  {
    // return the vertices in the necessary order
    int d=i;
    if(i>=2 && i<=5)
      d ^= 1;

    for (int c=0; c<3; ++c)
    {
      _vertices[i][c] = (1<<c)&d ? _min[c] : _max[c];
    }
  }
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
