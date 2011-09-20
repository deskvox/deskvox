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

#include "vvibrimage.h"

vvIbrImage::vvIbrImage(short h, short w, uchar* image, DepthPrecision dp)
: vvImage(h, w, image)
{
  _depthPrecision = dp;
  _pixeldepthUchar  = 0;
  _pixeldepthUshort = 0;
  _pixeldepthUint   = 0;
  _preprojectionMatrix.identity();
}

vvIbrImage::vvIbrImage()
: vvImage()
{
  _pixeldepthUchar  = 0;
  _pixeldepthUshort = 0;
  _pixeldepthUint   = 0;
  _preprojectionMatrix.identity();
}

vvIbrImage::~vvIbrImage()
{
  delete[] _pixeldepthUchar;
  delete[] _pixeldepthUshort;
  delete[] _pixeldepthUint;
}

uchar* vvIbrImage::getpixeldepthUchar()
{
  return _pixeldepthUchar;
}

ushort* vvIbrImage::getpixeldepthUshort()
{
  return _pixeldepthUshort;
}

uint* vvIbrImage::getpixeldepthUint()
{
  return _pixeldepthUint;
}

//----------------------------------------------------------------------------
/** Allocates momory for a new image and 2.5d-data
 */
void vvIbrImage::alloc_pd()
{
  delete[] _pixeldepthUchar;
  delete[] _pixeldepthUshort;
  delete[] _pixeldepthUint;

  _pixeldepthUchar  = NULL;
  _pixeldepthUshort = NULL;
  _pixeldepthUint   = NULL;
  switch(_depthPrecision)
  {
  case VV_UCHAR:
    _pixeldepthUchar = new uchar[width*height];
    break;
  case VV_USHORT:
    _pixeldepthUshort = new ushort[width*height];
    break;
  case VV_UINT:
    _pixeldepthUint = new uint[width*height];
    break;
  }
}

vvIbrImage::DepthPrecision vvIbrImage::getDepthPrecision()
{
  return _depthPrecision;
}

void vvIbrImage::setDepthPrecision(DepthPrecision dp)
{
  _depthPrecision = dp;
}

void vvIbrImage::setReprojectionMatrix(const vvMatrix& reprojectionMatrix)
{
  _preprojectionMatrix = reprojectionMatrix;
}

vvMatrix vvIbrImage::getReprojectionMatrix() const
{
  return _preprojectionMatrix;
}
