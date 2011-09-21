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

vvIbrImage::vvIbrImage(short h, short w, uchar* image, int dp)
: vvImage(h, w, image)
{
  _depthPrecision = dp;
  _pixeldepth = NULL;
  _preprojectionMatrix.identity();
}

vvIbrImage::vvIbrImage()
: vvImage()
{
  _pixeldepth = NULL;
  _preprojectionMatrix.identity();
}

vvIbrImage::~vvIbrImage()
{
  delete[] _pixeldepth;
}

uchar* vvIbrImage::getPixelDepth()
{
  return _pixeldepth;
}

//----------------------------------------------------------------------------
/** Allocates momory for a new image and 2.5d-data
 */
void vvIbrImage::alloc_pd()
{
  delete[] _pixeldepth;

  _pixeldepth = new uchar[width*height*_depthPrecision/8];
}

int vvIbrImage::getDepthPrecision() const
{
  return _depthPrecision;
}

void vvIbrImage::setDepthPrecision(int dp)
{
  assert(dp==8 || dp==16 || dp==32);
  _depthPrecision = dp;
  delete[] _pixeldepth;
  _pixeldepth = NULL;
}

void vvIbrImage::setReprojectionMatrix(const vvMatrix& reprojectionMatrix)
{
  _preprojectionMatrix = reprojectionMatrix;
}

vvMatrix vvIbrImage::getReprojectionMatrix() const
{
  return _preprojectionMatrix;
}
