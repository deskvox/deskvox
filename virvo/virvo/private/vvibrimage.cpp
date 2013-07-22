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
#include "vvcompress.h"


using virvo::IbrImage;
using virvo::PixelFormat;


IbrImage::IbrImage()
  : BaseType()
  , depth_(0)
  , depthFormat_(PF_UNSPECIFIED)
  , depthMin_(0.0f)
  , depthMax_(0.0f)
  , viewMatrix_(1.0f, 1.0f, 1.0f, 1.0f)
  , projMatrix_(1.0f, 1.0f, 1.0f, 1.0f)
  , viewport_(0, 0, 0, 0)
{
}


IbrImage::IbrImage(int w, int h, PixelFormat colorFormat, PixelFormat depthFormat)
  : BaseType(w, h, colorFormat)
  , depth_(w * h * getPixelSize(depthFormat))
  , depthFormat_(depthFormat)
  , depthMin_(0.0f)
  , depthMax_(1.0f)
  , viewMatrix_(1.0f, 1.0f, 1.0f, 1.0f)
  , projMatrix_(1.0f, 1.0f, 1.0f, 1.0f)
  , viewport_(0, 0, w, h)
{
}


IbrImage::~IbrImage()
{
}


bool IbrImage::compress()
{
  // Compress the color buffer
  if (!BaseType::compress())
    return false;

  // Compress the depth buffer
  if (!virvo::encodeSnappy(depth_))
    return false;

  return true;
}


bool IbrImage::decompress()
{
  // Decompress the color buffer
  if (!BaseType::decompress())
    return false;

  // Decompress the depth buffer
  if (!virvo::decodeSnappy(depth_))
    return false;

  return true;
}
