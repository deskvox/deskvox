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

#include <cassert>
#include <vvdebugmsg.h>
#include "vvibrimage.h"

vvIbrImage::vvIbrImage(short h, short w, uchar* image, int dp)
: vvImage(h, w, image)
, _depthPrecision(dp)
, _depthCodeType(0)
, _codedDepthSize(0)
, _pixeldepth(NULL)
, _codeddepth(NULL)
{
  _preprojectionMatrix.identity();
}

vvIbrImage::vvIbrImage()
: vvImage()
, _depthPrecision(0)
, _depthCodeType(0)
, _codedDepthSize(0)
, _pixeldepth(NULL)
, _codeddepth(NULL)
{
  _preprojectionMatrix.identity();
}

vvIbrImage::~vvIbrImage()
{
  if(t == VV_CLIENT)
    delete[] _pixeldepth;
  delete[] _codeddepth;
}

uchar* vvIbrImage::getPixelDepth() const
{
  return _pixeldepth;
}

uchar *vvIbrImage::getCodedDepth() const
{
  if(_depthCodeType == 0)
    return _pixeldepth;
  else
    return _codeddepth;
}

int vvIbrImage::getDepthCodetype() const
{
  return _depthCodeType;
}

void vvIbrImage::setDepthCodetype(int ct)
{
  _depthCodeType = ct;
}

//----------------------------------------------------------------------------
/** Allocates momory for a new image and 2.5d-data
 */
void vvIbrImage::alloc_pd()
{
  if(t == VV_CLIENT)
  {
    delete[] _pixeldepth;
    _pixeldepth = new uchar[width*height*_depthPrecision/8];
  }

  delete[] _codeddepth;
  _codeddepth = new uchar[width*height*_depthPrecision/8];
}

int vvIbrImage::getDepthPrecision() const
{
  return _depthPrecision;
}

int vvIbrImage::getDepthSize() const
{
  switch(_depthCodeType)
  {
  case 0:
    return width*height*(_depthPrecision/8);
  default:
    return _codedDepthSize;
  }
}

void vvIbrImage::setDepthPrecision(int dp)
{
  assert(dp==8 || dp==16 || dp==32);
  _depthPrecision = dp;
  if(t == VV_CLIENT)
  {
    delete[] _pixeldepth;
    _pixeldepth = NULL;
  }
}

void vvIbrImage::setNewDepthPtr(uchar *depth)
{
  _pixeldepth = depth;
  _depthCodeType = 0;
  _codedDepthSize = 0;
}

void vvIbrImage::setDepthSize(int size)
{
  _codedDepthSize = size;
}

void vvIbrImage::setReprojectionMatrix(const vvMatrix& reprojectionMatrix)
{
  _preprojectionMatrix = reprojectionMatrix;
}

vvMatrix vvIbrImage::getReprojectionMatrix() const
{
  return _preprojectionMatrix;
}

int vvIbrImage::encode(short ct, short sh, short eh, short sw, short ew)
{
  vvDebugMsg::msg(3, "vvIbrImage::encode: depth code type is ", ct);
  int err = vvImage::encode(ct, sh, eh, sw, ew);

  float cr = 1.f;
  switch(ct)
  {
  case 0:
    cr=1.f;
    _depthCodeType = 0;
    break;
  default:
    _codedDepthSize = gen_RLC_encode(_pixeldepth, _codeddepth, width*height, _depthPrecision/8, width*height*(_depthPrecision/8));
    if(_codedDepthSize < 0)
    {
      cr = 1.f;
      _depthCodeType = 0;
    }
    else
    {
      cr = (float)_codedDepthSize/width/height/(_depthPrecision/8);
      _depthCodeType = 1;
    }
    break;
  }

  vvDebugMsg::msg(3, "vvIbrImage::encode: depth compression ratio is ", cr);

  return err;
}

int vvIbrImage::decode()
{
  vvDebugMsg::msg(3, "vvIbrImage::decode: depth code type is ", _depthCodeType);
  int err = vvImage::decode();
  if(err)
    return err;

  switch(_depthCodeType)
  {
  case 0:
    break;
  default:
    err = gen_RLC_decode(_codeddepth, _pixeldepth, _codedDepthSize, _depthPrecision/8, width*height*(_depthPrecision/8));
    if(!err)
    {
      vvDebugMsg::msg(3, "vvIbrImage::decode: success, compressed size was ", _codedDepthSize);
      _depthCodeType = 0;
      _codedDepthSize = 0;
    }
    else
    {
      vvDebugMsg::msg(1, "vvIbrImage::decode: failed to decode depth");
    }
    break;
  }

  return err;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
