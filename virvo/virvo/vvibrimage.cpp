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
#include "vvdebugmsg.h"
#include "vvibrimage.h"
#include "vvibr.h"

using std::cerr;
using std::cout;
using std::endl;

vvIbrImage::vvIbrImage(short h, short w, uchar* image, int dp)
: vvImage(h, w, image)
, _depthPrecision(dp)
, _depthCodeType(VV_RAW)
, _codedDepthSize(0)
, _pixeldepth(NULL)
, _codeddepth(NULL)
, _viewport(0, 0, w, h)
{
  _projectionMatrix.identity();
  _modelViewMatrix.identity();
  _depthMin = _depthMax = 0.f;
}

vvIbrImage::vvIbrImage()
: vvImage()
, _depthPrecision(0)
, _depthCodeType(VV_RAW)
, _codedDepthSize(0)
, _pixeldepth(NULL)
, _codeddepth(NULL)
, _viewport(0, 0, 0, 0)
{
  _projectionMatrix.identity();
  _modelViewMatrix.identity();
  _depthMin = _depthMax = 0.f;
}

vvIbrImage::~vvIbrImage()
{
  if(t == VV_CLIENT)
  {
    delete[] _pixeldepth;
  }
  delete[] _codeddepth;
}

uchar* vvIbrImage::getPixelDepth() const
{
  return _pixeldepth;
}

uchar *vvIbrImage::getCodedDepth() const
{
  if(_depthCodeType == VV_RAW)
    return _pixeldepth;
  else
    return _codeddepth;
}

vvImage::CodeType vvIbrImage::getDepthCodetype() const
{
  return _depthCodeType;
}

void vvIbrImage::setDepthCodetype(vvImage::CodeType ct)
{
  _depthCodeType = ct;
}

//----------------------------------------------------------------------------
/** Allocates memory for a new image and 2.5d-data
 */
void vvIbrImage::alloc_pd()
{
  if(t == VV_CLIENT)
  {
    delete[] _pixeldepth;
    _pixeldepth = new uchar[width*height*_depthPrecision/8];
  }

  delete[] _codeddepth;
  _codeddepth = new uchar[width*height*_depthPrecision/8*2];
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
  _depthCodeType = VV_RAW;
  _codedDepthSize = 0;
}

void vvIbrImage::setDepthSize(int size)
{
  _codedDepthSize = size;
}

void vvIbrImage::setModelViewMatrix(const vvMatrix &mv)
{
  _modelViewMatrix = mv;
}

vvMatrix vvIbrImage::getModelViewMatrix() const
{
  return _modelViewMatrix;
}

void vvIbrImage::setProjectionMatrix(const vvMatrix &pm)
{
  _projectionMatrix = pm;
}

vvMatrix vvIbrImage::getProjectionMatrix() const
{
  return _projectionMatrix;
}

void vvIbrImage::setDepthRange(const float dmin, const float dmax)
{
  _depthMin = dmin;
  _depthMax = dmax;
}

void vvIbrImage::getDepthRange(float *dmin, float *dmax) const
{
  *dmin = _depthMin;
  *dmax = _depthMax;
}

void vvIbrImage::setViewport(const virvo::Viewport &vp)
{
  _viewport = vp;
}

virvo::Viewport vvIbrImage::getViewport() const
{
  return _viewport;
}

vvMatrix vvIbrImage::getReprojectionMatrix() const
{
  return vvIbr::calcImgMatrix(_projectionMatrix, _modelViewMatrix, _viewport, _depthMin, _depthMax);
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
    _depthCodeType = VV_RAW;
    break;
  default:
    {
      CodecFunc enc = gen_RLC_encode;
      CodeType ctused = VV_RLE;
      if(ct == VV_SNAPPY)
      {
        enc = snappyEncode;
        ctused = VV_SNAPPY;
      }

      _codedDepthSize = enc(_pixeldepth, _codeddepth, width*height*(_depthPrecision/8), width*height*(_depthPrecision/8)*2, _depthPrecision/8);
      if(_codedDepthSize < 0)
      {
        cr = 1.f;
        _depthCodeType = VV_RAW;
      }
      else
      {
        cr = (float)_codedDepthSize/width/height/(_depthPrecision/8);
        _depthCodeType = ctused;
      }
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
  case VV_RAW:
    break;
  default:
    {
      CodecFunc dec = gen_RLC_decode;
      if(_depthCodeType == VV_SNAPPY)
        dec = snappyDecode;

      err = dec(_codeddepth, _pixeldepth, _codedDepthSize, width*height*(_depthPrecision/8), _depthPrecision/8);
      if(!err)
      {
        vvDebugMsg::msg(3, "vvIbrImage::decode: success, compressed size for depth was ", _codedDepthSize);
        _depthCodeType = VV_RAW;
        _codedDepthSize = 0;
      }
      else
      {
        vvDebugMsg::msg(1, "vvIbrImage::decode: failed to decode depth");
      }
    }
    break;
  }

  return err;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
