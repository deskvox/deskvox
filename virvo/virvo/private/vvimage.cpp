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


#include "vvimage.h"
#include "vvcompress.h"


using virvo::Image;
using virvo::PixelFormat;


Image::Image(int w, int h, PixelFormat format, int stride)
{
  resize(w, h, format, stride);
}


Image::Image(unsigned char* data, int w, int h, PixelFormat format, int stride)
{
  assign(data, w, h, format, stride);
}


Image::~Image()
{
}


void Image::resize(int w, int h, PixelFormat format, int stride)
{
  init(w, h, format, stride);
  data_.resize(size());
}


void Image::assign(unsigned char* data, int w, int h, PixelFormat format, int stride)
{
  assert( data );

  init(w, h, format, stride);
  data_.assign(data, data + size());
}


bool Image::compress()
{
  assert( size() == data_.size() );

  return virvo::compress(data_);
}


bool Image::decompress()
{
  if (!virvo::decompress(data_))
    return false;

  assert( size() == data_.size() );

  return true;
}


void Image::init(int w, int h, PixelFormat format, int stride)
{
  assert( w > 0 );
  assert( h > 0 );
  assert( format != PF_UNSPECIFIED );

  width_ = w;
  height_ = h;
  format_ = format;
  stride_ = stride <= 0 ? w * getPixelSize(format) : stride;
}
