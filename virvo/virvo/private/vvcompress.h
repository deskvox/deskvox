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


#ifndef VV_COMPRESS_H
#define VV_COMPRESS_H


#include "vvexport.h"
#include "../vvpixelformat.h"

#include <vector>


namespace virvo
{


class CompressedVector;


struct JPEGOptions
{
    // encode: [in] The pixel format of the input image
    // decode: [in] The pixel format of the output image
    PixelFormat format;
    // encode: [in] The width of the input image
    // decode: [out] The width of the output image
    int w;
    // encode: [in] The height of the input image
    // decode: [out] The height of the output image
    int h;
    // encode: [in] The width in bytes of a single scanline in the input image
    // decode: [out] The width in bytes of a single scanline in the output image
    int pitch;
    // encode: [in] The quality of the JPEG encoding. Must be in the range [0,100]
    // decode: [ignored]
    int quality;
};


VVAPI bool encodeSnappy(std::vector<unsigned char>& data);

VVAPI bool decodeSnappy(std::vector<unsigned char>& data);

VVAPI bool encodeSnappy(CompressedVector& data);

VVAPI bool decodeSnappy(CompressedVector& data);

VVAPI bool encodeJPEG(std::vector<unsigned char>& data, JPEGOptions const& options);

VVAPI bool decodeJPEG(std::vector<unsigned char>& data, JPEGOptions& options);

VVAPI bool encodeJPEG(CompressedVector& data, JPEGOptions const& options);

VVAPI bool decodeJPEG(CompressedVector& data, JPEGOptions& options);


} // namespace virvo


#endif
