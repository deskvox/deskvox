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


#ifndef VV_GL_FORMAT_H
#define VV_GL_FORMAT_H


#include "vvexport.h"

#include "types.h"


namespace virvo
{
namespace gl
{


    enum /*class*/ EFormat
    {
        EFormat_Unspecified = 0,

        // Color formats
        EFormat_R8,
        EFormat_RG8,
        EFormat_RGB8,
        EFormat_RGBA8,
        EFormat_R16F,
        EFormat_RG16F,
        EFormat_RGB16F,
        EFormat_RGBA16F,
        EFormat_R32F,
        EFormat_RG32F,
        EFormat_RGB32F,
        EFormat_RGBA32F,
        EFormat_R16I,
        EFormat_RG16I,
        EFormat_RGB16I,
        EFormat_RGBA16I,
        EFormat_R32I,
        EFormat_RG32I,
        EFormat_RGB32I,
        EFormat_RGBA32I,
        EFormat_R16UI,
        EFormat_RG16UI,
        EFormat_RGB16UI,
        EFormat_RGBA16UI,
        EFormat_R32UI,
        EFormat_RG32UI,
        EFormat_RGB32UI,
        EFormat_RGBA32UI,
        EFormat_BGR8,
        EFormat_BGRA8,
        EFormat_RGB10_A2,
        EFormat_R11F_G11F_B10F,

        // Depth formats
        EFormat_DEPTH_COMPONENT16,
        EFormat_DEPTH_COMPONENT24,
        EFormat_DEPTH_COMPONENT32,
        EFormat_DEPTH_COMPONENT32F,

        // Combined depth-stencil formats
        EFormat_DEPTH24_STENCIL8,
        EFormat_DEPTH32F_STENCIL8
    };


    // Internal GL representation of the texture formats above
    struct Format
    {
        GLenum internalFormat;
        GLenum format;
        GLenum type;

        Format(GLenum internalFormat, GLenum format, GLenum type)
            : internalFormat(internalFormat)
            , format(format)
            , type(type)
        {
        }
    };


    // Maps an EFormat enum to the internal GL formats
    VVAPI Format mapFormat(EFormat f);

    // Returns whether the given texture format is a color format
    VVAPI bool isColorFormat(EFormat f);

    // Returns whether the given texture format is a depth format
    VVAPI bool isDepthFormat(EFormat f);

    // Returns whether the given texture format is a depth-stencil format
    VVAPI bool isDepthStencilFormat(EFormat f);


} // namespace gl
} // namespace virvo


#endif // VV_GL_FORMAT_H
