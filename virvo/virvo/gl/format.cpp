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


#include "format.h"

#include <GL/glew.h>

#include <assert.h>


namespace gl = virvo::gl;

using gl::Format;


// NOTE:
// Keep in sync with EFormat defined in format.h!!!
// Adjust the isXXXFormat() below when adding new formats!!!
const Format kFormats[] =
{
    // Unspecified
    Format(0,0,0),

    // Color formats
    Format(GL_R8,                   GL_RED,                 GL_UNSIGNED_BYTE),
    Format(GL_RG8,                  GL_RG,                  GL_UNSIGNED_BYTE),
    Format(GL_RGB8,                 GL_RGB,                 GL_UNSIGNED_BYTE),
    Format(GL_RGBA8,                GL_RGBA,                GL_UNSIGNED_BYTE),
    Format(GL_R16F,                 GL_RED,                 GL_HALF_FLOAT),
    Format(GL_RG16F,                GL_RG,                  GL_HALF_FLOAT),
    Format(GL_RGB16F,               GL_RGB,                 GL_HALF_FLOAT),
    Format(GL_RGBA16F,              GL_RGBA,                GL_HALF_FLOAT),
    Format(GL_R32F,                 GL_RED,                 GL_HALF_FLOAT),
    Format(GL_RG32F,                GL_RG,                  GL_HALF_FLOAT),
    Format(GL_RGB32F,               GL_RGB,                 GL_HALF_FLOAT),
    Format(GL_RGBA32F,              GL_RGBA,                GL_HALF_FLOAT),
    Format(GL_R16I,                 GL_RED_INTEGER,         GL_INT),
    Format(GL_RG16I,                GL_RG_INTEGER,          GL_INT),
    Format(GL_RGB16I,               GL_RGB_INTEGER,         GL_INT),
    Format(GL_RGBA16I,              GL_RGBA_INTEGER,        GL_INT),
    Format(GL_R32I,                 GL_RED_INTEGER,         GL_INT),
    Format(GL_RG32I,                GL_RG_INTEGER,          GL_INT),
    Format(GL_RGB32I,               GL_RGB_INTEGER,         GL_INT),
    Format(GL_RGBA32I,              GL_RGBA_INTEGER,        GL_INT),
    Format(GL_R16UI,                GL_RED_INTEGER,         GL_UNSIGNED_INT),
    Format(GL_RG16UI,               GL_RG_INTEGER,          GL_UNSIGNED_INT),
    Format(GL_RGB16UI,              GL_RGB_INTEGER,         GL_UNSIGNED_INT),
    Format(GL_RGBA16UI,             GL_RGBA_INTEGER,        GL_UNSIGNED_INT),
    Format(GL_R32UI,                GL_RED_INTEGER,         GL_UNSIGNED_INT),
    Format(GL_RG32UI,               GL_RG_INTEGER,          GL_UNSIGNED_INT),
    Format(GL_RGB32UI,              GL_RGB_INTEGER,         GL_UNSIGNED_INT),
    Format(GL_RGBA32UI,             GL_RGBA_INTEGER,        GL_UNSIGNED_INT),
    Format(GL_RGB8,                 GL_BGR,                 GL_UNSIGNED_BYTE),
    Format(GL_RGBA8,                GL_BGRA,                GL_UNSIGNED_BYTE),
    Format(GL_RGB10_A2,             GL_RGBA,                GL_UNSIGNED_INT_10_10_10_2),
    Format(GL_R11F_G11F_B10F,       GL_RGB,                 GL_UNSIGNED_INT_10F_11F_11F_REV),

    // Depth formats
    Format(GL_DEPTH_COMPONENT16,    GL_DEPTH_COMPONENT,     GL_UNSIGNED_INT),
    Format(GL_DEPTH_COMPONENT24,    GL_DEPTH_COMPONENT,     GL_UNSIGNED_INT),
    Format(GL_DEPTH_COMPONENT32,    GL_DEPTH_COMPONENT,     GL_UNSIGNED_INT),
    Format(GL_DEPTH_COMPONENT32F,   GL_DEPTH_COMPONENT,     GL_FLOAT),

    // Combined depth-stencil formats
    Format(GL_DEPTH24_STENCIL8,     GL_DEPTH_STENCIL,       GL_UNSIGNED_INT_24_8),
    Format(GL_DEPTH32F_STENCIL8,    GL_DEPTH_STENCIL,       GL_FLOAT_32_UNSIGNED_INT_24_8_REV)
};

const size_t kNumFormats = sizeof(kFormats) / sizeof(kFormats[0]);


Format gl::mapFormat(EFormat f)
{
    // Invalid format specifier?
    assert( 0 <= f && f < kNumFormats );

    return kFormats[static_cast<size_t>(f)];
}


bool gl::isColorFormat(EFormat f)
{
    return EFormat_R8 <= f && f <= EFormat_R11F_G11F_B10F;
}


bool gl::isDepthFormat(EFormat f)
{
    return EFormat_DEPTH_COMPONENT16 <= f && f <= EFormat_DEPTH32F_STENCIL8;
}


bool gl::isDepthStencilFormat(EFormat f)
{
    return EFormat_DEPTH24_STENCIL8 <= f && f <= EFormat_DEPTH32F_STENCIL8;
}
