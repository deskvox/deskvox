// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2012 University of Stuttgart, 2004-2005 Brown University
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


#include "vvcudarendertarget.h"


#ifdef HAVE_CUDA


#include "vvcuda.h"
#include "gl/util.h"

#include <GL/glew.h>

#include <assert.h>
#include <stdio.h>

#include <stdexcept>


namespace gl = virvo::gl;


using virvo::PixelUnpackBufferRT;
using virvo::DeviceBufferRT;


//--------------------------------------------------------------------------------------------------
// PixelUnpackBufferRT
//--------------------------------------------------------------------------------------------------


static GLenum GetInternalFormat(unsigned bits)
{
    switch (bits / 8)
    {
    case 1: return GL_R8;
    case 2: return GL_RG8;
    case 3: return GL_RGB8;
    case 4: return GL_RGBA8;
    }

    assert(!"bit depth not supported");
    return 0;
}

static GLenum GetFormat(unsigned bits)
{
    switch (bits / 8)
    {
    case 1: return GL_RED;
    case 2: return GL_RG;
    case 3: return GL_RGB;
    case 4: return GL_RGBA;
    }

    assert(!"bit depth not supported");
    return 0;
}


PixelUnpackBufferRT::PixelUnpackBufferRT(unsigned ColorBits, unsigned DepthBits)
    : ColorBits(ColorBits)
    , DepthBits(DepthBits)
    , Buffer(0)
    , Texture(0)
{
    if (!vvCuda::initGlInterop())
        throw std::runtime_error("Could not initialize CUDA-GL interop");
}


PixelUnpackBufferRT::~PixelUnpackBufferRT()
{
}


bool PixelUnpackBufferRT::BeginFrameImpl(unsigned clearMask)
{
    if (clearMask & CLEAR_COLOR)
    {
    }

    if (clearMask & CLEAR_DEPTH)
    {
    }

    return Resource.map() != 0;
}


bool PixelUnpackBufferRT::EndFrameImpl()
{
    Resource.unmap();

    // Download the depth buffer
    if (!DepthBuffer.download())
        return false;

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glBindTexture(GL_TEXTURE_2D, Texture.get());

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Buffer.get());

    // Copy the buffer data into the texture
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width(), height(), GetFormat(ColorBits), GL_UNSIGNED_BYTE, 0);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (GL_NO_ERROR != VV_GET_GL_ERROR())
        return false;

    return true;
}


bool PixelUnpackBufferRT::ResizeImpl(int w, int h)
{
    if (DepthBits != 0 && !DepthBuffer.resize(ComputeBufferSize(w, h, DepthBits)))
        return false;

    return CreateGLBuffers(w, h);
}


bool PixelUnpackBufferRT::CreateGLBuffers(int w, int h, bool linearInterpolation)
{
    // Create the texture
    Texture.reset( gl::createTexture() );

    glBindTexture(GL_TEXTURE_2D, Texture.get());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, GetInternalFormat(ColorBits), w, h, 0, GetFormat(ColorBits), GL_UNSIGNED_BYTE, 0);

    if (GL_NO_ERROR != VV_GET_GL_ERROR())
        return false;

    // Create the buffer
    Buffer.reset( gl::createBuffer() );

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Buffer.get());
    glBufferData(GL_PIXEL_UNPACK_BUFFER, ComputeBufferSize(w, h, ColorBits), 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (GL_NO_ERROR != VV_GET_GL_ERROR())
        return false;

    // Register the buffer object for use with CUDA
    if (!Resource.registerBuffer(Buffer.get(), cudaGraphicsRegisterFlagsWriteDiscard))
        return false;

    return true;
}


void PixelUnpackBufferRT::displayColorBuffer() const
{
    gl::blendTexture(Texture.get());
}


//--------------------------------------------------------------------------------------------------
// DeviceBufferRT
//--------------------------------------------------------------------------------------------------


DeviceBufferRT::DeviceBufferRT(unsigned ColorBits, unsigned DepthBits)
    : ColorBits(ColorBits)
    , DepthBits(DepthBits)
{
}


DeviceBufferRT::~DeviceBufferRT()
{
}


bool DeviceBufferRT::BeginFrameImpl(unsigned clearMask)
{
    if (clearMask & CLEAR_COLOR)
        ColorBuffer.fill(0);

    if (clearMask & CLEAR_DEPTH)
        DepthBuffer.fill(0);

    return true;
}


bool DeviceBufferRT::EndFrameImpl()
{
    return ColorBuffer.download() && DepthBuffer.download();
}


bool DeviceBufferRT::ResizeImpl(int w, int h)
{
    ColorBuffer.reset();
    DepthBuffer.reset();

    if (!ColorBuffer.resize(ComputeBufferSize(w, h, ColorBits)))
        return false;

    if (DepthBits != 0 && !DepthBuffer.resize(ComputeBufferSize(w, h, DepthBits)))
    {
        ColorBuffer.reset();
        return false;
    }

    return true;
}


#endif // HAVE_CUDA
