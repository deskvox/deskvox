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


#include <GL/glew.h>

#include <assert.h>

#include <stdexcept>

namespace gl = virvo::gl;


using virvo::PixelUnpackBufferRT;
using virvo::DeviceBufferRT;


//--------------------------------------------------------------------------------------------------
// PixelUnpackBufferRT
//--------------------------------------------------------------------------------------------------


PixelUnpackBufferRT::PixelUnpackBufferRT(unsigned ColorBits, unsigned DepthBits)
    : ColorBits(ColorBits)
    , DepthBits(DepthBits)
    , Buffer(0)
    , Texture(0)
{
}


PixelUnpackBufferRT::~PixelUnpackBufferRT()
{
}


bool PixelUnpackBufferRT::BeginFrameImpl()
{
    return Resource.map() != 0;
}


bool PixelUnpackBufferRT::EndFrameImpl()
{
    Resource.unmap();

    // Download the depth buffer
    if (!DepthBuffer.download())
        return false;

    // Bind the pixel-unpack buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Buffer.get());

    // Copy the buffer data into the texture
    glBindTexture(GL_TEXTURE_2D, Texture.get());

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width(), height(), GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, 0);

#if 0
    // Render the texture into the current frame buffer
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 1.0f);
    glEnd();
#endif

    return glGetError() == GL_NO_ERROR;
}


bool PixelUnpackBufferRT::ResizeImpl(int w, int h)
{
    // Resize the depth buffer
    if (DepthBits != 0)
    {
        if (!DepthBuffer.resize(ComputeBufferSize(w, h, DepthBits)))
            return false;
    }

    // Create an OpenGL buffer if not already done...
    if (Buffer.get() == 0)
        Buffer.reset(gl::createBuffer());

    // Reallocate the buffer storage
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Buffer.get());
    glBufferData(GL_PIXEL_UNPACK_BUFFER, ComputeBufferSize(w, h, ColorBits), 0, GL_STREAM_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Create an OpenGL texture if not already done...
    if (Texture.get() == 0)
        Texture.reset(gl::createTexture());

    // Reallocate the texture storage
    glBindTexture(GL_TEXTURE_2D, Texture.get());
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, 0);

    return glGetError() == GL_NO_ERROR;
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


bool DeviceBufferRT::BeginFrameImpl()
{
    return true;
}


bool DeviceBufferRT::EndFrameImpl()
{
    return ColorBuffer.download() && DepthBuffer.download();
}


bool DeviceBufferRT::ResizeImpl(int w, int h)
{
    ColorBuffer.clear();
    DepthBuffer.clear();

    if (!ColorBuffer.resize(ComputeBufferSize(w, h, ColorBits)))
        return false;

    if (DepthBits != 0 && !DepthBuffer.resize(ComputeBufferSize(w, h, DepthBits)))
    {
        ColorBuffer.clear();
        return false;
    }

    return true;
}


#endif // HAVE_CUDA
