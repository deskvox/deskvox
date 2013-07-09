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

#include "cuda/utils.h"

#include "cuda/graphics_resource.h"
#include "cuda/host_device_array.h"

#include "gl/handle.h"
#include "gl/util.h"

#include <GL/glew.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <stdexcept>


namespace cu = virvo::cuda;
namespace gl = virvo::gl;


using virvo::RenderTarget;
using virvo::PixelUnpackBufferRT;
using virvo::DeviceBufferRT;


static unsigned ComputeBufferSize(unsigned w, unsigned h, unsigned bits)
{
    return w * h * (bits / 8);
}


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


//--------------------------------------------------------------------------------------------------
// PixelUnpackBufferRT
//--------------------------------------------------------------------------------------------------


struct PixelUnpackBufferRT::Impl
{
    // The precision of the color buffer
    unsigned ColorBits;
    // The precision of the depth buffer
    unsigned DepthBits;
    // The CUDA graphics resource
    cu::GraphicsResource Resource;
    // The OpenGL buffer object
    gl::Buffer Buffer;
    // The OpenGL texture object
    gl::Texture Texture;
    // The depth buffer
    cu::HostDeviceArray DepthBuffer;

    Impl(unsigned ColorBits, unsigned DepthBits)
        : ColorBits(ColorBits)
        , DepthBits(DepthBits)
    {
    }
};


PixelUnpackBufferRT::PixelUnpackBufferRT(unsigned ColorBits, unsigned DepthBits)
    : impl(new Impl(ColorBits, DepthBits))
{
}


RenderTarget* PixelUnpackBufferRT::create(unsigned ColorBits, unsigned DepthBits)
{
	if (!virvo::cuda::initGlInterop())
		return 0;

    return new PixelUnpackBufferRT(ColorBits, DepthBits);
}


PixelUnpackBufferRT::~PixelUnpackBufferRT()
{
}


unsigned PixelUnpackBufferRT::colorBits() const
{
    return impl->ColorBits;
}


unsigned PixelUnpackBufferRT::depthBits() const
{
    return impl->DepthBits;
}


void* PixelUnpackBufferRT::deviceColor()
{
    return impl->Resource.devPtr();
}


void* PixelUnpackBufferRT::deviceDepth()
{
    return impl->DepthBuffer.devicePtr();
}


void const* PixelUnpackBufferRT::hostDepth() const
{
    return impl->DepthBuffer.hostPtr();
}


GLuint PixelUnpackBufferRT::buffer() const
{
    return impl->Buffer.get();
}


GLuint PixelUnpackBufferRT::texture() const
{
    return impl->Texture.get();
}


bool PixelUnpackBufferRT::BeginFrameImpl(unsigned clearMask)
{
    if (clearMask & CLEAR_COLOR)
    {
    }

    if (clearMask & CLEAR_DEPTH)
    {
    }

    return impl->Resource.map() != 0;
}


bool PixelUnpackBufferRT::EndFrameImpl()
{
    impl->Resource.unmap();

    // Download the depth buffer
    if (!impl->DepthBuffer.download())
        return false;

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glBindTexture(GL_TEXTURE_2D, impl->Texture.get());

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl->Buffer.get());

    // Copy the buffer data into the texture
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width(), height(), GetFormat(impl->ColorBits), GL_UNSIGNED_BYTE, 0);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (GL_NO_ERROR != VV_GET_GL_ERROR())
        return false;

    return true;
}


bool PixelUnpackBufferRT::ResizeImpl(int w, int h)
{
    if (impl->DepthBits != 0 && !impl->DepthBuffer.resize(ComputeBufferSize(w, h, impl->DepthBits)))
        return false;

    return CreateGLBuffers(w, h);
}


bool PixelUnpackBufferRT::CreateGLBuffers(int w, int h, bool linearInterpolation)
{
    // Create the texture
    impl->Texture.reset( gl::createTexture() );

    glBindTexture(GL_TEXTURE_2D, impl->Texture.get());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linearInterpolation ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, GetInternalFormat(impl->ColorBits), w, h, 0, GetFormat(impl->ColorBits), GL_UNSIGNED_BYTE, 0);

    if (GL_NO_ERROR != VV_GET_GL_ERROR())
        return false;

    // Create the buffer
    impl->Buffer.reset( gl::createBuffer() );

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl->Buffer.get());
    glBufferData(GL_PIXEL_UNPACK_BUFFER, ComputeBufferSize(w, h, impl->ColorBits), 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (GL_NO_ERROR != VV_GET_GL_ERROR())
        return false;

    // Register the buffer object for use with CUDA
    if (!impl->Resource.registerBuffer(impl->Buffer.get(), cudaGraphicsRegisterFlagsWriteDiscard))
        return false;

    return true;
}


bool PixelUnpackBufferRT::DisplayColorBufferImpl() const
{
    gl::blendTexture(impl->Texture.get());
    return true;
}


bool PixelUnpackBufferRT::DownloadColorBufferImpl(unsigned char* buffer, size_t bufferSize) const
{
    size_t bytes = ComputeBufferSize(width(), height(), impl->ColorBits);

    assert( bufferSize >= bytes );

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl->Buffer.get());
    glGetBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, bytes, buffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    return GL_NO_ERROR == VV_GET_GL_ERROR();

    static_cast<void>(bufferSize); // unused
}


bool PixelUnpackBufferRT::DownloadDepthBufferImpl(unsigned char* buffer, size_t bufferSize) const
{
    assert( bufferSize >= impl->DepthBuffer.size() );

    memcpy(&buffer[0], impl->DepthBuffer.hostPtr(), impl->DepthBuffer.size() );
    return true;

    static_cast<void>(bufferSize); // unused
}


//--------------------------------------------------------------------------------------------------
// DeviceBufferRT
//--------------------------------------------------------------------------------------------------


struct DeviceBufferRT::Impl
{
    // The precision of the color buffer
    unsigned ColorBits;
    // The precision of the depth buffer
    unsigned DepthBits;
    // The color buffer
    cu::HostDeviceArray ColorBuffer;
    // The depth buffer
    cu::HostDeviceArray DepthBuffer;

    Impl(unsigned ColorBits, unsigned DepthBits)
        : ColorBits(ColorBits)
        , DepthBits(DepthBits)
    {
    }
};


DeviceBufferRT::DeviceBufferRT(unsigned ColorBits, unsigned DepthBits)
    : impl(new Impl(ColorBits, DepthBits))
{
}


RenderTarget* DeviceBufferRT::create(unsigned ColorBits, unsigned DepthBits)
{
    return new DeviceBufferRT(ColorBits, DepthBits);
}


DeviceBufferRT::~DeviceBufferRT()
{
}


unsigned DeviceBufferRT::colorBits() const
{
    return impl->ColorBits;
}


unsigned DeviceBufferRT::depthBits() const
{
    return impl->DepthBits;
}


void* DeviceBufferRT::deviceColor()
{
    return impl->ColorBuffer.devicePtr();
}


void* DeviceBufferRT::deviceDepth()
{
    return impl->DepthBuffer.devicePtr();
}


void const* DeviceBufferRT::hostColor() const
{
    return impl->ColorBuffer.hostPtr();
}


void const* DeviceBufferRT::hostDepth() const
{
    return impl->DepthBuffer.hostPtr();
}


unsigned DeviceBufferRT::getColorBufferSize() const
{
    return ComputeBufferSize(width(), height(), impl->ColorBits);
}


unsigned DeviceBufferRT::getDepthBufferSize() const
{
    return ComputeBufferSize(width(), height(), impl->DepthBits);
}


bool DeviceBufferRT::BeginFrameImpl(unsigned clearMask)
{
    if (clearMask & CLEAR_COLOR)
        impl->ColorBuffer.fill(0);

    if (clearMask & CLEAR_DEPTH)
        impl->DepthBuffer.fill(0);

    return true;
}


bool DeviceBufferRT::EndFrameImpl()
{
    return impl->ColorBuffer.download() && impl->DepthBuffer.download();
}


bool DeviceBufferRT::ResizeImpl(int w, int h)
{
    impl->ColorBuffer.reset();
    impl->DepthBuffer.reset();

    if (!impl->ColorBuffer.resize(ComputeBufferSize(w, h, impl->ColorBits)))
        return false;

    if (impl->DepthBits != 0 && !impl->DepthBuffer.resize(ComputeBufferSize(w, h, impl->DepthBits)))
    {
        impl->ColorBuffer.reset();
        return false;
    }

    return true;
}


bool DeviceBufferRT::DisplayColorBufferImpl() const
{
    gl::blendPixels(width(), height(), GetFormat(impl->ColorBits), GL_UNSIGNED_BYTE, impl->ColorBuffer.hostPtr());
    return true;
}


bool DeviceBufferRT::DownloadColorBufferImpl(unsigned char* buffer, size_t bufferSize) const
{
    assert( bufferSize >= impl->ColorBuffer.size() );

    memcpy(&buffer[0], impl->ColorBuffer.hostPtr(), impl->ColorBuffer.size());
    return true;

    static_cast<void>(bufferSize); // unused
}


bool DeviceBufferRT::DownloadDepthBufferImpl(unsigned char* buffer, size_t bufferSize) const
{
    assert( bufferSize >= impl->DepthBuffer.size() );

    memcpy(&buffer[0], impl->DepthBuffer.hostPtr(), impl->DepthBuffer.size());
    return true;

    static_cast<void>(bufferSize); // unused
}
