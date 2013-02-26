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


#include "vvrendertarget.h"
#include "gl/util.h"

#include <GL/glew.h>

#include <assert.h>


namespace gl = virvo::gl;

using virvo::RenderTarget;
using virvo::DefaultFramebufferRT;
using virvo::FramebufferObjectRT;
using virvo::HostBufferRT;


//--------------------------------------------------------------------------------------------------
// RenderTarget
//--------------------------------------------------------------------------------------------------


RenderTarget::RenderTarget()
    : Width(0)
    , Height(0)
    , Bound(false)
{
}


RenderTarget::RenderTarget(int Width, int Height)
    : Width(Width)
    , Height(Height)
    , Bound(false)
{
}


RenderTarget::~RenderTarget()
{
}


bool RenderTarget::beginFrame(unsigned clearMask)
{
    assert( Bound == false && "already bound" );

    Bound = BeginFrameImpl(clearMask);
    return Bound;
}


bool RenderTarget::endFrame()
{
    assert( Bound == true && "not bound" );

    bool Success = EndFrameImpl();

    Bound = false;

    return Success;
}


bool RenderTarget::resize(int w, int h)
{
    assert( Bound == false && "resize while bound" );

    if (Width == w && Height == h)
        return true;

    bool Success = ResizeImpl(w, h);

    if (Success)
    {
        Width = w;
        Height = h;
    }

    return Success;
}


//--------------------------------------------------------------------------------------------------
// DefaultFramebufferRT
//--------------------------------------------------------------------------------------------------


DefaultFramebufferRT::~DefaultFramebufferRT()
{
}


bool DefaultFramebufferRT::BeginFrameImpl(unsigned clearMask)
{
#if 0
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0);
#endif
    glClear(clearMask);
    return true;
}


bool DefaultFramebufferRT::EndFrameImpl()
{
    return true;
}


bool DefaultFramebufferRT::ResizeImpl(int /*w*/, int /*h*/)
{
    return true;
}


//--------------------------------------------------------------------------------------------------
// FramebufferObjectRT
//--------------------------------------------------------------------------------------------------


FramebufferObjectRT::FramebufferObjectRT(gl::EFormat ColorBufferFormat, gl::EFormat DepthBufferFormat)
    : ColorBufferFormat(ColorBufferFormat)
    , DepthBufferFormat(DepthBufferFormat)
{
    assert( gl::isColorFormat(ColorBufferFormat) );
    assert( DepthBufferFormat == gl::EFormat_Unspecified || gl::isDepthFormat(DepthBufferFormat) );
}


FramebufferObjectRT::~FramebufferObjectRT()
{
}


bool FramebufferObjectRT::BeginFrameImpl(unsigned clearMask)
{
    assert( Framebuffer.get() != 0 );

    // Save current viewport
    glPushAttrib(GL_VIEWPORT_BIT);

    // Bind the framebuffer for rendering
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, Framebuffer.get());

    // Set the viewport
    glViewport(0, 0, width(), height());

    // Clear the render targets
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0);
    glClear(clearMask);

    return true;
}


bool FramebufferObjectRT::EndFrameImpl()
{
    // Unbind the framebuffer
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    // Restore the viewport
    glPopAttrib();

    return true;
}


bool FramebufferObjectRT::ResizeImpl(int w, int h)
{
    gl::Format cf = gl::mapFormat(ColorBufferFormat);
    gl::Format df = gl::mapFormat(DepthBufferFormat);

    // Delete current color and depth buffers
    ColorBuffer.reset();
    DepthBuffer.reset();

    //
    // Create the framebuffer object (if not already done...)
    //

    if (Framebuffer.get() == 0)
        Framebuffer.reset( gl::createFramebuffer() );

    glBindFramebuffer(GL_FRAMEBUFFER, Framebuffer.get());

    //
    // Create the color-buffer
    //

    ColorBuffer.reset( gl::createTexture() );

    glBindTexture(GL_TEXTURE_2D, ColorBuffer.get());

    // Initialize texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, cf.internalFormat, w, h, 0, cf.format, cf.type, 0);

    if (VV_GET_GL_ERROR() != GL_NO_ERROR)
        return false;

    // Attach to framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ColorBuffer.get(), 0);

    if (VV_GET_GL_ERROR() != GL_NO_ERROR)
        return false;

    //
    // Create the depth-buffer
    //

    if (DepthBufferFormat != gl::EFormat_Unspecified)
    {
        DepthBuffer.reset( gl::createRenderbuffer() );

        glBindRenderbuffer(GL_RENDERBUFFER, DepthBuffer.get());

        glRenderbufferStorage(GL_RENDERBUFFER, df.internalFormat, w, h);

        if (VV_GET_GL_ERROR() != GL_NO_ERROR)
            return false;

        GLenum attachment =
            gl::isDepthStencilFormat(DepthBufferFormat)
                ? GL_DEPTH_STENCIL_ATTACHMENT
                : GL_DEPTH_ATTACHMENT;

        // Attach as depth (and stencil) target
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, DepthBuffer.get());

        if (VV_GET_GL_ERROR() != GL_NO_ERROR)
            return false;
    }

    //
    // Check for errors
    //

    GLenum status = VV_GET_FRAMEBUFFER_STATUS(GL_FRAMEBUFFER);

    // Unbind the framebuffer object!!!
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return status == GL_FRAMEBUFFER_COMPLETE;
}


void FramebufferObjectRT::displayColorBuffer() const
{
    gl::blendTexture(ColorBuffer.get());
}


//--------------------------------------------------------------------------------------------------
// HostBufferRT
//--------------------------------------------------------------------------------------------------


HostBufferRT::HostBufferRT(unsigned ColorBits, unsigned DepthBits)
    : ColorBits(ColorBits)
    , DepthBits(DepthBits)
{
}


HostBufferRT::~HostBufferRT()
{
}


bool HostBufferRT::BeginFrameImpl(unsigned clearMask)
{
    if (clearMask & CLEAR_COLOR)
        std::fill(ColorBuffer.begin(), ColorBuffer.end(), 0);

    if (clearMask & CLEAR_DEPTH)
        std::fill(DepthBuffer.begin(), DepthBuffer.end(), 0);

    return true;
}


bool HostBufferRT::EndFrameImpl()
{
    return true;
}


bool HostBufferRT::ResizeImpl(int w, int h)
{
    ColorBuffer.resize(ComputeBufferSize(w, h, ColorBits));

    if (DepthBits > 0)
        DepthBuffer.resize(ComputeBufferSize(w, h, DepthBits));

    return true;
}


void HostBufferRT::displayColorBuffer() const
{
    gl::blendPixels(width(), height(), GL_RGBA, GL_FLOAT, &ColorBuffer[0]);
}
