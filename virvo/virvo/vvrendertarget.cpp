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


bool DefaultFramebufferRT::BeginFrameImpl(unsigned /*clearMask*/)
{
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


bool FramebufferObjectRT::BeginFrameImpl(unsigned /*clearMask*/)
{
    assert( Framebuffer.get() != 0 );

    // Save current viewport
    glPushAttrib(GL_VIEWPORT_BIT);

    // Bind the framebuffer for rendering
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, Framebuffer.get());

    // Set the viewport
    glViewport(0, 0, width(), height());

    // Clear the render targets
    // XXX: Save the clear-mask as a member in FramebufferObjectRT!!!
    if (gl::isDepthFormat(DepthBufferFormat))
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    else
        glClear(GL_COLOR_BUFFER_BIT);

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

    // Attach to framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ColorBuffer.get(), 0);

    //
    // Create the depth-buffer
    //

    if (DepthBufferFormat != gl::EFormat_Unspecified)
    {
        DepthBuffer.reset( gl::createRenderbuffer() );

        glBindRenderbuffer(GL_RENDERBUFFER, DepthBuffer.get());

        glRenderbufferStorage(GL_RENDERBUFFER, df.internalFormat, w, h);

        GLenum attachment =
            gl::isDepthStencilFormat(DepthBufferFormat)
                ? GL_DEPTH_STENCIL_ATTACHMENT
                : GL_DEPTH_ATTACHMENT;

        // Attach as depth (and stencil) target
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, DepthBuffer.get());
    }

    //
    // Check for errors
    //

    bool success = GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER);

    // Unbind the framebuffer object!!!
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return success;
}


void FramebufferObjectRT::displayColorBuffer() const
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

    glActiveTexture(GL_TEXTURE0);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_TEXTURE);
    glPushMatrix();
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, ColorBuffer.get());

    glDepthMask(GL_FALSE);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();

    glMatrixMode(GL_TEXTURE);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glPopAttrib();
    glPopClientAttrib();
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


bool HostBufferRT::BeginFrameImpl(unsigned /*clearMask*/)
{
    return true;
}


bool HostBufferRT::EndFrameImpl()
{
    return true;
}


bool HostBufferRT::ResizeImpl(int w, int h)
{
    ColorBuffer.resize(ComputeBufferSize(w, h, ColorBits));
    DepthBuffer.resize(ComputeBufferSize(w, h, DepthBits));

    return true;
}
