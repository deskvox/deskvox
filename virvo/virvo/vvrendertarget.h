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


#ifndef VV_RENDER_TARGET_H
#define VV_RENDER_TARGET_H


#include "vvexport.h"

#include <vector>

#include "gl/handle.h"
#include "gl/format.h"


namespace virvo
{


    //----------------------------------------------------------------------------------------------
    // RenderTarget
    //----------------------------------------------------------------------------------------------
    class RenderTarget
    {
    public:
        RenderTarget()
            : Width(0)
            , Height(0)
            , Bound(false)
        {
        }

        RenderTarget(int Width, int Height)
            : Width(Width)
            , Height(Height)
            , Bound(false)
        {
        }

        VVAPI virtual ~RenderTarget();

        // Returns the width of the render target
        int width() const { return Width; }

        // Returns the height of the render target
        int height() const { return Height; }

        // Returns whether the render-target is currently bound for rendering
        bool bound() const { return Bound; }

        // Prepare for rendering
        VVAPI bool beginFrame();

        // Finish rendering
        VVAPI bool endFrame();

        // Resize the render target
        VVAPI bool resize(int w, int h);

        // Returns a pointer to the device color buffer - if any
        virtual void* deviceColor() { return 0; }

        // Returns a pointer to the device depth buffer - if any
        virtual void* deviceDepth() { return 0; }

        // Returns a pointer to the host color buffer - if any
        virtual void* hostColor() { return 0; }

        // Returns a pointer to the host depth buffer - if any
        virtual void* hostDepth() { return 0; }

    private:
        virtual bool BeginFrameImpl() = 0;
        virtual bool EndFrameImpl() = 0;
        virtual bool ResizeImpl(int w, int h) = 0;

    private:
        // The width of the render-target
        int Width;
        // The height of the render-target
        int Height;
        // Whether the render-target is currently bound for rendering
        bool Bound;
    };


    //----------------------------------------------------------------------------------------------
    // DefaultFramebufferRT
    //----------------------------------------------------------------------------------------------
    class DefaultFramebufferRT : public RenderTarget
    {
    public:
        VVAPI virtual ~DefaultFramebufferRT();

    private:
        VVAPI virtual bool BeginFrameImpl();
        VVAPI virtual bool EndFrameImpl();
        VVAPI virtual bool ResizeImpl(int /*w*/, int /*h*/);
    };


    //----------------------------------------------------------------------------------------------
    // FramebufferObjectRT
    //----------------------------------------------------------------------------------------------
    class FramebufferObjectRT : public RenderTarget
    {
    public:
        // Construct a new framebuffer object
        //
        // NOTE: The actual framebuffer object (and the color- and depth-buffer) is constructed
        // when resize() is called for the first time.
        VVAPI FramebufferObjectRT(
            gl::EFormat ColorBufferFormat = gl::EFormat_RGBA8,
            gl::EFormat DepthBufferFormat = gl::EFormat_DEPTH24_STENCIL8
            );

        VVAPI virtual ~FramebufferObjectRT();

        // Returns the color buffer format
        gl::EFormat colorBufferFormat() { return ColorBufferFormat; }

        // Returns the depth(-stencil) buffer format
        gl::EFormat depthBufferFormat() { return DepthBufferFormat; }

        // Returns the framebuffer object
        GLuint framebuffer() { return Framebuffer.get(); }

        // Returns the color texture
        GLuint colorTexture() { return ColorBuffer.get(); }

        // Returns the depth(-stencil) renderbuffer
        GLuint depthRenderbuffer() { return DepthBuffer.get(); }

        // Render the color buffer into the current draw buffer
        void displayColorBuffer() const;

    private:
        VVAPI virtual bool BeginFrameImpl();
        VVAPI virtual bool EndFrameImpl();
        VVAPI virtual bool ResizeImpl(int w, int h);

    private:
        // Color buffer format
        gl::EFormat ColorBufferFormat;
        // Depth buffer format
        gl::EFormat DepthBufferFormat;
        // The framebuffer object
        gl::Framebuffer Framebuffer;
        // Color buffer
        gl::Texture ColorBuffer;
        // Depth buffer
        gl::Renderbuffer DepthBuffer;
    };


    //----------------------------------------------------------------------------------------------
    // HostBufferRT
    //----------------------------------------------------------------------------------------------
    class HostBufferRT : public RenderTarget
    {
    public:
        // Construct a render target
        VVAPI HostBufferRT(unsigned ColorBits = 32, unsigned DepthBits = 8);

        // Clean up
        VVAPI virtual ~HostBufferRT();

        // Returns the precision of the color buffer
        unsigned colorBits() const { return ColorBits; }

        // Returns the precision of the depth buffer
        unsigned depthBits() const { return DepthBits; }

        // Returns a pointer to the device color buffer - if any
        virtual void* deviceColor() { return &ColorBuffer[0]; }

        // Returns a pointer to the device depth buffer - if any
        virtual void* deviceDepth() { return &DepthBuffer[0]; }

        // Returns a pointer to the host color buffer - if any
        virtual void* hostColor() { return &ColorBuffer[0]; }

        // Returns a pointer to the host depth buffer - if any
        virtual void* hostDepth() { return &DepthBuffer[0]; }

    private:
        VVAPI virtual bool BeginFrameImpl();
        VVAPI virtual bool EndFrameImpl();
        VVAPI virtual bool ResizeImpl(int w, int h);

    private:
        static unsigned ComputeBufferSize(unsigned w, unsigned h, unsigned bits) {
            return w * h * (bits / 8);
        }

        // The precision of the color buffer
        unsigned ColorBits;
        // The precision of the depth buffer
        unsigned DepthBits;
        // The color buffer
        std::vector<unsigned char> ColorBuffer;
        // The depth buffer
        std::vector<unsigned char> DepthBuffer;
    };


} // namespace virvo


#endif
