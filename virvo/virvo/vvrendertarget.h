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


    enum ClearMask
    {
        CLEAR_NONE      = 0,
        CLEAR_DEPTH     = 0x00000100, // = GL_DEPTH_BUFFER_BIT
        CLEAR_ACCUM     = 0x00000200, // = GL_ACCUM_BUFFER_BIT
        CLEAR_STENCIL   = 0x00000400, // = GL_STENCIL_BUFFER_BIT
        CLEAR_COLOR     = 0x00004000, // = GL_COLOR_BUFFER_BIT
        CLEAR_ALL       = CLEAR_COLOR | CLEAR_DEPTH | CLEAR_STENCIL | CLEAR_ACCUM
    };


    //----------------------------------------------------------------------------------------------
    // RenderTarget
    //----------------------------------------------------------------------------------------------
    class RenderTarget
    {
    protected:
        RenderTarget();

    public:
        virtual ~RenderTarget();

        // Returns the width of the render target
        int width() const { return Width; }

        // Returns the height of the render target
        int height() const { return Height; }

        // Returns whether the render-target is currently bound for rendering
        bool bound() const { return Bound; }

        // Prepare for rendering
        VVAPI bool beginFrame(unsigned clearMask = 0);

        // Finish rendering
        VVAPI bool endFrame();

        // Resize the render target
        VVAPI bool resize(int w, int h);

        // Render the color buffer into the current draw buffer
        // NOTE: Must not be supported by all render targets
        VVAPI bool displayColorBuffer() const;

        // Copy the color buffer into the given vector
        VVAPI bool downloadColorBuffer(std::vector<unsigned char>& buffer) const;

        // Copy the depth buffer into the given vector
        VVAPI bool downloadDepthBuffer(std::vector<unsigned char>& buffer) const;

        // Returns the number of bits per pixel in the color buffer
        virtual unsigned colorBits() const { return 0; }

        // Returns the number of bits per pixel in the depth buffer
        virtual unsigned depthBits() const { return 0; }

        // Returns a pointer to the device color buffer - if any
        virtual void* deviceColor() { return 0; }

        // Returns a pointer to the device depth buffer - if any
        virtual void* deviceDepth() { return 0; }

        // Returns a pointer to the host color buffer - if any
        virtual void* hostColor() { return 0; }

        // Returns a pointer to the host depth buffer - if any
        virtual void* hostDepth() { return 0; }

    private:
        virtual bool BeginFrameImpl(unsigned clearMask) = 0;
        virtual bool EndFrameImpl() = 0;
        virtual bool ResizeImpl(int w, int h) = 0;

        virtual bool DisplayColorBufferImpl() const = 0;

        virtual bool DownloadColorBufferImpl(std::vector<unsigned char>& buffer) const = 0;
        virtual bool DownloadDepthBufferImpl(std::vector<unsigned char>& buffer) const = 0;

    private:
        // The width of the render-target
        int Width;
        // The height of the render-target
        int Height;
        // Whether the render-target is currently bound for rendering
        bool Bound;
    };


    //----------------------------------------------------------------------------------------------
    // NullRT
    //----------------------------------------------------------------------------------------------
    class NullRT : public RenderTarget
    {
        NullRT();

    public:
        static VVAPI RenderTarget* create();

        VVAPI virtual ~NullRT();

    private:
        virtual bool BeginFrameImpl(unsigned clearMask);
        virtual bool EndFrameImpl();
        virtual bool ResizeImpl(int /*w*/, int /*h*/);

        virtual bool DisplayColorBufferImpl() const;

        virtual bool DownloadColorBufferImpl(std::vector<unsigned char>& buffer) const;
        virtual bool DownloadDepthBufferImpl(std::vector<unsigned char>& buffer) const;
    };


    //----------------------------------------------------------------------------------------------
    // FramebufferObjectRT
    //----------------------------------------------------------------------------------------------
    class FramebufferObjectRT : public RenderTarget
    {
        FramebufferObjectRT(gl::EFormat ColorBufferFormat, gl::EFormat DepthBufferFormat);

    public:
        // Construct a new framebuffer object
        static VVAPI RenderTarget* create(gl::EFormat ColorBufferFormat = gl::EFormat_RGBA8,
                                          gl::EFormat DepthBufferFormat = gl::EFormat_DEPTH24_STENCIL8);

        VVAPI virtual ~FramebufferObjectRT();

        // Returns the number of bits per pixel in the color buffer
        virtual unsigned colorBits() const { return 8 * gl::mapFormat(ColorBufferFormat).sizeInBytes; }

        // Returns the number of bits per pixel in the depth buffer
        virtual unsigned depthBits() const { return 8 * gl::mapFormat(DepthBufferFormat).sizeInBytes; }

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

    private:
        virtual bool BeginFrameImpl(unsigned clearMask);
        virtual bool EndFrameImpl();
        virtual bool ResizeImpl(int w, int h);

        virtual bool DisplayColorBufferImpl() const;

        virtual bool DownloadColorBufferImpl(std::vector<unsigned char>& buffer) const;
        virtual bool DownloadDepthBufferImpl(std::vector<unsigned char>& buffer) const;

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
        HostBufferRT(unsigned ColorBits, unsigned DepthBits);

    public:
        // Construct a render target
        static VVAPI RenderTarget* create(unsigned ColorBits = 32, unsigned DepthBits = 8);

        // Clean up
        VVAPI virtual ~HostBufferRT();

        // Returns the precision of the color buffer
        virtual unsigned colorBits() const { return ColorBits; }

        // Returns the precision of the depth buffer
        virtual unsigned depthBits() const { return DepthBits; }

        // Returns a pointer to the device color buffer - if any
        virtual void* deviceColor() { return &ColorBuffer[0]; }

        // Returns a pointer to the device depth buffer - if any
        virtual void* deviceDepth() { return &DepthBuffer[0]; }

        // Returns a pointer to the host color buffer - if any
        virtual void* hostColor() { return &ColorBuffer[0]; }

        // Returns a pointer to the host depth buffer - if any
        virtual void* hostDepth() { return &DepthBuffer[0]; }

    private:
        virtual bool BeginFrameImpl(unsigned clearMask);
        virtual bool EndFrameImpl();
        virtual bool ResizeImpl(int w, int h);

        virtual bool DisplayColorBufferImpl() const;

        virtual bool DownloadColorBufferImpl(std::vector<unsigned char>& buffer) const;
        virtual bool DownloadDepthBufferImpl(std::vector<unsigned char>& buffer) const;

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
