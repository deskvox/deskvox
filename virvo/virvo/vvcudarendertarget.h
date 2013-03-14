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


#ifndef VV_CUDA_RENDER_TARGET_H
#define VV_CUDA_RENDER_TARGET_H


#include "vvrendertarget.h"

#include <memory>


namespace virvo
{


    //----------------------------------------------------------------------------------------------
    // PixelUnpackBufferRT
    //----------------------------------------------------------------------------------------------
    class PixelUnpackBufferRT : public RenderTarget
    {
        PixelUnpackBufferRT(unsigned ColorBits = 32, unsigned DepthBits = 8);

    public:
        // Construct a render target
        static VVAPI RenderTarget* create(unsigned ColorBits = 32, unsigned DepthBits = 8);

        // Clean up
        VVAPI virtual ~PixelUnpackBufferRT();

        // Returns the precision of the color buffer
        VVAPI virtual unsigned colorBits() const;

        // Returns the precision of the depth buffer
        VVAPI virtual unsigned depthBits() const;

        // Returns a pointer to the device color buffer
        VVAPI virtual void* deviceColor();

        // Returns a pointer to the device depth buffer
        VVAPI virtual void* deviceDepth();

        // Returns a pointer to the host depth buffer
        VVAPI virtual void const* hostDepth() const;

        // Returns the pixel-unpack buffer
        VVAPI GLuint buffer() const;

        // Returns the texture
        VVAPI GLuint texture() const;

    private:
        virtual bool BeginFrameImpl(unsigned clearMask);
        virtual bool EndFrameImpl();
        virtual bool ResizeImpl(int w, int h);

        virtual bool DisplayColorBufferImpl() const;

        virtual bool DownloadColorBufferImpl(std::vector<unsigned char>& buffer) const;
        virtual bool DownloadDepthBufferImpl(std::vector<unsigned char>& buffer) const;

        // (Re-)create the render buffers (but not the depth buffer)
        bool CreateGLBuffers(int w, int h, bool linearInterpolation = false);

    private:
        struct Impl;
        std::auto_ptr<Impl> impl;
    };


    //----------------------------------------------------------------------------------------------
    // DeviceBufferRT
    //----------------------------------------------------------------------------------------------
    class DeviceBufferRT : public RenderTarget
    {
        DeviceBufferRT(unsigned ColorBits = 32, unsigned DepthBits = 8);

    public:
        // Construct a render target
        static VVAPI RenderTarget* create(unsigned ColorBits = 32, unsigned DepthBits = 8);

        // Clean up
        VVAPI virtual ~DeviceBufferRT();

        // Returns the precision of the color buffer
        VVAPI virtual unsigned colorBits() const;

        // Returns the precision of the depth buffer
        VVAPI virtual unsigned depthBits() const;

        // Returns a pointer to the device color buffer
        VVAPI virtual void* deviceColor();

        // Returns a pointer to the device depth buffer
        VVAPI virtual void* deviceDepth();

        // Returns a pointer to the host color buffer
        VVAPI virtual void const* hostColor() const;

        // Returns a pointer to the host depth buffer
        VVAPI virtual void const* hostDepth() const;

        // Returns the size of the color buffer in bytes
        VVAPI unsigned getColorBufferSize() const;

        // Returns the size of the depth buffer in bytes
        VVAPI unsigned getDepthBufferSize() const;

    private:
        virtual bool BeginFrameImpl(unsigned clearMask);
        virtual bool EndFrameImpl();
        virtual bool ResizeImpl(int w, int h);

        virtual bool DisplayColorBufferImpl() const;

        virtual bool DownloadColorBufferImpl(std::vector<unsigned char>& buffer) const;
        virtual bool DownloadDepthBufferImpl(std::vector<unsigned char>& buffer) const;

    private:
        struct Impl;
        std::auto_ptr<Impl> impl;
    };


} // namespace virvo


#endif
