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


#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif


#ifdef HAVE_CUDA


#include "vvrendertarget.h"

#include "cuda/graphics_resource.h"
#include "cuda/host_device_array.h"

#include "gl/handle.h"


namespace virvo
{


    //----------------------------------------------------------------------------------------------
    // PixelUnpackBufferRT
    //----------------------------------------------------------------------------------------------
    class PixelUnpackBufferRT : public RenderTarget
    {
    public:
        // Construct a render target
        VVAPI PixelUnpackBufferRT(unsigned ColorBits = 32, unsigned DepthBits = 8);

        // Clean up
        VVAPI virtual ~PixelUnpackBufferRT();

        // Returns the precision of the color buffer
        unsigned colorBits() const { return ColorBits; }

        // Returns the precision of the depth buffer
        unsigned depthBits() const { return DepthBits; }

        // Returns the pixel-unpack buffer
        GLuint buffer() const { return Buffer.get(); }

        // Returns the texture
        GLuint texture() const { return Texture.get(); }

        // Returns a pointer to the device color buffer
        virtual void* deviceColor() { return Resource.devPtr(); }

        // Returns a pointer to the device depth buffer
        virtual void* deviceDepth() { return DepthBuffer.devicePtr(); }

        // Returns a pointer to the host depth buffer
        virtual void* hostDepth() { return DepthBuffer.hostPtr(); }

        // Render the color buffer into the current draw buffer
        void displayColorBuffer() const;

    private:
        VVAPI virtual bool BeginFrameImpl(unsigned clearMask);
        VVAPI virtual bool EndFrameImpl();
        VVAPI virtual bool ResizeImpl(int w, int h);

        // (Re-)create the render buffers (but not the depth buffer)
        bool CreateGLBuffers(int w, int h, bool linearInterpolation = false);

    private:
        static unsigned ComputeBufferSize(unsigned w, unsigned h, unsigned bits) {
            return w * h * (bits / 8);
        }

        // The precision of the color buffer
        unsigned ColorBits;
        // The precision of the depth buffer
        unsigned DepthBits;
        // The CUDA graphics resource
        cuda::GraphicsResource Resource;
        // The OpenGL buffer object
        gl::Buffer Buffer;
        // The OpenGL texture object
        gl::Texture Texture;
        // The depth buffer
        cuda::HostDeviceArray DepthBuffer;
    };


    //----------------------------------------------------------------------------------------------
    // DeviceBufferRT
    //----------------------------------------------------------------------------------------------
    class DeviceBufferRT : public RenderTarget
    {
    public:
        // Construct a render target
        VVAPI DeviceBufferRT(unsigned ColorBits = 32, unsigned DepthBits = 8);

        // Clean up
        VVAPI virtual ~DeviceBufferRT();

        // Returns the precision of the color buffer
        unsigned colorBits() const { return ColorBits; }

        // Returns the precision of the depth buffer
        unsigned depthBits() const { return DepthBits; }

        // Returns the size of the color buffer in bytes
        unsigned getColorBufferSize() const {
            return ComputeBufferSize(width(), height(), ColorBits);
        }

        // Returns the size of the depth buffer in bytes
        unsigned getDepthBufferSize() const {
            return ComputeBufferSize(width(), height(), DepthBits);
        }

        // Returns a pointer to the device color buffer
        virtual void* deviceColor() { return ColorBuffer.devicePtr(); }

        // Returns a pointer to the device depth buffer
        virtual void* deviceDepth() { return DepthBuffer.devicePtr(); }

        // Returns a pointer to the host color buffer
        virtual void* hostColor() { return ColorBuffer.hostPtr(); }

        // Returns a pointer to the host depth buffer
        virtual void* hostDepth() { return DepthBuffer.hostPtr(); }

    private:
        VVAPI virtual bool BeginFrameImpl(unsigned clearMask);
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
        cuda::HostDeviceArray ColorBuffer;
        // The depth buffer
        cuda::HostDeviceArray DepthBuffer;
    };


} // namespace virvo


#endif // HAVE_CUDA


#endif
