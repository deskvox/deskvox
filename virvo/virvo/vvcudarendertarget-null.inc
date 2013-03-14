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


using virvo::RenderTarget;
using virvo::PixelUnpackBufferRT;
using virvo::DeviceBufferRT;


//--------------------------------------------------------------------------------------------------
// PixelUnpackBufferRT
//--------------------------------------------------------------------------------------------------


struct PixelUnpackBufferRT::Impl
{
};


PixelUnpackBufferRT::PixelUnpackBufferRT(unsigned /*ColorBits*/, unsigned /*DepthBits*/)
{
}


RenderTarget* PixelUnpackBufferRT::create(unsigned /*ColorBits*/, unsigned /*DepthBits*/)
{
    return 0;
}


PixelUnpackBufferRT::~PixelUnpackBufferRT()
{
}


unsigned PixelUnpackBufferRT::colorBits() const
{
    return 0;
}


unsigned PixelUnpackBufferRT::depthBits() const
{
    return 0;
}


void* PixelUnpackBufferRT::deviceColor()
{
    return 0;
}


void* PixelUnpackBufferRT::deviceDepth()
{
    return 0;
}


void const* PixelUnpackBufferRT::hostDepth() const
{
    return 0;
}


GLuint PixelUnpackBufferRT::buffer() const
{
    return 0;
}


GLuint PixelUnpackBufferRT::texture() const
{
    return 0;
}


bool PixelUnpackBufferRT::BeginFrameImpl(unsigned /*clearMask*/)
{
    return false;
}


bool PixelUnpackBufferRT::EndFrameImpl()
{
    return false;
}


bool PixelUnpackBufferRT::ResizeImpl(int /*w*/, int /*h*/)
{
    return false;
}


bool PixelUnpackBufferRT::CreateGLBuffers(int /*w*/, int /*h*/, bool /*linearInterpolation*/)
{
    return false;
}


bool PixelUnpackBufferRT::DisplayColorBufferImpl() const
{
    return false;
}


bool PixelUnpackBufferRT::DownloadColorBufferImpl(std::vector<unsigned char>& /*buffer*/) const
{
    return false;
}


bool PixelUnpackBufferRT::DownloadDepthBufferImpl(std::vector<unsigned char>& /*buffer*/) const
{
    return false;
}


//--------------------------------------------------------------------------------------------------
// DeviceBufferRT
//--------------------------------------------------------------------------------------------------


struct DeviceBufferRT::Impl
{
};


DeviceBufferRT::DeviceBufferRT(unsigned /*ColorBits*/, unsigned /*DepthBits*/)
{
}


RenderTarget* DeviceBufferRT::create(unsigned /*ColorBits*/, unsigned /*DepthBits*/)
{
    return 0;
}


DeviceBufferRT::~DeviceBufferRT()
{
}


unsigned DeviceBufferRT::colorBits() const
{
    return 0;
}


unsigned DeviceBufferRT::depthBits() const
{
    return 0;
}


void* DeviceBufferRT::deviceColor()
{
    return 0;
}


void* DeviceBufferRT::deviceDepth()
{
    return 0;
}


void const* DeviceBufferRT::hostColor() const
{
    return 0;
}


void const* DeviceBufferRT::hostDepth() const
{
    return 0;
}


unsigned DeviceBufferRT::getColorBufferSize() const
{
    return 0;
}


unsigned DeviceBufferRT::getDepthBufferSize() const
{
    return 0;
}


bool DeviceBufferRT::BeginFrameImpl(unsigned /*clearMask*/)
{
    return false;
}


bool DeviceBufferRT::EndFrameImpl()
{
    return false;
}


bool DeviceBufferRT::ResizeImpl(int /*w*/, int /*h*/)
{
    return false;
}


bool DeviceBufferRT::DisplayColorBufferImpl() const
{
    return false;
}


bool DeviceBufferRT::DownloadColorBufferImpl(std::vector<unsigned char>& /*buffer*/) const
{
    return false;
}


bool DeviceBufferRT::DownloadDepthBufferImpl(std::vector<unsigned char>& /*buffer*/) const
{
    return false;
}
