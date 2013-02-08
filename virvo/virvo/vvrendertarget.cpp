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


using virvo::RenderTarget;
using virvo::DefaultFramebufferRT;
using virvo::HostBufferRT;


//--------------------------------------------------------------------------------------------------
// RenderTarget
//--------------------------------------------------------------------------------------------------


RenderTarget::~RenderTarget()
{
}


bool RenderTarget::beginFrame()
{
    assert( Bound == false && "already bound" );

    Bound = BeginFrameImpl();
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


bool DefaultFramebufferRT::BeginFrameImpl()
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


bool HostBufferRT::BeginFrameImpl()
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
