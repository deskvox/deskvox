// Virvo - Virtual Reality Volume Rendering
// Contact: Stefan Zellmann, zellmans@uni-koeln.de
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

#include "vvrendertarget.h"

#include <iostream>

using std::cerr;
using std::endl;

vvRenderTarget::vvRenderTarget()
{
  _type = VV_RENDER_TARGET;
}

vvRenderTarget::~vvRenderTarget()
{

}

void vvRenderTarget::initForRender()
{

}

void vvRenderTarget::writeBack(const int, const int)
{

}

void vvRenderTarget::resize(const int, const int)
{

}

void vvRenderTarget::clearBuffer()
{

}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
