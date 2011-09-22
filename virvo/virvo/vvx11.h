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

#ifndef _VVX11_H_
#define _VVX11_H_

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#if !defined(_WIN32) && !defined(__APPLE__)
#define HAVE_X11
#elif defined(HAVE_XLIBS) && !defined(HAVE_GL_FRAMEWORK)
#define HAVE_X11
#endif

// xlib:
#ifdef HAVE_X11
#include <GL/glx.h>
#include <X11/Xlib.h>
#endif

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
