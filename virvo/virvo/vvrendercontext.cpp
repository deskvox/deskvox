// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
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

#include "vvdebugmsg.h"
#include "vvrendercontext.h"

#include <iostream>

#include "vvx11.h"

using namespace std;

struct ContextArchData
{
#ifdef HAVE_X11
  GLXContext glxContext;
  Display* display;
  Drawable drawable;
#endif
};

vvRenderContext::vvRenderContext(const char* displayName)
{
  _archData = new ContextArchData;
  _initialized = false;
  init(displayName);
}

vvRenderContext::~vvRenderContext()
{
  vvDebugMsg::msg(1, "vvRenderContext::~vvRenderContext");

  if (_initialized)
  {
#ifdef HAVE_X11
    glXDestroyContext(_archData->display, _archData->glxContext);
    XCloseDisplay(_archData->display);
#endif
  }
  delete _archData;
}

bool vvRenderContext::makeCurrent() const
{
  if (_initialized)
  {
#ifdef HAVE_X11
    return glXMakeCurrent(_archData->display, _archData->drawable, _archData->glxContext);
#endif
  }
  return false;
}

void vvRenderContext::init(const char* displayName)
{
#ifdef HAVE_X11
  _archData->display = XOpenDisplay(displayName);

  if (_archData->display != NULL)
  {
    const Drawable parent = RootWindow(_archData->display, DefaultScreen(_archData->display));

    int attrList[] = { GLX_RGBA,
                       GLX_RED_SIZE, 8,
                       GLX_GREEN_SIZE, 8,
                       GLX_BLUE_SIZE, 8,
                       GLX_ALPHA_SIZE, 8,
                       GLX_DEPTH_SIZE, 24,
                       None};

    XVisualInfo* vi = glXChooseVisual(_archData->display,
                                      DefaultScreen(_archData->display),
                                      attrList);

    XSetWindowAttributes wa = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    wa.colormap = XCreateColormap(_archData->display, parent, vi->visual, AllocNone);
    wa.background_pixmap = None;
    wa.border_pixel = 0;

    if (vvDebugMsg::getDebugLevel() == 0)
    {
      wa.override_redirect = true;
    }
    else
    {
      wa.override_redirect = false;
    }

    _archData->glxContext = glXCreateContext(_archData->display, vi, NULL, True);

    if (_archData->glxContext != 0)
    {

      int windowWidth = 1;
      int windowHeight = 1;
      if (vvDebugMsg::getDebugLevel() > 0)
      {
        windowWidth = 512;
        windowHeight = 512;
      }

      _archData->drawable = XCreateWindow(_archData->display, parent, 0, 0, windowWidth, windowHeight, 0,
                                          vi->depth, InputOutput, vi->visual,
                                          CWBackPixmap|CWBorderPixel|CWColormap|CWOverrideRedirect, &wa);
      XMapWindow(_archData->display, _archData->drawable);
      XFlush(_archData->display);
      _initialized = true;
    }
    else
    {
      cerr << "Couldn't create OpenGL context" << endl;
      _initialized = false;
    }

    delete vi;
  }
  else
  {
    cerr << "Couldn't open X display" << endl;
    _initialized = false;
  }
#endif
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
