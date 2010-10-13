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

vvRenderContext::vvRenderContext()
  : vvRenderTarget()
{
  _archData = new ContextArchData;
  _initialized = false;
  init();
}

vvRenderContext::~vvRenderContext()
{
  if (_initialized)
  {
#ifdef HAVE_X11
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

void vvRenderContext::init()
{
#ifdef HAVE_X11
  // TODO: make this configurable.
  _archData->display = XOpenDisplay(NULL);

  if (_archData->display != NULL)
  {
    const Drawable parent = RootWindow(_archData->display, 0);

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
    cerr << "Couldn't open X display" << endl;
    _initialized = false;
  }
#endif
}
