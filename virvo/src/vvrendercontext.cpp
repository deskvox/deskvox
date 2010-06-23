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

#include "vvrendercontext.h"

#include <iostream>

#if !defined(_WIN32) && !defined(__APPLE__)
#define HAVE_X11
#endif

#ifdef HAVE_X11
#include <GL/glx.h>
#include <X11/Xlib.h>
#endif

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
  delete _archData;
}

bool vvRenderContext::makeCurrent() const
{
  if (_initialized)
  {
    return glXMakeCurrent(_archData->display, _archData->drawable, _archData->glxContext);
  }
  return false;
}

void vvRenderContext::init()
{
#ifdef HAVE_X11
  // TODO: make this configurable.
  _archData->display = XOpenDisplay(":0");

  const bool debug = true;
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

    wa.override_redirect = !debug;

    _archData->glxContext = glXCreateContext(_archData->display, vi, NULL, True);

    int windowWidth = 1;
    int windowHeight = 1;
    if (debug)
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
