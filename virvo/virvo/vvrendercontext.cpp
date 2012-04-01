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

#include "vvcocoaglcontext.h"
#include "vvdebugmsg.h"
#include "vvrendercontext.h"
#include "vvx11.h"

#include <sstream>

struct ContextArchData
{
#ifdef USE_COCOA
  vvCocoaGLContext* cocoaContext;
#endif

#ifdef HAVE_X11
  GLXContext glxContext;
  Display* display;
  Drawable drawable;
#endif
};

vvRenderContext::vvRenderContext(vvContextOptions * co)
{
  _archData = new ContextArchData;
  _initialized = false;
  _options = co;
  init();
}

vvRenderContext::~vvRenderContext()
{
  vvDebugMsg::msg(1, "vvRenderContext::~vvRenderContext");

  if (_initialized)
  {
#ifdef USE_COCOA
    delete _archData->cocoaContext;
#endif

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
#ifdef USE_COCOA
    return _archData->cocoaContext->makeCurrent();
#endif

#ifdef HAVE_X11
    return glXMakeCurrent(_archData->display, _archData->drawable, _archData->glxContext);
#endif
  }
  return false;
}

void vvRenderContext::swapBuffers() const
{
  if (_initialized)
  {
#ifdef USE_COCOA
    _archData->cocoaContext->swapBuffers();
#endif

#ifdef HAVE_X11
    glXSwapBuffers(_archData->display, _archData->drawable);
#endif
  }
}

void vvRenderContext::resize(const int w, const int h)
{
  if (_initialized)
  {
#ifdef USE_COCOA
    _archData->cocoaContext->resize(w, h);
#endif

#ifdef HAVE_X11
    std::cerr << "Function not implemented yet: vvRenderContext::resize() with X11" << std::endl;
#endif
  }
}

void vvRenderContext::init()
{
#ifdef USE_COCOA
  _archData->cocoaContext = new vvCocoaGLContext(_options);
  _initialized = true;
#endif

#ifdef HAVE_X11
  _archData->display = XOpenDisplay(_options->displayName.c_str());

  if(_archData->display != NULL)
  {
    switch(_options->type)
    {
    case vvContextOptions::VV_PBUFFER:
      {
        int attrList[] = { GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8, None};

        int nelements;
        GLXFBConfig* configs = glXChooseFBConfig(_archData->display, DefaultScreen(_archData->display),
                                                 attrList, &nelements);
        if (configs && (nelements > 0))
        {
          // TODO: find the nicest fbconfig.
          int pbAttrList[] = { GLX_PBUFFER_WIDTH, _options->width, GLX_PBUFFER_HEIGHT, _options->height, None };
          GLXPbuffer pbuffer = glXCreatePbuffer(_archData->display, configs[0], pbAttrList);
          _archData->glxContext = glXCreateNewContext(_archData->display, configs[0], GLX_RGBA_TYPE, 0, True);
          _archData->drawable = pbuffer;
          _initialized = true;
          return;
        }
      }
      // purposely no break; here
    case vvContextOptions::VV_WINDOW:
    default:
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

          int windowWidth = _options->width;
          int windowHeight = _options->height;
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
          vvDebugMsg::msg( 0, "Couldn't create OpenGL context");
          _initialized = false;
        }
        _initialized = true;
        delete vi;
      }
    }
  }
  else
  {
    _initialized = false;
    std::ostringstream errmsg;
    errmsg << "vvRenderContext::init() error: Could not open display " << _options->displayName;
    vvDebugMsg::msg(0, errmsg.str().c_str());
  }
#endif
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
