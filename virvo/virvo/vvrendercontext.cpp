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
#include <vector>

struct ContextArchData
{
#ifdef USE_COCOA
  vvCocoaGLContext* cocoaContext;
#endif

#if defined(HAVE_X11) && defined(USE_X11)
  GLXContext glxContext;
  Display* display;
  Drawable drawable;
  std::vector<int> attributes;
  GLXFBConfig* fbConfigs;
#endif

#ifdef _WIN32
  HGLRC wglContext;
  HWND window;
  HDC deviceContext;

  static LRESULT CALLBACK func(HWND hWnd, UINT Msg, WPARAM wParam, LPARAM lParam)
  {
    switch(Msg)
    {
    case WM_DESTROY:
      PostQuitMessage(WM_QUIT);
      break;
    default:
      return DefWindowProc(hWnd, Msg, wParam, lParam);
    }
    return 0;
  }
#endif
};

vvRenderContext::vvRenderContext(const vvContextOptions& co)
  : _options(co)
{
  _archData = new ContextArchData;
  _initialized = false;
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

#if defined(HAVE_X11) && defined(USE_X11)
    glXDestroyContext(_archData->display, _archData->glxContext);
    switch (_options.type)
    {
    case vvContextOptions::VV_PBUFFER:
      glXDestroyPbuffer(_archData->display, _archData->drawable);
      break;
    case vvContextOptions::VV_WINDOW:
      // fall through
    default:
      XDestroyWindow(_archData->display, _archData->drawable);
      break;
    }
    XCloseDisplay(_archData->display);
#endif

#ifdef _WIN32
    wglMakeCurrent(_archData->deviceContext, NULL);
    wglDeleteContext(_archData->wglContext);
    DestroyWindow(_archData->window);
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

#if defined(HAVE_X11) && defined(USE_X11)
    return glXMakeCurrent(_archData->display, _archData->drawable, _archData->glxContext);
#endif

#ifdef _WIN32
    return (wglMakeCurrent(_archData->deviceContext, _archData->wglContext) == TRUE);
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

#if defined(HAVE_X11) && defined(USE_X11)
    glXSwapBuffers(_archData->display, _archData->drawable);
#endif

#ifdef _WIN32
    SwapBuffers(_archData->deviceContext);
#endif
  }
}

void vvRenderContext::resize(const int w, const int h)
{
  if ((_options.width != w) || (_options.height != h))
  {
    _options.width = w;
    _options.height = h;
    if (_initialized)
    {
#ifdef USE_COCOA
      _archData->cocoaContext->resize(w, h);
#endif

#if defined(HAVE_X11) && defined(USE_X11)
      switch (_options.type)
      {
      case vvContextOptions::VV_PBUFFER:
      {
        glXDestroyPbuffer(_archData->display, _archData->drawable);
        initPbuffer();
        makeCurrent();
        break;
      }
      case vvContextOptions::VV_WINDOW:
        // fall through
      default:
        XResizeWindow(_archData->display, _archData->drawable,
                      static_cast<uint>(w),
                      static_cast<uint>(h));
        XSync(_archData->display, False);
        break;
      }
#endif
    }
  }
}

void vvRenderContext::init()
{
#ifdef USE_COCOA
  _archData->cocoaContext = new vvCocoaGLContext(_options);
  _initialized = true;
#endif

#if defined(HAVE_X11) && defined(USE_X11)
  _archData->display = XOpenDisplay(_options.displayName.c_str());

  _archData->attributes.push_back(GLX_RGBA);
  _archData->attributes.push_back(GLX_RED_SIZE);
  _archData->attributes.push_back(8);
  _archData->attributes.push_back(GLX_GREEN_SIZE);
  _archData->attributes.push_back(8);
  _archData->attributes.push_back(GLX_BLUE_SIZE);
  _archData->attributes.push_back(8);
  _archData->attributes.push_back(GLX_ALPHA_SIZE);
  _archData->attributes.push_back(8);
  _archData->attributes.push_back(GLX_DEPTH_SIZE);
  _archData->attributes.push_back(24);
  _archData->attributes.push_back(None);

  if(_archData->display != NULL)
  {
    switch(_options.type)
    {
    case vvContextOptions::VV_PBUFFER:
      if (initPbuffer())
      {
         _archData->glxContext = glXCreateNewContext(_archData->display, _archData->fbConfigs[0], GLX_RGBA_TYPE, 0, True);
        _initialized = true;
        return;
      }
      else
      {
        _options.type = vvContextOptions::VV_WINDOW;
      }
      // no pbuffer created - fall through
    case vvContextOptions::VV_WINDOW:
      // fall through
    default:
      {
        const Drawable parent = RootWindow(_archData->display, DefaultScreen(_archData->display));

        XVisualInfo* vi = glXChooseVisual(_archData->display,
                                          DefaultScreen(_archData->display),
                                          &(_archData->attributes)[0]);

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

          int windowWidth = _options.width;
          int windowHeight = _options.height;
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
          return;
        }
        _initialized = true;
        delete vi;
        break;
      }
    }
  }
  else
  {
    _initialized = false;
    std::ostringstream errmsg;
    errmsg << "vvRenderContext::init() error: Could not open display " << _options.displayName;
    vvDebugMsg::msg(0, errmsg.str().c_str());
  }
#endif

#ifdef _WIN32
  LPCTSTR dev = NULL; // TODO: configurable
  DWORD devNum = 0; // TODO: configurable
  DWORD dwFlags = 0;
  DISPLAY_DEVICE dispDev;
  DEVMODE devMode;
  dispDev.cb = sizeof(DISPLAY_DEVICE);

  if (EnumDisplayDevices(dev, devNum, &dispDev, dwFlags))
  {
    EnumDisplaySettings(dispDev.DeviceName, ENUM_CURRENT_SETTINGS, &devMode);
  }
  else
  {
    _initialized = false;
    return;
  }

  switch (_options.type)
  {
  case vvContextOptions::VV_PBUFFER:
    std::cerr << "WGL Pbuffers not implemented yet" << std::endl;
    // fall through
  case vvContextOptions::VV_WINDOW:
    // fall through
  default:
    {
      HINSTANCE hInstance = GetModuleHandle(0);
      WNDCLASSEX WndClsEx;
      ZeroMemory(&WndClsEx, sizeof(WNDCLASSEX));

      LPCTSTR ClsName = TEXT("Virvo");
      LPCTSTR WndName = TEXT("Render Context");

      WndClsEx.cbSize        = sizeof(WNDCLASSEX);
      WndClsEx.style         = CS_HREDRAW | CS_VREDRAW;
      WndClsEx.lpfnWndProc   = ContextArchData::func;
      WndClsEx.cbClsExtra    = 0;
      WndClsEx.cbWndExtra    = 0;
      WndClsEx.hIcon         = LoadIcon(NULL, IDI_APPLICATION);
      WndClsEx.hCursor       = LoadCursor(NULL, IDC_ARROW);
      WndClsEx.lpszMenuName  = NULL;
      WndClsEx.lpszClassName = ClsName;
      WndClsEx.hInstance     = hInstance;
      WndClsEx.hbrBackground = 0;
      WndClsEx.hIconSm       = LoadIcon(NULL, IDI_APPLICATION);
      RegisterClassEx(&WndClsEx);

      PIXELFORMATDESCRIPTOR pfd;
      ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));
      pfd.nSize = sizeof(pfd);
      pfd.nVersion = 1;
      pfd.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW;
      if (_options.doubleBuffering)
      {
        pfd.dwFlags |= PFD_DOUBLEBUFFER;
      }
      pfd.iPixelType = PFD_TYPE_RGBA; // TODO: configurable
      pfd.cColorBits = 32; // TODO: configurable
      pfd.cDepthBits = 16; // TODO: configurable
      pfd.iLayerType = PFD_MAIN_PLANE;
      HWND hWnd = CreateWindow(ClsName,
                               WndName,
                               WS_OVERLAPPEDWINDOW,
                               _options.left < 0 ? CW_USEDEFAULT : _options.left,
                               _options.top < 0 ? CW_USEDEFAULT : _options.top,
                               _options.width,
                               _options.height,
                               NULL,      // handle of parent window
                               NULL,      // handle of menu
                               hInstance,
                               NULL);     // for MDI client windows

      if(!hWnd)
      {
        vvDebugMsg::msg(0, "vvRenderContext::init(): error CreateWindow()");
        _initialized = false;
        return;
      }
      else
      {
        _archData->window = hWnd;
        _archData->deviceContext = GetDC(hWnd);
        int pf = ChoosePixelFormat(_archData->deviceContext, &pfd);
        if (pf != 0)
        {
          if (SetPixelFormat(_archData->deviceContext, pf, &pfd))
          {
            _archData->wglContext = wglCreateContext(_archData->deviceContext);
          }
          else
          {
            vvDebugMsg::msg(0, "vvRenderContext::init(): error SetPixelFormat()");
            _initialized = false;
            return;
          }
        }
        ShowWindow(hWnd, SW_SHOWNORMAL);
      }

      _initialized = true;
    }
    break;
  }
#endif
}

bool vvRenderContext::initPbuffer()
{
#if defined(HAVE_X11) && defined(USE_X11)
  int nelements;
  _archData->fbConfigs = glXChooseFBConfig(_archData->display, DefaultScreen(_archData->display),
                                           &(_archData->attributes)[1], &nelements); // first entry (GLX_RGBA) in attributes list confuses pbuffers
  if ((_archData->fbConfigs != NULL) && (nelements > 0))
  {
    // TODO: find the nicest fbconfig.
    int pbAttrList[] = { GLX_PBUFFER_WIDTH, _options.width, GLX_PBUFFER_HEIGHT, _options.height, None };
    _archData->drawable = glXCreatePbuffer(_archData->display, _archData->fbConfigs[0], pbAttrList);
    if (!_archData->drawable)
    {
      std::cerr << "No pbuffer created" << std::endl;
      return false;
    }
    return true;
  }
#endif
  return false;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
