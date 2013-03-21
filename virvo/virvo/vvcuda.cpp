// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 2010 University of Cologne
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

#include "vvcuda.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include "vvcudatools.h"

#include <string>
#include <cstring>
#include <iostream>

#ifdef HAVE_CUDA
#include <cuda_gl_interop.h>
#endif

#ifdef USE_X11
#include <GL/glx.h>
#include <X11/Xlib.h>
#endif

#include "vvdebugmsg.h"

bool vvCuda::init()
{
#ifdef HAVE_CUDA
  bool ok = true;
  if(!vvCudaTools::checkError(&ok, cudaSetDeviceFlags(cudaDeviceMapHost), "vvCuda::init (set device flags)", false))
    return false;

  return true;
#else
  std::cerr << "Cuda not found!" << std::endl;
  return false;
#endif
}

bool vvCuda::initGlInterop()
{
#ifdef HAVE_CUDA
  static bool done = false;
  if (done)
    return true;

#ifdef USE_X11
  GLXContext ctx = glXGetCurrentContext();
  Display *dsp = XOpenDisplay(NULL);
  if(dsp)
    XSynchronize(dsp, True);
  if(!dsp || !glXIsDirect(dsp, ctx))
  {
    vvDebugMsg::msg(1, "no CUDA/GL interop possible");
    return false;
  }
#endif

  cudaDeviceProp prop;
  memset(&prop, 0, sizeof(prop));
  prop.major = 1;
  prop.minor = 0;

  int dev;
  bool ok = true;
  if(!vvCudaTools::checkError(&ok, cudaChooseDevice(&dev, &prop), "vvCuda::initGlInterop (choose device)", false))
    return false;
  if(!vvCudaTools::checkError(&ok, cudaGLSetGLDevice(dev), "vvCuda::initGlInterop (set device)", false))
    return false;

  done = true;
  return true;
#else
  return false;
#endif
}

//============================================================================
// End of File
//============================================================================

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
