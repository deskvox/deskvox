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

#include "vvcomparisonrend.h"
#include "vvgltools.h"
#include "vvdebugmsg.h"
#include "vvibrimage.h"
#include "vvimage.h"
#include "vvimageclient.h"
#include "vvsocketio.h"

#include <cassert>
#include <cmath>

using std::cerr;
using std::cout;
using std::endl;

template<typename ReferenceRend, typename TestRend>
vvComparisonRend<ReferenceRend, TestRend>::vvComparisonRend(vvVolDesc *vd, vvRenderState renderState)
  : vvRenderer(vd, renderState)
  , _ref(new ReferenceRend(vd, renderState))
  , _test(new TestRend(vd, renderState))
{
  vvDebugMsg::msg(1, "vvComparisonRend::vvComparisonRend()");
}

template<typename ReferenceRend, typename TestRend>
vvComparisonRend<ReferenceRend, TestRend>::~vvComparisonRend()
{
  vvDebugMsg::msg(1, "vvComparisonRend::~vvComparisonRend()");
  delete _ref;
  delete _test;
}

template<typename ReferenceRend, typename TestRend>
void vvComparisonRend<ReferenceRend, TestRend>::renderVolumeGL()
{
  vvDebugMsg::msg(3, "vvComparisonRend::renderVolumeGL()");
}

//----------------------------------------------------------------------------
/// Specialication for vvImageClient as reference renderer and IBR CLIENT as test renderer

template<typename TestRend>
vvComparisonRend<vvImageClient, TestRend>::vvComparisonRend(vvVolDesc *vd, vvRenderState renderState,
                                                            const char* refHostName, int refPort,
                                                            const char* refFileName,
                                                            const char* testHostName, int testPort,
                                                            const char* testFileName)
  : vvRenderer(vd, renderState)
{
  vvDebugMsg::msg(1, "vvComparisonRend<vvImageClient, TestRend>::vvComparisonRend()");

  _ref = new vvImageClient(vd, renderState, refHostName, refPort, refFileName);
  _test = new TestRend(vd, renderState, testHostName, testPort, testFileName);
}

template<typename TestRend>
vvComparisonRend<vvImageClient, TestRend>::~vvComparisonRend()
{
  vvDebugMsg::msg(1, "vvComparisonRend<vvImageClient, TestRend>::~vvComparisonRend()");
  delete _ref;
  delete _test;
}

template<typename TestRend>
void vvComparisonRend<vvImageClient, TestRend>::renderVolumeGL()
{
  vvDebugMsg::msg(3, "vvComparisonRend<vvImageClient, TestRend>::renderVolumeGL()");

  // Process img from image client.
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);

  if ((vp[2] <= 0) || (vp[3] <= 0))
  {
    return;
  }

  if(vp[2] != _ref->_viewportWidth || vp[3] != _ref->_viewportHeight)
  {
    _ref->resize(vp[2], vp[3]);
  }

  vvGLTools::getModelviewMatrix(&_ref->_currentMv);
  vvGLTools::getProjectionMatrix(&_ref->_currentPr);

  vvImage refImg;
  vvRemoteClient::ErrorType err = _ref->requestFrame();
  if(err != vvRemoteClient::VV_OK)
  {
    vvDebugMsg::msg(0, "vvComparisonRend<vvImageClient, TestRend>::renderVolumeGL: remote client error in function _requestFrame()");
    return;// err;
  }

  if(!_ref->_socket)
    return;// vvRemoteClient::VV_SOCKET_ERROR;
  vvSocketIO::ErrorType sockerr = _ref->_socket->getImage(&refImg);
  if(sockerr != vvSocketIO::VV_OK)
  {
    std::cerr << "vvComparisonRend<vvImageClient, TestRend>::renderVolumeGL: socket error ("
              << sockerr << ") - exiting..." << std::endl;
    return;// vvRemoteClient::VV_SOCKET_ERROR;
  }

  refImg.decode();

  // Process img from ibr client.
  glClear(GL_STENCIL_BUFFER_BIT);
  glEnable(GL_STENCIL_TEST);
  glStencilFunc(GL_ALWAYS, 0, 0);
  glStencilOp(GL_KEEP, GL_INCR, GL_INCR);

  _test->renderVolumeGL();

  const int chan = 4;

  // Depth complexity from ibr rendering.
  const size_t size = static_cast<size_t>(vp[2] * vp[3]);
  uchar* ibrDepthCompl = new uchar[size];
  glReadPixels(vp[0], vp[1], vp[2], vp[3], GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, ibrDepthCompl);
  uchar* ibrImg = new uchar[size * chan];
  glReadPixels(vp[0], vp[1], vp[2], vp[3], GL_RGBA, GL_UNSIGNED_BYTE, ibrImg);

  // Compare images.
  float* noise = new float[size * chan];
  int nonHolePixels = 0;
  float mseNonHoles = 0.0f; // mean squared error only for non-hole pixels
  for (size_t i = 0; i < size; ++i)
  {
    if (ibrDepthCompl[i] > 0)
    {
      for (size_t c = 0; c < chan; ++c)
      {
        const size_t idx = i * chan + c;
#define POWF(x) ((static_cast<float>(x)) * (static_cast<float>(x)))
        noise[idx] = POWF(refImg.getImagePtr()[idx] - ibrImg[idx]);
#undef POWF

        if (c < 3)
        {
          mseNonHoles += noise[idx];
        }
      }
      ++nonHolePixels;
    }
    else
    {
      for (size_t c = 0; c < chan; ++c)
      {
        const size_t idx = i * chan + c;
        noise[idx] = 0.0f;
      }
    }
  }

  if (nonHolePixels > 0)
  {
    mseNonHoles /= (static_cast<float>(nonHolePixels) * 3.0f);
  }
  const double MAX = 255.0;
  const float psnr = float(mseNonHoles != 0.0f ? 10 * log10((MAX * MAX) / mseNonHoles) : 0.0f);
  std::cerr << "Drawn pixels: " << nonHolePixels << std::endl;
  std::cerr << "Peak signal-to-noise ratios: " << psnr << std::endl;

  delete[] noise;
  delete[] ibrDepthCompl;
  delete[] ibrImg;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
