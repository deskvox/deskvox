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

}

//----------------------------------------------------------------------------
/// Specialication for vvImageClient as reference renderer and IBR CLIENT as test renderer

template<typename TestRend>
vvComparisonRend<vvImageClient, TestRend>::vvComparisonRend(vvVolDesc *vd, vvRenderState renderState)
  : vvRenderer(vd, renderState)
{
  vvDebugMsg::msg(1, "vvComparisonRend<vvImageClient, TestRend>::vvComparisonRend()");

  _ref = new vvImageClient(vd, renderState, "localhost", 31050);
  _test = new TestRend(vd, renderState, "localhost", 31051);
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
    return;// err;

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

  // Depth complexity from ibr rendering.
  uchar* ibrDepthCompl = new uchar[vp[2] * vp[3]];
  glReadPixels(0, 0, vp[2], vp[3], GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, ibrDepthCompl);

  // Compare images.
  const size_t size = static_cast<size_t>(vp[2] * vp[3]);
  float* noise = new float[size];
  float* sigToNoiseRatio = new float[size];
  float peak = 0.0f;
  for (size_t i = 0; i < size; ++i)
  {
    if (ibrDepthCompl[i] > 0)
    {
      noise[i] = static_cast<float>(abs(refImg.getImagePtr()[i] - ibrDepthCompl[i]));
      sigToNoiseRatio[i] = static_cast<float>(ibrDepthCompl[i])
                             / (noise[i] != 0.0f) ? noise[i] : 1.0f;
      if (sigToNoiseRatio[i] > peak)
      {
        peak = sigToNoiseRatio[i];
      }
    }
    else
    {
      noise[i] = 0.0f;
      sigToNoiseRatio[i] = 0.0f;
    }
  }

  glDrawPixels(vp[2], vp[3], GL_LUMINANCE, GL_UNSIGNED_BYTE, noise);

  float meanSigToNoise = 0.0f;
  for (size_t i = 0; i < size; ++i)
  {
    meanSigToNoise += sigToNoiseRatio[i];
  }
  meanSigToNoise /= static_cast<float>(size);
  std::cerr << peak << std::endl;
  std::cerr << log10(static_cast<double>(meanSigToNoise)) << std::endl;

  delete[] noise;
  delete[] sigToNoiseRatio;
  delete[] ibrDepthCompl;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
