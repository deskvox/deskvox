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

#include <cmath>

#include "vvgltools.h"
#include "vvibr.h"
#include "vvibrserver.h"
#include "vvrayrend.h"
#include "vvcudaimg.h"
#include "vvdebugmsg.h"
#include "vvvoldesc.h"
#include "vvibrimage.h"
#include "vvsocketio.h"
#include "vvtoolshed.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

vvIbrServer::vvIbrServer(vvSocketIO *socket)
: vvRemoteServer(socket)
, _ibrMode(vvRenderer::VV_MAX_GRADIENT)
, _image(NULL)
, _pixels(NULL)
, _depth(NULL)
, _uncertainty(NULL)
{
}

vvIbrServer::~vvIbrServer()
{
  delete _image;
  delete[] _pixels;
  delete[] _depth;
  delete[] _uncertainty;
}

//----------------------------------------------------------------------------
/** Perform remote rendering, read back pixel data and send it over socket
    connections using a vvImage instance.
*/
void vvIbrServer::renderImage(vvMatrix& pr, vvMatrix& mv, vvRenderer* renderer)
{
#ifdef HAVE_CUDA
  vvDebugMsg::msg(3, "vvIsaServer::renderImage()");

  // Render volume:
  float matrixGL[16];

  glMatrixMode(GL_PROJECTION);
  pr.getGL(matrixGL);
  glLoadMatrixf(matrixGL);

  glMatrixMode(GL_MODELVIEW);
  mv.getGL(matrixGL);
  glLoadMatrixf(matrixGL);

  vvRayRend* rayRend = dynamic_cast<vvRayRend*>(renderer);
  assert(rayRend != NULL);

  float drMin = 0.0f;
  float drMax = 0.0f;
  vvIbr::calcDepthRange(pr, mv, renderer->getVolDesc()->getBoundingBox(), drMin, drMax);

  rayRend->setDepthRange(drMin, drMax);

  int dp = rayRend->getParameter(vvRenderer::VV_IBR_DEPTH_PREC);
  int up = rayRend->getParameter(vvRenderer::VV_IBR_UNCERTAINTY_PREC);
  rayRend->compositeVolume();

  // Fetch rendered image
  vvGLTools::Viewport vp = vvGLTools::getViewport();
  const int w = vp[2];
  const int h = vp[3];
  if(!_image || _image->getWidth() != w || _image->getHeight() != h || _image->getDepthPrecision() != dp)
  {
    delete[] _pixels;
    delete[] _depth;
    delete[] _uncertainty;
    _pixels = new uchar[w*h*4];
    _depth = new uchar[w*h*(dp/8)];
    _uncertainty = new uchar[w*h*(up/8)];
    if(_image)
    {
      _image->setDepthPrecision(dp);
      _image->setNewImage(h, w, _pixels);
    }
    else
    {
      // for now uncertainty precision same as depth precision
      const int up = dp;
      _image = new vvIbrImage(h, w, _pixels, dp, up);
    }
    _image->alloc_pd();
  }
  else
  {
    _image->setNewImagePtr(_pixels);
  }
  _image->setNewDepthPtr(_depth);
  _image->setNewUncertaintyPtr(_uncertainty);

  cudaMemcpy(_pixels, rayRend->getDeviceImg(), w*h*4, cudaMemcpyDeviceToHost);
  cudaMemcpy(_depth, rayRend->getDeviceDepth(), w*h*(dp/8), cudaMemcpyDeviceToHost);
  cudaMemcpy(_uncertainty, rayRend->getDeviceUncertainty(), w*h*(up/8), cudaMemcpyDeviceToHost);

  _image->setModelViewMatrix(mv);
  _image->setProjectionMatrix(pr);
  _image->setViewport(vp);
  _image->setDepthRange(drMin, drMax);
  _image->encode(_codetype, 0, h-1, 0, w-1);
  _socket->putIbrImage(_image);
#endif
}

void vvIbrServer::resize(const int w, const int h)
{
  glViewport(0, 0, w, h);
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
