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

#ifdef HAVE_BONJOUR
#include "vvbonjour/vvbonjourregistrar.h"
#endif

#ifdef HAVE_CUDA
vvIbrServer::vvIbrServer(const vvRayRend::IbrMode mode)
  : vvRemoteServer(), _ibrMode(mode)
{
}

vvIbrServer::~vvIbrServer()
{

}

//----------------------------------------------------------------------------
/** Perform remote rendering, read back pixel data and send it over socket
    connections using a vvImage instance.
*/
void vvIbrServer::renderImage(vvMatrix& pr, vvMatrix& mv, vvRenderer* renderer)
{
  vvDebugMsg::msg(3, "vvIsaServer::renderImage()");

  // Render volume:
  float matrixGL[16];

  glMatrixMode(GL_PROJECTION);
  pr.makeGL(matrixGL);
  glLoadMatrixf(matrixGL);

  glMatrixMode(GL_MODELVIEW);
  mv.makeGL(matrixGL);
  glLoadMatrixf(matrixGL);

  vvRect* screenRect = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect();

  vvRayRend* rayRend = dynamic_cast<vvRayRend*>(renderer);
  assert(rayRend != NULL);

  float drMin = 0.0f;
  float drMax = 0.0f;
  vvIbr::calcDepthRange(pr, mv, renderer->getVolDesc()->getBoundingBox(), drMin, drMax);

  rayRend->setDepthRange(drMin, drMax);

  int dp = rayRend->getParameter(vvRenderer::VV_IBR_DEPTH_PREC);
  rayRend->compositeVolume();

  // Fetch rendered image
  vvGLTools::Viewport vp = vvGLTools::getViewport();
  uchar* pixels = new uchar[vp[2]*vp[3]*4];
  cudaMemcpy(pixels, dynamic_cast<vvCudaImg*>(rayRend->intImg)->getDeviceImg(), vp[2]*vp[3]*4, cudaMemcpyDeviceToHost);

  vvIbrImage* img = new vvIbrImage(vp[3], vp[2], (uchar*)pixels, 8);
  img->setDepthPrecision(dp);
  img->alloc_pd();
  uchar* depth = img->getPixelDepth();
  cudaMemcpy(depth, rayRend->getDeviceDepth(), vp[2]*vp[3]*dp/8, cudaMemcpyDeviceToHost);

  img->setReprojectionMatrix(vvIbr::calcImgMatrix(pr, mv, vp, drMin, drMax));
  img->encode(_codetype, 0, vp[2]-1, 0, vp[3]-1);
  _socket->putIbrImage(img);

  delete[] pixels;
  delete img;
}

void vvIbrServer::resize(const int w, const int h)
{
  glViewport(0, 0, w, h);
}

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
