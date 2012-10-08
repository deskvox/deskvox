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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include <limits>

#include "vvimageclient.h"
#include "vvgltools.h"
#include "float.h"
#include "vvshaderfactory.h"
#include "vvshaderprogram.h"
#include "vvtoolshed.h"
#include "vvsocketio.h"
#include "vvdebugmsg.h"
#include "vvimage.h"

using std::cerr;
using std::endl;

vvImageClient::vvImageClient(vvVolDesc *vd, vvRenderState renderState,
                             vvTcpSocket* socket, const std::string& filename)
  : vvRemoteClient(vd, renderState, socket, filename)
  , _image(NULL)
{
  vvDebugMsg::msg(1, "vvImageClient::vvImageClient()");

  rendererType = REMOTE_IMAGE;

  glGenTextures(1, &_rgbaTex);
  _image = new vvImage;
}

vvImageClient::~vvImageClient()
{
  vvDebugMsg::msg(1, "vvImageClient::~vvImageClient()");

  glDeleteTextures(1, &_rgbaTex);
}

vvRemoteClient::ErrorType vvImageClient::render()
{
  vvDebugMsg::msg(1, "vvImageClient::render()");

  vvRemoteClient::ErrorType err = requestFrame();
  if(err != vvRemoteClient::VV_OK)
    return err;

  if(!_socketIO)
    return vvRemoteClient::VV_SOCKET_ERROR;

  vvSocket::ErrorType sockerr = _socketIO->getImage(_image);
  if(sockerr != vvSocket::VV_OK)
  {
    std::cerr << "vvImageClient::render: socket error (" << sockerr << ") - exiting..." << std::endl;
    return vvRemoteClient::VV_SOCKET_ERROR;
  }

  _image->decode();

  const int h = _image->getHeight();
  const int w = _image->getWidth();

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glPushAttrib(GL_COLOR_BUFFER_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT
               | GL_ENABLE_BIT | GL_TEXTURE_BIT | GL_TRANSFORM_BIT);

  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

  glEnable(GL_TEXTURE_2D);

  // get pixel and depth-data
  glBindTexture(GL_TEXTURE_2D, _rgbaTex);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, _image->getImagePtr());

  vvGLTools::drawViewAlignedQuad();

  glPopAttrib();

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  return VV_OK;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
