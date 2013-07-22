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

#include "vvimageclient.h"
#include "vvsocketio.h"
#include "vvdebugmsg.h"

#include "gl/util.h"

#include "private/vvgltools.h"
#include "private/vvimage.h"

using std::cerr;
using std::endl;

vvImageClient::vvImageClient(vvVolDesc *vd, vvRenderState renderState,
                             vvTcpSocket* socket, const std::string& filename)
  : vvRemoteClient(vd, renderState, socket, filename)
{
  vvDebugMsg::msg(1, "vvImageClient::vvImageClient()");

  rendererType = REMOTE_IMAGE;
}

vvImageClient::~vvImageClient()
{
  vvDebugMsg::msg(1, "vvImageClient::~vvImageClient()");
}

vvRemoteClient::ErrorType vvImageClient::render()
{
  vvDebugMsg::msg(1, "vvImageClient::render()");

  // Send the request
  vvRemoteClient::ErrorType err = requestFrame();
  if(err != VV_OK)
    return err;

  virvo::Image image;

  // Get the image
  vvSocket::ErrorType sockerr = _socketIO->getImage(image);
  if (vvSocket::VV_OK != sockerr)
  {
    std::cerr << "vvImageClient::render: socket error (" << sockerr << ") - exiting..." << std::endl;
    return vvRemoteClient::VV_SOCKET_ERROR;
  }

  // Decompress the image
  if (!image.decompress())
    return VV_BAD_IMAGE;

  // Display the image
  virvo::PixelFormatInfo f = mapPixelFormat(image.format());
  virvo::gl::blendPixels(image.width(), image.height(), f.format, f.type, image.data());

  return VV_OK;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
