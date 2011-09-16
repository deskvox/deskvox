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

#include "vvdebugmsg.h"
#include "vvremoteclient.h"

using std::cerr;
using std::endl;

vvRemoteClient::vvRemoteClient(vvVolDesc *vd, vvRenderState renderState,
                               const char *slaveName, int slavePort,
                               const char *slaveFileName)
   : vvRenderer(vd, renderState),
    _slaveName(slaveName),
    _slavePort(slavePort),
    _slaveFileName(slaveFileName),
    _changes(true)
{
  vvDebugMsg::msg(1, "vvRemoteClient::vvRemoteClient()");

  initSocket(vd);
}

vvRemoteClient::~vvRemoteClient()
{
  vvDebugMsg::msg(1, "vvRemoteClient::~vvRemoteClient()");

  clearImages();
}

vvRemoteClient::ErrorType vvRemoteClient::initSocket(vvVolDesc*& vd)
{
  vvDebugMsg::msg(1, "vvRemoteClient::initSocket()");

  _socket = new vvSocketIO(_slavePort, _slaveName, vvSocket::VV_TCP);
  _socket->set_debuglevel(vvDebugMsg::getDebugLevel());

  if (_socket->init() == vvSocket::VV_OK)
  {
    _socket->no_nagle();
    _socket->putBool(_slaveFileName!=NULL);

    if (_slaveFileName)
    {
      _socket->putFileName(_slaveFileName);
      _socket->getVolumeAttributes(vd);
    }
    else
    {
      switch (_socket->putVolume(vd))
      {
        case vvSocket::VV_OK:
          cerr << "Volume transferred successfully" << endl;
          break;
        case vvSocket::VV_ALLOC_ERROR:
          cerr << "Not enough memory" << endl;
          return VV_SOCKET_ERROR;
        default:
          cerr << "Cannot write volume to socket" << endl;
          return VV_SOCKET_ERROR;
      }
    }
  }
  else
  {
    cerr << "No connection to remote rendering server established at: " << _slaveName << endl;
    return VV_SOCKET_ERROR;
  }
  return VV_OK;
}

void vvRemoteClient::resize(const int w, const int h)
{
  vvDebugMsg::msg(1, "vvRemoteClient::resize()");
  _changes = true;

  if (_socket->putCommReason(vvSocketIO::VV_RESIZE) == vvSocket::VV_OK)
  {
    _socket->putWinDims(w, h);
  }
}

void vvRemoteClient:: setCurrentFrame(const int index)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setCurrentFrame()");
  _changes = true;

  if (_socket->putCommReason(vvSocketIO::VV_CURRENT_FRAME) == vvSocket::VV_OK)
  {
    _socket->putInt32(index);
  }
}

void vvRemoteClient::setMipMode(const int mipMode)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setMipMode()");
  _changes = true;

  if (_socket->putCommReason(vvSocketIO::VV_MIPMODE) == vvSocket::VV_OK)
  {
    _socket->putInt32(mipMode);
  }
}

void vvRemoteClient::setObjectDirection(const vvVector3* od)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setObjectDirection()");
  _changes = true;

  if (_socket->putCommReason(vvSocketIO::VV_OBJECT_DIRECTION) == vvSocket::VV_OK)
  {
    _socket->putVector3(*od);
  }
}

void vvRemoteClient::setViewingDirection(const vvVector3* vd)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setViewingDirection()");
  _changes = true;

  if (_socket->putCommReason(vvSocketIO::VV_VIEWING_DIRECTION) == vvSocket::VV_OK)
  {
    _socket->putVector3(*vd);
  }
}

void vvRemoteClient::setPosition(const vvVector3* p)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setPosition()");
  _changes = true;

  if (_socket->putCommReason(vvSocketIO::VV_POSITION) == vvSocket::VV_OK)
  {
    _socket->putVector3(*p);
  }
}

void vvRemoteClient::setROIEnable(const bool roiEnabled)
{
  vvDebugMsg::msg(1, "vvRemoteClient::setROIEnable()");
  _changes = true;

  if (_socket->putCommReason(vvSocketIO::VV_TOGGLE_ROI) == vvSocket::VV_OK)
  {
    _socket->putBool(roiEnabled);
  }
}

void vvRemoteClient::setProbePosition(const vvVector3* pos)
{
  vvDebugMsg::msg(1, "vvRemoteClient::setProbePosition()");
  _changes = true;

  if (_socket->putCommReason(vvSocketIO::VV_ROI_POSITION) == vvSocket::VV_OK)
  {
    _socket->putVector3(*pos);
  }
}

void vvRemoteClient::setProbeSize(const vvVector3* newSize)
{
  vvDebugMsg::msg(1, "vvRemoteClient::setProbeSize()");
  _changes = true;

  if (_socket->putCommReason(vvSocketIO::VV_ROI_SIZE) == vvSocket::VV_OK)
  {
    _socket->putVector3(*newSize);
  }
}

void vvRemoteClient::updateTransferFunction(vvTransFunc& tf)
{
  vvDebugMsg::msg(1, "vvRemoteClient::updateTransferFunction()");
  _changes = true;

  if (_socket->putCommReason(vvSocketIO::VV_TRANSFER_FUNCTION) == vvSocket::VV_OK)
  {
    _socket->putTransferFunction(tf);
  }
}

void vvRemoteClient::setParameter(const vvRenderer::ParameterType param, const float newValue)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setParameter()");
  _changes = true;
  switch (param)
  {
  case vvRenderer::VV_QUALITY:
    adjustQuality(newValue);
    break;
  case vvRenderer::VV_SLICEINT:
    setInterpolation((newValue != 0.0f));
    break;
  default:
    vvRenderer::setParameter(param, newValue);
    break;
  }
}

void vvRemoteClient::adjustQuality(const float quality)
{
  vvDebugMsg::msg(3, "vvRemoteClient::adjustQuality()");
  if (_socket->putCommReason(vvSocketIO::VV_QUALITY) == vvSocket::VV_OK)
  {
    _socket->putFloat(quality);
  }
}

void vvRemoteClient::setInterpolation(const bool interpolation)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setInterpolation()");
    if (_socket->putCommReason(vvSocketIO::VV_INTERPOLATION) == vvSocket::VV_OK)
    {
      _socket->putBool(interpolation);
    }
}

void vvRemoteClient::clearImages()
{
  vvDebugMsg::msg(3, "vvRemoteClient::clearImages()");
  for (std::vector<vvImage*>::const_iterator it = _images.begin();
      it != _images.end();
      ++it)
  {
    delete (*it);
  }
  _images.clear();
}
