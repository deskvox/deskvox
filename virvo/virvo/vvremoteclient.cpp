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
#include "vvsocketio.h"
#include "vvopengl.h"
#include "vvvoldesc.h"
#include "vvimage.h"
#include "vvtoolshed.h"

using std::cerr;
using std::endl;

vvRemoteClient::vvRemoteClient(vvVolDesc *vd, vvRenderState renderState, uint32_t type,
                               const char *slaveName, int slavePort,
                               const char *slaveFileName)
   : vvRenderer(vd, renderState),
    _type(type),
    _slaveName(slaveName),
    _slavePort(slavePort),
    _slaveFileName(slaveFileName),
    _changes(true),
    _viewportWidth(-1),
    _viewportHeight(-1)
{
  vvDebugMsg::msg(1, "vvRemoteClient::vvRemoteClient()");

  initSocket(vd);
}

vvRemoteClient::~vvRemoteClient()
{
  vvDebugMsg::msg(1, "vvRemoteClient::~vvRemoteClient()");
}

void vvRemoteClient::renderVolumeGL()
{
  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  if(vp[2] != _viewportWidth || vp[3] != _viewportHeight)
  {
    resize(vp[2], vp[3]);
  }

  vvGLTools::getModelviewMatrix(&_currentMv);
  vvGLTools::getProjectionMatrix(&_currentPr);

  if (render() != vvRemoteClient::VV_OK)
  {
    vvDebugMsg::msg(0, "vvRemoteClient::renderVolumeGL(): remote rendering error");
  }
  vvRenderer::renderVolumeGL();
}

vvRemoteClient::ErrorType vvRemoteClient::initSocket(vvVolDesc*& vd)
{
  vvDebugMsg::msg(3, "vvRemoteClient::initSocket()");

  const int defaultPort = 31050;
  int port = _slavePort;
  char *serverName = NULL;

  if(!_slaveName)
  {
    if(const char *s = getenv("VV_SERVER"))
    {
      vvDebugMsg::msg(1, "remote rendering server from environment: ", s);
      port = vvToolshed::parsePort(s);
      if(port == -1)
        _slaveName = s;
      else
        serverName = vvToolshed::stripPort(s);
    }
  }

  if(port == -1)
    port = defaultPort;

  if(!_slaveName && !serverName)
  {
    vvDebugMsg::msg(1, "no server specified");
    return VV_SOCKET_ERROR;
  }

  _socket = new vvSocketIO(port, serverName ? serverName : _slaveName, vvSocket::VV_TCP);
  _socket->set_debuglevel(vvDebugMsg::getDebugLevel());

  if (_socket->init() == vvSocket::VV_OK)
  {
    delete serverName;
    _socket->no_nagle();
    _socket->putInt32(_type);
    _socket->putBool(_slaveFileName!=NULL);

    if (_slaveFileName)
    {
      _socket->putFileName(_slaveFileName);
      _socket->getVolumeAttributes(vd);
      vvTransFunc tf;
      tf._widgets.removeAll();
      if ((_socket->getTransferFunction(tf)) == vvSocket::VV_OK)
      {
        vd->tf = tf;
      }
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
    delete serverName;
    cerr << "No connection to remote rendering server established at: " << _slaveName << endl;
    return VV_SOCKET_ERROR;
  }
  return VV_OK;
}

void vvRemoteClient::resize(const int w, const int h)
{
  vvDebugMsg::msg(1, "vvRemoteClient::resize()");
  _changes = true;
  _viewportWidth = w;
  _viewportHeight = h;

  if(!_socket)
    return;

  if (_socket->putCommReason(vvSocketIO::VV_RESIZE) == vvSocket::VV_OK)
  {
    _socket->putWinDims(w, h);
  }
}

void vvRemoteClient:: setCurrentFrame(const int index)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setCurrentFrame()");
  _changes = true;

  if(!_socket)
    return;

  if (_socket->putCommReason(vvSocketIO::VV_CURRENT_FRAME) == vvSocket::VV_OK)
  {
    _socket->putInt32(index);
  }
}

void vvRemoteClient::setObjectDirection(const vvVector3* od)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setObjectDirection()");
  _changes = true;

  if(!_socket)
    return;

  if (_socket->putCommReason(vvSocketIO::VV_OBJECT_DIRECTION) == vvSocket::VV_OK)
  {
    _socket->putVector3(*od);
  }
}

void vvRemoteClient::setViewingDirection(const vvVector3* vd)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setViewingDirection()");
  _changes = true;

  if(!_socket)
    return;

  if (_socket->putCommReason(vvSocketIO::VV_VIEWING_DIRECTION) == vvSocket::VV_OK)
  {
    _socket->putVector3(*vd);
  }
}

void vvRemoteClient::setPosition(const vvVector3* p)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setPosition()");
  _changes = true;

  if(!_socket)
    return;

  if (_socket->putCommReason(vvSocketIO::VV_POSITION) == vvSocket::VV_OK)
  {
    _socket->putVector3(*p);
  }
}

void vvRemoteClient::updateTransferFunction()
{
  vvDebugMsg::msg(1, "vvRemoteClient::updateTransferFunction()");
  _changes = true;

  if(!_socket)
    return;

  if (_socket->putCommReason(vvSocketIO::VV_TRANSFER_FUNCTION) == vvSocket::VV_OK)
  {
    _socket->putTransferFunction(vd->tf);
  }
}

void vvRemoteClient::setParameter(const vvRenderer::ParameterType param, const float newValue)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setParameter()");
  _changes = true;
  vvRenderer::setParameter(param, newValue);

  if(!_socket)
    return;

  if (_socket->putCommReason(vvSocketIO::VV_PARAMETER_1) == vvSocket::VV_OK)
  {
    _socket->putInt32((int32_t)param);
    _socket->putFloat(newValue);
  }
}

void vvRemoteClient::setParameterV3(const vvRenderer::ParameterType param, const vvVector3 &newValue)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setParameter()");
  _changes = true;
  vvRenderer::setParameterV3(param, newValue);

  if(!_socket)
    return;

  if (_socket->putCommReason(vvSocketIO::VV_PARAMETER_3) == vvSocket::VV_OK)
  {
    _socket->putInt32((int32_t)param);
    _socket->putVector3(newValue);
  }
}

void vvRemoteClient::setParameterV4(const vvRenderer::ParameterType param, const vvVector4 &newValue)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setParameter()");
  _changes = true;

  if(!_socket)
    return;

  vvRenderer::setParameterV4(param, newValue);
  if (_socket->putCommReason(vvSocketIO::VV_PARAMETER_3) == vvSocket::VV_OK)
  {
    _socket->putInt32((int32_t)param);
    _socket->putVector4(newValue);
  }
}

vvRemoteClient::ErrorType vvRemoteClient::requestFrame() const
{
  vvDebugMsg::msg(1, "vvRemoteClient::requestFrame()");

  if(!_socket)
    return vvRemoteClient::VV_SOCKET_ERROR;

  if(_socket->putCommReason(vvSocketIO::VV_MATRIX) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

  if(_socket->putMatrix(&_currentPr) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

  if(_socket->putMatrix(&_currentMv) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

  return vvRemoteClient::VV_OK;
}

void vvRemoteClient::exit()
{
  vvDebugMsg::msg(1, "vvRemoteClient::exit()");

  if(_socket)
    _socket->putCommReason(vvSocketIO::VV_EXIT);
  delete _socket;
  _socket = NULL;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
