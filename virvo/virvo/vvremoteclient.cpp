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
#include "vvtcpsocket.h"
#include "vvopengl.h"
#include "vvvoldesc.h"
#include "vvimage.h"
#include "vvtoolshed.h"

using std::cerr;
using std::endl;

vvRemoteClient::vvRemoteClient(vvVolDesc *vd, vvRenderState renderState, uint32_t type,
                               const char *slaveName, int port,
                               const char *slaveFileName)
   : vvRenderer(vd, renderState),
    _type(type),
    _slaveName(slaveName),
    _port(port),
    _slaveFileName(slaveFileName),
    _socket(NULL),
    _socketIO(NULL),
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

  if(!_slaveName || !_slaveName[0])
  {
    if(const char *s = getenv("VV_SERVER"))
    {
      vvDebugMsg::msg(1, "remote rendering server from environment: ", s);
      _slaveName = s;
    }
  }

  if(!_slaveName || !_slaveName[0])
  {
    vvDebugMsg::msg(0, "no server specified");
    return VV_SOCKET_ERROR;
  }

  char *serverName = NULL;
  int port = vvToolshed::parsePort(_slaveName);
  if(port != -1)
  {
    serverName = vvToolshed::stripPort(_slaveName);
  }

  if(_port != -1)
    port = _port;
  if(port == -1)
    port = defaultPort;

  _socket = new vvTcpSocket;
  _socketIO = new vvSocketIO(_socket);

  if (_socket->connectToHost(serverName ? serverName : _slaveName, port) == vvSocket::VV_OK)
  {
    delete serverName;
    _socket->setParameter(vvSocket::VV_NO_NAGLE, true);
    _socketIO->putInt32(_type);

    if (_slaveFileName && _slaveFileName[0])
    {
      _socketIO->putBool(true);
      _socketIO->putFileName(_slaveFileName);
      _socketIO->getVolumeAttributes(vd);
      vvTransFunc tf;
      tf._widgets.removeAll();
      if ((_socketIO->getTransferFunction(tf)) == vvSocket::VV_OK)
      {
        vd->tf = tf;
      }
    }
    else
    {
      _socketIO->putBool(false);
      switch (_socketIO->putVolume(vd))
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
    delete _socket;
    delete _socketIO;
    _socket = NULL;
    _socketIO = NULL;
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

  if(!_socketIO)
    return;

  if (_socketIO->putCommReason(vvSocketIO::VV_RESIZE) == vvSocket::VV_OK)
  {
    _socketIO->putWinDims(w, h);
  }
}

void vvRemoteClient:: setCurrentFrame(const int index)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setCurrentFrame()");
  _changes = true;

  if(!_socketIO)
    return;

  if (_socketIO->putCommReason(vvSocketIO::VV_CURRENT_FRAME) == vvSocket::VV_OK)
  {
    _socketIO->putInt32(index);
  }
}

void vvRemoteClient::setObjectDirection(const vvVector3* od)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setObjectDirection()");
  _changes = true;

  if(!_socketIO)
    return;

  if (_socketIO->putCommReason(vvSocketIO::VV_OBJECT_DIRECTION) == vvSocket::VV_OK)
  {
    _socketIO->putVector3(*od);
  }
}

void vvRemoteClient::setViewingDirection(const vvVector3* vd)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setViewingDirection()");
  _changes = true;

  if(!_socketIO)
    return;

  if (_socketIO->putCommReason(vvSocketIO::VV_VIEWING_DIRECTION) == vvSocket::VV_OK)
  {
    _socketIO->putVector3(*vd);
  }
}

void vvRemoteClient::setPosition(const vvVector3* p)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setPosition()");
  _changes = true;

  if(!_socketIO)
    return;

  if (_socketIO->putCommReason(vvSocketIO::VV_POSITION) == vvSocket::VV_OK)
  {
    _socketIO->putVector3(*p);
  }
}

void vvRemoteClient::updateTransferFunction()
{
  vvDebugMsg::msg(1, "vvRemoteClient::updateTransferFunction()");
  _changes = true;

  if(!_socketIO)
    return;

  if (_socketIO->putCommReason(vvSocketIO::VV_TRANSFER_FUNCTION) == vvSocket::VV_OK)
  {
    _socketIO->putTransferFunction(vd->tf);
  }
}

void vvRemoteClient::setParameter(const vvRenderer::ParameterType param, const float newValue)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setParameter()");
  _changes = true;
  vvRenderer::setParameter(param, newValue);

  if(!_socketIO)
    return;

  if (_socketIO->putCommReason(vvSocketIO::VV_PARAMETER_1) == vvSocket::VV_OK)
  {
    _socketIO->putInt32((int32_t)param);
    _socketIO->putFloat(newValue);
  }
}

void vvRemoteClient::setParameterV3(const vvRenderer::ParameterType param, const vvVector3 &newValue)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setParameter()");
  _changes = true;
  vvRenderer::setParameterV3(param, newValue);

  if(!_socketIO)
    return;

  if (_socketIO->putCommReason(vvSocketIO::VV_PARAMETER_3) == vvSocket::VV_OK)
  {
    _socketIO->putInt32((int32_t)param);
    _socketIO->putVector3(newValue);
  }
}

void vvRemoteClient::setParameterV4(const vvRenderer::ParameterType param, const vvVector4 &newValue)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setParameter()");
  _changes = true;

  if(!_socketIO)
    return;

  vvRenderer::setParameterV4(param, newValue);
  if (_socketIO->putCommReason(vvSocketIO::VV_PARAMETER_3) == vvSocket::VV_OK)
  {
    _socketIO->putInt32((int32_t)param);
    _socketIO->putVector4(newValue);
  }
}

vvRemoteClient::ErrorType vvRemoteClient::requestFrame() const
{
  vvDebugMsg::msg(1, "vvRemoteClient::requestFrame()");

  if(!_socketIO)
    return vvRemoteClient::VV_SOCKET_ERROR;

  if(_socketIO->putCommReason(vvSocketIO::VV_MATRIX) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

  if(_socketIO->putMatrix(&_currentPr) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

  if(_socketIO->putMatrix(&_currentMv) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

  return vvRemoteClient::VV_OK;
}

void vvRemoteClient::exit()
{
  vvDebugMsg::msg(1, "vvRemoteClient::exit()");

  if(_socketIO)
    _socketIO->putCommReason(vvSocketIO::VV_EXIT);
  delete _socket;
  _socket = NULL;
  delete _socketIO;
  _socketIO = NULL;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
