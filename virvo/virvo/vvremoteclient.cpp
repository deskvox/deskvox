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
                               vvTcpSocket* socket, const std::string &filename)
   : vvRenderer(vd, renderState)
   , _type(type)
   , _socket(socket)
   , _filename(filename)
   , _socketIO(NULL)
   , _changes(true)
   , _viewportWidth(-1)
   , _viewportHeight(-1)
{
  vvDebugMsg::msg(1, "vvRemoteClient::vvRemoteClient()");

  initSocket(vd);
}

vvRemoteClient::~vvRemoteClient()
{
  vvDebugMsg::msg(1, "vvRemoteClient::~vvRemoteClient()");

  quit();
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

  delete _socketIO;
  _socketIO = new vvSocketIO(_socket);

  _socketIO->putInt32(_type);

  if (!_filename.empty())
  {
    _socketIO->putBool(true);
    _socketIO->putFileName(_filename.c_str());
    _socketIO->getVolumeAttributes(vd);
    vvTransFunc tf;
    for (std::vector<vvTFWidget*>::const_iterator it = tf._widgets.begin();
         it != tf._widgets.end(); ++it)
    {
      delete *it;
    }
    tf._widgets.clear();
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

void vvRemoteClient::setObjectDirection(const vvVector3& od)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setObjectDirection()");
  _changes = true;

  if(!_socketIO)
    return;

  if (_socketIO->putCommReason(vvSocketIO::VV_OBJECT_DIRECTION) == vvSocket::VV_OK)
  {
    _socketIO->putVector3(od);
  }
}

void vvRemoteClient::setViewingDirection(const vvVector3& vd)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setViewingDirection()");
  _changes = true;

  if(!_socketIO)
    return;

  if (_socketIO->putCommReason(vvSocketIO::VV_VIEWING_DIRECTION) == vvSocket::VV_OK)
  {
    _socketIO->putVector3(vd);
  }
}

void vvRemoteClient::setPosition(const vvVector3& p)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setPosition()");
  _changes = true;

  if(!_socketIO)
    return;

  if (_socketIO->putCommReason(vvSocketIO::VV_POSITION) == vvSocket::VV_OK)
  {
    _socketIO->putVector3(p);
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

void vvRemoteClient::setParameter(ParameterType param, const vvParam& value)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setParameter()");
  _changes = true;
  vvRenderer::setParameter(param, value);

  if(!_socketIO)
    return;

  if (value.isa(vvParam::VV_BOOL))
  {
    if (_socketIO->putCommReason(vvSocketIO::VV_PARAMETER_1B) == vvSocket::VV_OK)
    {
      _socketIO->putInt32((int32_t)param);
      _socketIO->putBool(value);
    }
    return;
  }

  if (value.isa(vvParam::VV_INT))
  {
    if (_socketIO->putCommReason(vvSocketIO::VV_PARAMETER_1I) == vvSocket::VV_OK)
    {
      _socketIO->putInt32((int32_t)param);
      _socketIO->putInt32((int32_t)value);
    }
    return;
  }

  if (value.isa(vvParam::VV_FLOAT))
  {
    if (_socketIO->putCommReason(vvSocketIO::VV_PARAMETER_1F) == vvSocket::VV_OK)
    {
      _socketIO->putInt32((int32_t)param);
      _socketIO->putFloat(value);
    }
    return;
  }

  if (value.isa(vvParam::VV_VEC3))
  {
    if (_socketIO->putCommReason(vvSocketIO::VV_PARAMETER_3F) == vvSocket::VV_OK)
    {
      _socketIO->putInt32((int32_t)param);
      _socketIO->putVector3(value);
    }
    return;
  }

  if (value.isa(vvParam::VV_VEC4))
  {
    if (_socketIO->putCommReason(vvSocketIO::VV_PARAMETER_4F) == vvSocket::VV_OK)
    {
      _socketIO->putInt32((int32_t)param);
      _socketIO->putVector4(value);
    }
    return;
  }

  if (value.isa(vvParam::VV_AABBI))
  {
    if (_socketIO->putCommReason(vvSocketIO::VV_PARAMETER_AABBI) == vvSocket::VV_OK)
    {
      _socketIO->putInt32((int32_t)param);
      _socketIO->putAABBi(value);
    }
    return;
  }

  assert( 0 && "Parameter not handled" );
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

void vvRemoteClient::quit()
{
  vvDebugMsg::msg(1, "vvRemoteClient::quit()");

  if(_socketIO)
  {
    _socketIO->putCommReason(vvSocketIO::VV_QUIT);
    delete _socketIO;
    _socketIO = NULL;
  }
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
