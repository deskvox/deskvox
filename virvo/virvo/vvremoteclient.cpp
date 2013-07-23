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

#include "private/vvgltools.h"

namespace
{
  virvo::Viewport viewport;
}

vvRemoteClient::vvRemoteClient(vvVolDesc *vd, vvRenderState renderState,
                               vvTcpSocket* socket, const std::string &filename)
   : vvRenderer(vd, renderState)
   , _socket(socket)
   , _filename(filename)
   , _socketIO(NULL)
   , _changes(true)
{
  vvDebugMsg::msg(1, "vvRemoteClient::vvRemoteClient()");

  _socketIO = new vvSocketIO(_socket);
  sendVolume(vd);
}

vvRemoteClient::~vvRemoteClient()
{
  vvDebugMsg::msg(1, "vvRemoteClient::~vvRemoteClient()");

  delete _socketIO;
}

void vvRemoteClient::renderVolumeGL()
{
  vvGLTools::getModelviewMatrix(&_currentMv);
  vvGLTools::getProjectionMatrix(&_currentPr);

  if (render() != vvRemoteClient::VV_OK)
  {
    vvDebugMsg::msg(0, "vvRemoteClient::renderVolumeGL(): remote rendering error");
  }
}

bool vvRemoteClient::resize(int w, int h)
{
  if (vvSocket::VV_OK != this->_socketIO->putEvent(virvo::WindowResize))
  {
  }

  if (vvSocket::VV_OK != this->_socketIO->putWinDims(w, h))
  {
  }

  return BaseType::resize(w, h);
}

bool vvRemoteClient::present() const
{
  return true;
}

vvRemoteClient::ErrorType vvRemoteClient::sendVolume(vvVolDesc*& vd)
{
  if (!_filename.empty())
  {
    _socketIO->putEvent(virvo::VolumeFile);
    _socketIO->putFileName(_filename);
  }
  else
  {
    if (_socketIO->putEvent(virvo::Volume) == vvSocket::VV_OK)
    {
      switch (_socketIO->putVolume(vd))
      {
        case vvSocket::VV_OK:
          vvDebugMsg::msg(1, "Volume transferred successfully");
          break;
        case vvSocket::VV_ALLOC_ERROR:
          vvDebugMsg::msg(0, "Not enough memory to accomodate volume");
          return VV_SOCKET_ERROR;
        default:
          vvDebugMsg::msg(0, "Unknown error writing volume to socket");
          return VV_SOCKET_ERROR;
      }
    }
    else
    {
      vvDebugMsg::msg(0, "Unknown socket error");
      return VV_SOCKET_ERROR;
    }
  }
  return VV_OK;
}

void vvRemoteClient::setCurrentFrame(size_t index)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setCurrentFrame()");
  _changes = true;

  if (_socketIO == NULL)
  {
    return;
  }

  if (_socketIO->putEvent(virvo::CurrentFrame) == vvSocket::VV_OK)
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

  if (_socketIO->putEvent(virvo::ObjectDirection) == vvSocket::VV_OK)
  {
    _socketIO->putVector3(od);
  }
}

void vvRemoteClient::setViewingDirection(const vvVector3& vd)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setViewingDirection()");
  _changes = true;

  if (!_socketIO)
  {
    return;
  }

  if (_socketIO->putEvent(virvo::ViewingDirection) == vvSocket::VV_OK)
  {
    _socketIO->putVector3(vd);
  }
}

void vvRemoteClient::setPosition(const vvVector3& p)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setPosition()");
  _changes = true;

  if (!_socketIO)
  {
    return;
  }

  if (_socketIO->putEvent(virvo::Position) == vvSocket::VV_OK)
  {
    _socketIO->putVector3(p);
  }
}

void vvRemoteClient::updateTransferFunction()
{
  vvDebugMsg::msg(1, "vvRemoteClient::updateTransferFunction()");
  _changes = true;

  if (!_socketIO)
  {
    return;
  }

  if (_socketIO->putEvent(virvo::TransFunc) == vvSocket::VV_OK)
  {
    _socketIO->putTransferFunction(vd->tf);
  }
}

void vvRemoteClient::setParameter(ParameterType param, const vvParam& value)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setParameter()");
  _changes = true;
  vvRenderer::setParameter(param, value);

  if (_socketIO == NULL)
  {
    return;
  }

  if (value.isa(vvParam::VV_BOOL))
  {
    if (_socketIO->putEvent(virvo::ParameterBool) == vvSocket::VV_OK)
    {
      _socketIO->putInt32((int32_t)param);
      _socketIO->putBool(value);
    }
    return;
  }

  if (value.isa(vvParam::VV_INT))
  {
    if (_socketIO->putEvent(virvo::ParameterInt32) == vvSocket::VV_OK)
    {
      _socketIO->putInt32((int32_t)param);
      _socketIO->putInt32((int32_t)value);
    }
    return;
  }

  if (value.isa(vvParam::VV_UINT))
  {
    if (_socketIO->putEvent(virvo::ParameterUint32) == vvSocket::VV_OK)
    {
      _socketIO->putInt32(static_cast<int32_t>(param));
      _socketIO->putUint32(value);
    }
    return;
  }

  if (value.isa(vvParam::VV_ULONG))
  {
#ifdef __LP64__
    if (_socketIO->putEvent(virvo::ParameterUint64) == vvSocket::VV_OK)
    {
      _socketIO->putInt32(static_cast<int32_t>(param));
      _socketIO->putUint64(value);
    }
    return;
#else
    assert(0);
#endif
  }

  if (value.isa(vvParam::VV_FLOAT))
  {
    if (_socketIO->putEvent(virvo::ParameterFloat) == vvSocket::VV_OK)
    {
      _socketIO->putInt32((int32_t)param);
      _socketIO->putFloat(value);
    }
    return;
  }

  if (value.isa(vvParam::VV_VEC3))
  {
    if (_socketIO->putEvent(virvo::ParameterVec3) == vvSocket::VV_OK)
    {
      _socketIO->putInt32((int32_t)param);
      _socketIO->putVector3(value);
    }
    return;
  }

  if (value.isa(vvParam::VV_VEC4))
  {
    if (_socketIO->putEvent(virvo::ParameterVec4) == vvSocket::VV_OK)
    {
      _socketIO->putInt32((int32_t)param);
      _socketIO->putVector4(value);
    }
    return;
  }

  if (value.isa(vvParam::VV_COLOR))
  {
    if (_socketIO->putEvent(virvo::ParameterColor) == vvSocket::VV_OK)
    {
      _socketIO->putInt32((int32_t)param);
      _socketIO->putColor(value);
    }
    return;
  }

  if (value.isa(vvParam::VV_SIZE3))
  {
#ifdef __LP64__
    if (_socketIO->putEvent(virvo::ParameterSize3) == vvSocket::VV_OK)
    {
      _socketIO->putInt32(static_cast<int32_t>(param));
      _socketIO->putUint32(static_cast<uint32_t>(sizeof(size_t)));
      _socketIO->putUint64(value.asSize3()[0]);
      _socketIO->putUint64(value.asSize3()[1]);
      _socketIO->putUint64(value.asSize3()[2]);
    }
    return;
#else
    assert(0);
#endif
  }

  if (value.isa(vvParam::VV_AABBI))
  {
    if (_socketIO->putEvent(virvo::ParameterAABBI) == vvSocket::VV_OK)
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

  if (!_socketIO)
  {
    return vvRemoteClient::VV_SOCKET_ERROR;
  }

  if (_socketIO->putEvent(virvo::CameraMatrix) != vvSocket::VV_OK)
  {
    return vvRemoteClient::VV_SOCKET_ERROR;
  }

  if (_socketIO->putMatrix(&_currentPr) != vvSocket::VV_OK)
  {
    return vvRemoteClient::VV_SOCKET_ERROR;
  }

  if (_socketIO->putMatrix(&_currentMv) != vvSocket::VV_OK)
  {
    return vvRemoteClient::VV_SOCKET_ERROR;
  }

  return vvRemoteClient::VV_OK;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
