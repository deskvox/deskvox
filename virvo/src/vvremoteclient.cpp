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

vvRemoteClient::vvRemoteClient(std::vector<const char*>& slaveNames, std::vector<int>& slavePorts,
                               std::vector<const char*>& slaveFileNames,
                               const char* fileName)
  // : vvRenderer(/* */),
  : _fileName(fileName), _slaveNames(slaveNames),
    _slavePorts(slavePorts),
    _slaveFileNames(slaveFileNames)
{

}

vvRemoteClient::~vvRemoteClient()
{
  clearImages();
  delete _images;
}

vvRemoteClient::ErrorType vvRemoteClient::initSockets(const int defaultPort, const bool redistributeVolData,
                                                      vvVolDesc*& vd)
{
  const bool loadVolumeFromFile = !redistributeVolData;
  for (int s=0; s<_slaveNames.size(); ++s)
  {
    if (_slavePorts[s] == -1)
    {
        _sockets.push_back(new vvSocketIO(defaultPort, _slaveNames[s], vvSocket::VV_TCP));
    }
    else
    {
      _sockets.push_back(new vvSocketIO(_slavePorts[s], _slaveNames[s], vvSocket::VV_TCP));
    }
    _sockets[s]->set_debuglevel(vvDebugMsg::getDebugLevel());

    if (_sockets[s]->init() == vvSocket::VV_OK)
    {
      _sockets[s]->no_nagle();
      _sockets[s]->putBool(loadVolumeFromFile);

      if (loadVolumeFromFile)
      {
        const bool allFileNamesAreEqual = (_slaveFileNames.size() == 0);
        if (allFileNamesAreEqual)
        {
          _sockets[s]->putFileName(_fileName);
        }
        else
        {
          if (_slaveFileNames.size() > s)
          {
            _sockets[s]->putFileName(_slaveFileNames[s]);
          }
          else
          {
            // Not enough file names specified, try this one.
            _sockets[s]->putFileName(_fileName);
          }
        }
      }
      else
      {
        switch (_sockets[s]->putVolume(vd))
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
      cerr << "No connection to remote rendering server established at: " << _slaveNames[s] << endl;
      cerr << "Falling back to local rendering" << endl;
      return VV_SOCKET_ERROR;
    }
  }
  createImageVector();
  createThreads();
  return VV_OK;
}

void vvRemoteClient::setBackgroundColor(const vvVector3& bgColor)
{
  _bgColor = bgColor;
}

void vvRemoteClient::setCurrentFrame(const int /* index */)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setCurrentFrame()");
  //vvRenderer::setCurrentFrame(index);
}

void vvRemoteClient::setObjectDirection(const vvVector3* /* od */)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setObjectDirection()");
  //vvRenderer::setObjectDirection(od);
}

void vvRemoteClient::setROIEnable(const bool /* flag */)
{
  vvDebugMsg::msg(1, "vvRemoteClient::setROIEnable()");
  //vvRenderer::setROIEnable(flag);
}

void vvRemoteClient::setPosition(const vvVector3* p)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setPosition()");
  //vvRenderer::setPosition(p);
}

void vvRemoteClient::setProbePosition(const vvVector3* /* probePos */)
{
  vvDebugMsg::msg(1, "vvRemoteClient::setProbePosition()");
  //vvRenderer::setProbePositin(probePos);
}

void vvRemoteClient::setProbeSize(const vvVector3* /* newSize */)
{
  vvDebugMsg::msg(1, "vvRemoteClient::setProbeSize()");
  //vvRenderer::setProbeSize(newSize);
}

void vvRemoteClient::toggleBoundingBox()
{
  vvDebugMsg::msg(3, "vvRemoteClient::toggleBoundingBox()");
}

void vvRemoteClient::updateTransferFunction(vvTransFunc& /* tf */)
{
  vvDebugMsg::msg(1, "vvRemoteClient::updateTransferFunction()");
}

void vvRemoteClient::setParameter(const vvRenderer::ParameterType /* param */, const float /* newValue */, const char*)
{
  vvDebugMsg::msg(3, "vvRemoteClient::setParameter()");
  //vvRenderer::setParameter(param, newValue);
}

void vvRemoteClient::clearImages()
{
  for (std::vector<vvImage*>::const_iterator it = _images->begin(); it != _images->end();
       ++it)
  {
    delete (*it);
  }
}

void vvRemoteClient::createImageVector()
{
  _images = new std::vector<vvImage*>(_sockets.size());
}
