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

vvRemoteClient::vvRemoteClient(const char* fileName)
  // : vvRenderer(/* */),
  :_fileName(fileName)
{

}

vvRemoteClient::~vvRemoteClient()
{

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
