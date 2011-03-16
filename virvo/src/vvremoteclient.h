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

#ifndef _VV_REMOTECLIENT_H_
#define _VV_REMOTECLIENT_H_

#include "vvexport.h"
#include "vvrenderer.h"
#include "vvsocketio.h"
#include "vvtransfunc.h"
#include "vvvecmath.h"

class VIRVOEXPORT vvRemoteClient // : public vvRenderer // TODO: derive this from vvRenderer
{
public:
  enum ErrorType
  {
    VV_OK = 0,
    VV_WRONG_RENDERER,
    VV_SOCKET_ERROR,
    VV_MUTEX_ERROR,
    VV_BAD_IMAGE
  };

  vvRemoteClient(std::vector<const char*>& slaveNames, std::vector<int>& slavePorts,
                 std::vector<const char*>& slaveFileNames,
                 const char* fileName);
  virtual ~vvRemoteClient();

  ErrorType initSockets(const int port, const bool redistributeVolData,
                        vvVolDesc*& vd);

  virtual ErrorType setRenderer(vvRenderer*)
  {
    throw "vvRemoteClient::setRenderer(): Not implemented by inherited class\n";
  }

  virtual ErrorType render() = 0;
  virtual void setBackgroundColor(const vvVector3& bgColor);

  void resize(int w, int h);
  void setCurrentFrame(int index);
  void setMipMode(int mipMode);
  void setObjectDirection(const vvVector3* od);
  void setViewingDirection(const vvVector3* vd);
  void setPosition(const vvVector3* p);
  void setROIEnable(bool roiEnabled);
  void setProbePosition(const vvVector3* pos);
  void setProbeSize(const vvVector3* roiSize);
  void toggleBoundingBox();
  void updateTransferFunction(vvTransFunc& tf);
  void setParameter(vvRenderer::ParameterType param, float newValue, const char* = NULL);

protected:
  const char* _fileName;

  std::vector<const char*> _slaveNames;
  std::vector<int> _slavePorts;
  std::vector<const char*> _slaveFileNames;
  std::vector<vvSocketIO*> _sockets;
  std::vector<vvImage*>* _images;

  vvRenderer* _renderer;

  vvVector3 _bgColor;

  void adjustQuality(float quality);
  void setInterpolation(bool interpolation);

  void clearImages();
  void createImageVector();
private:
  virtual void createThreads() { }
  virtual void destroyThreads() { }
};

#endif
