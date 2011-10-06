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
#include "vvvecmath.h"

class vvSocketIO;
class vvImage;

class VIRVOEXPORT vvRemoteClient : public vvRenderer
{
public:
  enum ErrorType
  {
    VV_OK = 0,
    VV_WRONG_RENDERER,
    VV_SOCKET_ERROR,
    VV_MUTEX_ERROR,
    VV_SHADER_ERROR,
    VV_GL_ERROR,
    VV_BAD_IMAGE
  };

  vvRemoteClient(vvVolDesc *vd, vvRenderState renderState, uint32_t type,
                 const char* slaveName, int slavePort,
                 const char* slaveFileName);
  virtual ~vvRemoteClient();

  virtual ErrorType render() = 0;
  void renderVolumeGL();

  void setCurrentFrame(int index);
  void setObjectDirection(const vvVector3* od);
  void setViewingDirection(const vvVector3* vd);
  void setPosition(const vvVector3* p);
  virtual void updateTransferFunction();
  virtual void setParameter(vvRenderer::ParameterType param, float newValue);
  virtual void setParameterV3(vvRenderer::ParameterType param, const vvVector3 &newValue);
  virtual void setParameterV4(vvRenderer::ParameterType param, const vvVector4 &newValue);
  virtual ErrorType requestFrame() const;

protected:
  virtual void exit();

  uint32_t _type;
  const char* _slaveName;
  int _slavePort;
  const char* _slaveFileName;
  vvSocketIO* _socket;

  bool _changes; ///< indicate if a new rendering is required
  int _viewportWidth, _viewportHeight;
  vvMatrix _currentMv;                                    ///< Current modelview matrix
  vvMatrix _currentPr;                                    ///< Current projection matrix
private:
  void resize(int w, int h);
  virtual void destroyThreads() { }

  ErrorType initSocket(vvVolDesc*& vd);
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
