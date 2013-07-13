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
class vvTcpSocket;
class vvImage;
class vvIbrClient;

class vvRemoteClient : public vvRenderer
{
  typedef vvRenderer BaseType;

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

public:
  VVAPI vvRemoteClient(vvVolDesc *vd, vvRenderState renderState, vvTcpSocket* socket, const std::string &filename);
  VVAPI virtual ~vvRemoteClient();

  VVAPI virtual ErrorType render() = 0;

  VVAPI virtual void renderVolumeGL() VV_OVERRIDE;

  VVAPI virtual bool resize(int w, int h) VV_OVERRIDE;

  VVAPI virtual bool present() const VV_OVERRIDE;

  VVAPI virtual void setCurrentFrame(size_t index) VV_OVERRIDE;
  VVAPI virtual void setObjectDirection(const vvVector3& od) VV_OVERRIDE;
  VVAPI virtual void setViewingDirection(const vvVector3& vd) VV_OVERRIDE;
  VVAPI virtual void setPosition(const vvVector3& p) VV_OVERRIDE;

  VVAPI virtual void updateTransferFunction() VV_OVERRIDE;

  VVAPI virtual void setParameter(ParameterType param, const vvParam& value) VV_OVERRIDE;

  VVAPI virtual ErrorType requestFrame() const;

protected:
  vvTcpSocket* _socket;
  std::string _filename;
  vvSocketIO  *_socketIO;

  bool _changes; ///< indicate if a new rendering is required
  vvMatrix _currentMv;                                    ///< Current modelview matrix
  vvMatrix _currentPr;                                    ///< Current projection matrix

private:
  ErrorType sendVolume(vvVolDesc*& vd);
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
