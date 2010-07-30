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

#ifndef VV_RENDERSLAVE_H
#define VV_RENDERSLAVE_H

#include "vvexport.h"
#include "vvoffscreenbuffer.h"
#include "vvsocketio.h"
#include "vvtexrend.h"

class VIRVOEXPORT vvRenderSlave
{
public:
  enum ErrorType
  {
    VV_OK = 0,
    VV_SOCKET_ERROR,
    VV_FILEIO_ERROR
  };

  vvRenderSlave();
  ~vvRenderSlave();

  vvRenderSlave::ErrorType initRemoteRenderingSocket(const int port, vvSocket::SocketType st);
  vvRenderSlave::ErrorType initRemoteRenderingData(vvVolDesc*& vd);
  vvRenderSlave::ErrorType initRemoteRenderingBricks(std::vector<vvBrick*>& bricks);
  void  remoteRenderingLoop(vvTexRend* renderer);
private:
  vvOffscreenBuffer* _remoteRenderingBuffer;    ///< offscreen buffer for remote rendering
  vvSocketIO* _remoteRenderingSocket;           ///< socket for remote rendering
};

#endif
