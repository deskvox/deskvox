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

#ifndef VVREMOTESERVER_H
#define VVREMOTESERVER_H

#include "vvexport.h"
#include "vvoffscreenbuffer.h"
#include "vvsocketio.h"

class vvRenderer;

class VIRVOEXPORT vvRemoteServer
{
public:
  enum ErrorType
  {
    VV_OK = 0,
    VV_SOCKET_ERROR,
    VV_FILEIO_ERROR
  };

  vvRemoteServer();
  virtual ~vvRemoteServer();

  bool getLoadVolumeFromFile() const;

  vvRemoteServer::ErrorType initSocket(int port, vvSocket::SocketType st);
  vvRemoteServer::ErrorType initData(vvVolDesc*& vd);

  virtual void renderLoop(vvRenderer* renderer);
protected:
  vvSocketIO* _socket;                    ///< socket for remote rendering

  bool _loadVolumeFromFile;

  virtual void renderImage(vvMatrix& pr, vvMatrix& mv, vvRenderer* renderer) = 0;
  virtual void resize(int w, int h) = 0;
};

#endif // VVREMOTESERVER_H
