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

#ifndef _VV_RENDERMASTER_H_
#define _VV_RENDERMASTER_H_

#include "vvbsptree.h"
#include "vvexport.h"
#include "vvopengl.h"
#include "vvsocketio.h"
#include "vvvoldesc.h"

#include <vector>

class vvSlaveVisitor;
class vvTexRend;

class VIRVOEXPORT vvRenderMaster
{
public:
  enum ErrorType
  {
    VV_OK = 0,
    VV_SOCKET_ERROR
  };

  vvRenderMaster(std::vector<char*>& slaveNames, std::vector<int>& slavePorts,
                 std::vector<char*>& slaveFileNames,
                 const char* fileName);
  ~vvRenderMaster();

  ErrorType initSockets(const int port, vvSocket::SocketType st,
                        const bool redistributeVolData,
                        vvVolDesc*& vd);
  ErrorType initBricks(vvTexRend* renderer);
  void render(const float bgColor[3]) const;

  void adjustQuality(const float quality);
  void resize(const int w, const int h);
private:
  std::vector<char*> _slaveNames;
  std::vector<int> _slavePorts;
  std::vector<char*> _slaveFileNames;
  std::vector<vvSocketIO*>  _sockets;

  const char* _fileName;

  vvTexRend* _renderer;
  vvBspTree* _bspTree;
  vvSlaveVisitor* _visitor;
};

#endif
