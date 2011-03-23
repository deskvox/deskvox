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

#ifndef _WIN32

#include "vvdebugmsg.h"
#include "vvsocketmonitor.h"

#include <fcntl.h>
#include <iostream>

using std::cerr;
using std::endl;

vvSocketMonitor::vvSocketMonitor(std::vector<vvSocket*>& sockets)
  : _sockets(sockets)
{

}

vvSocket* vvSocketMonitor::wait()
{
  vvDebugMsg::msg(3, "vvSocketMonitor::wait()");

  fd_set sockfds;

  FD_ZERO(&sockfds);

  int highestSocketNum = -1;
  for (std::vector<vvSocket*>::const_iterator it = _sockets.begin(); it != _sockets.end(); ++it)
  {
    vvSocket* socket = (*it);
    FD_SET(socket->get_sockfd(), &sockfds);

    if (socket->get_sockfd() > highestSocketNum)
    {
      highestSocketNum = socket->get_sockfd();
    }
  }

  if (select(highestSocketNum + 1, &sockfds, 0, 0, 0) > 0)
  {
    for (std::vector<vvSocket*>::const_iterator it = _sockets.begin(); it != _sockets.end(); ++it)
    {
      vvSocket* socket = (*it);
      if (FD_ISSET(socket->get_sockfd(), &sockfds))
      {
        return socket;
      }
    }
  }

  return NULL;
}

void vvSocketMonitor::clear()
{
  for (std::vector<vvSocket*>::const_iterator it = _sockets.begin(); it != _sockets.end(); ++it)
  {
    delete (*it);
  }
}

#endif
