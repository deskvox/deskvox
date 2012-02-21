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

#ifndef _VVTCPSOCKET_H_
#define _VVTCPSOCKET_H_

#include <iostream>

#include "vvplatform.h"
#ifndef _WIN32
# include <netdb.h>
# include <unistd.h>
# include <arpa/inet.h>
# include <fcntl.h>
# include <netinet/in.h>
# include <netinet/tcp.h>
# include <sys/time.h>
# include <sys/errno.h>
# include <sys/param.h>
# include <sys/ioctl.h>
# include <sys/socket.h>
# include <sys/wait.h>
# include <errno.h>
#endif
#include <string.h>
#include <signal.h>
#include <stdlib.h>
#ifdef __sun
#include <sys/filio.h>
#endif

#include "vvexport.h"
#include "vvinttypes.h"
#include "vvsocket.h"

//----------------------------------------------------------------------------
/** This class provides basic socket functionality. It is used for TCP and UDP
    sockets. For example code see documentation about vvSocket  <BR>
*/
class VIRVOEXPORT vvTcpSocket : public vvSocket
{
  public:
    vvTcpSocket();
    ~vvTcpSocket();

    ErrorType connectToHost(const std::string host, const ushort port, const int clminport = 0, const int clmaxport = 0);
    ErrorType disconnectFromHost();

  private:
    ssize_t readn(char*, size_t);
    ssize_t writen(const char*, size_t);
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
