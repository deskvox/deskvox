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

#include <sstream>

#include "vvdebugmsg.h"
#include "vvsocketmonitor.h"
#include "vvtcpserver.h"
#include "vvtcpsocket.h"


vvTcpServer::vvTcpServer(ushort port)
{
#ifdef _WIN32
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2,0), &wsaData) != 0)
    vvDebugMsg::msg(1, "WSAStartup failed!");
#endif

#ifdef _WIN32
  char optval=1;
#else
  int optval=1;
#endif

  if ((_sockfd = socket(AF_INET, SOCK_STREAM, 0 )) < 0)
  {
    vvDebugMsg::msg(1, "Error: socket()", true);
    _ready = false;
  }

  if (setsockopt(_sockfd, SOL_SOCKET, SO_REUSEADDR, &optval,sizeof(optval)))
  {
    vvDebugMsg::msg(1, "Error: setsockopt()");
    _ready = false;
  }

  memset((char *) &_hostAddr, 0, sizeof(_hostAddr));
  _hostAddr.sin_family = AF_INET;
  _hostAddr.sin_port = htons((unsigned short)port);
  _hostAddr.sin_addr.s_addr = INADDR_ANY;
  _hostAddrlen = sizeof(_hostAddr);
  if  (bind(_sockfd, (struct sockaddr *)&_hostAddr, _hostAddrlen))
  {
    vvDebugMsg::msg(1, "Error: bind()");
    _ready = false;
  }

  if (listen(_sockfd, 1))
  {
    vvDebugMsg::msg(1, "Error: listen()");
    _ready = false;
  }
  _ready = true;
}

vvTcpServer::~vvTcpServer()
{
#ifdef _WIN32
  if(_sockfd >= 0)
    if(closesocket(_sockfd))
      if (WSAGetLastError() ==  WSAEWOULDBLOCK)
        vvDebugMsg::msg(1, "Linger time expires");
  WSACleanup();
#else
  if(_sockfd >= 0)
    if (close(_sockfd))
      if (errno ==  EWOULDBLOCK)
        vvDebugMsg::msg(1, "Linger time expires");
#endif
}

vvTcpSocket* vvTcpServer::nextConnection(double to)
{
  if(!_ready)
  {
    vvDebugMsg::msg(2, "vvTcpServer::nextConnection() error: server not correctly initialized");
  }

#ifdef _WIN32
  SOCKET n;
#else
  int n;
#endif

#ifndef _WIN32
  int flags = fcntl(_sockfd, F_GETFL, 0);
  if(flags < 0)
  {
    vvDebugMsg::msg(1, "vvTcpServer::nextConnection() error: Getting flags of server-socket failed");
    return NULL;
  }
#endif

  if (to < 0.0 ? false : true)
  {
#ifdef _WIN32
    unsigned long tru = 1;
    ioctlsocket(_sockfd, FIONBIO, &tru);
#else
    if(fcntl(_sockfd, F_SETFL, flags|O_NONBLOCK))
    {
      vvDebugMsg::msg(1, "vvTcpServer::nextConnection() error: setting O_NONBLOCK on server-socket failed");
    }
#endif
    vvTcpSocket sock;
    sock.setSockfd(_sockfd);

    std::vector<vvSocket*> socks;
    socks.push_back(&sock);

    vvSocketMonitor sm;
    sm.setReadFds(socks);

    vvSocket* ready;
    sm.wait(&ready, &to);

    sock.setSockfd(0);

    if(ready == NULL)
      return NULL;
  }
  else
  {
#ifdef _WIN32
    unsigned long tru = 0;
    ioctlsocket(_sockfd, FIONBIO, &tru);
#else
    if(fcntl(_sockfd, F_SETFL, flags & (~O_NONBLOCK)))
    {
      vvDebugMsg::msg(1, "vvTcpServer::nextConnection() error: removing O_NONBLOCK from server-socket failed.");
      return NULL;
    }
#endif
  }

  if ( (n = accept(_sockfd, (struct sockaddr *)&_hostAddr, &_hostAddrlen)) < 0)
  {
    vvDebugMsg::msg(1, "Error: accept()");
    return NULL;
  }

  vvTcpSocket *next = new vvTcpSocket();
  next->setSockfd(n);

  std::ostringstream errmsg;
  errmsg << "Incoming connection from " << inet_ntoa(_hostAddr.sin_addr);
  vvDebugMsg::msg(2, errmsg.str().c_str());

  return next;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
