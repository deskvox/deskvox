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
#include "vvinttypes.h"
#include "vvmulticast2.h"
#include "vvsocketmonitor.h"

//#include <algorithm>

#include "vvsocket.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

vvMulticast2::vvMulticast2(const char* addr, const ushort port, const MulticastType type)
: _type(type)
{
  /* Create a socket on. */
  sd = socket(AF_INET, SOCK_DGRAM, 0);
  if(sd < 0)
  {
    vvDebugMsg::msg(2, "Opening socket error", true);
    return;
  }

#ifndef _WIN32
  // set to 0 for non-blocking
  int nFlags = fcntl(sd, F_GETFL, 0);
  nFlags |= O_NONBLOCK;
  if (fcntl(sd, F_SETFL, nFlags) == -1)
  {
    vvDebugMsg::msg(2, "Set socket non-blocking error:");
    return;
  }
#else
  // set to 1 for non-blocking
  u_long iMode = 1;
  int nFlags = ioctlsocket(sd, FIONBIO, &iMode);
  if (nFlags != NO_ERROR)
  {
    vvDebugMsg::msg(2, "Set socket non-blocking error", true);
    return;
  }
#endif

  if(VV_SENDER == _type)
  {
    /* Initialize the group sockaddr structure with a */
    memset((char *) &groupSock, 0, sizeof(groupSock));
    groupSock.sin_family = AF_INET;
    groupSock.sin_addr.s_addr = inet_addr(addr);
    groupSock.sin_port = htons(port);

    /* Disable loopback so you do not receive your own datagrams. */
    char loopch = 0;
    int err = setsockopt(sd, IPPROTO_IP, IP_MULTICAST_LOOP, (char *)&loopch, sizeof(loopch));
    if(err != 0)
    {
      vvDebugMsg::msg(2, "Setting IP_MULTICAST_LOOP error", true);
#ifndef _WIN32
      close(sd);
#else
      closesocket(sd);
#endif
    }

    /* Set local interface for outbound multicast datagrams. */
    /* The IP address specified must be associated with a local, */
    /* multicast capable interface. */
    localInterface.s_addr = INADDR_ANY;
    if(setsockopt(sd, IPPROTO_IP, IP_MULTICAST_IF, (char *)&localInterface, sizeof(localInterface)) < 0)
    {
      vvDebugMsg::msg(2, "Setting local interface error", true);
      return;
    }
  }
  else
  {
    // RECEIVER -------------

    /* Enable SO_REUSEADDR to allow multiple instances of this */
    /* application to receive copies of the multicast datagrams. */
    int reuse = 1;
    if(setsockopt(sd, SOL_SOCKET, SO_REUSEADDR, (char *)&reuse, sizeof(reuse)) < 0)
    {
      vvDebugMsg::msg(2, "Setting SO_REUSEADDR error", true);
#ifndef _WIN32
      close(sd);
#else
      closesocket(sd);
#endif
    }

    /* Bind to the proper port number with the IP address */
    /* specified as INADDR_ANY. */
    memset((char *) &localSock, 0, sizeof(localSock));
    localSock.sin_family = AF_INET;
    localSock.sin_port = htons(port);
    localSock.sin_addr.s_addr = INADDR_ANY;
    if(bind(sd, (struct sockaddr*)&localSock, sizeof(localSock)))
    {
      vvDebugMsg::msg(2, "Binding datagram socket error", true);
#ifndef _WIN32
      close(sd);
#else
      closesocket(sd);
#endif
    }

    /* Join the multicast group on the local INADDR_ANY */
    /* interface. Note that this IP_ADD_MEMBERSHIP option must be */
    /* called for each local interface over which the multicast */
    /* datagrams are to be received. */
    group.imr_multiaddr.s_addr = inet_addr(addr);
    group.imr_interface.s_addr = INADDR_ANY;
    if(setsockopt(sd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *)&group, sizeof(group)) < 0)
    {
      vvDebugMsg::msg(2, "Adding multicast group error", true);
#ifndef _WIN32
      close(sd);
#else
      closesocket(sd);
#endif
    }
  }
}

//vvMulticast2::~vvMulticast2()
//{
//}

ssize_t vvMulticast2::write(const uchar* bytes, const size_t size, double timeout)
{
  vvDebugMsg::msg(3, "vvMulticast2::write()");

  ssize_t sentBytes = sendto(sd, (char*)bytes, size, 0, (struct sockaddr*)&groupSock, sizeof(groupSock));

//  vvSocketMonitor monitor;
//  monitor.setWriteFds(sd);



//  while(keepGoing)
//  {
//    ready = monitor.wait(timeout);
//    if(NULL == ready)
//    {
//      vvDebugMsg::msg(2, "vvMulticast::write() error or timeout reached!");
//      return 0;
//    }
//    else
//    {
//      NormGetNextEvent(_instance, &theEvent);
//      switch(theEvent.type)
//      {
//      case NORM_CC_ACTIVE:
//        vvDebugMsg::msg(3, "vvMulticast::write() NORM_CC_ACTIVE: transmission still active");
//        break;
//      case NORM_TX_FLUSH_COMPLETED:
//      case NORM_LOCAL_SENDER_CLOSED:
//      case NORM_TX_OBJECT_SENT:
//        vvDebugMsg::msg(3, "vvMulticast::write() NORM_TX_FLUSH_COMPLETED: tile-transfer completed.");
//        bytesSent += size_t(NormObjectGetSize(theEvent.object));
//        keepGoing = false;
//        break;
//      default:
//        {
//          std::string eventmsg = std::string("vvMulticast::write() Norm-Event: ");
//          eventmsg += theEvent.type;
//          vvDebugMsg::msg(3, eventmsg.c_str());
//          break;
//        }
//      }
//    }
//  }

  // TODO:
  // HIER MUSS SELECT REIN!!
  // +++++++++++++++++++++++++++++++++++++++++++++++++

  if(sentBytes < 0)
  {
    vvDebugMsg::msg(2, "vvMulticast2::write() error", true);
  }
  else
  {
//    printf("Sending datagram message...OK\n");
    return sentBytes;
  }

  (void)timeout;
  return -1;
}

ssize_t vvMulticast2::read(uchar* data, const size_t size, double timeout)
{
  vvDebugMsg::msg(3, "vvMulticast2::read()");

  /* Read from the socket. */
  ssize_t bytes = recv(sd, (char*)data, size, 0);

  // TODO:
  // HIER MUSS SELECT REIN!!
  // +++++++++++++++++++++++++++++++++++++++++++++++++
  // ( http://www.sockets.com/winsock.htm )

  if(bytes < 0)
  {
    vvDebugMsg::msg(2, "Reading datagram message error", true);
#ifndef _WIN32
      close(sd);
#else
      closesocket(sd);
#endif
  }
  else
  {
//    printf("Reading datagram message...OK.\n");
    return bytes;
  }
  (void)timeout;
  return -1;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
