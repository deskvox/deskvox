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

#ifndef _VVUDPSOCKET_H_
#define _VVUDPSOCKET_H_

#include "vvexport.h"
#include "vvinttypes.h"
#include "vvsocket.h"

//----------------------------------------------------------------------------
/** UDP Sockets
*/
class VIRVOEXPORT vvUdpSocket : public vvSocket
{
public:

  enum MulticastType
  {
    VV_MC_SENDER,
    VV_MC_RECEIVER
  };

  vvUdpSocket();
  ~vvUdpSocket();

  ErrorType bind(const std::string hostname, const ushort port, const int clmin = 0, const int clmax = 0);
  ErrorType bind(const ushort port);
  ErrorType unbind();

  ErrorType multicast(const std::string hostname, const ushort port, const MulticastType type);

  ErrorType readData (      uchar*, size_t, ssize_t *ret = NULL);
  ErrorType writeData(const uchar*, size_t, ssize_t *ret = NULL);

private:
  ssize_t readn(char*, size_t);
  ssize_t writen(const char*, size_t);

  ErrorType getClientAddr();

  uint _maxSendSize;
  int retValue;

  // multicasting...
  bool _mc;
  sockaddr_in _localSock;
  ip_mreq _mcGroup;
  sockaddr_in _groupSock;
  in_addr _localInterface;
};

#endif
