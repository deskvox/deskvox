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

#ifndef _VV_MULTICAST_H_
#define _VV_MULTICAST_H_

#include "vvexport.h"
#include "vvinttypes.h"

#include <string>

typedef const void* NormInstanceHandle;
typedef const void* NormSessionHandle;
typedef const void* NormObjectHandle;

/** Wrapper class for NormAPI.
  This class can be used for lossless multicast communication.

  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
 */
class VIRVOEXPORT vvMulticast
{
public:
  enum MulticastType
  {
    VV_SENDER = 0,
    VV_RECEIVER
  };

  /** Constructor creating a sending or receiving multicast-unit

    \param addr Must be an adress within the range of 224.0.0.0 to 239.255.255.255.
    \note Some addresses are reserved! (See IPv4-documentation for further informations.)
    \param port Desired port number
    \param type Defined by VV_SENDER or VV_RECEIVER
    */
  vvMulticast(const char* addr, const ushort port, const MulticastType type);
  ~vvMulticast();

  /**
    send bytes to multicast-adress
    \param bytes pointer to stored data
    \param size size of data in bytes
    \param timeout timeout in seconds or negative for no timeout
    */
  ssize_t write(const uchar* bytes, const uint size, double timeout = -1.0);
  /**
    read until "size" bytes or timeout is reached
    \param size expected size of data in bytes
    \param bytes pointer for data to be written to
    \param timeout timeout in seconds or negative for no timeout
    \return number of bytes actually read
    */
  ssize_t read(const uint size, uchar*& data, double timeout = -1.0);

private:
  MulticastType      _type;
  NormInstanceHandle _instance;
  NormSessionHandle  _session;
  NormObjectHandle   _object;
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
