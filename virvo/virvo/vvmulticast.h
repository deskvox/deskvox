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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_NORM

#include "vvexport.h"
#include "vvinttypes.h"

#include <string>
#include <normApi.h>

class VIRVOEXPORT vvMulticast
{
public:
  enum MulticastType
  {
    VV_SENDER = 0,
    VV_RECEIVER
  };

  vvMulticast(const char* addr, const ushort port, const MulticastType type);
  ~vvMulticast();

  bool write(const uchar* bytes, const uint size, const double timeout = -1.0);  // send bytes to multicast-adress
  int read(const uint size, uchar*& data, const double timeout = -1.0);          // read until "size" bytes or timeout is reached. Return number of read bytes.

private:
  MulticastType      _type;
  NormInstanceHandle _instance;
  NormSessionHandle  _session;
  NormObjectHandle   _object;
};

#endif

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
