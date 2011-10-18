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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_NORM

#include "vvdebugmsg.h"
#include "vvinttypes.h"
#include "vvmulticast.h"
#include "vvsocketmonitor.h"

#include <normApi.h>
#include <stdlib.h>
#include <sys/select.h>

vvMulticast::vvMulticast(const char* addr, const ushort port, const MulticastType type)
{
  _type = type;
  _instance = NormCreateInstance();
  _session = NormCreateSession(_instance, addr, port, NORM_NODE_ANY);

  NormSetCongestionControl(_session, true);
  if(VV_SENDER == type)
  {
    NormSessionId sessionId = (NormSessionId)rand();
    // TODO: Adjust these numbers depending on the used network topology
    NormStartSender(_session, sessionId, 1024*1024, 1400, 64, 16);
    NormSetTxSocketBuffer(_session, 512000);
  }
  else if(VV_RECEIVER == type)
  {
    NormStartReceiver(_session, 1024*1024);
  }
};

vvMulticast::~vvMulticast()
{
  if(VV_SENDER == _type)
  {
    NormStopSender(_session);
  }
  else if(VV_RECEIVER == _type)
  {
    NormStopReceiver(_session);
  }
  NormDestroySession(_session);
  NormDestroyInstance(_instance);
}

bool vvMulticast::write(const uchar* bytes, const uint size, const double timeout)
{
  vvDebugMsg::msg(3, "vvMulticast::write()");
  _object = NormDataEnqueue(_session, (char*)bytes, size);

  if(NORM_OBJECT_INVALID ==_object)
  {
    vvDebugMsg::msg(2, "vvMulticast::write(): Norm Object is invalid!");
    return false;
  }

  NormDescriptor normDesc = NormGetDescriptor(_instance);

  vvSocketMonitor* monitor = new vvSocketMonitor;

  std::vector<vvSocket*> sock;
  sock.push_back(new vvSocket(normDesc, vvSocket::VV_UDP));
  monitor->setReadFds(sock);

  vvSocket* ready;
  NormEvent theEvent;
  while(true)
  {
    ready = monitor->wait(timeout);
    if(NULL == ready)
    {
      vvDebugMsg::msg(2, "vvMulticast::write() error or timeout reached!");
      return 0;
    }
    else
    {
      NormGetNextEvent(_instance, &theEvent);
      switch(theEvent.type)
      {
      case NORM_CC_ACTIVE:
        vvDebugMsg::msg(3, "vvMulticast::write() NORM_CC_ACTIVE: transmission still active");
        break;
      case NORM_TX_FLUSH_COMPLETED:
      case NORM_LOCAL_SENDER_CLOSED:
      case NORM_TX_OBJECT_SENT:
        vvDebugMsg::msg(3, "vvMulticast::write() NORM_TX_FLUSH_COMPLETED: transfer completed.");
        return true;
        break;
      default:
        {
          std::string eventmsg = std::string("vvMulticast::write() Norm-Event: ");
          eventmsg += theEvent.type;
          vvDebugMsg::msg(3, eventmsg.c_str());
          break;
        }
      }
    }
  }
}

int vvMulticast::read(const uint size, uchar*& data, const double timeout)
{
  vvDebugMsg::msg(3, "vvMulticast::read()");
  NormDescriptor normDesc = NormGetDescriptor(_instance);

  vvSocketMonitor* monitor = new vvSocketMonitor;

  std::vector<vvSocket*> sock;
  sock.push_back(new vvSocket(normDesc, vvSocket::VV_UDP));
  monitor->setReadFds(sock);

  vvSocket* ready;
  NormEvent theEvent;
  uint bytesReceived;
  bool keepGoing = true;
  while(keepGoing)
  {
    ready = monitor->wait(timeout);
    if(NULL == ready)
    {
      vvDebugMsg::msg(2, "vvMulticast::read() error or timeout reached!");
      return 0;
    }
    else
    {
      NormGetNextEvent(_instance, &theEvent);
      switch(theEvent.type)
      {
      case NORM_RX_OBJECT_UPDATED:
        vvDebugMsg::msg(3, "vvMulticast::read() NORM_RX_OBJECT_UPDATED: the identified receive object has newly received data content.");
        bytesReceived = NormObjectGetSize(theEvent.object) - NormObjectGetBytesPending(theEvent.object);
        break;
      case NORM_RX_OBJECT_COMPLETED:
        vvDebugMsg::msg(3, "vvMulticast::read() NORM_RX_OBJECT_COMPLETED: transfer completed.");
        bytesReceived = NormObjectGetSize(theEvent.object) - NormObjectGetBytesPending(theEvent.object);
        keepGoing = false;
        break;
      case NORM_RX_OBJECT_ABORTED:
        vvDebugMsg::msg(2, "vvMulticast::read() NORM_RX_OBJECT_ABORTED: transfer incomplete!");
        bytesReceived = NormObjectGetSize(theEvent.object) - NormObjectGetBytesPending(theEvent.object);
        keepGoing = false;
        break;
      default:
        {
          std::string eventmsg = std::string("vvMulticast::read() Norm-Event: ");
          eventmsg += theEvent.type;
          vvDebugMsg::msg(3, eventmsg.c_str());
          break;
        }
      }
    }
    if(bytesReceived >= size) keepGoing = false;
  }
  delete monitor;
  data = (uchar*)NormDataDetachData(theEvent.object);
  return bytesReceived;
}

#endif

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
