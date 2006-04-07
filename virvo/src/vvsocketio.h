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

#ifndef _VVSOCKETIO_H_
#define _VVSOCKETIO_H_

#include "vvexport.h"
#include "vvsocket.h"
#include "vvvoldesc.h"
#include "vvtoolshed.h"
#include "vvimage.h"
#include "vvvecmath.h"
#include "vvdebugmsg.h"

/** This class provides specific data transfer through sockets.
  It requires the class vvSocket.<BR>
  Here is an example code fragment for a TCP sever which reads
  a volume from a socket and a TCP client which writes a volume to the
  socket:<BR>
  <PRE>

  Server:

  // Create a new TCP socket class instance, which is a server listening on port 17171

vvSocketIO* sio = new vvSocketIO(17171 , vvSocket::VV_TCP);
vvVolDesc* vd = new vvVolDesc();

//Set the parameters of the socket( e.g. connection timer 3 sec, transfer
timer 1.5 sec, socket buffer 65535 bytes, debuglevel 0)
sio->set_sock_param(3.0f, 1.5f, 65535 , 0)

// Get a volume
switch (sio->getVolume(vd))
{
case vvSocket::VV_OK:
cerr << "Volume transferred successfully" << endl;
break;
case vvSocket::VV_ALLOC_ERROR:
cerr << "Not enough memory" << endl;
break;
default:
cerr << "Cannot read volume from socket" << endl;
break;
}
delete sio;

Client:

// Create a new TCP socket class instance, which is a client and connects
// to a server listening on port 17171

char* servername = "buxdehude";
vvSocketIO* sio = new vvSocketIO(17171 , servername, vvSocket::VV_TCP);
vvVolDesc* vd = new vvVolDesc();

//Set the parameters of the socket( e.g. connection timer 3 sec, transfer
timer 1.5 sec, socket buffer 65535 bytes, debuglevel 0)
sio->set_sock_param(3.0f, 1.5f, 65535 , 0);
// Put a volume
switch (sio->putVolume(vd))
{
case vvSocket::VV_OK:
cerr << "Volume transferred successfully" << endl;
break;
case vvSocket::VV_ALLOC_ERROR:
cerr << "Not enough memory" << endl;
break;
default:
cerr << "Cannot write volume to socket" << endl;
break;
}
delete sio;

</PRE>
@see vvSocket
@author Michael Poehnl
*/
class VIRVOEXPORT vvSocketIO : public vvSocket
{
  public:
    enum DataType                                 /// data type for get/putData
    {
      VV_UCHAR,
      VV_INT,
      VV_FLOAT
    };

    vvSocketIO(int, char*, vvSocket::SocketType, int clminport=0, int clmaxport=0);
    vvSocketIO(int, vvSocket::SocketType);
    ~vvSocketIO();
    ErrorType init();
    bool sock_action();
    ErrorType getVolume(vvVolDesc*);
    ErrorType putVolume(vvVolDesc*);
    ErrorType getImage(vvImage*);
    ErrorType putImage(vvImage*);
    ErrorType getData(uchar**, int&);             //  unknown number and type
    ErrorType putData(uchar*, int);
    ErrorType getMatrix(vvMatrix*);
    ErrorType putMatrix(vvMatrix*);
    ErrorType getData(void*, int, DataType);      // known number and type
    ErrorType putData(void*, int, DataType);
    void set_sock_param(float, float, int=65536, int=0);
};
#endif
