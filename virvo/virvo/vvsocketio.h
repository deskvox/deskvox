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

#if !defined(VV_LIBRARY_BUILD) && !defined(VV_APPLICATION_BUILD)
#error "vvsocketio.h is meant for internal use only"
#endif

#ifndef _VVSOCKETIO_H_
#define _VVSOCKETIO_H_

#include <vector>

#include "vvexport.h"
#include "vvgpu.h"
#include "vvsocket.h"
#include "vvinttypes.h"
#include "vvvecmath.h"
#include "vvtransfunc.h"
#include "vvgltools.h"

struct vvMulticastParameters;
class vvBrick;
class vvImage;
class vvIbrImage;
class vvVolDesc;

/** This class provides specific data transfer through sockets.
  It requires a socket of type vvSocket.<BR>
  Here is an example code fragment for a TCP sever which reads
  a volume from a socket and a TCP client which writes a volume to the
  socket:<BR>
  <PRE>

  Server:

  // Create a new TCP socket class instance, which is a server listening on port 17171
  vvSocket* sock = new vvSocket(17171 , vvSocket::VV_TCP);
  vvVolDesc* vd = new vvVolDesc();

  // Initialize the socket with the parameters and wait for a server
  if (sock->init() != vvSocket::VV_OK)
  {
    delete sock;
    return -1;
  }

  // Assign the socket to socketIO which extens the socket with more functions
  vvSocketIO* sio = new vvSocketIO(sock);

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
  delete sock;
  delete sio;

  Client:

  // Create a new TCP socket class instance, which is a client and connects
  // to a server listening on port 17171

  char* servername = "buxdehude";
  vvSocket* sock = new vvSocket(17171 , servername, vvSocket::VV_TCP);
  vvVolDesc* vd = new vvVolDesc();

  // Initialize the socket with the parameters and connect to the server.
  if (sio->init() != vvSocket::VV_OK)
  {
    delete sock;
    return -1;
  }

  // Assign the socket to socketIO
  vvSocketIO* sio = new vvSocketIO(sock);

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
  delete sock;
  delete sio;

</PRE>
@see vvSocket
@author Michael Poehnl
*/
class VIRVOEXPORT vvSocketIO
{
  public:
    enum DataType                                 /// data type for get/putData
    {
      VV_UCHAR,
      VV_USHORT,
      VV_INT,
      VV_FLOAT
    };

    enum CommReason                               /// before sending data, let the recipient know what to expect
    {
      VV_QUIT = 0,
      VV_MATRIX,
      VV_CURRENT_FRAME,
      VV_IMAGE,
      VV_OBJECT_DIRECTION,
      VV_POSITION,
      VV_RESIZE,
      VV_TRANSFER_FUNCTION,
      VV_VIEWING_DIRECTION,
      VV_VOLUME,
      VV_PARAMETER_1B,
      VV_PARAMETER_1I,
      VV_PARAMETER_1F,
      VV_PARAMETER_3I,
      VV_PARAMETER_3F,
      VV_PARAMETER_4I,
      VV_PARAMETER_4F,
      VV_PARAMETER_AABBI
    };

    vvSocketIO(vvSocket* sock);
    ~vvSocketIO();
    bool sock_action();
    vvSocket::ErrorType getVolumeAttributes(vvVolDesc* vd);
    vvSocket::ErrorType getVolume(vvVolDesc*, vvMulticastParameters *mcParam = NULL);
    vvSocket::ErrorType putVolumeAttributes(const vvVolDesc*);
    vvSocket::ErrorType putVolume(const vvVolDesc*, bool tryMC = false, bool mcMaster = false, vvMulticastParameters *mcParam = NULL);
    vvSocket::ErrorType getTransferFunction(vvTransFunc& tf);
    vvSocket::ErrorType putTransferFunction(vvTransFunc& tf);
    vvSocket::ErrorType getBrick(vvBrick* brick);
    vvSocket::ErrorType putBrick(const vvBrick* brick);
    vvSocket::ErrorType getBricks(std::vector<vvBrick*>& bricks);
    vvSocket::ErrorType putBricks(const std::vector<vvBrick*>& bricks);
    vvSocket::ErrorType getImage(vvImage*);
    vvSocket::ErrorType putImage(const vvImage*);
    vvSocket::ErrorType getIbrImage(vvIbrImage*);
    vvSocket::ErrorType putIbrImage(const vvIbrImage*);
    vvSocket::ErrorType getFileName(char*& fn);
    vvSocket::ErrorType putFileName(const char* fn);
    vvSocket::ErrorType allocateAndGetData(uchar**, int&);             //  unknown number and type
    vvSocket::ErrorType putData(uchar*, int);
    vvSocket::ErrorType getMatrix(vvMatrix*);
    vvSocket::ErrorType putMatrix(const vvMatrix*);
    vvSocket::ErrorType getBool(bool& val);
    vvSocket::ErrorType putBool(const bool val);
    vvSocket::ErrorType getInt32(int& val);
    vvSocket::ErrorType putInt32(const int val);
    vvSocket::ErrorType getFloat(float& val);
    vvSocket::ErrorType putFloat(const float val);
    vvSocket::ErrorType getVector3(vvVector3& val);
    vvSocket::ErrorType putVector3(const vvVector3& val);
    vvSocket::ErrorType getVector4(vvVector4& val);
    vvSocket::ErrorType putVector4(const vvVector4& val);
    vvSocket::ErrorType getAABBi(vvAABBi& val);
    vvSocket::ErrorType putAABBi(const vvAABBi& val);
    vvSocket::ErrorType getViewport(vvGLTools::Viewport &val);
    vvSocket::ErrorType putViewport(const vvGLTools::Viewport &val);
    vvSocket::ErrorType getCommReason(CommReason& val);
    vvSocket::ErrorType putCommReason(const CommReason val);
    vvSocket::ErrorType getWinDims(int& w, int& h);
    vvSocket::ErrorType putWinDims(const int w, const int h);
    vvSocket::ErrorType getData(void*, int, DataType);      // known number and type
    vvSocket::ErrorType putData(void*, int, DataType);
    vvSocket::ErrorType getGpuInfo(vvGpuInfo& ginfo);
    vvSocket::ErrorType putGpuInfo(const vvGpuInfo& ginfo);

    vvSocket* getSocket() const;
private:
    int sizeOfBrick() const;

    vvSocket *_socket;
};
#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
