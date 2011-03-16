#ifndef VVREMOTESERVER_H
#define VVREMOTESERVER_H

#include "vvexport.h"
#include "vvoffscreenbuffer.h"
#include "vvsocketio.h"

class vvRenderer;

class VIRVOEXPORT vvRemoteServer
{
public:
  enum ErrorType
  {
    VV_OK = 0,
    VV_SOCKET_ERROR,
    VV_FILEIO_ERROR
  };

  vvRemoteServer();
  virtual ~vvRemoteServer();

  bool getLoadVolumeFromFile() const;

  vvRemoteServer::ErrorType initSocket(int port, vvSocket::SocketType st);
  vvRemoteServer::ErrorType initData(vvVolDesc*& vd);

  virtual void renderLoop(vvRenderer* renderer);
protected:
  vvSocketIO* _socket;                    ///< socket for remote rendering

  bool _loadVolumeFromFile;

  virtual void renderImage(vvMatrix& pr, vvMatrix& mv, vvRenderer* renderer) = 0;
  virtual void resize(int w, int h) = 0;
};

#endif // VVREMOTESERVER_H
