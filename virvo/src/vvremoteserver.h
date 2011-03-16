#ifndef VVREMOTESERVER_H
#define VVREMOTESERVER_H

#include "vvexport.h"
#include "vvoffscreenbuffer.h"
#include "vvsocketio.h"
//#include "vvtexrend.h"
//#include "vvremoteserver.h"

class VIRVOEXPORT vvRemoteServer
{
public:
  vvRemoteServer();
  virtual ~vvRemoteServer();

  bool getLoadVolumeFromFile() const;
protected:
  bool _loadVolumeFromFile;
};

#endif // VVREMOTESERVER_H