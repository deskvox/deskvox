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

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvresourcemanager.h"
#include "vvserver.h"

#include <virvo/vvdebugmsg.h>
#include <virvo/vvibrserver.h>
#include <virvo/vvimageserver.h>
#include <virvo/vvremoteserver.h>
#include <virvo/vvrendererfactory.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvtcpserver.h>
#include <virvo/vvtcpsocket.h>
#include <virvo/vvtoolshed.h>
#include <virvo/vvvirvo.h>
#include <virvo/vvvoldesc.h>

#include <iostream>
#include <limits>
#include <pthread.h>

const int vvServer::DEFAULTSIZE  = 512;
const int vvServer::DEFAULT_PORT = 31050;

using std::cerr;
using std::endl;

vvServer::vvServer()
{
  _port       = vvServer::DEFAULT_PORT;
  _sm         = SERVER;
  _useBonjour = false;
}

vvServer::~vvServer()
{
}

int vvServer::run(int argc, char** argv)
{
  vvDebugMsg::msg(3, "vvServer::run()");

  cerr << "Virvo server " << virvo::getVersionMajor() << "." << virvo::getReleaseCounter() << endl;
  cerr << "(c) " << virvo::getYearOfRelease() << " Juergen Schulze (schulze@cs.brown.edu)" << endl;
  cerr << "Brown University" << endl << endl;

  if(!parseCommandLine(argc, argv))
    return 1;

  if(serverLoop())
    return 0;
  else
    return 1;
}

void vvServer::displayHelpInfo()
{
  vvDebugMsg::msg(3, "vvServer::displayHelpInfo()");

  cerr << "Syntax:" << endl;
  cerr << endl;
  cerr << "  vserver [options]" << endl;
  cerr << endl;
  cerr << "Available options:" << endl;
  cerr << endl;
  cerr << "-port" << endl;
  cerr << " Don't use the default port (" << DEFAULT_PORT << "), but the specified one" << endl;
  cerr << endl;
  cerr << "-mode" << endl;
  cerr << " Start vvServer with one of the following modes:" << endl;
  cerr << " s     single server (default)" << endl;
  cerr << " rm    resource manager" << endl;
  cerr << " rm+s  server and resource manager simultanously" << endl;
  cerr << endl;
#ifdef HAVE_BONJOUR
  cerr << "-bonjour" << endl;
  cerr << " use bonjour to broadcast this service. options:" << endl;
  cerr << " on" << endl;
  cerr << " off (default)" << endl;
  cerr << endl;
#endif
  cerr << "-debug" << endl;
  cerr << " Set debug level" << endl;
  cerr << endl;
}

bool vvServer::parseCommandLine(int argc, char** argv)
{
  vvDebugMsg::msg(1, "vvServer::parseCommandLine()");

  for (int arg=1; arg<argc; ++arg)
  {
    if (vvToolshed::strCompare(argv[arg], "-help")==0 ||
        vvToolshed::strCompare(argv[arg], "-h")==0    ||
        vvToolshed::strCompare(argv[arg], "-?")==0    ||
        vvToolshed::strCompare(argv[arg], "/?")==0)
    {
      displayHelpInfo();
      return false;
    }
    else if (vvToolshed::strCompare(argv[arg], "-port")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "No port specified" << endl;
        return false;
      }
      else
      {
        int inport = atoi(argv[arg]);
        if(inport > std::numeric_limits<ushort>::max() || inport <= std::numeric_limits<ushort>::min())
        {
          cerr << "Specified port is out of range. Falling back to default: " << vvServer::DEFAULT_PORT << endl;
          _port = vvServer::DEFAULT_PORT;
        }
        else
          _port = inport;
      }
    }
    else if (vvToolshed::strCompare(argv[arg], "-mode")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Mode type missing." << endl;
        return false;
      }
      if     (vvToolshed::strCompare(argv[arg], "s")==0)
      {
        _sm = SERVER;
      }
      else if(vvToolshed::strCompare(argv[arg], "rm")==0)
      {
        _sm = RM;
      }
      else if(vvToolshed::strCompare(argv[arg], "rm+s")==0)
      {
        _sm = RM_WITH_SERVER;
      }
      else
      {
        cerr << "Unknown mode type." << endl;
        return false;
      }
    }
#ifdef HAVE_BONJOUR
    else if (vvToolshed::strCompare(argv[arg], "-bonjour")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Bonjour setting missing." << endl;
        return false;
      }
      if     (vvToolshed::strCompare(argv[arg], "on")==0)
      {
        _useBonjour = true;
      }
      else if(vvToolshed::strCompare(argv[arg], "off")==0)
      {
        _useBonjour = false;
      }
      else
      {
        cerr << "Unknown bonjour setting." << endl;
        return false;
      }
    }
#endif
    else if (vvToolshed::strCompare(argv[arg], "-debug")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Debug level missing." << endl;
        return false;
      }
      int level = atoi(argv[arg]);
      if (level>=0 && level<=3)
        vvDebugMsg::setDebugLevel(level);
      else
      {
        cerr << "Invalid debug level." << endl;
        return false;
      }
    }
    else
    {
      cerr << "Unknown option/parameter: \"" << argv[arg] << "\", use -help for instructions" << endl;
      return false;
    }
  }
  return true;
}

bool vvServer::serverLoop()
{
  vvTcpServer tcpServ = vvTcpServer(_port);

  if(!tcpServ.initStatus())
  {
    cerr << "Failed to initialize server-socket on port " << _port << "." << endl;
    return false;
  }
  else
  {
#ifdef HAVE_BONJOUR
    if(_useBonjour) registerToBonjour();
#endif
  }

  vvResourceManager *rm = NULL;

  if(RM == _sm)
  {
    rm = new vvResourceManager();
  }
  else if(RM_WITH_SERVER == _sm)
  {
    rm = new vvResourceManager(this);
  }

  while (1)
  {
    cerr << "Listening on port " << _port << endl;

    vvTcpSocket *sock = NULL;

    while((sock = tcpServ.nextConnection()) == NULL)
    {
      vvDebugMsg::msg(3, "vvServer::serverLoop() Listening socket blocked, retry...");
    }

    if(sock == NULL)
    {
      cerr << "vvServer::serverLoop() Failed to initialize server-socket on port " << _port << endl;
      break;
    }
    else
    {
      cerr << "Incoming connection..." << endl;

      if(RM == _sm || RM_WITH_SERVER == _sm)
      {
        rm->addJob(sock);
      }
      else
      {
        handleClient(sock);
      }
    }
  }

#ifdef HAVE_BONJOUR
  if(_useBonjour) unregisterFromBonjour();
#endif

  delete rm;

  return true;
}

#ifdef HAVE_BONJOUR
DNSServiceErrorType vvServer::registerToBonjour()
{
  vvDebugMsg::msg(3, "vvServer::registerToBonjour()");
  vvBonjourEntry entry = vvBonjourEntry("Virvo Server", "_vserver._tcp", "");
  return _registrar.registerService(entry, _port);
}

void vvServer::unregisterFromBonjour()
{
  vvDebugMsg::msg(3, "vvServer::unregisterFromBonjour()");
  _registrar.unregisterService();
}
#endif

void vvServer::handleClient(vvTcpSocket *sock)
{
  vvServerThreadArgs *args = new vvServerThreadArgs;
  args->_sock = sock;
  args->_exitFunc = NULL;

  pthread_t pthread;
  pthread_create(&pthread,  NULL, handleClientThread, args);
  pthread_detach(pthread);
}

void * vvServer::handleClientThread(void *param)
{
  vvServerThreadArgs *args = reinterpret_cast<vvServerThreadArgs*>(param);

  if(args->_exitFunc == NULL)
  {
    args->_exitFunc = &vvServer::exitCallback;
  }

  pthread_cleanup_push(args->_exitFunc, NULL);

  vvTcpSocket *sock = args->_sock;

  vvSocketIO *sockio = new vvSocketIO(sock);

  vvRemoteServer* server = NULL;

  sockio->putBool(true);  // let client know we are ready

  int getType;
  sockio->getInt32(getType);
  vvRenderer::RendererType rType = (vvRenderer::RendererType)getType;
  switch(rType)
  {
    case vvRenderer::REMOTE_IMAGE:
      server = new vvImageServer(sockio);
      break;
    case vvRenderer::REMOTE_IBR:
      server = new vvIbrServer(sockio);
      break;
    default:
      cerr << "Unknown remote rendering type " << rType << std::endl;
      break;
  }
  if(!server)
  {
    pthread_exit(NULL);
  #ifdef _WIN32
    return NULL;
  #endif
  }

  vvVolDesc *vd = NULL;
  if(server->initRenderContext(DEFAULTSIZE, DEFAULTSIZE) != vvRemoteServer::VV_OK)
  {
    cerr << "Couldn't initialize render context" << std::endl;
    goto cleanup;
  }
  vvGLTools::enableGLErrorBacktrace();
  if(server->initData(vd) != vvRemoteServer::VV_OK)
  {
    cerr << "Could not initialize volume data" << endl;
    cerr << "Continuing with next client..." << endl;
    goto cleanup;
  }

  if(vd != NULL)
  {
    if(server->getLoadVolumeFromFile())
    {
      vd->printInfoLine();
    }

    // Set default color scheme if no TF present:
    if(vd->tf.isEmpty())
    {
      vd->tf.setDefaultAlpha(0, 0.0, 1.0);
      vd->tf.setDefaultColors((vd->chan==1) ? 0 : 2, 0.0, 1.0);
    }

    vvRenderState rs;
    vvRenderer *renderer = vvRendererFactory::create(vd,
        rs,
        rType==vvRenderer::REMOTE_IBR ? "rayrend" : "default",
        "");

    if(rType == vvRenderer::REMOTE_IBR)
      renderer->setParameter(vvRenderer::VV_USE_IBR, 1.f);

    while(1)
    {
      if(!server->processEvents(renderer))
      {
        delete renderer;
        renderer = NULL;
        break;
      }
    }
  }

cleanup:
  server->destroyRenderContext();

  // Frames vector with bricks is deleted along with the renderer.
  // Don't free them here.
  // see setRenderer().

  delete server;
  server = NULL;

  delete sockio;
  sockio = NULL;

  sock->disconnectFromHost();
  delete sock;
  sock = NULL;

  delete vd;
  vd = NULL;

  pthread_exit(NULL);
  pthread_cleanup_pop(0);
#ifdef _WIN32
  return NULL;
#endif
}

/*vvServer *s = NULL;

void sigproc(int )
{
  // NOTE some versions of UNIX will reset signal to default
  // after each call. So for portability set signal each time
  signal(SIGINT, sigproc);

  cerr << "you have pressed ctrl-c" << endl;

  delete s;
  s = NULL;
}*/

//-------------------------------------------------------------------
/// Main entry point.
int main(int argc, char** argv)
{
//  signal(SIGINT, sigproc);
#ifdef VV_DEBUG_MEMORY
  int flag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);// Get current flag
  flag |= _CRTDBG_LEAK_CHECK_DF;                 // Turn on leak-checking bit
  flag |=  _CRTDBG_CHECK_ALWAYS_DF;
  _CrtSetDbgFlag(flag);                          // Set flag to the new value
#endif

#ifdef VV_DEBUG_MEMORY
  _CrtMemState s1, s2, s3;
  _CrtCheckMemory();
  _CrtMemCheckpoint( &s1 );
#endif

  // do stuff to test memory difference for

#ifdef VV_DEBUG_MEMORY
  _CrtMemCheckpoint( &s2 );
  if ( _CrtMemDifference( &s3, &s1, &s2 ) ) _CrtMemDumpStatistics( &s3 );
  _CrtCheckMemory();
#endif

  // do stuff to verify memory status after

#ifdef VV_DEBUG_MEMORY
  _CrtCheckMemory();
#endif

  //vvDebugMsg::setDebugLevel(vvDebugMsg::NO_MESSAGES);
  vvServer vserver;
  int error = vserver.run(argc, argv);

#ifdef VV_DEBUG_MEMORY
  _CrtDumpMemoryLeaks();                         // display memory leaks, if any
#endif

  return error;
}

//===================================================================
// End of File
//===================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
