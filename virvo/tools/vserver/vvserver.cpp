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

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvresourcemanager.h"
#include "vvserver.h"

#include <virvo/vvdebugmsg.h>
#include <virvo/vvfileio.h>
#include <virvo/vvibrserver.h>
#include <virvo/vvimageserver.h>
#include <virvo/vvremoteserver.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvtcpserver.h>
#include <virvo/vvtcpsocket.h>
#include <virvo/vvtoolshed.h>
#include <virvo/vvvirvo.h>
#include <virvo/vvvoldesc.h>

#include <iostream>
#include <limits>
#include <sstream>

#ifndef _WIN32
#include <signal.h>
#include <syslog.h>
#include <unistd.h>
#endif

const int            vvServer::DEFAULTSIZE  = 512;
const unsigned short vvServer::DEFAULT_PORT = 31050;

vvServer::ThreadData::ThreadData()
  : renderContext(NULL)
  , server(NULL)
  , remoteServerType(vvRenderer::REMOTE_IMAGE)
  , renderertype("default")
  , renderer(NULL)
  , vd(NULL)
  , request(NULL)
{}

vvServer::ThreadData::~ThreadData()
{
  delete renderContext;
  delete server;
  delete renderer;
  delete vd;
}

vvServer::vvServer(bool useBonjour)
  : _port(vvServer::DEFAULT_PORT)
  , _sm(SERVER)
  , _useBonjour(useBonjour)
  , _daemonize(false)
  , _daemonName("voxserver")
{
}

vvServer::~vvServer()
{}

int vvServer::run(int argc, char** argv)
{
  vvDebugMsg::msg(3, "vvServer::run()");

  cerr << "Virvo server " << virvo::version() << endl;
  cerr << "(c) " << VV_VERSION_YEAR << " Juergen Schulze (schulze@cs.brown.edu)" << endl;
  cerr << "Brown University" << endl << endl;

  if(!parseCommandLine(argc, argv))
    return 1;

#ifndef _WIN32
  if (_daemonize)
  {
    signal(SIGHUP, handleSignal);
    signal(SIGTERM, handleSignal);
    signal(SIGINT, handleSignal);
    signal(SIGQUIT, handleSignal);

    setlogmask(LOG_UPTO(LOG_INFO));
    openlog(_daemonName.c_str(), LOG_CONS, LOG_USER);

    pid_t pid;
    pid_t sid;

    syslog(LOG_INFO, "Starting %s.", _daemonName.c_str());

    pid = fork();
    if (pid < 0)
    {
      exit(EXIT_FAILURE);
    }

    if (pid > 0)
    {
      exit(EXIT_SUCCESS);
    }

    umask(0);

    sid = setsid();
    if (sid < 0)
    {
      exit(EXIT_FAILURE);
    }

    // change working directory to /, which cannot be unmounted
    if (chdir("/") < 0)
    {
      exit(EXIT_FAILURE);
    }

    // close file descriptors for standard output
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);
  }
#endif

  const bool res = serverLoop();

#ifndef _WIN32
  syslog(LOG_INFO, "%s exiting.", _daemonName.c_str());
#endif

  return (int)(!res);
}

void vvServer::setPort(unsigned short port)
{
  _port = port;
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
  if (virvo::hasFeature("bonjour"))
  {
    cerr << "-mode" << endl;
    cerr << " Start vvServer with one of the following modes:" << endl;
    cerr << " s     single server (default)" << endl;
    cerr << " rm    resource manager" << endl;
    cerr << " rm+s  server and resource manager simultanously" << endl;
    cerr << endl;
    cerr << "-bonjour" << endl;
    cerr << " use bonjour to broadcast this service. options:" << endl;
    cerr << " on" << endl;
    cerr << " off (default)" << endl;
    cerr << endl;
  }
#ifndef _WIN32
  cerr << "-daemon" << endl;
  cerr << " Start in background as a daemon" << endl;
  cerr << endl;
#endif
  cerr << "-debug" << endl;
  cerr << " Set debug level" << endl;
  cerr << endl;
}

bool vvServer::parseCommandLine(int argc, char** argv)
{
  vvDebugMsg::msg(1, "vvServer::parseCommandLine()");

  const static bool HaveBonjour = virvo::hasFeature("bonjour");

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
    else if (HaveBonjour && vvToolshed::strCompare(argv[arg], "-mode")==0)
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
    else if (HaveBonjour && vvToolshed::strCompare(argv[arg], "-bonjour")==0)
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
#ifndef _WIN32
    else if (vvToolshed::strCompare(argv[arg], "-daemon")==0)
    {
      _daemonize = true;
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

  while (1)
  {
    cerr << "Listening on port " << _port << endl;

    vvTcpSocket *sock = NULL;

    while((sock = tcpServ.nextConnection()) == NULL)
    {
      vvDebugMsg::msg(3, "vvServer::serverLoop() Listening socket busy, retry...");
    }

    if(sock == NULL)
    {
      cerr << "vvServer::serverLoop() Failed to initialize server-socket on port " << _port << endl;
      break;
    }
    else
    {
      cerr << "Incoming connection..." << endl;
      handleNextConnection(sock);
    }
  }

  return true;
}

bool vvServer::handleEvent(ThreadData *tData, const virvo::RemoteEvent event, const vvSocketIO& io)
{
  switch (event)
  {
  case virvo::Volume:
    delete tData->vd;
    tData->vd = new vvVolDesc;

    switch (io.getVolume(tData->vd))
    {
    case vvSocket::VV_OK:
      vvDebugMsg::msg(1, "Volume transferred successfully");
      return true;
    case vvSocket::VV_ALLOC_ERROR:
      vvDebugMsg::msg(0, "Not enough memory to accomodate volume");
      delete tData->vd;
      tData->vd = NULL;
      return true;
    default:
      vvDebugMsg::msg(0, "Unknown error while reading volume from socket");
      delete tData->vd;
      tData->vd = NULL;
      return true;
    }

    // if remote server is already created, create a new one
    if (tData->server != NULL)
    {
      if (!createRemoteServer(tData, static_cast<vvTcpSocket*>(io.getSocket())))
      {
        vvDebugMsg::msg(0, "Couldn't create remote server");
        return true;
      }
    }
    return true;
  case virvo::VolumeFile:
    {
      std::string fn;
      io.getFileName(fn);
      vvDebugMsg::msg(1, "Load volume from file: ", fn.c_str());
      delete tData->vd;
      tData->vd = new vvVolDesc(fn.c_str());

      vvFileIO fio;
      if (fio.loadVolumeData(tData->vd) != vvFileIO::OK)
      {
        vvDebugMsg::msg(0, "Error loading volume file");
        delete tData->vd;
        tData->vd = NULL;
        return true;
      }
      else
      {
        tData->vd->printInfoLine();
      }

      // set default color scheme if no TF present
      if (tData->vd->tf.isEmpty())
      {
        tData->vd->tf.setDefaultAlpha(0, 0.0, 1.0);
        tData->vd->tf.setDefaultColors((tData->vd->chan == 1) ? 0 : 2, 0.0, 1.0);
      }

      // if remote server is already created, create a new one
      if (tData->server != NULL)
      {
        if (!createRemoteServer(tData, static_cast<vvTcpSocket*>(io.getSocket())))
        {
          vvDebugMsg::msg(0, "Couldn't create remote server");
          return true;
        }
      }

      // if remote server is already created, create a new one
      if (tData->server != NULL)
      {
        if (!createRemoteServer(tData, static_cast<vvTcpSocket*>(io.getSocket())))
        {
          vvDebugMsg::msg(0, "Couldn't create remote server");
          return true;
        }
      }
    }
    return true;
  case virvo::CameraMatrix:
  case virvo::Parameter1B:
  case virvo::Parameter1I:
  case virvo::Parameter1F:
  case virvo::Parameter3F:
  case virvo::Parameter4F:
  case virvo::ParameterColor:
  case virvo::ParameterAABBI:
  case virvo::CurrentFrame:
  case virvo::ObjectDirection:
  case virvo::ViewingDirection:
  case virvo::Position:
  case virvo::TransFunc:
    // if no render context exists, create one
    if (tData->renderContext == NULL && !createRenderContext(tData, DEFAULTSIZE, DEFAULTSIZE))
    {
      vvDebugMsg::msg(0, "Couldn't create render context");
      return true;
    }

    // if no remote server exists, create one
    if (tData->server == NULL && !createRemoteServer(tData, static_cast<vvTcpSocket*>(io.getSocket())))
    {
      vvDebugMsg::msg(0, "Couldn't create remote server");
      return true;
    }

    if (tData->server != NULL && tData->vd != NULL && tData->renderer != NULL)
    {
      return tData->server->processEvent(event, tData->renderer);
    }
    else
    {
      vvDebugMsg::msg(0, "Cannot process remote rendering event");
      return false;
    }
    return true;
  case virvo::RemoteServerType:
    if (io.getRendererType(tData->remoteServerType) != vvSocket::VV_OK)
    {
      vvDebugMsg::msg(0, "Cannot get remote server type");
      return false;
    }

    // if an old server exists, we need to create a new server
    if (tData->server != NULL)
    {
      if (!createRemoteServer(tData, static_cast<vvTcpSocket*>(io.getSocket())))
      {
        vvDebugMsg::msg(0, "Couldn't create remote server");
        return true;
      }
    }
    return true;
  case virvo::ServerInfo:
    {
      vvServerInfo info;

      // if no render context exists, create one
      if (tData->renderContext == NULL && !createRenderContext(tData, DEFAULTSIZE, DEFAULTSIZE))
      {
        vvDebugMsg::msg(0, "Couldn't create render context");

        // send an empty server info
        io.putServerInfo(info);
        return true;
      }

      // assemble renderers
      std::vector<std::string> renderers;
      renderers.push_back("rayrend");
      renderers.push_back("softrayrend");
      renderers.push_back("slices");
      renderers.push_back("cubic2d");
      renderers.push_back("planar");
      renderers.push_back("bricks");
      renderers.push_back("spherical");
      renderers.push_back("serbrick");
      renderers.push_back("parbrick");

      // query available renderers
      std::stringstream rendstr;
      for (std::vector<std::string>::const_iterator it = renderers.begin();
           it != renderers.end(); ++it)
      {
        if (vvRendererFactory::hasRenderer(*it))
        {
          if (rendstr.str() != "")
          {
            rendstr << ",";
          }
          rendstr << *it;
        }
      }
      info.renderers = rendstr.str();
      io.putServerInfo(info);
    }
    return true;
  case virvo::WindowResize:
    {
      int w;
      int h;
      io.getWinDims(w, h);
      if (tData->renderContext == NULL && !createRenderContext(tData, w, h))
      {
        vvDebugMsg::msg(0, "Cannot resize remote rendering context");
        return true;
      }
      tData->renderContext->resize(static_cast<uint>(w), static_cast<uint>(h));
    }
    return true;
  case virvo::Disconnect:
    delete tData->renderer;
    delete tData->vd;
    delete tData->server;
    delete tData->renderContext;
    tData->renderer = NULL;
    tData->vd = NULL;
    tData->server = NULL;
    tData->renderContext = NULL;
    return true;
  default:
    assert(0 && "Event not handled");
    return false;
  }
}

bool vvServer::createRenderContext(ThreadData *tData, const int w, const int h)
{
  delete tData->renderContext;
  tData->contextOptions.type = vvContextOptions::VV_PBUFFER;
  tData->contextOptions.width = w;
  tData->contextOptions.height = h;
  tData->contextOptions.displayName = "";
  tData->renderContext = new vvRenderContext(tData->contextOptions);
  return tData->renderContext->makeCurrent();
}

bool vvServer::createRemoteServer(ThreadData *tData, vvTcpSocket* sock)
{
  vvDebugMsg::msg(3, "vvServer::createRemoteServer() Enter");

  // need a render context. either reuse existing or create a new one
  if (tData->renderContext == NULL && !createRenderContext(tData, DEFAULTSIZE, DEFAULTSIZE))
  {
    vvDebugMsg::msg(0, "Couldn't create render context");
    return false;
  }

  delete tData->server;
  switch (tData->remoteServerType)
  {
  case vvRenderer::REMOTE_IMAGE:
    tData->server = new vvImageServer(sock);
    break;
  case vvRenderer::REMOTE_IBR:
    tData->server = new vvIbrServer(sock);
    break;
  default:
    vvDebugMsg::msg(0, "Unknown remote rendering type");
    break;
  }

  if (tData->server == NULL)
  {
    vvDebugMsg::msg(0, "Couldn't create remote server");
    return false;
  }

  vvGLTools::enableGLErrorBacktrace();

  if (tData->vd != NULL)
  {
    // set default color scheme if no TF present
    if (tData->vd->tf.isEmpty())
    {
      tData->vd->tf.setDefaultAlpha(0, 0.0, 1.0);
      tData->vd->tf.setDefaultColors((tData->vd->chan == 1) ? 0 : 2, 0.0, 1.0);
    }

    vvRenderState rs;


    tData->renderer = vvRendererFactory::create(tData->vd,
      rs,
      tData->renderertype.c_str(),
      tData->opt);

    tData->renderer->setParameter(vvRenderer::VV_USE_IBR, tData->remoteServerType == vvRenderer::REMOTE_IBR);
    return true;
  }
  else
  {
    vvDebugMsg::msg(0, "No volume loaded");
    return false;
  }
}


//-------------------------------------------------------------------
/// Handle signals a daemon receives.
void vvServer::handleSignal(int sig)
{
#ifndef _WIN32
  switch (sig)
  {
  case SIGHUP:
    syslog(LOG_WARNING, "Got SIGHUP, quitting.");
    break;
  case SIGTERM:
    syslog(LOG_WARNING, "Got SIGTERM, quitting.");
    break;
  default:
    syslog(LOG_WARNING, "Got unhandled signal %s, quitting.", strsignal(sig));
    break;
  }
#endif
}

//-------------------------------------------------------------------
/// Main entry point.
int main(int argc, char** argv)
{
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
//  vvDebugMsg::setDebugLevel(5);
  vvResourceManager vvRM;
  int error = vvRM.run(argc, argv);

#ifdef VV_DEBUG_MEMORY
  _CrtDumpMemoryLeaks();                         // display memory leaks, if any
#endif

  return error;
}

//===================================================================
// End of File
//===================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
