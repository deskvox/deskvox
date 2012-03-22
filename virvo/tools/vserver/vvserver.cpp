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

#include <iostream>

using std::cerr;
using std::endl;

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include <virvo/vvvirvo.h>
#include <virvo/vvtoolshed.h>
#include <virvo/vvibrserver.h>
#include <virvo/vvimageserver.h>
#include <virvo/vvremoteserver.h>
#include <virvo/vvrendercontext.h>
#include <virvo/vvrendererfactory.h>
#include <virvo/vvvoldesc.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvtcpserver.h>
#include <virvo/vvcuda.h>
#include <virvo/vvpthread.h>

/**
 * Virvo Server main class.
 *
 * Server unit for remote rendering.
 *
 * Options -port: set port, else default will be used.
 *
 * @author Juergen Schulze (schulze@cs.brown.de)
 * @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
 */

class vvServer
{
  private:
    /// Remote rendering type
    enum RemoteRenderingType
    {
      RR_NONE,
      RR_IMAGE,
      RR_IBR
    };
    static const int DEFAULTSIZE;               ///< default window size (width and height) in pixels
    static const int DEFAULT_PORT;              ///< default port for socket connections
    static vvServer* ds;                        ///< one instance of vvServer is always present
    int   winWidth, winHeight;                  ///< window size in pixels
    int port;                                   ///< port the server renderer uses to listen for incoming connections
  public:
    vvServer();
    ~vvServer();
    int run(int, char**);

  private:
    void displayHelpInfo();
    bool parseCommandLine(int argc, char *argv[]);
    void serverLoop();
    static void * handleClient(void * attribs);
};

const int vvServer::DEFAULTSIZE = 512;
const int vvServer::DEFAULT_PORT = 31050;

//----------------------------------------------------------------------------
/// Constructor
vvServer::vvServer()
{
  winWidth = winHeight = DEFAULTSIZE;
  port                 = vvServer::DEFAULT_PORT;
}

//----------------------------------------------------------------------------
/// Destructor.
vvServer::~vvServer()
{
}

void vvServer::serverLoop()
{
  pthread_t thread;

  vvTcpServer tcpServ = vvTcpServer(port);

  if(!tcpServ.initStatus())
  {
    cerr << "Failed to initialize server-socket on port " << port << "." << endl;
    return;
  }

  while (1)
  {
    cerr << "Listening on port " << port << endl;

    vvSocket *sock = NULL;

    while((sock = tcpServ.nextConnection()) == NULL)
    {
      // retry
      cerr << "Socket blocked, retry..." << endl;
    }
    if(sock == NULL)
    {
      cerr << "Failed to initialize server socket on port " << port << endl;
      delete sock;
      break;
    }
    else
    {
      pthread_create(&thread, NULL, handleClient, sock);
    }
  }
}

void * vvServer::handleClient(void *attribs)
{
  vvRenderContext::ContextOptions contextOptions;
  contextOptions.displayName = "";
  contextOptions.height = DEFAULTSIZE;
  contextOptions.width = DEFAULTSIZE;
  contextOptions.type = vvRenderContext::VV_PBUFFER;
  vvRenderContext renderContext = vvRenderContext(&contextOptions);
  renderContext.makeCurrent();

  vvCuda::initGlInterop();

  vvTcpSocket *sock = reinterpret_cast<vvTcpSocket*>(attribs);
  vvSocketIO *sockio = new vvSocketIO(sock);

  vvRemoteServer* server = NULL;
  int getType;
  sockio->getInt32(getType);
  RemoteRenderingType remoteType = RR_NONE;
  vvRenderer::RendererType rType = (vvRenderer::RendererType)getType;
  switch(rType)
  {
    case vvRenderer::REMOTE_IMAGE:
      server = new vvImageServer(sockio);
      remoteType = RR_IMAGE;
      break;
    case vvRenderer::REMOTE_IBR:
      server = new vvIbrServer(sockio);
      remoteType = RR_IBR;
      break;
    default:
      std::cerr << "Unknown remote rendering type " << rType << std::endl;
      break;
  }

  if(!server)
  {
    return NULL;
  }

  vvVolDesc *vd = NULL;
  if (server->initData(vd) != vvRemoteServer::VV_OK)
  {
    cerr << "Could not initialize volume data" << endl;
    cerr << "Continuing with next client..." << endl;
    goto cleanup;
  }

  if (vd != NULL)
  {
    if (server->getLoadVolumeFromFile())
    {
      vd->printInfoLine();
    }

    // Set default color scheme if no TF present:
    if (vd->tf.isEmpty())
    {
      vd->tf.setDefaultAlpha(0, 0.0, 1.0);
      vd->tf.setDefaultColors((vd->chan==1) ? 0 : 2, 0.0, 1.0);
    }

    vvRenderState rs;
    vvRenderer *renderer = vvRendererFactory::create(vd,
        rs,
        remoteType==RR_IBR ? "rayrend" : "default",
        "");

    if(remoteType == RR_IBR)
      renderer->setParameter(vvRenderer::VV_USE_IBR, 1.f);

    while (1)
    {
      if (!server->processEvents(renderer))
      {
        break;
      }
//      if (vvDebugMsg::getDebugLevel() > 0)
//      {
//        renderContext.swapBuffers();
//      }
    }
    delete renderer;
  }

  // Frames vector with bricks is deleted along with the renderer.
  // Don't free them here.
  // see setRenderer().

cleanup:
  delete server;
  server = NULL;

  sock->disconnectFromHost();
  delete sock;
  sock = NULL;

  delete vd;
  vd = NULL;

  return NULL;
}

//----------------------------------------------------------------------------
/// Display command usage help on the command line.
void vvServer::displayHelpInfo()
{
  vvDebugMsg::msg(1, "vvServer::displayHelpInfo()");

  cerr << "Syntax:" << endl;
  cerr << endl;
  cerr << "  vserver [options]" << endl;
  cerr << endl;
  cerr << "Available options:" << endl;
  cerr << endl;
  cerr << "-port" << endl;
  cerr << " Don't use the default port (" << DEFAULT_PORT << "), but the specified one" << endl;
  cerr << endl;
  cerr << "-size <width> <height>" << endl;
  cerr << " Set the window size to <width> * <height> pixels." << endl;
  cerr << " The default window size is " << DEFAULTSIZE << " * " << DEFAULTSIZE <<
          " pixels" << endl;
  cerr << endl;
  cerr << "-debug" << endl;
  cerr << " Set debug level" << endl;
  cerr << endl;
}

//----------------------------------------------------------------------------
/** Parse command line arguments.
  @param argc,argv command line arguments
  @return true if parsing ok, false on error
*/
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
    else if (vvToolshed::strCompare(argv[arg], "-size")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Window width missing." << endl;
        return false;
      }
      winWidth = atoi(argv[arg]);
      if ((++arg)>=argc)
      {
        cerr << "Window height missing." << endl;
        return false;
      }
      winHeight = atoi(argv[arg]);
      if (winWidth<1 || winHeight<1)
      {
        cerr << "Invalid window size." << endl;
        return false;
      }
    }
    else if (vvToolshed::strCompare(argv[arg], "-port")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "No port specified, defaulting to: " << vvServer::DEFAULT_PORT << endl;
        port = vvServer::DEFAULT_PORT;
        return false;
      }
      else
      {
        port = atoi(argv[arg]);
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

//----------------------------------------------------------------------------
/** Main Virvo server routine.
  @param argc,argv command line arguments
  @return 0 if the program finished ok, 1 if an error occurred
*/
int vvServer::run(int argc, char** argv)
{
  vvDebugMsg::msg(1, "vvServer::run()");

  cerr << "Virvo server " << virvo::getVersionMajor() << "." << virvo::getReleaseCounter() << endl;
  cerr << "(c) " << virvo::getYearOfRelease() << " Juergen Schulze (schulze@cs.brown.edu)" << endl;
  cerr << "Brown University" << endl << endl;

  if (parseCommandLine(argc, argv) == false)
    return 1;

  serverLoop();
  return 0;
}

//----------------------------------------------------------------------------
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
  vvServer vserver;
  int error = vserver.run(argc, argv);

#ifdef VV_DEBUG_MEMORY
  _CrtDumpMemoryLeaks();                         // display memory leaks, if any
#endif

  return error;
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
