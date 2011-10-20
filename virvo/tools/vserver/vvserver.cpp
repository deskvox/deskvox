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
using std::ios;

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#ifdef FREEGLUT
// For glutInitContextFlags(GLUT_DEBUG), needed for GL_ARB_debug_output
#include <GL/freeglut.h>
#endif
#endif

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include <virvo/vvvirvo.h>
#include <virvo/vvtoolshed.h>
#include <virvo/vvibrserver.h>
#include <virvo/vvimageserver.h>
#include <virvo/vvimage.h>
#include <virvo/vvremoteserver.h>
#include <virvo/vvrendererfactory.h>
#include <virvo/vvvoldesc.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvtexrend.h>
#include <virvo/vvcuda.h>

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
    enum
    {
      RR_NONE = 0,
      RR_CLUSTER,
      RR_IMAGE,
      RR_IBR
    };
    static const int DEFAULTSIZE;               ///< default window size (width and height) in pixels
    static const int DEFAULT_PORT;              ///< default port for socket connections
    static vvServer* ds;                        ///< one instance of vvServer is always present
    int   winWidth, winHeight;                  ///< window size in pixels
    int  rrMode;                                ///< memory remote rendering mode
    int port;                                   ///< port the server renderer uses to listen for incoming connections
  public:
    vvServer();
    ~vvServer();
    int run(int, char**);
    static void cleanup();

  private:
    static void reshapeCallback(int, int);
    static void displayCallback();
    static void timerCallback(int);
    void initGraphics(int argc, char *argv[]);
    void displayHelpInfo();
    bool parseCommandLine(int argc, char *argv[]);
    void mainLoop(int argc, char *argv[]);
    void serverLoop();
};

const int vvServer::DEFAULTSIZE = 512;
const int vvServer::DEFAULT_PORT = 31050;
vvServer* vvServer::ds = NULL;

//----------------------------------------------------------------------------
/// Constructor
vvServer::vvServer()
{
  winWidth = winHeight = DEFAULTSIZE;
  ds = this;
  rrMode                = RR_NONE;
  port                  = vvServer::DEFAULT_PORT;
}


//----------------------------------------------------------------------------
/// Destructor.
vvServer::~vvServer()
{
  ds = NULL;
}

void vvServer::serverLoop()
{
  vvGLTools::enableGLErrorBacktrace();

  while (1)
  {
    cerr << "Listening on port " << port << endl;

    vvSocketIO *sock = new vvSocketIO(port, vvSocket::VV_TCP);
    sock->set_debuglevel(vvDebugMsg::getDebugLevel());
    if(sock->init() != vvSocket::VV_OK)
    {
      std::cerr << "Failed to initialize server socket on port " << port << std::endl;
      delete sock;
      break;
    }

    vvRemoteServer* server = NULL;
    int type;
    sock->getInt32(type);
    switch(type)
    {
      case vvRenderer::REMOTE_IMAGE:
        rrMode = RR_IMAGE;
        server = new vvImageServer(sock);
        break;
      case vvRenderer::REMOTE_IBR:
        rrMode = RR_IBR;
        server = new vvIbrServer(sock);
        break;
      default:
        std::cerr << "Unknown remote rendering type " << type << std::endl;
        delete sock;
        break;
    }

    if(!server)
    {
      break;
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
          rrMode==RR_IBR ? "rayrend" : "default",
          "");
      if(rrMode == RR_IBR)
        renderer->setParameter(vvRenderer::VV_USE_IBR, 1.f);

      server->renderLoop(renderer);
      delete renderer;
    }

    // Frames vector with bricks is deleted along with the renderer.
    // Don't free them here.
    // see setRenderer().

cleanup:
    delete server;
    server = NULL;

    delete vd;
    vd = NULL;
  }
}

//----------------------------------------------------------------------------
/** Virvo server main loop.
  @param filename  volume file to display
*/
void vvServer::mainLoop(int argc, char *argv[])
{
  vvDebugMsg::msg(2, "vvServer::mainLoop()");

  initGraphics(argc, argv);
  vvCuda::initGlInterop();

  glutTimerFunc(1, timerCallback, 0);

  glutMainLoop();
}

//----------------------------------------------------------------------------
/** Callback method for window resizes.
    @param w,h new window width and height
*/
void vvServer::reshapeCallback(int w, int h)
{
  vvDebugMsg::msg(2, "vvServer::reshapeCallback(): ", w, h);

  ds->winWidth  = w;
  ds->winHeight = h;

  // Resize OpenGL viewport:
  glViewport(0, 0, ds->winWidth, ds->winHeight);

  glDrawBuffer(GL_FRONT_AND_BACK);               // select all buffers
                                                 // set clear color
  glClearColor(0., 0., 0., 0.);
                                                 // clear window
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}


//----------------------------------------------------------------------------
/// Callback method for window redraws.
void vvServer::displayCallback()
{
  vvDebugMsg::msg(3, "vvServer::displayCallback()");

  vvGLTools::printGLError("enter vvServer::displayCallback()");

  glDrawBuffer(GL_BACK);
  glClearColor(0., 0., 0., 0.);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Draw volume:
  glMatrixMode(GL_MODELVIEW);

  glDrawBuffer(GL_BACK);

  glutSwapBuffers();

  vvGLTools::printGLError("leave vvServer::displayCallback()");
}

//----------------------------------------------------------------------------
/** Timer callback method, triggered by glutTimerFunc().
*/
void vvServer::timerCallback(int)
{
  vvDebugMsg::msg(3, "vvServer::timerCallback()");

  ds->serverLoop();
  exit(0);
}

//----------------------------------------------------------------------------
/// Initialize the GLUT window and the OpenGL graphics context.
void vvServer::initGraphics(int argc, char *argv[])
{
  vvDebugMsg::msg(1, "vvServer::initGraphics()");

  cerr << "Number of CPUs found: " << vvToolshed::getNumProcessors() << endl;
  cerr << "Initializing GLUT." << endl;

  glutInit(&argc, argv);                // initialize GLUT

// Other glut versions than freeglut currently don't support
// debug context flags.
#if defined(FREEGLUT) && defined(GLUT_INIT_MAJOR_VERSION)
  glutInitContextFlags(GLUT_DEBUG);
#endif // FREEGLUT

  // create double buffering context
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  if (!glutGet(GLUT_DISPLAY_MODE_POSSIBLE))
  {
    cerr << "Error: Virvo server needs a double buffering OpenGL context with alpha channel." << endl;
    exit(1);
  }

  glutInitWindowSize(winWidth, winHeight);       // set initial window size

  // Create window title.
  // Don't use sprintf, it won't work with macros on Irix!
  char  title[1024];                              // window title
  sprintf(title, "Virvo Server V%s.%s (Port %d)",
      virvo::getVersionMajor(), virvo::getReleaseCounter(), port);

  glutCreateWindow(title);              // open window and set window title
  glutSetWindowTitle(title);

  // Set GL state:
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glEnable(GL_DEPTH_TEST);

  // Set Glut callbacks:
  glutDisplayFunc(displayCallback);
  glutReshapeFunc(reshapeCallback);

  const char *version = (const char *)glGetString(GL_VERSION);
  cerr << "Found OpenGL version: " << version << endl;
  if (strncmp(version,"1.0",3)==0)
  {
    cerr << "Virvo server requires OpenGL version 1.1 or greater." << endl;
  }

  vvGLTools::checkOpenGLextensions();

  if (vvDebugMsg::isActive(2))
  {
    cerr << "\nSupported OpenGL extensions:" << endl;
    vvGLTools::displayOpenGLextensions(vvGLTools::ONE_BY_ONE);
  }
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
  cerr << "-port (-p)" << endl;
  cerr << " Don't use the default port (" << DEFAULT_PORT << "), but the specified one" << endl;
  cerr << endl;
  cerr << "-size <width> <height>" << endl;
  cerr << " Set the window size to <width> * <height> pixels." << endl;
  cerr << " The default window size is " << DEFAULTSIZE << " * " << DEFAULTSIZE <<
          " pixels" << endl;
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
        vvToolshed::strCompare(argv[arg], "-h")==0 ||
        vvToolshed::strCompare(argv[arg], "-?")==0 ||
        vvToolshed::strCompare(argv[arg], "/?")==0)
    {
      displayHelpInfo();
      return false;
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
        cerr << "test----------------------"<<endl;
        port = atoi(argv[arg]);
      }
    }
    else if (vvToolshed::strCompare(argv[arg], "-display")==0
        || vvToolshed::strCompare(argv[arg], "-geometry")==0)
    {
      // handled by GLUT
      if ((++arg)>=argc)
      {
        cerr << "Required argument unspecified" << endl;
        return false;
      }
    }
    else if (vvToolshed::strCompare(argv[arg], "-iconic")==0
        || vvToolshed::strCompare(argv[arg], "-direct")==0
        || vvToolshed::strCompare(argv[arg], "-indirect")==0
        || vvToolshed::strCompare(argv[arg], "-gldebug")==0
        || vvToolshed::strCompare(argv[arg], "-sync")==0)
    {
      // handled by GLUT
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

  mainLoop(argc, argv);
  return 0;
}

void vvServer::cleanup()
{
  delete ds;
  ds = NULL;
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

  atexit(vvServer::cleanup);

  //vvDebugMsg::setDebugLevel(vvDebugMsg::NO_MESSAGES);
  int error = (new vvServer())->run(argc, argv);

#ifdef VV_DEBUG_MEMORY
  _CrtDumpMemoryLeaks();                         // display memory leaks, if any
#endif

  return error;
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
