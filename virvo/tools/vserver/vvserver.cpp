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

#include <virvo/vvglew.h>
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
    enum                                       ///  Timer callback types
    {
      ANIMATION_TIMER = 0,                     ///< volume animation timer callback
      ROTATION_TIMER  = 1,                     ///< rotation animation timer callback
      BENCHMARK_TIMER  = 2,                    ///< benchmark timer callback
      SERVER_TIMER  = 3                        ///< server mode timer callback
    };
    /// Remote rendering type
    enum
    {
      RR_NONE = 0,
      RR_CLUSTER,
      RR_IMAGE,
      RR_IBR
    };
    static const int DEFAULTSIZE;               ///< default window size (width and height) in pixels
    static const float OBJ_SIZE;                ///< default object size
    static const int DEFAULT_PORT;              ///< default port for socket connections
    static vvServer* ds;                        ///< one instance of VView is always present
    vvRenderer* renderer;                       ///< rendering engine
    vvRenderState renderState;                  ///< renderer state
    vvVolDesc* vd;                              ///< volume description
    int   window;                               ///< GLUT window handle
    int   winWidth, winHeight;                  ///< window size in pixels
    std::string currentRenderer;                ///< current renderer/rendering geometry
    std::string currentOptions;                 ///< current options/voxel type
    vvVector3 bgColor;                          ///< background color (R,G,B in [0..1])
    std::vector<std::string> rendererName;      ///< strings for renderer types
    vvVector3 pos;                              ///< volume position in object space
    int  rrMode;                                ///< memory remote rendering mode
    int codec;                                  ///< code type/codec for images sent over the network
    bool dbgOutputExtSet;                       ///< callback func for gl debug output was registered or can't be registered
    bool showBt;                                ///< Show backtrace if execution stopped due to OpenGL errors
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
    static void debugCallbackARB(GLenum source, GLenum type, GLuint id, GLenum severity,
                                 GLsizei length, GLchar const* message, GLvoid* userParam);
    void initGraphics(int argc, char *argv[]);
    void initARBDebugOutput();
    void createRenderer(std::string renderertype, std::string options,
                     std::vector<std::vector<vvBrick*> > *bricks = NULL,
                     const int maxBrickSizeX = 64,
                     const int maxBrickSizeY = 64, const int maxBrickSizeZ = 64);
    void applyRendererParameters();
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
  renderer = NULL;
  vd = NULL;
  currentRenderer = "default";
  currentOptions = "default";
  bgColor[0] = bgColor[1] = bgColor[2] = 0.0f;
  window = 0;
  pos.zero();
  codec                 = vvImage::VV_RLE;
  rrMode                = RR_NONE;
  dbgOutputExtSet       = false;
  showBt                = true;
  port                  = vvServer::DEFAULT_PORT;

  // keep in sync with vvrenderer.h
  rendererName.push_back("TexRend");
  rendererName.push_back("Parallel software shear-warp");
  rendererName.push_back("Perspective software shear-warp");
  rendererName.push_back("Software shear-warp");
  rendererName.push_back("Parallel GPU shear-warp");
  rendererName.push_back("Perspective GPU shear-warp");
  rendererName.push_back("GPU shear-warp");
  rendererName.push_back("CUDA ray caster");
  rendererName.push_back("VolPack");
  rendererName.push_back("VolPack");
  rendererName.push_back("Simian");
  rendererName.push_back("Imgrend");
  rendererName.push_back("Unknown");
  rendererName.push_back("Stingray");
  rendererName.push_back("Out of core texture renderer");
  rendererName.push_back("Image based remote rendering");
  rendererName.push_back("Remote rendering");
  assert(rendererName.size() == vvRenderer::NUM_RENDERERS);

}


//----------------------------------------------------------------------------
/// Destructor.
vvServer::~vvServer()
{
  delete renderer;
  delete vd;
  ds = NULL;
}

void vvServer::serverLoop()
{
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
        //currentRenderer = "planar";
        server = new vvImageServer(sock);
        break;
      case vvRenderer::REMOTE_IBR:
#ifdef HAVE_CUDA
        rrMode = RR_IBR;
        currentRenderer = "rayrend";
        server = new vvIbrServer(sock);
#else
        std::cerr << "Image based remote rendering requires CUDA." << std::endl;
#endif
        break;
      default:
        std::cerr << "Unknown remote rendering type " << type << std::endl;
        break;
    }

    if(!server)
    {
      delete sock;
      break;
    }

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

      delete renderer;
      renderer = NULL;

      createRenderer(currentRenderer, currentOptions);
      srand(time(NULL));

      server->renderLoop(renderer);
      cerr << "Exiting..." << endl;

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
/** VView main loop.
  @param filename  volume file to display
*/
void vvServer::mainLoop(int argc, char *argv[])
{
  vvDebugMsg::msg(2, "vvView::mainLoop()");

  initGraphics(argc, argv);
#ifdef HAVE_CUDA
  vvCuda::initGlInterop();
#endif

  // Set window title:
  char title[1024];
  sprintf(title, "vvView Server: %d", port);
  glutSetWindowTitle(title);

  glutTimerFunc(1, timerCallback, SERVER_TIMER);

  glutMainLoop();
}

//----------------------------------------------------------------------------
/** Callback method for window resizes.
    @param w,h new window width and height
*/
void vvServer::reshapeCallback(int w, int h)
{
  vvDebugMsg::msg(2, "vvView::reshapeCallback(): ", w, h);

  ds->winWidth  = w;
  ds->winHeight = h;

  // Resize OpenGL viewport:
  glViewport(0, 0, ds->winWidth, ds->winHeight);

  glDrawBuffer(GL_FRONT_AND_BACK);               // select all buffers
                                                 // set clear color
  glClearColor(ds->bgColor[0], ds->bgColor[1], ds->bgColor[2], 1.0f);
                                                 // clear window
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}


//----------------------------------------------------------------------------
/// Callback method for window redraws.
void vvServer::displayCallback(void)
{
  vvDebugMsg::msg(3, "vvView::displayCallback()");

  vvGLTools::printGLError("enter vvView::displayCallback()");

  glDrawBuffer(GL_BACK);
  glClearColor(ds->bgColor[0], ds->bgColor[1], ds->bgColor[2], 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if(!ds->renderer)
    return;

  // Draw volume:
  glMatrixMode(GL_MODELVIEW);

  glDrawBuffer(GL_BACK);
#ifdef CLIPPING_TEST
  ds->renderClipObject();
  ds->renderQuad();
#endif

#ifdef FBO_WITH_GEOMETRY_TEST
  ds->renderCube();
  ds->renderer->_renderState._opaqueGeometryPresent = true;
#endif
  ds->renderer->renderVolumeGL();

  glutSwapBuffers();

  vvGLTools::printGLError("leave vvView::displayCallback()");
}

void vvServer::applyRendererParameters()
{
  renderer->setPosition(&pos);
  renderer->setParameter(vvRenderState::VV_CODEC, codec);

  if(rrMode == RR_IBR)
    renderer->setParameter(vvRenderer::VV_USE_IBR, 1.);
}


//----------------------------------------------------------------------------
/** Set active rendering algorithm.
 */
void vvServer::createRenderer(std::string type, std::string options,
                         std::vector<BrickList>* bricks, const int maxBrickSizeX,
                         const int maxBrickSizeY, const int maxBrickSizeZ)
{
  vvDebugMsg::msg(3, "vvView::setRenderer()");

  glewInit();
  ds->initARBDebugOutput();

  ds->currentRenderer = type;
  ds->currentOptions = options;

  if(renderer)
  renderState = *renderer;
  delete renderer;
  renderer = NULL;
  vvVector3 maxBrickSize(maxBrickSizeX, maxBrickSizeY, maxBrickSizeZ);
  renderState.setParameterV3(vvRenderState::VV_MAX_BRICK_SIZE, maxBrickSize);

  if(bricks)
  {
    renderer = new vvTexRend(vd, renderState, vvTexRend::VV_BRICKS, vvTexRend::VV_PIX_SHD, bricks);
  }
  else
  {
    renderer = vvRendererFactory::create(vd, renderState, type.c_str(), options.c_str());
  }

  applyRendererParameters();
}

//----------------------------------------------------------------------------
/** Timer callback method, triggered by glutTimerFunc().
  @param id  timer ID: 0=animation, 1=rotation
*/
void vvServer::timerCallback(int id)
{
  vvDebugMsg::msg(3, "vvView::timerCallback()");

  switch(id)
  {
  case SERVER_TIMER:
    ds->serverLoop();
    exit(0);
    break;
  default:
    break;
    }
}

/** Callback function for gl errors.
  If the extension GL_ARB_debug_output is available, this callback
  function will be called automatically if an opengl error is
  generated
*/
void vvServer::debugCallbackARB(GLenum /*source*/, GLenum /*type*/, GLuint /*id*/, GLenum /*severity*/,
                      GLsizei /*length*/, GLchar const* message, GLvoid* /*userParam*/)
{
  cerr << "=======================================================" << endl;
  cerr << "Execution stopped because an OpenGL error was detected." << endl;
  cerr << "Message: " << message << endl;
  if (ds->showBt)
  {
    cerr << "Backtrace is following" << endl;
  }
  cerr << "=======================================================" << endl;
  if (ds->showBt)
  {
    vvToolshed::printBacktrace();
  }
  exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------
/// Initialize the GLUT window and the OpenGL graphics context.
void vvServer::initGraphics(int argc, char *argv[])
{
  vvDebugMsg::msg(1, "vvView::initGraphics()");

  char* version;
  char  title[128];                              // window title

  cerr << "Number of CPUs found: " << vvToolshed::getNumProcessors() << endl;
  cerr << "Initializing GLUT." << endl;

  glutInit(&argc, argv);                // initialize GLUT

// Other glut versions than freeglut currently don't support
// debug context flags.
#if defined(FREEGLUT) && defined(GLUT_INIT_MAJOR_VERSION)
  glutInitContextFlags(GLUT_DEBUG);
#endif // FREEGLUT

  glutInitWindowSize(winWidth, winHeight);       // set initial window size

  // Create window title.
  // Don't use sprintf, it won't work with macros on Irix!
  strcpy(title, "Virvo File Viewer V");
  strcat(title, virvo::getVersionMajor());
  strcat(title, ".");
  strcat(title, virvo::getReleaseCounter());
  window = glutCreateWindow(title);              // open window and set window title

  // Set GL state:
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glEnable(GL_DEPTH_TEST);

  // Set Glut callbacks:
  glutDisplayFunc(displayCallback);
  glutReshapeFunc(reshapeCallback);

  version = (char*)glGetString(GL_VERSION);
  cerr << "Found OpenGL version: " << version << endl;
  if (strncmp(version,"1.0",3)==0)
  {
    cerr << "VView requires OpenGL version 1.1 or greater." << endl;
  }

  vvGLTools::checkOpenGLextensions();

  if (vvDebugMsg::isActive(2))
  {
    cerr << "\nSupported OpenGL extensions:" << endl;
    vvGLTools::displayOpenGLextensions(vvGLTools::ONE_BY_ONE);
  }
}


void vvServer::initARBDebugOutput()
{
  if (dbgOutputExtSet)
  {
    return;
  }

// As of January 2011, only freeglut supports glutInitContextFlags
// with GLUT_DEBUG. This may be outdated in the meantime as may be
// those checks!
#ifdef FREEGLUT
#ifdef GL_ARB_debug_output
#if !defined(__GNUC__) || !defined(_WIN32)
  if (glDebugMessageCallbackARB != NULL)
  {
    cerr << "Init callback function for GL_ARB_debug_output extension" << endl;
    glDebugMessageCallbackARB(debugCallbackARB, NULL);
  }
  else
#endif
  {
    cerr << "glDebugMessageCallbackARB not available" << endl;
  }
#else
  cerr << "Consider installing GLEW >= 1.5.7 for extension GL_ARB_debug_output" << endl;
#endif // GL_ARB_debug_output
#endif // FREEGLUT

  dbgOutputExtSet = true;
}

//----------------------------------------------------------------------------
/// Display command usage help on the command line.
void vvServer::displayHelpInfo()
{
  vvDebugMsg::msg(1, "vvView::displayHelpInfo()");

  cerr << "Syntax:" << endl;
  cerr << endl;
  cerr << "  vserver [options]" << endl;
  cerr << endl;
  cerr << "Available options:" << endl;
  cerr << endl;
  cerr << "-port (-p)" << endl;
  cerr << " Don't use the default port (31050), but the specified one" << endl;
  cerr << endl;
}

//----------------------------------------------------------------------------
/** Parse command line arguments.
  @param argc,argv command line arguments
  @return true if parsing ok, false on error
*/
bool vvServer::parseCommandLine(int argc, char** argv)
{
  vvDebugMsg::msg(1, "vvView::parseCommandLine()");

  int arg;                                       // index of currently processed command line argument

  arg = 0;
  for (;;)
  {
    if ((++arg)>=argc) return true;
    if (vvToolshed::strCompare(argv[arg], "-help")==0 ||
        vvToolshed::strCompare(argv[arg], "-h")==0 ||
        vvToolshed::strCompare(argv[arg], "-?")==0 ||
        vvToolshed::strCompare(argv[arg], "/?")==0)
    {
      displayHelpInfo();
      return false;
    }
    else if (vvToolshed::strCompare(argv[arg], "-port")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "No port specified, defaulting to: " << vvServer::DEFAULT_PORT << endl;
        port = vvServer::DEFAULT_PORT;
        return false;
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
      }
      else
      {
        cerr << "test----------------------"<<endl;
        port = atoi(argv[arg]);
      }
    }
    else
    {
      cerr << "Unknown option/parameter:" << argv[arg] << endl;
      return false;
    }
  }
}

//----------------------------------------------------------------------------
/** Main VView routine.
  @param argc,argv command line arguments
  @return 0 if the program finished ok, 1 if an error occurred
*/
int vvServer::run(int argc, char** argv)
{
  vvDebugMsg::msg(1, "vvView::run()");

  cerr << "VView " << virvo::getVersionMajor() << "." << virvo::getReleaseCounter() << endl;
  cerr << "(c) " << virvo::getYearOfRelease() << " Juergen Schulze (schulze@cs.brown.edu)" << endl;
  cerr << "Brown University" << endl << endl;

  if (argc<2)
  {
    cerr << "VView (=Virvo View) is a utility to display volume files." << endl;
    cerr << "The Virvo volume rendering system was developed at the University of Stuttgart." << endl;
    cerr << "Please find more information at http://www.cs.brown.edu/people/schulze/virvo/." << endl;
    cerr << endl;
    cerr << "Syntax:" << endl;
    cerr << "  vview [<volume_file.xxx>] [options]" << endl;
    cerr << endl;
    cerr << "For a list of options type:" << endl;
    cerr << "  vview -help" << endl;
    cerr << endl;
  }
  else if (parseCommandLine(argc, argv) == false) return 1;

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
