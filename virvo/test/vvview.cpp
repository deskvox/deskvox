//****************************************************************************
// Project:         Virvo (Virtual Reality Volume Renderer)
// Copyright:       (c) 1999-2004 Jurgen P. Schulze. All rights reserved.
// Author's E-Mail: schulze@cs.brown.edu
// Affiliation:     Brown University, Department of Computer Science
//****************************************************************************

#define IMAGESPACE_APPROX
#ifndef HLRS
#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif
#endif
#include <virvo/vvglew.h>

#include <iostream>
#include <fstream>
#include <istream>
#include <string>
#include <iomanip>
using std::cerr;
using std::endl;
using std::ios;

#include <stdlib.h>
#include <stdio.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#ifdef FREEGLUT
// For glutInitContextFlags(GLUT_DEBUG), needed for GL_ARB_debug_output
#include <GL/freeglut.h>
#endif
#endif
#include <time.h>
#include <assert.h>
#include <math.h>

#ifdef WIN32
#ifndef HLRS
#include <winsock2.h>
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include <virvo/vvvirvo.h>
#include <virvo/vvgltools.h>
#include <virvo/vvtoolshed.h>
#include <virvo/vvoffscreenbuffer.h>
#include <virvo/vvprintgl.h>
#include <virvo/vvstopwatch.h>
#include <virvo/vvrendercontext.h>
//#include <virvo/vvclusterclient.h>
//#include <virvo/vvclusterserver.h>
#include <virvo/vvibrclient.h>
#include <virvo/vvibrserver.h>
#include <virvo/vvimageclient.h>
#include <virvo/vvimageserver.h>
#include <virvo/vvimage.h>
#include <virvo/vvremoteserver.h>
#include <virvo/vvrenderer.h>
#include <virvo/vvfileio.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvtexrend.h>
#include <virvo/vvsoftpar.h>
#include <virvo/vvsoftper.h>
#include <virvo/vvcudasw.h>
#include <virvo/vvcuda.h>

#ifdef HAVE_CUDA
#include <virvo/vvrayrend.h>
#endif
#ifdef HAVE_VOLPACK
#include <virvo/vvrendervp.h>
#endif
#include <virvo/vvbonjour/vvbonjourbrowser.h>
#include <virvo/vvbonjour/vvbonjourresolver.h>

#include "vvobjview.h"
#include "vvperformancetest.h"
#include "vvview.h"

const int vvView::ROT_TIMER_DELAY = 20;
const int vvView::DEFAULTSIZE = 512;
const float vvView::OBJ_SIZE  = 1.0f;
const int vvView::DEFAULT_PORT = 31050;
vvView* vvView::ds = NULL;

//#define CLIPPING_TEST
//#define FBO_WITH_GEOMETRY_TEST

//----------------------------------------------------------------------------
/// Constructor
vvView::vvView()
{
  lastWidth = lastHeight = DEFAULTSIZE;
  winWidth = winHeight = DEFAULTSIZE;
  lastPosX = lastPosY = 50;
  pressedButton = NO_BUTTON;
  ds = this;
  renderer = NULL;
  vd = NULL;
  ov = NULL;
  currentGeom = vvTexRend::VV_AUTO;              // vvTexRend::VV_SLICES;
  currentVoxels = vvTexRend::VV_BEST;            // vvTexRend::VV_RGBA;
  rendererType = vvRenderer::TEXREND;
  bgColor[0] = bgColor[1] = bgColor[2] = 0.0f;
  frame = 0;
  filename = NULL;
  window = 0;
  draftQuality = 1.0f;
  highQuality = 5.0f;
  onOff[0] = "off";
  onOff[1] = "on";
  pos.zero();
  numDisplays = 0;
  displayNames = NULL;
  animating             = false;
  rotating              = false;
  activeStereoCapable   = false;
  tryQuadBuffer         = false;
  boundariesMode        = false;
  orientationMode       = false;
  fpsMode               = false;
  stereoMode            = 0;
  fullscreenMode        = false;
  interpolMode          = true;
  warpInterpolMode      = true;
  preintMode            = false;
  paletteMode           = false;
  emptySpaceLeapingMode = false;
  earlyRayTermination   = false;
  perspectiveMode       = true;
  timingMode            = false;
  opCorrMode            = true;
  gammaMode             = false;
  mipMode               = 0;
  rotationMode          = false;
  refinement            = false;
  hqMode                = false;
  animSpeed             = 1.0f;
  iconMode              = false;
  gpuproxygeo           = true;
  useOffscreenBuffer    = false;
  bufferPrecision       = 8;
  useHeadLight          = false;
  clientMode            = false;
  serverMode            = false;
  slavePort             = vvView::DEFAULT_PORT;
  ibrPrecision          = 8;
  ibrMode               = vvRenderer::VV_MAX_GRADIENT;
  codec                 = 4;
  rrMode                = RR_NONE;
  clipBuffer            = NULL;
  framebufferDump       = NULL;
  benchmark             = false;
  testSuiteFileName     = NULL;
  showBricks            = false;
  recordMode            = false;
  matrixFile            = NULL;
  roiEnabled            = false;
  sphericalROI          = false;
  clipPlane             = false;
  clipPerimeter         = false;
  mvScale               = 1.0f;
  dbgOutputExtSet       = false;
  showBt                = true;


  // keep in sync with vvrenderer.h
  rendererName.push_back("TexRend");
  rendererName.push_back("Parallel software shear-warp");
  rendererName.push_back("Perspective software shear-warp");
  rendererName.push_back("Parallel GPU shear-warp");
  rendererName.push_back("Perspective GPU shear-warp");
  rendererName.push_back("CUDA ray caster");
  rendererName.push_back("VolPack");
  rendererName.push_back("Simian");
  rendererName.push_back("Imgrend");
  rendererName.push_back("Unknown");
  rendererName.push_back("Stingray");
  rendererName.push_back("Out of core texture renderer");
  rendererName.push_back("Image based remote rendering");
  rendererName.push_back("Remote rendering");
  assert(rendererName.size() == vvRenderer::NUM_RENDERERS);
  rayRenderer           = false;

}


//----------------------------------------------------------------------------
/// Destructor.
vvView::~vvView()
{
  if (recordMode && matrixFile)
  {
    fclose(matrixFile);
  }
  delete renderer;
  delete ov;
  delete vd;
  ds = NULL;
}

void vvView::serverLoop()
{
  while (1)
  {
    cerr << "Listening on port " << slavePort << endl;

    vvSocketIO *sock = new vvSocketIO(slavePort, vvSocket::VV_TCP);
    sock->set_debuglevel(vvDebugMsg::getDebugLevel());
    if(sock->init() != vvSocket::VV_OK)
    {
      std::cerr << "Failed to initialize server socket on port " << slavePort << std::endl;
      delete sock;
      break;
    }

#ifdef HAVE_BONJOUR
    // Register the bonjour service for the slave.
    vvBonjourRegistrar* registrar = new vvBonjourRegistrar();
    const vvBonjourEntry entry = vvBonjourEntry("Virvo render slave",
        "_distrendering._tcp", "");
    registrar->registerService(entry, slavePort);
#endif

    vvRemoteServer* server = NULL;
    int type;
    sock->getInt32(type);
    switch(type)
    {
      case vvRenderer::REMOTE_IMAGE:
        rrMode = RR_IMAGE;
        //setRendererType(vvRenderer::TEXREND);
        server = new vvImageServer(sock);
        break;
      case vvRenderer::REMOTE_IBR:
#ifdef HAVE_CUDA
        rrMode = RR_IBR;
        setRendererType(vvRenderer::RAYREND);
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
      delete vd;
      vd = NULL;
      cerr << "Could not initialize volume data" << endl;
      cerr << "Continuing with next client..." << endl;
      continue;
    }

#if 0
    // Get bricks to render
    std::vector<BrickList>* frames = new std::vector<BrickList>();
    if (rrMode == RR_CLUSTER)
    {
      vvClusterServer* clusterServer = dynamic_cast<vvClusterServer*>(server);
      assert(clusterServer != NULL);

      bool ok = true;
      for (int f=0; f<vd->frames; ++f)
      {
        BrickList bricks;
        if (clusterServer->initBricks(bricks) != vvClusterServer::VV_OK)
        {
          ok = false;
          delete frames;
          cerr << "Could not initialize brick structure" << endl;
          cerr << "Continuing with next client..." << endl;
          break;
        }
        frames->push_back(bricks);
      }

      if(!ok)
      {
        delete vd;
        vd = NULL;
        continue;
      }
    }
    setRenderer(currentGeom, currentVoxels, frames);
#endif

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

      ov = new vvObjView();
      setProjectionMode(perspectiveMode);
      setRenderer(currentGeom, currentVoxels);
      srand(time(NULL));

      server->renderLoop(renderer);
      cerr << "Exiting..." << endl;

      delete ov;
      ov = NULL;

      delete vd;
      vd = NULL;
    }

    // Frames vector with bricks is deleted along with the renderer.
    // Don't free them here.
    // see setRenderer().

    delete server;
    server = NULL;
  }
}

//----------------------------------------------------------------------------
/** VView main loop.
  @param filename  volume file to display
*/
void vvView::mainLoop(int argc, char *argv[])
{
  vvDebugMsg::msg(2, "vvView::mainLoop()");

  if (serverMode)
  {
    initGraphics(argc, argv);
#ifdef HAVE_CUDA
    vvCuda::initGlInterop();
#endif

    // Set window title:
    char title[1024];
    sprintf(title, "vvView Server: %d", slavePort);
    glutSetWindowTitle(title);

    glutTimerFunc(1, timerCallback, SERVER_TIMER);

    glutMainLoop();
  }
  else
  {
    if (filename!=NULL && strlen(filename)==0) filename = NULL;
    if (filename!=NULL)
    {
      vd = new vvVolDesc(filename);
      cerr << "Loading volume file: " << filename << endl;
    }
    else
    {
      vd = new vvVolDesc();
      cerr << "Using default volume" << endl;
    }

    vvFileIO fio;
    if (fio.loadVolumeData(vd) != vvFileIO::OK)
    {
      cerr << "Error loading volume file" << endl;
      delete vd;
      vd = NULL;
      return;
    }
    else vd->printInfoLine();

    // Set default color scheme if no TF present:
    if (vd->tf.isEmpty())
    {
      vd->tf.setDefaultAlpha(0, 0.0, 1.0);
      vd->tf.setDefaultColors((vd->chan==1) ? 0 : 2, 0.0, 1.0);
    }

    if (slaveNames.size() == 0)
    {
#ifdef HAVE_BONJOUR
      vvBonjourBrowser* bonjourBrowser = new vvBonjourBrowser();
      bonjourBrowser->browseForServiceType("_distrendering._tcp");
      int timeout = 1000;
      while (bonjourBrowser->expectingServices() && timeout > 0)
      {
        --timeout;
        sleep(1);
      }
      std::vector<vvBonjourEntry> entries = bonjourBrowser->getBonjourEntries();
      for (std::vector<vvBonjourEntry>::const_iterator it = entries.begin(); it != entries.end(); ++it)
      {
        vvBonjourResolver* bonjourResolver = new vvBonjourResolver();
        bonjourResolver->resolveBonjourEntry((*it));
      }
#endif
    }

    initGraphics(argc, argv);
#ifdef HAVE_CUDA
    vvCuda::initGlInterop();
#endif

    if (rrMode == RR_IBR)
    {
      setRendererType(vvRenderer::REMOTE_IBR);
    }
    else if(rrMode == RR_IMAGE)
    {
      setRendererType(vvRenderer::REMOTE_IMAGE);
    }
    else if(rrMode == RR_CLUSTER)
    {
      //_remoteClient = new vvClusterClient(ds->renderState, slaveNames, slavePorts, slaveFileNames, filename);
    }

    setRenderer(currentGeom, currentVoxels);

    const vvVector3 size = vd->getSize();
    const float maxedge = ts_max(size[0], size[1], size[2]);

    mvScale = 1.0f / maxedge;
    cerr << "Scale modelview matrix by " << mvScale << endl;

    animSpeed = vd->dt;
    createMenus();

    ov = new vvObjView();
    ds->ov->mv.scale(mvScale);

    setProjectionMode(perspectiveMode);

    // Set window title:
    if (filename!=NULL) glutSetWindowTitle(filename);

    srand(time(NULL));
    if(benchmark)
    {
      glutTimerFunc(1, timerCallback, BENCHMARK_TIMER);
    }

    if (playMode)
    {
      renderMotion();
    }
    else
    {
      glutMainLoop();
    }

    delete vd;
    vd = NULL;
  }
}

//----------------------------------------------------------------------------
/** Add a display to render. This is relevant if a multi threaded renderer
    is used which distributes its work loads among multiple gpus.
    @param name The descriptor for the x-display (e.g. host:0.1 <==>
                host = host,
                display = 0,
                screen = 1)
*/
void vvView::addDisplay(const char* name)
{
  unsigned int i;

  ++numDisplays;
  const char** tmp = new const char*[numDisplays];

  for (i = 0; i < (numDisplays - 1); ++i)
  {
    tmp[i] = displayNames[i];
  }
  tmp[i] = name;

  delete[] displayNames;
  displayNames = tmp;
}


//----------------------------------------------------------------------------
/** Callback method for window resizes.
    @param w,h new window width and height
*/
void vvView::reshapeCallback(int w, int h)
{
  vvDebugMsg::msg(2, "vvView::reshapeCallback(): ", w, h);

  ds->winWidth  = w;
  ds->winHeight = h;

  // Resize OpenGL viewport:
  glViewport(0, 0, ds->winWidth, ds->winHeight);

  // Set new aspect ratio:
  if (ds->winHeight > 0 && ds->ov) ds->ov->setAspectRatio((float)ds->winWidth / (float)ds->winHeight);

  glDrawBuffer(GL_FRONT_AND_BACK);               // select all buffers
                                                 // set clear color
  glClearColor(ds->bgColor[0], ds->bgColor[1], ds->bgColor[2], 1.0f);
                                                 // clear window
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}


//----------------------------------------------------------------------------
/// Callback method for window redraws.
void vvView::displayCallback(void)
{
  vvDebugMsg::msg(3, "vvView::displayCallback()");

  vvGLTools::printGLError("enter vvView::displayCallback()");

  glDrawBuffer(GL_BACK);
  glClearColor(ds->bgColor[0], ds->bgColor[1], ds->bgColor[2], 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if(!ds->renderer)
    return;

  ds->renderer->setParameter(vvRenderState::VV_QUALITY, ((ds->hqMode) ? ds->highQuality : ds->draftQuality));

  ds->renderer->setParameter(vvRenderer::VV_MEASURETIME, (float)ds->fpsMode);

  // Draw volume:
  glMatrixMode(GL_MODELVIEW);
  if (ds->stereoMode>0)                          // stereo mode?
  {
    if (ds->stereoMode==1)                      // active stereo?
    {
      // Draw right image:
      glDrawBuffer(GL_BACK_RIGHT);
      ds->ov->updateModelviewMatrix(vvObjView::RIGHT_EYE);
      ds->renderer->renderVolumeGL();

      // Draw left image:
      glDrawBuffer(GL_BACK_LEFT);
      ds->ov->updateModelviewMatrix(vvObjView::LEFT_EYE);
      ds->renderer->renderVolumeGL();
    }
    // passive stereo?
    else if (ds->stereoMode==2 || ds->stereoMode==3)
    {
      ds->ov->setAspectRatio((float)ds->winWidth / 2 / (float)ds->winHeight);
      for (int i=0; i<2; ++i)
      {
        // Specify eye to draw:
        if (i==0) ds->ov->updateModelviewMatrix(vvObjView::LEFT_EYE);
        else      ds->ov->updateModelviewMatrix(vvObjView::RIGHT_EYE);

        // Specify where to draw it:
        if ((ds->stereoMode==2 && i==0) || (ds->stereoMode==3 && i==1))
        {
          // right
          glViewport(ds->winWidth / 2, 0, ds->winWidth / 2, ds->winHeight);
        }
        else
        {
          // left
          glViewport(0, 0, ds->winWidth / 2, ds->winHeight);
        }
        ds->renderer->renderVolumeGL();
      }

      // Reset viewport and aspect ratio:
      glViewport(0, 0, ds->winWidth, ds->winHeight);
      ds->ov->setAspectRatio((float)ds->winWidth / (float)ds->winHeight);
    }
  }
  else                                           // mono mode
  {
    glDrawBuffer(GL_BACK);
    ds->ov->updateModelviewMatrix(vvObjView::LEFT_EYE);
#ifdef CLIPPING_TEST
    ds->renderClipObject();
    ds->renderQuad();
#endif

#ifdef FBO_WITH_GEOMETRY_TEST
    ds->renderCube();
    ds->renderer->_renderState._opaqueGeometryPresent = true;
#endif
    ds->renderer->renderVolumeGL();
    //ds->renderCube();
  }

  if (ds->iconMode)
  {
    if (ds->vd->iconSize>0)
    {
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      glLoadIdentity();
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();
      glRasterPos2f(-1.0f, -0.0f);
      glPixelZoom(1.0f, -1.0f);
      glDrawPixels(ds->vd->iconSize, ds->vd->iconSize, GL_RGBA, GL_UNSIGNED_BYTE, ds->vd->iconData);
      glPopMatrix();
      glMatrixMode(GL_PROJECTION);
      glPopMatrix();
      glMatrixMode(GL_MODELVIEW);
    }
    else
    {
      cerr << "No icon stored" << endl;
    }
  }

  if (ds->recordMode)
  {
    ds->ov->saveMV(ds->matrixFile);
    // store time since program start
    fprintf(ds->matrixFile, "# %f\n", ds->stopWatch.getTime());
  }
  glutSwapBuffers();

  vvGLTools::printGLError("leave vvView::displayCallback()");
}


//----------------------------------------------------------------------------
/** Callback method for mouse button actions.
    @param button clicked button ID (left, middle, right)
    @param state  new button state (up or down)
    @param x,y    new mouse coordinates
*/
void vvView::buttonCallback(int button, int state, int x, int y)
{
  vvDebugMsg::msg(3, "vvView::buttonCallback()");

  const int ROTATION_THRESHOLD = 4;              // empirical threshold for auto rotation
  int dist, dx, dy;

  if (state==GLUT_DOWN)
  {
    ds->hqMode = false;
    switch (button)
    {
    case GLUT_LEFT_BUTTON:
      ds->pressedButton |= LEFT_BUTTON;
      ds->rotating = false;
      break;
    case GLUT_MIDDLE_BUTTON:
      ds->pressedButton |= MIDDLE_BUTTON;
      break;
    case GLUT_RIGHT_BUTTON:
      ds->pressedButton |= RIGHT_BUTTON;
      break;
    default: break;
    }
    ds->curX = ds->lastX = x;
    ds->curY = ds->lastY = y;
  }
  else if (state==GLUT_UP)
  {
    if (ds->refinement) ds->hqMode = true;
    switch (button)
    {
    case GLUT_LEFT_BUTTON:
                                                // only do something if button was pressed before
      if ((ds->pressedButton & LEFT_BUTTON) != 0)
      {
        ds->pressedButton &= ~LEFT_BUTTON;

        // Compute length of last mouse movement:
        dx = ts_abs(ds->curX - ds->lastX);
        dy = ts_abs(ds->curY - ds->lastY);
        dist = int(sqrt(float(dx*dx + dy*dy)));

                                                // auto-rotate if threshold was exceeded
        if (dist > ROTATION_THRESHOLD && ds->rotationMode)
        {
          ds->rotating = true;
          ds->x1 = ds->lastX;
          ds->y1 = ds->lastY;
          ds->x2 = ds->curX;
          ds->y2 = ds->curY;
          glutTimerFunc(ROT_TIMER_DELAY, timerCallback, ROTATION_TIMER);
          ds->hqMode = false;
        }
      }
      break;
    case GLUT_MIDDLE_BUTTON:
      ds->pressedButton &= ~MIDDLE_BUTTON;
      break;
    case GLUT_RIGHT_BUTTON:
      ds->pressedButton &= ~RIGHT_BUTTON;
      break;
    default: break;
    }
    if (ds->refinement) glutPostRedisplay();
  }
}


//----------------------------------------------------------------------------
/** Callback for mouse motion.
    @param x,y new mouse coordinates
*/
void vvView::motionCallback(int x, int y)
{
  vvDebugMsg::msg(3, "vvView::motionCallback()");

  int dx, dy;
  float factor;

  ds->lastX = ds->curX;                          // save current mouse coordinates for next call
  ds->lastY = ds->curY;
  ds->curX = x;
  ds->curY = y;
  dx = ds->curX - ds->lastX;
  dy = ds->curY - ds->lastY;

  switch (ds->pressedButton)
  {
  case LEFT_BUTTON:
    ds->ov->trackballRotation(ds->winWidth, ds->winHeight,
                              ds->lastX, ds->lastY, ds->curX, ds->curY);
    break;

  case MIDDLE_BUTTON:
  case LEFT_BUTTON | RIGHT_BUTTON:
    if (ds->perspectiveMode==false)
      ds->ov->mv.translate((float)dx * 0.01f, -(float)dy * 0.01f, 0.0f);
    else
      ds->ov->mv.translate(0.0f, 0.0f, (float)dy / 10.0f);
    break;

  case RIGHT_BUTTON:
    factor = 1.0f + ((float)dy) * 0.01f;
    if (factor > 2.0f) factor = 2.0f;
    if (factor < 0.5f) factor = 0.5f;
    ds->ov->mv.scale(factor, factor, factor);
    break;

  default: break;
  }

  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Set software renderering flag.
 */
void vvView::setRendererType(enum vvRenderer::RendererType type)
{
  rendererType = type;
}

void vvView::applyRendererParameters()
{
  renderer->setParameter(vvRenderState::VV_BOUNDARIES, boundariesMode);
  renderer->setPosition(&pos);
  renderer->setParameter(vvRenderState::VV_SLICEINT, (interpolMode) ? 1.0f : 0.0f);
  renderer->setParameter(vvRenderer::VV_WARPINT, (warpInterpolMode) ? 1.0f : 0.0f);
  renderer->setParameter(vvRenderer::VV_PREINT, (preintMode) ? 1.0f : 0.0f);
  renderer->setParameter(vvRenderState::VV_MIP_MODE, mipMode);
  renderer->setParameter(vvRenderState::VV_QUALITY, (hqMode) ? highQuality : draftQuality);
  renderer->setParameter(vvRenderer::VV_LEAPEMPTY, emptySpaceLeapingMode);
  renderer->setParameter(vvRenderer::VV_TERMINATEEARLY, earlyRayTermination);
  renderer->setParameter(vvRenderer::VV_LIGHTING, useHeadLight);
  renderer->setParameter(vvRenderer::VV_GPUPROXYGEO, gpuproxygeo);
  renderer->setParameter(vvRenderer::VV_OFFSCREENBUFFER, useOffscreenBuffer);
  renderer->setParameter(vvRenderer::VV_IMG_PRECISION, bufferPrecision);
  renderer->setParameter(vvRenderState::VV_SHOW_BRICKS, showBricks);
  renderer->setParameter(vvRenderState::VV_CODEC, codec);
}


//----------------------------------------------------------------------------
/** Set active rendering algorithm.
 */
void vvView::setRenderer(vvTexRend::GeometryType gt, vvTexRend::VoxelType vt,
                         std::vector<BrickList>* bricks, const int maxBrickSizeX,
                         const int maxBrickSizeY, const int maxBrickSizeZ)
{
  vvDebugMsg::msg(3, "vvView::setRenderer()");

  glewInit();
  ds->initARBDebugOutput();

  currentGeom = gt;
  currentVoxels = vt;

  if(renderer)
  renderState = *renderer;
  delete renderer;
  renderer = NULL;
  vvVector3 maxBrickSize(maxBrickSizeX, maxBrickSizeY, maxBrickSizeZ);
  renderState.setParameterV3(vvRenderState::VV_MAX_BRICK_SIZE, maxBrickSize);

  switch(rendererType)
  {
    case vvRenderer::SOFTPAR:
    case vvRenderer::SOFTPER:
      if(perspectiveMode)
        renderer = new vvSoftPer(vd, renderState);
      else
        renderer = new vvSoftPar(vd, renderState);
      break;
    case vvRenderer::CUDAPAR:
    case vvRenderer::CUDAPER:
#ifdef HAVE_CUDA
      if(perspectiveMode)
        renderer = new vvCudaPer(vd, renderState);
      else
        renderer = new vvCudaPar(vd, renderState);
#endif
      break;
    case vvRenderer::RAYREND:
#ifdef HAVE_CUDA
      if(rrMode == RR_IBR)
        renderState.setParameter(vvRenderer::VV_USE_IBR, 1.);
      renderer = new vvRayRend(vd, renderState);
#endif
      break;
#ifdef HAVE_VOLPACK
    case vvRenderer::VOLPACK:
      renderer = new vvRenderVP(vd, renderState);
      break;
#endif
    case vvRenderer::REMOTE_IMAGE:
      renderer = new vvImageClient(vd, renderState, slaveNames[0], slavePorts[0]==-1 ? slavePort : slavePorts[0],
              slaveFileNames.empty() ? NULL : slaveFileNames[0]);
      break;
    case vvRenderer::REMOTE_IBR:
      renderer = new vvIbrClient(vd, renderState, slaveNames[0], slavePorts[0]==-1 ? slavePort : slavePorts[0],
              slaveFileNames.empty() ? NULL : slaveFileNames[0]);
      renderer->setParameter(vvRenderer::VV_IBR_DEPTH_PREC, ibrPrecision);
      renderer->setParameter(vvRenderer::VV_IBR_UNCERTAINTY_PREC, ibrPrecision); // both precisions the same for now
      break;
    default:
      if (numDisplays > 0)
      {
        cerr << "Running in threaded mode using the following displays:" << endl;
        for (unsigned int i=0; i<numDisplays; ++i)
        {
          cerr << displayNames[i] << endl;
        }
        renderer = new vvTexRend(vd, renderState, currentGeom, currentVoxels, bricks, displayNames, numDisplays);
      }
      else
      {
        renderer = new vvTexRend(vd, renderState, currentGeom, currentVoxels, bricks);
      }
  }

  if(!serverMode)
  {
    renderer->setROIEnable(roiEnabled);
    printROIMessage();
  }

  //static_cast<vvTexRend *>(renderer)->setTexMemorySize( 4 );
  //static_cast<vvTexRend *>(renderer)->setComputeBrickSize( false );
  //static_cast<vvTexRend *>(renderer)->setBrickSize( 64 );
  applyRendererParameters();
}


//----------------------------------------------------------------------------
/** Callback method for keyboard action.
    @param key ASCII code of pressed key
*/
void vvView::keyboardCallback(unsigned char key, int, int)
{
  vvDebugMsg::msg(3, "vvView::keyboardCallback()");

  switch((char)key)
  {
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9': ds->rendererMenuCallback(key - '0'); break;
  case '-': ds->rendererMenuCallback(98); break;
  case '+':
  case '=': ds->rendererMenuCallback(99); break;
  case 'a': ds->animMenuCallback(2);  break;
  case 'A': ds->optionsMenuCallback(14); break;
  case 'B': ds->optionsMenuCallback(12); break;
  case 'b': ds->viewMenuCallback(0);  break;
  case 'c': ds->viewMenuCallback(10); break;
  case 'C': ds->optionsMenuCallback(18);  break;
  case 'd': ds->mainMenuCallback(5);  break;
  case 'D': ds->mainMenuCallback(13);  break;
  case 'e': ds->mainMenuCallback(4);  break;
  case 'E': ds->clipMenuCallback(1); break;
  case 'f': ds->viewMenuCallback(2);  break;
  case 'g': ds->optionsMenuCallback(13);  break;
  case 'H': ds->optionsMenuCallback(5); break;
  case 'h': ds->optionsMenuCallback(6); break;
  case 'i': ds->optionsMenuCallback(0);  break;
  case 'I': ds->clipMenuCallback(0); break;
  case 'j': ds->transferMenuCallback(19); break;
  case 'J': ds->transferMenuCallback(20); break;
  case 'k': ds->transferMenuCallback(21); break;
  case 'K': ds->transferMenuCallback(22); break;
  case 'l': ds->transferMenuCallback(23); break;
  case 'L': ds->transferMenuCallback(24); break;
  case 'm': ds->mainMenuCallback(8);  break;
  case 'n': ds->animMenuCallback(0);  break;
  case 'N': ds->animMenuCallback(1);  break;
  case 'o': ds->viewMenuCallback(1);  break;
  case 'p': ds->mainMenuCallback(0);  break;
  case 'P': ds->optionsMenuCallback(1);  break;
  case 27:                                    // Escape
  case 'q': ds->mainMenuCallback(12); break;
  case 'r': ds->mainMenuCallback(7);  break;
  case 'R': ds->roiMenuCallback(0); break;
  case 's': ds->animMenuCallback(4);  break;
  case 'S': ds->animMenuCallback(5);  break;
  case 't': ds->mainMenuCallback(11); break;
  case 'u': ds->viewMenuCallback(8);  break;
  case 'v': ds->viewMenuCallback(9);  break;
  case 'w': ds->viewMenuCallback(6);  break;
  case 'W': ds->optionsMenuCallback(16);  break;
  case 'x':
    if (ds->clipEditMode)
    {
      ds->editClipPlane(PLANE_X, 0.05f);
    }
    else
    {
      ds->optionsMenuCallback(2);
    }
    break;
  case 'y':
    if (ds->clipEditMode)
    {
      ds->editClipPlane(PLANE_Y, 0.05f);
    }
    break;
  case 'z':
    if (ds->clipEditMode)
    {
      ds->editClipPlane(PLANE_Z, 0.05f);
    }
    else
    {
      ds->viewMenuCallback(5);
    }
    break;
  case '<': ds->transferMenuCallback(12);  break;
  case '>': ds->transferMenuCallback(13);  break;
  case '[': ds->roiMenuCallback(98); break;
  case ']': ds->roiMenuCallback(99); break;
  case '{': ds->optionsMenuCallback(11); break;
  case '}': ds->optionsMenuCallback(10); break;
  case '#': ds->optionsMenuCallback(17); break;
  case ' ': ds->optionsMenuCallback(8); break;
  default: cerr << "key '" << char(key) << "' has no function'" << endl; break;
  }
}


//----------------------------------------------------------------------------
/** Callback method for special keys.
    @param key ID of pressed key
*/
void vvView::specialCallback(int key, int, int)
{
  vvDebugMsg::msg(3, "vvView::specialCallback()");

  vvVector3 probePos;
  ds->renderer->getProbePosition(&probePos);

  const int modifiers = glutGetModifiers();
  const float delta = 0.1f / ds->mvScale;

  const vvVector3 clipPoint = ds->renderer->getParameterV3(vvRenderState::VV_CLIP_POINT);
  const vvVector3 clipNormal = ds->renderer->getParameterV3(vvRenderState::VV_CLIP_NORMAL);

  switch(key)
  {
  case GLUT_KEY_LEFT:
    if (ds->roiEnabled)
    {
      probePos[0] -= delta;
      ds->renderer->setProbePosition(&probePos);
    }
    else if (ds->clipEditMode)
    {
      ds->renderer->setParameterV3(vvRenderState::VV_CLIP_POINT, clipPoint - clipNormal * delta);
    }
    break;
  case GLUT_KEY_RIGHT:
    if (ds->roiEnabled)
    {
      probePos[0] += delta;
      ds->renderer->setProbePosition(&probePos);
    }
    else if (ds->clipEditMode)
    {
      ds->renderer->setParameterV3(vvRenderState::VV_CLIP_POINT, clipPoint + clipNormal * delta);
    }
    break;
  case GLUT_KEY_UP:
    if (ds->roiEnabled)
    {
      if (modifiers & GLUT_ACTIVE_SHIFT)
      {
        probePos[2] += delta;
      }
      else
      {
        probePos[1] += delta;
      }
      ds->renderer->setProbePosition(&probePos);
    }
    else if (ds->clipEditMode)
    {
      ds->renderer->setParameterV3(vvRenderState::VV_CLIP_POINT, clipPoint + clipNormal * delta);
    }
    break;
  case GLUT_KEY_DOWN:
    if (ds->roiEnabled)
    {
      if (modifiers & GLUT_ACTIVE_SHIFT)
      {
        probePos[2] -= delta;
      }
      else
      {
        probePos[1] -= delta;
      }
      ds->renderer->setProbePosition(&probePos);
    }
    else if (ds->clipEditMode)
    {
      ds->renderer->setParameterV3(vvRenderState::VV_CLIP_POINT, clipPoint - clipNormal * delta);
    }
    break;
  default: break;
  }

  glutPostRedisplay();
}

void vvView::runTest()
{
  double tmin = DBL_MAX;
  ds->applyRendererParameters();
  for(int i=0; i<3; ++i)
  {
    double t = performanceTest();
    if(t < tmin)
      tmin = t;
  }
  printf("%f\n", tmin);
}


//----------------------------------------------------------------------------
/** Timer callback method, triggered by glutTimerFunc().
  @param id  timer ID: 0=animation, 1=rotation
*/
void vvView::timerCallback(int id)
{
  vvDebugMsg::msg(3, "vvView::timerCallback()");

  switch(id)
  {
  case ANIMATION_TIMER:
    if (ds->animating)
    {
      ++ds->frame;
      ds->setAnimationFrame(ds->frame);
      glutTimerFunc(int(ds->animSpeed * 1000.0f), timerCallback, ANIMATION_TIMER);
    }
    break;
  case ROTATION_TIMER:
    if (ds->rotating)
    {
      ds->ov->trackballRotation(ds->winWidth, ds->winHeight,
                                ds->x1, ds->y1, ds->x2, ds->y2);
      glutPostRedisplay();
      glutTimerFunc(ROT_TIMER_DELAY, timerCallback, ROTATION_TIMER);
    }
    break;
  case BENCHMARK_TIMER:
    {
#if 0
        ds->vd->tf.setDefaultColors(0, 0.0, 1.0);
        ds->vd->tf.setDefaultAlpha(0, 0.0, 1.0);
        ds->setProjectionMode(false);

        ds->useOffscreenBuffer = false;
        ds->preintMode = false;
        ds->earlyRayTermination = false;
        ds->emptySpaceLeapingMode = false;
        ds->bufferPrecision = 8;

        // CUDA SW
        ds->setRendererType(vvRenderer::CUDAPAR);
        ds->setRenderer();
        runTest();

        ds->preintMode = true;
        runTest();
        ds->preintMode = false;

        ds->earlyRayTermination = true;
        runTest();
        ds->earlyRayTermination = false;

        ds->emptySpaceLeapingMode = true;
        runTest();
        ds->emptySpaceLeapingMode = false;

        ds->useOffscreenBuffer = true;
        ds->bufferPrecision = 32;
        runTest();

        ds->preintMode = true;
        runTest();
        ds->preintMode = false;

        ds->earlyRayTermination = true;
        runTest();
        ds->earlyRayTermination = false;

        ds->emptySpaceLeapingMode = true;
        runTest();
        ds->emptySpaceLeapingMode = false;

        // RAYCAST
        ds->setRendererType(vvRenderer::RAYREND);
        ds->setRenderer();
        runTest();

        ds->earlyRayTermination = true;
        runTest();
        ds->earlyRayTermination = false;

        // TEX2D
        ds->useOffscreenBuffer = false;
        ds->bufferPrecision = 8;
        ds->setRendererType(vvRenderer::TEXREND);
        ds->setRenderer(vvTexRend::VV_CUBIC2D);
        runTest();

        ds->useOffscreenBuffer = true;
        ds->bufferPrecision = 32;
        runTest();

        // TEX3D
        ds->useOffscreenBuffer = false;
        ds->bufferPrecision = 8;
        ds->setRendererType(vvRenderer::TEXREND);
        ds->setRenderer(vvTexRend::VV_VIEWPORT);
        runTest();

        ds->preintMode = true;
        runTest();
        ds->preintMode = false;

        ds->useOffscreenBuffer = true;
        ds->bufferPrecision = 32;
        runTest();

        ds->preintMode = true;
        runTest();
        ds->preintMode = false;

        // VOLPACK
        ds->setRendererType(vvRenderer::VOLPACK);
        ds->setRenderer();
#endif
        runTest();
    }
    exit(0);
    break;
  case SERVER_TIMER:
    ds->serverLoop();
    exit(0);
    break;
  default:
    break;
    }
}


//----------------------------------------------------------------------------
/** Callback for main menu.
  @param item selected menu item index
*/
void vvView::mainMenuCallback(int item)
{
  vvDebugMsg::msg(1, "vvView::mainMenuCallback()");

  switch (item)
  {
  case 0:                                     // projection mode
    ds->setProjectionMode(!ds->perspectiveMode);
    switch(ds->rendererType)
    {
      case vvRenderer::SOFTPAR:
      case vvRenderer::SOFTPER:
        delete ds->renderer;
        ds->renderer = NULL;
        if (ds->perspectiveMode)
          ds->renderer = new vvSoftPer(ds->vd, ds->renderState);
        else
          ds->renderer = new vvSoftPar(ds->vd, ds->renderState);
        break;
      case vvRenderer::CUDAPAR:
      case vvRenderer::CUDAPER:
#ifdef HAVE_CUDA
        delete ds->renderer;
        ds->renderer = NULL;
        if (ds->perspectiveMode)
          ds->renderer = new vvCudaPer(ds->vd, ds->renderState);
        else
          ds->renderer = new vvCudaPar(ds->vd, ds->renderState);
#endif
        break;
      default:
        break;
    }
    break;
  case 4:                                     // refinement mode
    if (ds->refinement) ds->refinement = false;
    else ds->refinement = true;
    ds->hqMode = ds->refinement;
    break;
  case 5:                                     // timing mode
    ds->timingMode = !ds->timingMode;
    //ds->renderer->setTimingMode(ds->timingMode);
    break;
  case 7:                                     // reset object position
    ds->ov->reset();
    ds->ov->mv.scale(ds->mvScale);
    ds->setProjectionMode(ds->perspectiveMode);
    break;
  case 8:                                     // menu/zoom mode
    if (ds->menuEnabled)
    {
      glutDetachMenu(GLUT_RIGHT_BUTTON);
      ds->menuEnabled = false;
    }
    else
    {
      glutSetMenu(ds->mainMenu);
      glutAttachMenu(GLUT_RIGHT_BUTTON);
      ds->menuEnabled = true;
    }
    break;
  case 9:                                     // save volume with transfer function
    vvFileIO* fio;
    ds->vd->setFilename("virvo-saved.xvf");
    fio = new vvFileIO();
    fio->saveVolumeData(ds->vd, true);
    delete fio;
    cerr << "Volume saved to file 'virvo-saved.xvf'." << endl;
    break;
  case 11:                                    // performance test
    performanceTest();
    break;
  case 12:                                    // quit
    glutDestroyWindow(ds->window);
    delete ds;
    exit(0);
    break;
  case 13:                                    // rotate debug level
    {
        int l = vvDebugMsg::getDebugLevel()+1;
        l %= 4;
        vvDebugMsg::setDebugLevel(l);
    }
    break;
  default: break;
  }
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for renderer menu.
  @param item selected menu item index
*/
void vvView::rendererMenuCallback(int item)
{
  vvDebugMsg::msg(1, "vvView::rendererMenuCallback()");

  const float QUALITY_CHANGE = 1.05f;            // quality modification unit
  const char QUALITY_NAMES[2][6] =               // quality mode names
  {
    "Draft", "High"
  };

  if (item>=0 && item<=5)
  {
    ds->setRendererType(vvRenderer::TEXREND);
  }

  if (item==0)
  {
    ds->setRenderer(vvTexRend::VV_AUTO, ds->currentVoxels);
  }
  else if (item==1)
  {
    ds->setRenderer(vvTexRend::VV_SLICES, ds->currentVoxels);
  }
  else if (item==2)
  {
    ds->setRenderer(vvTexRend::VV_CUBIC2D, ds->currentVoxels);
  }
  else if (item==3)
  {
    ds->setRenderer(vvTexRend::VV_VIEWPORT, ds->currentVoxels);
  }
  else if (item==4)
  {
    ds->setRenderer(vvTexRend::VV_SPHERICAL, ds->currentVoxels);
  }
  else if (item==5)
  {
    ds->setRenderer(vvTexRend::VV_BRICKS, ds->currentVoxels);
  }
  else if (item==6)
  {
    cerr << "Switched to software shear-warp renderer" << endl;
    if(ds->perspectiveMode)
      ds->setRendererType(vvRenderer::SOFTPAR);
    else
      ds->setRendererType(vvRenderer::SOFTPER);
    ds->setRenderer();
  }
#ifdef HAVE_CUDA
  else if (item==7)
  {
    cerr << "Switched to CUDA shear-warp renderer" << endl;
    if(ds->perspectiveMode)
      ds->setRendererType(vvRenderer::CUDAPAR);
    else
      ds->setRendererType(vvRenderer::CUDAPER);
    ds->setRenderer();
  }
#endif
#ifdef HAVE_CUDA
  else if (item == 8)
  {
    cerr << "Switched to CUDA ray casting renderer" << endl;
    ds->setRendererType(vvRenderer::RAYREND);
    ds->setRenderer();
  }
#endif
#ifdef HAVE_VOLPACK
  else if (item == 9)
  {
    cerr << "Switched to VolPack shear-warp renderer" << endl;
    ds->setRendererType(vvRenderer::VOLPACK);
    ds->setRenderer();
  }
#endif
  else if (item==98 || item==99)
  {
    if (item==98)
    {
      ((ds->hqMode) ? ds->highQuality : ds->draftQuality) /= QUALITY_CHANGE;
    }
    else if (item==99)
    {
      ((ds->hqMode) ? ds->highQuality : ds->draftQuality) *= QUALITY_CHANGE;
    }
    ds->renderer->setParameter(vvRenderState::VV_QUALITY, (ds->hqMode) ? ds->highQuality : ds->draftQuality);
    cerr << QUALITY_NAMES[ds->hqMode] << " quality: " <<
        ((ds->hqMode) ? ds->highQuality : ds->draftQuality) << endl;
  }
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for voxel type menu.
  @param item selected menu item index
*/
void vvView::voxelMenuCallback(int item)
{
  vvDebugMsg::msg(1, "vvView::voxelMenuCallback()");

  ds->setRenderer(ds->currentGeom, (vvTexRend::VoxelType)item);

  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for options menu.
  @param item selected menu item index
*/
void vvView::optionsMenuCallback(int item)
{
  vvVector3 size;

  vvDebugMsg::msg(1, "vvView::optionsMenuCallback()");

  //vvClusterClient* client = dynamic_cast<vvClusterClient*>(ds->_remoteClient);

  switch(item)
  {
  case 0:                                     // slice interpolation mode
    ds->interpolMode = !ds->interpolMode;
    ds->renderer->setParameter(vvRenderer::VV_SLICEINT, (ds->interpolMode) ? 1.0f : 0.0f);
    cerr << "Interpolation mode set to " << int(ds->interpolMode) << endl;
    break;
  case 1:
    ds->preintMode = !ds->preintMode;
    ds->renderer->setParameter(vvRenderer::VV_PREINT, (ds->preintMode) ? 1.0f : 0.0f);
    cerr << "Pre-integration set to " << int(ds->preintMode) << endl;
    break;
  case 2:                                     // min/maximum intensity projection
    ++ds->mipMode;
    if (ds->mipMode>2) ds->mipMode = 0;
    ds->renderer->setParameter(vvRenderState::VV_MIP_MODE, ds->mipMode);
    cerr << "MIP mode set to " << ds->mipMode << endl;
    break;
  case 3:                                     // opacity correction
    ds->opCorrMode = !ds->opCorrMode;
    ds->renderer->setParameter(vvRenderer::VV_OPCORR, (ds->opCorrMode) ? 1.0f : 0.0f);
    cerr << "Opacity correction set to " << int(ds->opCorrMode) << endl;
    break;
  case 4:                                     // gamma correction
    ds->gammaMode = !ds->gammaMode;
    ds->renderer->setGamma(vvRenderer::VV_RED, 2.2f);
    ds->renderer->setGamma(vvRenderer::VV_GREEN, 2.2f);
    ds->renderer->setGamma(vvRenderer::VV_BLUE, 2.2f);
    //ds->renderer->setParameter(vvRenderer::VV_GAMMA, (ds->gammaMode) ? 1.0f : 0.0f);
    cerr << "Gamma correction set to " << int(ds->gammaMode) << endl;
    break;
  case 5:
    ds->emptySpaceLeapingMode = !ds->emptySpaceLeapingMode;
    ds->renderer->setParameter(vvRenderer::VV_LEAPEMPTY, ds->emptySpaceLeapingMode);
    cerr << "Empty space leaping set to " << int(ds->emptySpaceLeapingMode) << endl;
    break;
  case 6:
    ds->useOffscreenBuffer = !ds->useOffscreenBuffer;
    ds->renderer->setParameter(vvRenderer::VV_OFFSCREENBUFFER, ds->useOffscreenBuffer);
    cerr << "Offscreen Buffering set to " << int(ds->useOffscreenBuffer) << endl;
    break;
  case 7:
    ds->useHeadLight = !ds->useHeadLight;
    ds->renderer->setParameter(vvRenderer::VV_LIGHTING, ds->useHeadLight);
    break;
  case 8:                                     // increase z size
    //ds->renderer->getVoxelSize(&size);
    size[2] *= 1.05f;
    //ds->renderer->setVoxelSize(&size);
    cerr << "Z size set to " << size[2] << endl;
    break;
  case 9:                                     // decrease z size
    //ds->renderer->getVoxelSize(&size);
    size[2] *= 0.95f;
    //ds->renderer->setVoxelSize(&size);
    cerr << "Z size set to " << size[2] << endl;
    break;
  case 10:                                     // increase precision of visual
    switch(ds->rendererType)
    {
      case vvRenderer::SOFTPAR:
      case vvRenderer::SOFTPER:
      case vvRenderer::CUDAPAR:
      case vvRenderer::CUDAPER:
      case vvRenderer::RAYREND:
        if (ds->bufferPrecision < 32)
        {
          ds->bufferPrecision = 32;
          cerr << "Buffer precision set to 32 bit floating point" << endl;
        }
        else
        {
          cerr << "Highest precision reached" << endl;
        }
        break;
      case vvRenderer::TEXREND:
        if (ds->useOffscreenBuffer)
        {
          if (ds->bufferPrecision == 8)
          {
            ds->bufferPrecision = 16;
            cerr << "Buffer precision set to 16bit" << endl;
          }
          else if (ds->bufferPrecision == 16)
          {
            ds->bufferPrecision = 32;
            cerr << "Buffer precision set to 32bit" << endl;
          }
          else
          {
            cerr << "Highest precision reached" << endl;
          }
        }
        else
        {
        cerr << "Enable offscreen buffering to change visual precision" << endl;
        }
        break;
      default:
        break;
    }
    ds->renderer->setParameter(vvRenderer::VV_IMG_PRECISION, ds->bufferPrecision);
    break;
  case 11:                                    // decrease precision of visual
    switch(ds->rendererType)
    {
      case vvRenderer::SOFTPAR:
      case vvRenderer::SOFTPER:
      case vvRenderer::CUDAPAR:
      case vvRenderer::CUDAPER:
      case vvRenderer::RAYREND:
        if (ds->bufferPrecision == 32)
        {
          ds->bufferPrecision = 8;
          cerr << "Buffer precision set to 8 bit fixed point" << endl;
        }
        else
        {
          cerr << "Lowest precision reached" << endl;
        }
        break;
      case vvRenderer::TEXREND:
        if (ds->useOffscreenBuffer)
        {
          if (ds->bufferPrecision == 32)
          {
            ds->bufferPrecision = 16;
            cerr << "Buffer precision set to 16bit" << endl;
          }
          else if (ds->bufferPrecision == 16)
          {
            ds->bufferPrecision = 8;
            cerr << "Buffer precision set to 8bit" << endl;
          }
          else
          {
            cerr << "Lowest precision reached" << endl;
          }
        }
        else
        {
          cerr << "Enable offscreen buffering to change visual precision" << endl;
        }
        break;
      default:
        break;
    }
    ds->renderer->setParameter(vvRenderer::VV_IMG_PRECISION, ds->bufferPrecision);
    break;
  case 12:                                     // toggle showing of bricks
    {
      vvTexRend *rend = dynamic_cast<vvTexRend *>(ds->renderer);
      if (rend != NULL)
      {
        ds->showBricks = !ds->showBricks;
        ds->renderer->setParameter(vvRenderState::VV_SHOW_BRICKS, ds->showBricks);
        cerr << (!ds->showBricks?"not ":"") << "showing bricks" << endl;
      }
      else
      {
        cerr << "Option not available for this renderer" << endl;
      }
    }
    break;
  case 13:
    ds->gpuproxygeo = !ds->gpuproxygeo;
    ds->renderer->setParameter(vvRenderer::VV_GPUPROXYGEO, ds->gpuproxygeo);
    cerr << "Switched to proxy geometry generation on the ";
    if (ds->gpuproxygeo)
    {
      cerr << "GPU";
    }
    else
    {
      cerr << "CPU";
    }
    cerr << endl;
    break;
  case 14:
    {
      if(vvTexRend *rend = dynamic_cast<vvTexRend *>(ds->renderer))
      {
        int shader = rend->getCurrentShader()+1;
        rend->setCurrentShader(shader);
        cerr << "shader set to " << rend->getCurrentShader() << endl;
      }
    }
    break;
  case 15:
    ds->earlyRayTermination = !ds->earlyRayTermination;
    ds->renderer->setParameter(vvRenderer::VV_TERMINATEEARLY, ds->earlyRayTermination);
    cerr << "Early ray termination set to " << int(ds->earlyRayTermination) << endl;
    break;
  case 16:
    ds->warpInterpolMode = !ds->warpInterpolMode;
    ds->renderer->setParameter(vvRenderer::VV_WARPINT, ds->warpInterpolMode);
    cerr << "Warp interpolation set to " << int(ds->warpInterpolMode) << endl;
    break;
  case 17:
    {
      int tmp = ds->ibrMode;
      ++tmp;
      ds->ibrMode = static_cast<vvRenderState::IbrMode>(tmp);
      if (ds->ibrMode == vvRenderState::VV_NONE)
      {
        ds->ibrMode = static_cast<vvRenderState::IbrMode>(0);
      }
      ds->renderer->setParameter(vvRenderer::VV_IBR_MODE, ds->ibrMode);
      cerr << "Set IBR mode to " << int(ds->ibrMode) << endl;
    }
    break;
  case 18:
    ++ds->codec;
    if(ds->codec > 10)
      ds->codec = 0;
    cerr << "Codec set to " << ds->codec << endl;
    ds->renderer->setParameter(vvRenderer::VV_CODEC, ds->codec);
    break;
  default: break;
  }
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for transfer function menu.
  @param item selected menu item index
*/
void vvView::transferMenuCallback(int item)
{
  static float peakPosX = 0.0f;
  float gamma, chan4;

  vvDebugMsg::msg(1, "vvView::transferMenuCallback()");

  switch(item)
  {
  case 0:                                     // bright colors
    ds->vd->tf.setDefaultColors(0, 0.0, 1.0);
    break;
  case 1:                                     // HSV colors
    ds->vd->tf.setDefaultColors(1, 0.0, 1.0);
    break;
  case 2:                                     // grayscale
    ds->vd->tf.setDefaultColors(2, 0.0, 1.0);
    break;
  case 3:                                     // white
    ds->vd->tf.setDefaultColors(3, 0.0, 1.0);
    break;
  case 4:                                     // white
    ds->vd->tf.setDefaultColors(4, 0.0, 1.0);
    break;
  case 5:                                     // white
    ds->vd->tf.setDefaultColors(5, 0.0, 1.0);
    break;
  case 6:                                     // white
    ds->vd->tf.setDefaultColors(6, 0.0, 1.0);
    break;
  case 7:                                     // cool to warm
    ds->vd->tf.setDefaultColors(7, 0.0, 1.0);
    break;
  case 8:                                     // alpha: ascending
    ds->vd->tf.setDefaultAlpha(0, 0.0, 1.0);
    break;
  case 9:                                     // alpha: descending
    ds->vd->tf.setDefaultAlpha(1, 0.0, 1.0);
    break;
  case 10:                                     // alpha: opaque
    ds->vd->tf.setDefaultAlpha(2, 0.0, 1.0);
    break;
  case 11:                                    // alpha: display peak
  case 12:                                    // alpha: shift left peak
  case 13:                                    // alpha: shift right peak
    if(item == 12)
      peakPosX -= .05;
    else if(item == 13)
      peakPosX += .05;
    if (peakPosX < 0.0f) peakPosX += 1.0f;
    if (peakPosX > 1.0f) peakPosX -= 1.0f;
    cerr << "Peak position: " << peakPosX << endl;

    ds->vd->tf.deleteWidgets(vvTFWidget::TF_PYRAMID);
    ds->vd->tf.deleteWidgets(vvTFWidget::TF_BELL);
    ds->vd->tf.deleteWidgets(vvTFWidget::TF_CUSTOM);
    ds->vd->tf.deleteWidgets(vvTFWidget::TF_SKIP);
    ds->vd->tf._widgets.append(
          new vvTFPyramid(vvColor(1.f, 1.f, 1.f), false, 1.f, peakPosX, .2f, 0.f),
          vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    break;
  case 14:                                    // gamma red
  case 15:
    gamma = ds->renderer->getGamma(vvRenderer::VV_RED);
    gamma *= (item==14) ? 0.95f : 1.05f;
    ds->renderer->setGamma(vvRenderer::VV_RED, gamma);
    cerr << "gamma red = " << gamma << endl;
    break;
  case 16:                                    // gamma green
  case 17:
    gamma = ds->renderer->getGamma(vvRenderer::VV_GREEN);
    gamma *= (item==16) ? 0.95f : 1.05f;
    ds->renderer->setGamma(vvRenderer::VV_GREEN, gamma);
    cerr << "gamma green = " << gamma << endl;
    break;
  case 18:                                    // gamma blue
  case 19:
    gamma = ds->renderer->getGamma(vvRenderer::VV_BLUE);
    gamma *= (item==18) ? 0.95f : 1.05f;
    ds->renderer->setGamma(vvRenderer::VV_BLUE, gamma);
    cerr << "gamma blue = " << gamma << endl;
    break;
  case 20:                                    // channel 4 red
  case 21:
    chan4 = ds->renderer->getOpacityWeight(vvRenderer::VV_RED);
    chan4 *= (item==20) ? 0.95f : 1.05f;
    ds->renderer->setOpacityWeight(vvRenderer::VV_RED, chan4);
    cerr << "channel 4 red = " << chan4 << endl;
    break;
  case 22:                                    // channel 4 green
  case 23:
    chan4 = ds->renderer->getOpacityWeight(vvRenderer::VV_GREEN);
    chan4 *= (item==22) ? 0.95f : 1.05f;
    ds->renderer->setOpacityWeight(vvRenderer::VV_GREEN, chan4);
    cerr << "channel 4 green = " << chan4 << endl;
    break;
  case 24:                                    // channel 4 blue
  case 25:
    chan4 = ds->renderer->getOpacityWeight(vvRenderer::VV_BLUE);
    chan4 *= (item==24) ? 0.95f : 1.05f;
    ds->renderer->setOpacityWeight(vvRenderer::VV_BLUE, chan4);
    cerr << "channel 4 blue = " << chan4 << endl;
    break;
  default: break;
  }

  ds->renderer->updateTransferFunction();
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Display a specific animation frame.
    @param f Index of frame to display. First index = 0
*/
void vvView::setAnimationFrame(int f)
{
  vvDebugMsg::msg(3, "vvView::setAnimationFrame()");

  if (f >= vd->frames)
    frame = 0;
  else if (f<0)
    frame = vd->frames - 1;

  renderer->setCurrentFrame(frame);

  cerr << "Time step: " << (frame+1) << endl;
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for animation menu.
  @param item selected menu item index
*/
void vvView::animMenuCallback(int item)
{
   vvDebugMsg::msg(1, "vvView::animMenuCallback()");

  switch(item)
  {
  default:
  case 0:                                     // next frame
    ++ds->frame;
    ds->setAnimationFrame(ds->frame);
    break;
  case 1:                                     // previous frame
    --ds->frame;
    ds->setAnimationFrame(ds->frame);
    break;
  case 2:                                     // start/stop animation
    if (ds->animating) ds->animating = false;
    else
    {
      ds->animating = true;
                                            // trigger timer function
      glutTimerFunc(int(ds->ds->animSpeed * 1000.0f), timerCallback, ANIMATION_TIMER);
    }
    break;
  case 3:                                     // rewind
    ds->frame = 0;
    ds->setAnimationFrame(ds->frame);
    break;
  case 4:                                     // speed up
    ds->animSpeed *= 0.9f;
    cerr << "speed=" << ds->animSpeed << endl;
    break;
  case 5:                                     // speed down
    ds->animSpeed *= 1.1f;
    cerr << "speed=" << ds->animSpeed << endl;
    break;
  case 6:                                     // reset speed
    ds->animSpeed = ds->vd->dt;
    break;
  }
}


//----------------------------------------------------------------------------
/** Callback for roi menu.
  @param item selected menu item index
*/
void vvView::roiMenuCallback(const int item)
{
  vvDebugMsg::msg(1, "vvView::roiMenuCallback()");

  vvVector3 probeSize;

  switch (item)
  {
  case 0:                                    // toggle roi mode
    if (!ds->roiEnabled && !ds->sphericalROI)
    {
      // Cuboid roi.
      ds->roiEnabled = true;
      ds->sphericalROI = false;
    }
    else if (ds->roiEnabled && !ds->sphericalROI)
    {
      // Spherical roi.
      ds->roiEnabled = true;
      ds->sphericalROI = true;
    }
    else
    {
      // No roi.
      ds->roiEnabled = false;
      ds->sphericalROI = false;
    }
    ds->renderer->setROIEnable(ds->roiEnabled);
    ds->renderer->setSphericalROI(ds->sphericalROI);
    printROIMessage();
    break;
  case 98:
    if (ds->roiEnabled)
    {
      ds->renderer->getProbeSize(&probeSize);
      probeSize.sub(0.1f);
      const float size = probeSize[0];
      if (size <= 0.0f)
      {
        probeSize = vvVector3(0.00001f);
      }
      ds->renderer->setProbeSize(&probeSize);
    }
    else
    {
      cerr << "Function only available in ROI mode" << endl;
    }
    break;
  case 99:
    if (ds->roiEnabled)
    {
      ds->renderer->getProbeSize(&probeSize);
      probeSize.add(0.1f);
      const float size = probeSize[0];
      if (size > 1.0f)
      {
        probeSize = vvVector3(1.0f);
      }
      ds->renderer->setProbeSize(&probeSize);
    }
    else
    {
      cerr << "Function only available in ROI mode" << endl;
    }
    break;
  default:
    break;
  }
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for clip menu.
  @param item selected menu item index
*/
void vvView::clipMenuCallback(const int item)
{
  vvDebugMsg::msg(1, "vvView::clipMenuCallback()");

  switch (item)
  {
  case 0:
    ds->clipPlane = !ds->clipPlane;
    ds->renderer->setParameter(vvRenderState::VV_CLIP_MODE, ds->clipPlane);
    cerr << "Clipping " << ds->onOff[ds->boundariesMode] << endl;
    break;
  case 1:
    ds->clipEditMode = !ds->clipEditMode;
    if (ds->clipEditMode)
    {
      ds->clipPlane = true;
      ds->renderer->setParameter(vvRenderState::VV_CLIP_MODE, ds->clipPlane);
      cerr << "Clip edit mode activated" << endl;
      cerr << "x|y|z keys:\t\trotation along (x|y|z) axis" << endl;
      cerr << "Arrow down/left:\tmove in negative normal direction" << endl;
      cerr << "Arrow up/right:\tmove in positive normal direction" << endl;
    }
    else
    {
      cerr << "Clip edit mode deactivated" << endl;
    }
    break;
  }
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/** Callback for viewing window menu.
  @param item selected menu item index
*/
void vvView::viewMenuCallback(int item)
{
  vvDebugMsg::msg(1, "vvView::viewMenuCallback()");

  switch(item)
  {
  default:
  case 0:                                     // bounding box
    ds->boundariesMode = !ds->boundariesMode;
    ds->renderer->setParameter(vvRenderState::VV_BOUNDARIES, ds->boundariesMode);
    cerr << "Bounding box " << ds->onOff[ds->boundariesMode] << endl;
    break;
  case 1:                                     // axis orientation
    ds->orientationMode = !ds->orientationMode;
    ds->renderer->setParameter(vvRenderState::VV_ORIENTATION,
                                 !(ds->renderer->getParameter(vvRenderState::VV_ORIENTATION) != 0.0f));
    cerr << "Coordinate axes display " << ds->onOff[ds->orientationMode] << endl;
    break;
  case 2:                                     // frame rate
    ds->fpsMode = !ds->fpsMode;
    ds->renderer->setParameter(vvRenderState::VV_FPS_DISPLAY, !ds->renderer->getParameter(vvRenderState::VV_FPS_DISPLAY));
    cerr << "Frame rate display " << ds->onOff[ds->fpsMode] << endl;
    break;
  case 3:                                     // transfer function
    ds->paletteMode = !ds->paletteMode;
    ds->renderer->setParameter(vvRenderState::VV_PALETTE, !(ds->renderer->getParameter(vvRenderState::VV_PALETTE) != 0.0f));
    cerr << "Palette display " << ds->onOff[ds->paletteMode] << endl;
    break;
  case 4:                                     // stereo mode
    ++ds->stereoMode;
    if (ds->stereoMode > 3) ds->stereoMode = 0;
    if (ds->stereoMode==1 && !ds->activeStereoCapable) ds->stereoMode = 2;
    cerr << "Stereo mode: " << ds->stereoMode << endl;
    break;
  case 5:                                     // full screen
    if (ds->fullscreenMode)
    {
      glutPositionWindow(ds->lastPosX, ds->lastPosY);
      glutReshapeWindow(ds->lastWidth, ds->lastHeight);
      ds->fullscreenMode = false;
    }
    else
    {
      ds->lastWidth  = glutGet(GLUT_WINDOW_WIDTH);
      ds->lastHeight = glutGet(GLUT_WINDOW_HEIGHT);
      ds->lastPosX   = glutGet(GLUT_WINDOW_X);
      ds->lastPosY   = glutGet(GLUT_WINDOW_Y);
      glutFullScreen();
      ds->fullscreenMode = true;
    }
    cerr << "Fullscreen mode " << ds->onOff[ds->fullscreenMode] << endl;
    break;
  case 6:                                     // window color
    if (ds->bgColor[0] == 0.0f)
                                        // white
      ds->bgColor[0] = ds->bgColor[1] = ds->bgColor[2] = 1.0f;
                                        // black
    else
      ds->bgColor[0] = ds->bgColor[1] = ds->bgColor[2] = 0.0f;
    // background color is only a property of the display window
    //ds->renderer->setParameterV3(VV_BG_COLOR, ds->bgColor);
      // Use opposite color for object boundaries:
      //ds->renderer->setBoundariesColor(1.0f-ds->bgColor[0], 1.0f-ds->bgColor[1], 1.0f-ds->bgColor[2]);
break;
  case 7:                                     // auto-rotation mode
    ds->rotationMode = !ds->rotationMode;
    if (!ds->rotationMode)
    {
      ds->rotating = false;
      if (ds->refinement) ds->hqMode = true;
    }
    cerr << "Auto rotation " << ds->onOff[ds->rotationMode] << endl;
    break;
  case 8:                                     // save camera
    if (ds->ov->saveMV("virvo-camera.txt"))
      cerr << "Camera saved to file 'virvo-camera.txt'." << endl;
    else
      cerr << "Couldn't save camera to file 'virvo-camera.txt'." << endl;
    break;
  case 9:                                     // load camera
    if (ds->ov->loadMV("virvo-camera.txt"))
      cerr << "Camera loaded from file 'virvo-camera.txt'." << endl;
    else
      cerr << "Couldn't load camera from file 'virvo-camera.txt'." << endl;
    break;
  case 10:                                    // toggle icon
    ds->iconMode = !ds->iconMode;
    cerr << "Icon " << ds->onOff[ds->iconMode] << endl;
    break;
  }
  glutPostRedisplay();
}

/** Callback function for gl errors.
  If the extension GL_ARB_debug_output is available, this callback
  function will be called automatically if an opengl error is
  generated
*/
void vvView::debugCallbackARB(GLenum /*source*/, GLenum /*type*/, GLuint /*id*/, GLenum /*severity*/,
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
/** Do a performance test.
  Default behavior:
  The test resets the view position (but not the projection mode)
  and measures the time needed for a
  360 degrees rotation of the volume about its vertical axis.
  The image is drawn every 2 degrees.
  <br>
*/
double vvView::performanceTest()
{
  vvDebugMsg::msg(1, "vvView::performanceTest()");

  double total = 0.;

  vvTestSuite* testSuite = new vvTestSuite(ds->testSuiteFileName);
  if (testSuite->isInitialized())
  {
    std::vector<vvPerformanceTest*> tests = testSuite->getTests();
    std::vector<vvPerformanceTest*>::const_iterator it;
    float step = 2.0f * VV_PI / 180.0f;

    if (tests.size() < 1)
    {
      cerr << "No tests in test suite" << endl;
    }

    for(it = tests.begin(); it != tests.end(); ++it)
    {
      vvPerformanceTest* test = *it;

      // TODO: make dataset configurable.
      test->setDatasetName(ds->filename);

      std::vector<float> diffTimes;
      std::vector<vvMatrix> modelViewMatrices;

      vvStopwatch* totalTime = new vvStopwatch();
      totalTime->start();

      int framesRendered = 0;
      ds->setRenderer(test->getGeomType(), test->getVoxelType(), 0,
                      (int) test->getBrickDims()[0],
                      (int) test->getBrickDims()[1],
                      (int) test->getBrickDims()[2]);
      ds->hqMode = false;
      ds->draftQuality = test->getQuality();
      ds->ov->reset();
      ds->ov->resetMV();
      ds->ov->mv.scale(ds->mvScale);
      ds->perspectiveMode = (test->getProjectionType() == vvObjView::PERSPECTIVE);
      if (ds->perspectiveMode)
      {
        ds->ov->setProjection(vvObjView::PERSPECTIVE, 45.0f, 0.01f, 100.0f);
      }
      else
      {
        ds->ov->setProjection(vvObjView::ORTHO, 2.0f, -100.0, 100.0);
      }
      // Do this once to propagate the changes... .
      ds->displayCallback();
      ds->renderer->profileStart();

      if (test->getVerbose())
      {
        printProfilingInfo(test->getId(), tests.size());
      }

      for (int ite = 0; ite < test->getIterations(); ++ite)
      {
        for (int i = 0; i < test->getFrames(); ++i)
        {
          vvVector3 dir;
          switch (test->getTestAnimation())
          {
          case vvPerformanceTest::VV_ROT_X:
            ds->ov->mv.rotate(step, 1.0f, 0.0f, 0.0f);
            break;
          case vvPerformanceTest::VV_ROT_Y:
            ds->ov->mv.rotate(step, 0.0f, 1.0f, 0.0f);
            break;
          case vvPerformanceTest::VV_ROT_Z:
            ds->ov->mv.rotate(step, 0.0f, 0.0f, 1.0f);
            break;
          case vvPerformanceTest::VV_ROT_RAND:
            dir.random(0.0f, 1.0f);
            ds->ov->mv.rotate(step, &dir);
            break;
          default:
            break;
          }
          ds->displayCallback();
          diffTimes.push_back(totalTime->getDiff());
          modelViewMatrices.push_back(ds->ov->mv);

          ++framesRendered;
        }
      }
      total = totalTime->getTime();

      test->getTestResult()->setDiffTimes(diffTimes);
      test->getTestResult()->setModelViewMatrices(modelViewMatrices);
      test->writeResultFiles();

      if (test->getVerbose())
      {
        printProfilingResult(totalTime, framesRendered);
      }
      delete totalTime;
    }
  }
  else
  {
    vvStopwatch* totalTime;
    float step = 2.0f * VV_PI / 180.0f;
    int   angle;
    int   framesRendered = 0;

    // Prepare test:
    totalTime = new vvStopwatch();

    ds->hqMode = false;
    ds->ov->reset();
    ds->ov->mv.scale(ds->mvScale);
    ds->displayCallback();

    printProfilingInfo();

    // Perform test:
    totalTime->start();
    ds->renderer->profileStart();

    for (angle=0; angle<180; angle+=2)
    {
      ds->ov->mv.rotate(step, 0.0f, 1.0f, 0.0f);  // rotate model view matrix
      ds->displayCallback();
      ++framesRendered;
    }

    ds->ov->reset();
    ds->ov->mv.scale(ds->mvScale);
    for (angle=0; angle<180; angle+=2)
    {
      ds->ov->mv.rotate(step, 0.0f, 0.0f, 1.0f);  // rotate model view matrix
      ds->displayCallback();
      ++framesRendered;
    }

    ds->ov->reset();
    ds->ov->mv.scale(ds->mvScale);
    for (angle=0; angle<180; angle+=2)
    {
      ds->ov->mv.rotate(step, 1.0f, 0.0f, 0.0f);  // rotate model view matrix
      ds->displayCallback();
      ++framesRendered;
    }
    total = totalTime->getTime();

    printProfilingResult(totalTime, framesRendered);

    delete totalTime;
  }
  delete testSuite;

  return total;
}


//----------------------------------------------------------------------------
/** Print information about the most recent test run.
    @param testNr Number of the current test (1-based).
    @param testCnt Number of tests in total.
  */
void vvView::printProfilingInfo(const int testNr, const int testCnt)
{
  GLint viewport[4];                             // OpenGL viewport information (position and size)
  char  projectMode[2][16] = {"parallel","perspective"};
  char  interpolMode[2][32] = {"nearest neighbor","linear"};
  char  onOffMode[2][8] = {"off","on"};
  const int HOST_NAME_LEN = 80;
  char  localHost[HOST_NAME_LEN];
  glGetIntegerv(GL_VIEWPORT, viewport);
#ifdef _WIN32
  strcpy(localHost, "n/a");
#else
  if (gethostname(localHost, HOST_NAME_LEN-1)) strcpy(localHost, "n/a");
#endif

  // Print profiling info:
  cerr.setf(ios::fixed, ios::floatfield);
  cerr.precision(3);
  cerr << "*******************************************************************************" << endl;
  cerr << "Test (" << testNr << "/" << testCnt << ")" << endl;
  cerr << "Local host........................................" << localHost << endl;
  cerr << "Renderer.........................................." << ds->rendererName[ds->rendererType] << endl;
  cerr << "Renderer geometry................................." << ds->currentGeom << endl;
  cerr << "Voxel type........................................" << ds->currentVoxels << endl;
  cerr << "Volume file name.................................." << ds->vd->getFilename() << endl;
  cerr << "Volume size [voxels].............................." << ds->vd->vox[0] << " x " << ds->vd->vox[1] << " x " << ds->vd->vox[2] << endl;
  cerr << "Output image size [pixels]........................" << viewport[2] << " x " << viewport[3] << endl;
  cerr << "Image quality....................................." << ds->renderer->getParameter(vvRenderState::VV_QUALITY) << endl;
  cerr << "Projection........................................" << projectMode[ds->perspectiveMode] << endl;
  cerr << "Interpolation mode................................" << interpolMode[ds->interpolMode] << endl;
  cerr << "Empty space leaping for bricks...................." << onOffMode[ds->emptySpaceLeapingMode] << endl;
  cerr << "Early ray termination............................." << onOffMode[ds->earlyRayTermination] << endl;
  cerr << "Pre-integration..................................." << onOffMode[ds->preintMode] << endl;
  cerr << "Render to offscreen buffer........................" << onOffMode[ds->useOffscreenBuffer] << endl;
  cerr << "Precision of offscreen buffer....................." << ds->bufferPrecision << "bit" << endl;
  cerr << "Generate proxy geometry on GPU...................." << onOffMode[ds->gpuproxygeo] << endl;
  cerr << "Opacity correction................................" << onOffMode[ds->opCorrMode] << endl;
  cerr << "Gamma correction.................................." << onOffMode[ds->gammaMode] << endl;
}


//----------------------------------------------------------------------------
/** Conclude profiling info with the final result.
    @param totalTime A stop watch to read the profiling time from.
    @param framesRendered The number of frames rendered for the test.
  */
void vvView::printProfilingResult(vvStopwatch* totalTime, const int framesRendered)
{
  cerr << "Total profiling time [sec]........................" << totalTime->getTime() << endl;
  cerr << "Frames rendered..................................." << framesRendered << endl;
  cerr << "Average time per frame [sec]......................" << (float(totalTime->getTime()/framesRendered)) << endl;
  cerr << "*******************************************************************************" << endl;
}


//----------------------------------------------------------------------------
/** Print a short info how to interact with the probe in roi mode
  */
void vvView::printROIMessage()
{
  if (ds->roiEnabled)
  {
    cerr << "Region of interest mode enabled" << endl;
    if (ds->sphericalROI)
    {
      cerr << "Region of interest mode: spherical" << endl;
    }
    else
    {
      cerr << "Region of interest mode: cuboid" << endl;
    }
    cerr << "Arrow left:         -x, arrow right:      +x" << endl;
    cerr << "Arrow down:         -y, arrow up:         +y" << endl;
    cerr << "Arrow down + shift: -z, arrow up + shift: +z" << endl;
    cerr << endl;
    cerr << "Use '[‘ and ‘]‘ to resize probe" << endl;
  }
  else
  {
    cerr << "Region of interest mode disabled" << endl;
  }
}


//----------------------------------------------------------------------------
/// Create the pop-up menus.
void vvView::createMenus()
{
  int rendererMenu, voxelMenu, optionsMenu, transferMenu, animMenu, roiMenu, clipMenu, viewMenu;

  vvDebugMsg::msg(1, "vvView::createMenus()");

  // Rendering geometry menu:
  rendererMenu = glutCreateMenu(rendererMenuCallback);
  glutAddMenuEntry("Auto select [0]", 0);
  if (vvTexRend::isSupported(vvTexRend::VV_SLICES))    glutAddMenuEntry("2D textures - slices [1]", 1);
  if (vvTexRend::isSupported(vvTexRend::VV_CUBIC2D))   glutAddMenuEntry("2D textures - cubic [2]", 2);
  if (vvTexRend::isSupported(vvTexRend::VV_VIEWPORT))  glutAddMenuEntry("3D textures - viewport aligned [3]", 3);
  if (vvTexRend::isSupported(vvTexRend::VV_SPHERICAL)) glutAddMenuEntry("3D textures - spherical [4]", 4);
  if (vvTexRend::isSupported(vvTexRend::VV_BRICKS))    glutAddMenuEntry("3D textures - bricked [5]", 5);
  glutAddMenuEntry("CPU Shear-warp [6]", 6);
#if defined(HAVE_CUDA)
  glutAddMenuEntry("GPU Shear-warp [7]", 7);
  glutAddMenuEntry("GPU Ray casting [8]", 8);
#endif
  glutAddMenuEntry("Decrease quality [-]", 98);
  glutAddMenuEntry("Increase quality [+]", 99);

  // Voxel menu:
  voxelMenu = glutCreateMenu(voxelMenuCallback);
  glutAddMenuEntry("Auto select", 0);
  glutAddMenuEntry("RGBA", 1);
  if (vvTexRend::isSupported(vvTexRend::VV_SGI_LUT)) glutAddMenuEntry("SGI look-up table", 2);
  if (vvTexRend::isSupported(vvTexRend::VV_PAL_TEX)) glutAddMenuEntry("Paletted textures", 3);
  if (vvTexRend::isSupported(vvTexRend::VV_TEX_SHD)) glutAddMenuEntry("Texture shader", 4);
  if (vvTexRend::isSupported(vvTexRend::VV_PIX_SHD)) glutAddMenuEntry("Fragment shader", 5);
  if (vvTexRend::isSupported(vvTexRend::VV_FRG_PRG)) glutAddMenuEntry("ARB fragment program", 6);

  // Renderer options menu:
  optionsMenu = glutCreateMenu(optionsMenuCallback);
  glutAddMenuEntry("Toggle slice interpolation [i]", 0);
  glutAddMenuEntry("Toggle warp interpolation [W]", 16);
  if (vvTexRend::isSupported(vvTexRend::VV_FRG_PRG)
      && vvTexRend::isSupported(vvTexRend::VV_VIEWPORT)
      && vvGLTools::isGLextensionSupported("GL_ARB_multitexture"))
  {
    glutAddMenuEntry("Toggle pre-integration [P]", 1);
  }
  if (vvGLTools::isGLextensionSupported("GL_EXT_blend_minmax"))
  {
    glutAddMenuEntry("Toggle maximum intensity projection (MIP) [x]", 2);
  }
  glutAddMenuEntry("Toggle opacity correction", 3);
  glutAddMenuEntry("Toggle gamma correction", 4);
  glutAddMenuEntry("Toggle empty space leaping", 5);
  glutAddMenuEntry("Toggle early ray termination", 15);
  glutAddMenuEntry("Toggle offscreen buffering", 6);
  glutAddMenuEntry("Toggle head light", 7);
  glutAddMenuEntry("Increase z size [H]", 8);
  glutAddMenuEntry("Decrease z size [h]", 9);
  glutAddMenuEntry("Increase buffer precision", 10);
  glutAddMenuEntry("Decrease buffer precision", 11);
  glutAddMenuEntry("Show/hide bricks [B]", 12);
  if (vvTexRend::isSupported(vvTexRend::VV_BRICKS))
    glutAddMenuEntry("Bricks - generate proxy geometry on GPU [g]", 13);
  if (vvTexRend::isSupported(vvTexRend::VV_PIX_SHD))
    glutAddMenuEntry("Cycle shader [A]", 14);
  glutAddMenuEntry("Inc ibr mode [#]", 17);
  glutAddMenuEntry("Cycle codec [C]", 18);

  // Transfer function menu:
  transferMenu = glutCreateMenu(transferMenuCallback);
  glutAddMenuEntry("Colors: bright", 0);
  glutAddMenuEntry("Colors: HSV", 1);
  glutAddMenuEntry("Colors: grayscale", 2);
  glutAddMenuEntry("Colors: white", 3);
  glutAddMenuEntry("Colors: red", 4);
  glutAddMenuEntry("Colors: green", 5);
  glutAddMenuEntry("Colors: blue", 6);
  glutAddMenuEntry("Colors: cool to warm", 7);
  glutAddMenuEntry("Alpha: ascending", 8);
  glutAddMenuEntry("Alpha: descending", 9);
  glutAddMenuEntry("Alpha: opaque", 10);
  glutAddMenuEntry("Alpha: single peak", 11);
  glutAddMenuEntry("Alpha: shift peak left [<]", 12);
  glutAddMenuEntry("Alpha: shift peak right [>]", 13);
  glutAddMenuEntry("Gamma: less red [1]", 14);
  glutAddMenuEntry("Gamma: more red [2]", 15);
  glutAddMenuEntry("Gamma: less green [3]", 16);
  glutAddMenuEntry("Gamma: more green [4]", 17);
  glutAddMenuEntry("Gamma: less blue [5]", 18);
  glutAddMenuEntry("Gamma: more blue [6]", 19);
  glutAddMenuEntry("Channel 4: less red", 20);
  glutAddMenuEntry("Channel 4: more red", 21);
  glutAddMenuEntry("Channel 4: less green", 22);
  glutAddMenuEntry("Channel 4: more green", 23);
  glutAddMenuEntry("Channel 4: less blue", 24);
  glutAddMenuEntry("Channel 4: more blue", 25);

  // Animation menu:
  animMenu = glutCreateMenu(animMenuCallback);
  glutAddMenuEntry("Next frame [n]", 0);
  glutAddMenuEntry("Previous frame [N]", 1);
  glutAddMenuEntry("Start/stop animation [a]", 2);
  glutAddMenuEntry("Rewind", 3);
  glutAddMenuEntry("Animation speed up [s]", 4);
  glutAddMenuEntry("Animation speed down [S]", 5);
  glutAddMenuEntry("Reset speed", 6);

  // Region of interest menu:
  roiMenu = glutCreateMenu(roiMenuCallback);
  glutAddMenuEntry("Toggle region of interest mode [R]", 0);
  glutAddMenuEntry("size-- [[]", 98);
  glutAddMenuEntry("size++ []]", 99);

  // Clip menu:
  clipMenu = glutCreateMenu(clipMenuCallback);
  glutAddMenuEntry("Toggle clip mode [I]", 0);
  glutAddMenuEntry("Toggle clipping edit mode [E]", 1);

  // Viewing Window Menu:
  viewMenu = glutCreateMenu(viewMenuCallback);
  glutAddMenuEntry("Toggle bounding box [b]", 0);
  glutAddMenuEntry("Toggle axis orientation [o]", 1);
  glutAddMenuEntry("Toggle frame rate display [f]", 2);
  glutAddMenuEntry("Toggle transfer function display", 3);
  glutAddMenuEntry("Stereo mode", 4);
  glutAddMenuEntry("Toggle full screen zoom [z]", 5);
  glutAddMenuEntry("Toggle window color [w]", 6);
  glutAddMenuEntry("Toggle auto rotation", 7);
  glutAddMenuEntry("Save camera position [u]", 8);
  glutAddMenuEntry("Load camera position [v]", 9);
  glutAddMenuEntry("Toggle icon display [c]", 10);

  // Main menu:
  mainMenu = glutCreateMenu(mainMenuCallback);
  glutAddSubMenu("Rendering geometry", rendererMenu);
  glutAddSubMenu("Voxel representation", voxelMenu);
  glutAddSubMenu("Renderer options", optionsMenu);
  glutAddSubMenu("Transfer function", transferMenu);
  glutAddSubMenu("Animation", animMenu);
  glutAddSubMenu("Region of interest", roiMenu);
  glutAddSubMenu("Clipping", clipMenu);
  glutAddSubMenu("Viewing window", viewMenu);
  glutAddMenuEntry("Toggle perspective mode [p]", 0);
  glutAddMenuEntry("Toggle auto refinement [e]", 4);
  glutAddMenuEntry("Toggle rendering time display [d]", 5);
  glutAddMenuEntry("Reset object orientation", 7);
  glutAddMenuEntry("Toggle popup menu/zoom mode [m]", 8);
  glutAddMenuEntry("Save volume to file", 9);
  glutAddMenuEntry("Performance test [t]", 11);
  glutAddMenuEntry("Change debug level [D]", 13);
  glutAddMenuEntry("Quit [q]", 12);

  glutSetMenu(mainMenu);
  glutAttachMenu(GLUT_RIGHT_BUTTON);
  menuEnabled = true;
}


//----------------------------------------------------------------------------
/// Initialize the GLUT window and the OpenGL graphics context.
void vvView::initGraphics(int argc, char *argv[])
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

  if (tryQuadBuffer)
  {
    // create stereo context
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_STEREO | GLUT_DEPTH);
    if (!glutGet(GLUT_DISPLAY_MODE_POSSIBLE))
    {
      cerr << "Stereo mode not supported by display driver." << endl;
      tryQuadBuffer = false;
    }
  }
  if (!tryQuadBuffer)
  {
    // create double buffering context
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  }
  else
  {
    cerr << "Stereo mode found." << endl;
    activeStereoCapable = true;
  }
  if (!glutGet(GLUT_DISPLAY_MODE_POSSIBLE))
  {
    cerr << "Error: Glut needs a double buffering OpenGL context with alpha channel." << endl;
    exit(-1);
  }

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
  glutMouseFunc(buttonCallback);
  glutMotionFunc(motionCallback);
  glutKeyboardFunc(keyboardCallback);
  glutSpecialFunc(specialCallback);

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


void vvView::initARBDebugOutput()
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
  if (glDebugMessageCallbackARB != NULL)
  {
    cerr << "Init callback function for GL_ARB_debug_output extension" << endl;
    glDebugMessageCallbackARB(debugCallbackARB, NULL);
  }
  else
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
/** Set projection mode to perspective or parallel.
  @param newmode true = perspective projection, false = parallel projection
*/
void vvView::setProjectionMode(bool newmode)
{
  vvDebugMsg::msg(1, "vvView::setProjectionMode()");

  perspectiveMode = newmode;

  if (perspectiveMode)
  {
    ov->setProjection(vvObjView::PERSPECTIVE, 45.0f, 0.01f, 100.0f);
  }
  else
  {
    ov->setProjection(vvObjView::ORTHO, 2.0f, -100.0, 100.0);
  }
}

void vvView::setupClipBuffer()
{
  delete clipBuffer;
  clipBuffer = new vvOffscreenBuffer(1.0f, VV_FLOAT);
  clipBuffer->initForRender();
}

void vvView::renderClipObject()
{
  if (clipBuffer == NULL)
  {
    setupClipBuffer();
  }

  clipBuffer->bindFramebuffer();
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  renderCube();

  delete[] framebufferDump;
  const vvGLTools::Viewport viewport = vvGLTools::getViewport();
  framebufferDump = new GLfloat[viewport[2] * viewport[3] * 4];
  glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
  glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGBA, GL_FLOAT, framebufferDump);
  clipBuffer->unbindFramebuffer();
}




void vvView::renderCube() const
{


}

void vvView::renderQuad() const
{
  glEnable(GL_TEXTURE_2D);
  clipBuffer->bindTexture();
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  const vvGLTools::Viewport viewport = vvGLTools::getViewport();
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, viewport[2], viewport[3],
               0, GL_RGBA, GL_FLOAT, framebufferDump);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  vvGLTools::drawViewAlignedQuad();

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glEnable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);
}


void vvView::renderMotion() const
{
  FILE* fp;
  fp = fopen("motion.txt", "rb");

  while (ds->ov->loadMV(fp))
  {
    glDrawBuffer(GL_BACK);
    glClearColor(ds->bgColor[0], ds->bgColor[1], ds->bgColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ds->ov->updateModelviewMatrix(vvObjView::LEFT_EYE);

    ds->renderer->renderVolumeGL();
    glutSwapBuffers();
  }

  fclose(fp);
}


void vvView::editClipPlane(const int command, const float val)
{
  vvVector3 clipNormal = ds->renderer->getParameterV3(vvRenderState::VV_CLIP_NORMAL);
  switch (command)
  {
  case PLANE_X:
    {
      vvMatrix m;
      m.identity();
      const vvVector3 axis(1, 0, 0);
      m.rotate(val, &axis);
      clipNormal.multiply(&m);
    }
    break;
  case PLANE_Y:
    {
      vvMatrix m;
      m.identity();
      const vvVector3 axis(0, 1, 0);
      m.rotate(val, &axis);
      clipNormal.multiply(&m);
    }
    break;
  case PLANE_Z:
    {
      vvMatrix m;
      m.identity();
      const vvVector3 axis(0, 0, 1);
      m.rotate(val, &axis);
      clipNormal.multiply(&m);
    }
    break;
  default:
    cerr << "Unknown command" << endl;
    break;
  }
  ds->renderer->setParameterV3(vvRenderState::VV_CLIP_NORMAL, clipNormal);
  glutPostRedisplay();
}


//----------------------------------------------------------------------------
/// Display command usage help on the command line.
void vvView::displayHelpInfo()
{
  vvDebugMsg::msg(1, "vvView::displayHelpInfo()");

  cerr << "Syntax:" << endl;
  cerr << endl;
  cerr << "  vview [<volume_file.xxx>] [options]" << endl;
  cerr << endl;
  cerr << "<volume_file.xxx>" << endl;
  cerr << "Volume file to display. Please see VConv for a list of available" << endl;
  cerr << "file types." << endl;
  cerr << endl;
  cerr << "Available options:" << endl;
  cerr << endl;
  cerr << "-servermode (-s)" << endl;
  cerr << " Renderer is a server in mode <mode> and accepts assignments from a client" << endl;
  cerr << endl;
  cerr << "-clientmode <mode> (-c)" << endl;
  cerr << " Renderer is a client in mode <mode> and connects to server(s) given with -server" << endl;
  cerr << " Modes:" << endl;
  cerr << " cluster = cluster rendering (default)" << endl;
  cerr << " ibr     = image based rendering" << endl;
  cerr << " image   = remote rendering" << endl;
  cerr << endl;
  cerr << "-port" << endl;
  cerr << " Renderer is a render slave. Don't use the default port (31050), but the specified one" << endl;
  cerr << endl;
  cerr << "-renderer <num> (-r)" << endl;
  cerr << " Select the default renderer:" << endl;
  cerr << " 0  = Autoselect" << endl;
  cerr << " 1  = 2D Textures - Slices" << endl;
  cerr << " 2  = 2D Textures - Cubic" << endl;
  cerr << " 3  = 3D Textures - Viewport aligned" << endl;
  cerr << " 4  = 3D Textures - Bricks" << endl;
  cerr << " 5  = 3D Textures - Spherical" << endl;
  cerr << " 6  = Shear-warp (CPU)" << endl;
  cerr << " 7  = Shear-warp (GPU)" << endl;
  cerr << " 8  = Ray casting (GPU)" << endl;
#ifdef HAVE_VOLPACK
  cerr << " 9  = VolPack (CPU)" << endl;
#endif
  cerr << endl;
  cerr << "-voxeltype <num>" << endl;
  cerr << " Select the default voxel type:" << endl;
  cerr << " 0 = Autoselect" << endl;
  cerr << " 1 = RGBA" << endl;
  cerr << " 2 = SGI color look-up table" << endl;
  cerr << " 3 = OpenGL paletted textures" << endl;
  cerr << " 4 = Nvidia texture shader" << endl;
  cerr << " 5 = Nvidia pixel shader" << endl;
  cerr << " 6 = ARB fragment program" << endl;
  cerr << endl;
  cerr << "-quality <value> (-q)" << endl;
  cerr << "Set the render quality (default: 1.0)" << endl;
  cerr << endl;
  cerr << "-dsp <host:display.screen>" << endl;
  cerr << "  Add x-org display for additional rendering context" << endl;
  cerr << endl;
  cerr << "-server <url>[:port]" << endl;
  cerr << "  Add a server renderer connected to over tcp ip" << endl;
  cerr << endl;
  cerr << "-serverfilename <path to file>" << endl;
  cerr << "  Path to a file where the server can find its volume data" << endl;
  cerr << "  If this entry is -serverfilename n, the n'th server will try to load this file" << endl;
  cerr << endl;
  cerr << "-lighting" << endl;
  cerr << " Use headlight for local illumination" << endl;
  cerr << endl;
  cerr << "-benchmark" << endl;
  cerr << " Time 3 half rotations and exit" << endl;
  cerr << endl;
  cerr << "-help (-h)" << endl;
  cerr << "Display this help information" << endl;
  cerr << endl;
  cerr << "-size <width> <height>" << endl;
  cerr << " Set the window size to <width> * <height> pixels." << endl;
  cerr << " The default window size is " << DEFAULTSIZE << " * " << DEFAULTSIZE <<
          " pixels" << endl;
  cerr << endl;
  cerr << "-parallel (-p)" << endl;
  cerr << " Use parallel projection mode" << endl;
  cerr << endl;
  cerr << "-boundaries (-b)" << endl;
  cerr << " Draw volume data set boundaries" << endl;
  cerr << endl;
  cerr << "-quad (-q)" << endl;
  cerr << " Try to request a quad buffered visual" << endl;
  cerr << endl;
  cerr << "-astereo" << endl;
  cerr << " Enable active stereo mode (if available)" << endl;
  cerr << endl;
  cerr << "-pstereo1, -pstereo2" << endl;
  cerr << " Enable passive stereo mode. The two modes are for different left/right" << endl;
  cerr << " assignments." << endl;
  cerr << endl;
  cerr << "-orientation (-o)" << endl;
  cerr << " Display volume orientation axes" << endl;
  cerr << endl;
  cerr << "-fps (-f)" << endl;
  cerr << " Display rendering speed [frames per second]" << endl;
  cerr << endl;
  cerr << "-nobt" << endl;
  cerr << " Don't show backtrace on OpenGL error (GL_ARB_debug_output only)" << endl;
  cerr << endl;
  cerr << "-transfunc (-t)" << endl;
  cerr << " Display transfer function color bar. Only works with 8 and 16 bit volumes" << endl;
  cerr << endl;
  cerr << "-testsuitefilename" << endl;
  cerr << " Specify a file with performance tests" << endl;
  cerr << endl;
  cerr << "-showbricks" << endl;
  cerr << " Show the brick outlines \\wo volume when brick renderer is used" << endl;
  cerr << endl;
  cerr << "-rec" << endl;
  cerr << " Record camera motion to file" << endl;
  cerr << endl;
  cerr << "-play" << endl;
  cerr << " Play camera motion from file" << endl;
  cerr << endl;
  #ifndef WIN32
  cerr << endl;
  #endif
}


//----------------------------------------------------------------------------
/** Parse command line arguments.
  @param argc,argv command line arguments
  @return true if parsing ok, false on error
*/
bool vvView::parseCommandLine(int argc, char** argv)
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
    else if (vvToolshed::strCompare(argv[arg], "-s")==0 ||
             vvToolshed::strCompare(argv[arg], "-servermode")==0)
    {
      serverMode = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-c")==0 ||
             vvToolshed::strCompare(argv[arg], "-clientmode")==0)
    {
      clientMode = true;
      std::string val;
      if(argv[arg+1])
        val = argv[arg+1];

      if(val == "cluster")
      {
        rrMode = RR_CLUSTER;
        setRendererType(vvRenderer::TEXREND);
        arg++;
      }
      else if(val == "ibr")
      {
        rrMode = RR_IBR;
        setRendererType(vvRenderer::REMOTE_IBR);
        arg++;
      }
      else if(val == "image")
      {
        rrMode = RR_IMAGE;
        setRendererType(vvRenderer::REMOTE_IMAGE);
        arg++;
      }
      else
      {
        cerr << "Set default client mode: image based rendering" << endl;
        rrMode = RR_IBR;
        setRendererType(vvRenderer::REMOTE_IBR);
      }
    }
    else if (vvToolshed::strCompare(argv[arg], "-nobt")==0)
    {
      showBt = false;
    }
    else if (vvToolshed::strCompare(argv[arg], "-roi")==0)
    {
      roiEnabled = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-port")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "No port specified, defaulting to: " << vvView::DEFAULT_PORT << endl;
        slavePort = vvView::DEFAULT_PORT;
        return false;
      }
      else
      {
        slavePort = atoi(argv[arg]);
      }
    }
    else if (vvToolshed::strCompare(argv[arg], "-r")==0 ||
             vvToolshed::strCompare(argv[arg], "-renderer")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Renderer ID missing." << endl;
        return false;
      }
      int val = atoi(argv[arg]);
      if(val >= 0 && val <= 5)
      {
        currentGeom = (vvTexRend::GeometryType)val;
        setRendererType(vvRenderer::TEXREND);
      }
      else if(val == 6)
      {
        if(perspectiveMode)
          setRendererType(vvRenderer::SOFTPAR);
        else
          setRendererType(vvRenderer::SOFTPER);
      }
      else if(val == 7)
      {
        if(perspectiveMode)
          setRendererType(vvRenderer::CUDAPAR);
        else
          setRendererType(vvRenderer::CUDAPER);
      }
      else if(val == 8)
      {
        setRendererType(vvRenderer::RAYREND);
      }
#ifdef HAVE_VOLPACK
      else if(val == 9)
      {
        setRendererType(vvRenderer::VOLPACK);
      }
#endif
      else
      {
        cerr << "Invalid geometry type." << endl;
        return false;
      }
    }
    else if (vvToolshed::strCompare(argv[arg], "-voxeltype")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Voxel type missing." << endl;
        return false;
      }
      currentVoxels = (vvTexRend::VoxelType)atoi(argv[arg]);
      if (currentVoxels<0 || currentVoxels>6)
      {
        cerr << "Invalid voxel type." << endl;
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
    else if ((vvToolshed::strCompare(argv[arg], "-q") == 0)
          |  (vvToolshed::strCompare(argv[arg], "-quality") == 0))
    {
      if ((++arg)>=argc)
      {
        cerr << "Quality missing." << endl;
        return false;
      }
      ds->draftQuality = (float)strtod(argv[arg], NULL);
    }
    else if (vvToolshed::strCompare(argv[arg], "-dsp")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Display name unspecified." << endl;
        return false;
      }
      addDisplay(argv[arg]);
    }
    else if (vvToolshed::strCompare(argv[arg], "-lighting")==0)
    {
      useHeadLight = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-server")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Server unspecified." << endl;
        return false;
      }

      if (!clientMode)
      {
        clientMode = true;
        rrMode = RR_IBR;
        setRendererType(vvRenderer::REMOTE_IBR);
      }

      const int port = vvToolshed::parsePort(argv[arg]);
      slavePorts.push_back(port);

      char* sname;
      if (port != -1)
      {
        sname = vvToolshed::stripPort(argv[arg]);
      }
      else
      {
       sname = argv[arg];
      }
      slaveNames.push_back(sname);
    }
    else if (vvToolshed::strCompare(argv[arg], "-serverfilename")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Server file name unspecified" << endl;
        return false;
      }
      slaveFileNames.push_back(argv[arg]);
    }
    else if (vvToolshed::strCompare(argv[arg], "-testsuitefilename")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Test suite file name unspecified" << endl;
      }
      testSuiteFileName = argv[arg];
    }
    else if (vvToolshed::strCompare(argv[arg], "-benchmark")==0)
    {
      benchmark = true;
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
    else if (vvToolshed::strCompare(argv[arg], "-q")==0
            || vvToolshed::strCompare(argv[arg], "-quad")==0)
    {
      tryQuadBuffer = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-astereo")==0)
    {
      stereoMode = 1;
    }
    else if (vvToolshed::strCompare(argv[arg], "-pstereo1")==0)
    {
      stereoMode = 2;
    }
    else if (vvToolshed::strCompare(argv[arg], "-pstereo2")==0)
    {
      stereoMode = 3;
    }
    else if (vvToolshed::strCompare(argv[arg], "-parallel")==0 ||
             vvToolshed::strCompare(argv[arg], "-p")==0)
    {
      perspectiveMode = false;
    }
    else if (vvToolshed::strCompare(argv[arg], "-boundaries")==0 ||
             vvToolshed::strCompare(argv[arg], "-b")==0)
    {
      boundariesMode = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-orientation")==0 ||
             vvToolshed::strCompare(argv[arg], "-o")==0)
    {
      orientationMode = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-fps")==0 ||
             vvToolshed::strCompare(argv[arg], "-f")==0)
    {
      fpsMode = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-transfunc")==0 ||
             vvToolshed::strCompare(argv[arg], "-t")==0)
    {
      paletteMode = true;
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
    else if (vvToolshed::strCompare(argv[arg], "-showbricks")==0)
    {
      showBricks = true;
    }
    else if (vvToolshed::strCompare(argv[arg], "-rec")==0)
    {
      recordMode = true;
      stopWatch.start();
      matrixFile = fopen("motion.txt", "wab");
    }
    else if (vvToolshed::strCompare(argv[arg], "-play")==0)
    {
      playMode = true;
    }
    else
    {
      filename = argv[arg];
      if (filename==NULL || filename[0]=='-')
      {
        cerr << "File name expected." << endl;
        return false;
      }
      if (!vvToolshed::isFile(filename))       // check if file exists
      {
        cerr << "File not found: " << filename << endl;
        return false;
      }
    }
  }
}


//----------------------------------------------------------------------------
/** Main VView routine.
  @param argc,argv command line arguments
  @return 0 if the program finished ok, 1 if an error occurred
*/
int vvView::run(int argc, char** argv)
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
  int error = (new vvView())->run(argc, argv);

#ifdef VV_DEBUG_MEMORY
  _CrtDumpMemoryLeaks();                         // display memory leaks, if any
#endif

  return error;
}

//============================================================================
// End of File
//============================================================================
