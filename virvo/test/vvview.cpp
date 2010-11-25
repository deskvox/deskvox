//****************************************************************************
// Project:         Virvo (Virtual Reality Volume Renderer)
// Copyright:       (c) 1999-2004 Jurgen P. Schulze. All rights reserved.
// Author's E-Mail: schulze@cs.brown.edu
// Affiliation:     Brown University, Department of Computer Science
//****************************************************************************

#include "../src/vvglew.h"

#include <iostream>
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

#include "../src/vvvirvo.h"
#include "../src/vvgltools.h"
#include "../src/vvoffscreenbuffer.h"
#include "../src/vvprintgl.h"
#include "../src/vvstopwatch.h"
#include "../src/vvrendercontext.h"
#include "../src/vvrendermaster.h"
#include "../src/vvrenderslave.h"
#include "../src/vvrenderer.h"
#include "../src/vvfileio.h"
#include "../src/vvdebugmsg.h"
#include "../src/vvsocketio.h"
#include "../src/vvtexrend.h"
#include "../src/vvsoftpar.h"
#include "../src/vvsoftper.h"
#include "../src/vvcudapar.h"
#include "../src/vvcuda.h"
#include "../src/vvrayrend.h"
#include "../src/vvbonjour/vvbonjourbrowser.h"
#include "../src/vvbonjour/vvbonjourresolver.h"
#include "vvobjview.h"
#include "vvperformancetest.h"
#include "vvview.h"

#define vvCudaPer vvCudaPar

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
  softwareRenderer = false;
  cudaRenderer = false;
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
  boundariesMode        = false;
  orientationMode       = false;
  fpsMode               = false;
  stereoMode            = 0;
  fullscreenMode        = false;
  interpolMode          = true;
  preintMode            = false;
  paletteMode           = false;
  emptySpaceLeapingMode = true;
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
  slaveMode             = false;
  slavePort             = vvView::DEFAULT_PORT;
  remoteRendering       = true;
  clipBuffer            = NULL;
  framebufferDump       = NULL;
  redistributeVolData   = false;
  _renderMaster         = NULL;
  benchmark             = false;
  testSuiteFileName     = NULL;
  showBricks            = false;
  roiEnabled            = false;
  mvScale               = 1.0f;
  rayRenderer           = true;
}


//----------------------------------------------------------------------------
/// Destructor.
vvView::~vvView()
{
  delete renderer;
  delete ov;
  delete vd;
  delete _renderMaster;
}


//----------------------------------------------------------------------------
/** VView main loop.
  @param filename  volume file to display
*/
void vvView::mainLoop(int argc, char *argv[])
{
  vvDebugMsg::msg(0, "vvView::mainLoop()");

  if (slaveMode)
  {
    while (1)
    {
      cerr << "Renderer started in slave mode" << endl;
      vvRenderSlave* renderSlave = new vvRenderSlave();

      if (renderSlave->initSocket(slavePort, vvSocket::VV_TCP) != vvRenderSlave::VV_OK)
      {
        // TODO: Evaluate the error type, maybe don't even return but try again.
        cerr << "Couldn't initialize the socket connection" << endl;
        cerr << "Exiting..." << endl;
        return;
      }

      if (renderSlave->initData(vd) != vvRenderSlave::VV_OK)
      {
        cerr << "Exiting..." << endl;
        return;
      }

      // Get bricks to render
      std::vector<BrickList>* frames = new std::vector<BrickList>();
      BrickList bricks;

      if (renderSlave->initBricks(bricks) != vvRenderSlave::VV_OK)
      {
        delete frames;
        cerr << "Exiting..." << endl;
        return;
      }

      frames->push_back(bricks);
      if (vd != NULL)
      {
        vd->printInfoLine();

        // Set default color scheme if no TF present:
        if (vd->tf.isEmpty())
        {
          vd->tf.setDefaultAlpha(0, 0.0, 1.0);
          vd->tf.setDefaultColors((vd->chan==1) ? 0 : 2, 0.0, 1.0);
        }

        ov = new vvObjView();

        vvRenderContext* context = new vvRenderContext();
        if (context->makeCurrent())
        {
          ov = new vvObjView();
          setProjectionMode(perspectiveMode);
          setRenderer(currentGeom, currentVoxels, frames);
          srand(time(NULL));

          renderSlave->renderLoop(dynamic_cast<vvTexRend*>(renderer));
          cerr << "Exiting..." << endl;
        }
        delete context;
      }

      // Frames vector with bricks is deleted along with the renderer.
      // Don't free them here.
      // see setRenderer().

      delete renderSlave;
      renderSlave = NULL;
    }
  }
  else
  {
    vvFileIO* fio;

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

    fio = new vvFileIO();
    if (fio->loadVolumeData(vd) != vvFileIO::OK)
    {
      cerr << "Error loading volume file" << endl;
      delete vd;
      delete fio;
      return;
    }
    else vd->printInfoLine();
    delete fio;

    float div = 1.;
    const char* txt[] = { "sx", "sy", "sz" };
    for(int i=0; i<3; i++)
    {
      cerr << txt[i] << ": " << vd->dist[i]*vd->vox[i] << endl;
      if(vd->dist[i]*vd->vox[i]/div > 1.)
        div = vd->dist[i]*vd->vox[i];
    }
    mvScale = vd->dist[0] / div;
    cerr << "Scale modelview matrix by " << mvScale << endl;

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
      remoteRendering = false;
    }

    initGraphics(argc, argv);
#ifdef HAVE_CUDA
    vvCuda::initGlInterop();
#endif
    if (remoteRendering)
    {
      _renderMaster = new vvRenderMaster(slaveNames, slavePorts, slaveFileNames, filename);
      remoteRendering = (_renderMaster->initSockets(vvView::DEFAULT_PORT, vvSocket::VV_TCP,
                                             redistributeVolData, vd) == vvRenderMaster::VV_OK);
    }

    animSpeed = vd->dt;
    createMenus();

    ov = new vvObjView();
    ds->ov->mv.scale(mvScale);

    setProjectionMode(perspectiveMode);
    setRenderer(currentGeom, currentVoxels);

    if (remoteRendering)
    {
      remoteRendering = (_renderMaster->initBricks(dynamic_cast<vvTexRend*>(renderer)) == vvRenderMaster::VV_OK);
    }

    // Set window title:
    if (filename!=NULL) glutSetWindowTitle(filename);

    srand(time(NULL));
    if(benchmark)
    {
      glutTimerFunc(1, timerCallback, BENCHMARK_TIMER);
    }

    glutMainLoop();

    delete vd;
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
  if (ds->winHeight > 0) ds->ov->setAspectRatio((float)ds->winWidth / (float)ds->winHeight);

  glDrawBuffer(GL_FRONT_AND_BACK);               // select all buffers
                                                 // set clear color
  glClearColor(ds->bgColor[0], ds->bgColor[1], ds->bgColor[2], 1.0f);
                                                 // clear window
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (ds->remoteRendering)
  {
    ds->_renderMaster->resize(w, h);
  }
}


//----------------------------------------------------------------------------
/// Callback method for window redraws.
void vvView::displayCallback(void)
{
  vvDebugMsg::msg(3, "vvView::displayCallback()");

  if (ds->remoteRendering)
  {
    ds->ov->updateModelviewMatrix(vvObjView::LEFT_EYE);

    ds->_renderMaster->render(ds->bgColor);

    glutSwapBuffers();
  }
  else
  {
    glDrawBuffer(GL_BACK);
    glClearColor(ds->bgColor[0], ds->bgColor[1], ds->bgColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ds->renderer->_renderState._quality = ((ds->hqMode) ? ds->highQuality : ds->draftQuality);

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

    glutSwapBuffers();
  }
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
void vvView::setSoftwareRenderer(bool enable)
{
  softwareRenderer = enable;
}


//----------------------------------------------------------------------------
/** Set software renderering flag.
 */
void vvView::setCudaRenderer(bool enable)
{
  cudaRenderer = enable;
}


//----------------------------------------------------------------------------
/** Set ray rendering flag.
 */
void vvView::setRayRenderer(const bool enable)
{
  rayRenderer = enable;
}


//----------------------------------------------------------------------------
/** Set active rendering algorithm.
 */
void vvView::setRenderer(vvTexRend::GeometryType gt, vvTexRend::VoxelType vt,
                         std::vector<BrickList>* bricks, const int maxBrickSizeX,
                         const int maxBrickSizeY, const int maxBrickSizeZ)
{
  vvDebugMsg::msg(3, "vvView::setRenderer()");

  currentGeom = gt;
  currentVoxels = vt;

  if(renderer)
  renderState = renderer->_renderState;
  delete renderer;
  renderer = NULL;
  renderState._maxBrickSize[0] = maxBrickSizeX;
  renderState._maxBrickSize[1] = maxBrickSizeY;
  renderState._maxBrickSize[2] = maxBrickSizeZ;

  // Multi threading parameters.
  // These are needed before construction of the renderer so that
  // additional rendering contexts and x-windows can be created.

  if (cudaRenderer)
  {
    if(perspectiveMode)
      renderer = new vvCudaPer(vd, renderState);
    else
      renderer = new vvCudaPar(vd, renderState);
  }
  else if (softwareRenderer)
  {
    if(perspectiveMode)
      renderer = new vvSoftPer(vd, renderState);
    else
      renderer = new vvSoftPar(vd, renderState);
  }
  else if (rayRenderer)
  {
    renderer = new vvRayRend(vd, renderState);
  }
  else if (numDisplays > 0)
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

  if (!slaveMode)
  {
    renderer->setROIEnable(roiEnabled);
    printROIMessage();
  }

  //static_cast<vvTexRend *>(renderer)->setTexMemorySize( 4 );
  //static_cast<vvTexRend *>(renderer)->setComputeBrickSize( false );
  //static_cast<vvTexRend *>(renderer)->setBrickSize( 64 );

  //renderer->setBoundariesMode(boundariesMode);
  renderer->_renderState._boundaries = boundariesMode;
  //renderer->setBoundariesColor(1.0f-bgColor[0], 1.0f-bgColor[1], 1.0f-bgColor[2]);
  //renderer->resizeEdgeMax(OBJ_SIZE);
  renderer->setPosition(&pos);
  //renderer->setOrientationMode(orientationMode);
  //renderer->setTimingMode(timingMode);
  //renderer->setPaletteMode(paletteMode);
  //renderer->setParameter(vvRenderer::VV_SLICEINT, (interpolMode) ? 1.0f : 0.0f);
  renderer->setParameter(vvRenderer::VV_PREINT, (preintMode) ? 1.0f : 0.0f);
  renderer->_renderState._mipMode = mipMode;
  //renderer->setParameter(vvRenderer::VV_GAMMA, (gammaMode) ? 1.0f : 0.0f);
  //renderer->setParameter(vvRenderer::VV_OPCORR, (opCorrMode) ? 1.0f : 0.0f);
  //renderer->setClippingMode(false);
  renderer->_renderState._quality = (hqMode) ? highQuality : draftQuality;
  renderer->setParameter(vvRenderer::VV_LEAPEMPTY, emptySpaceLeapingMode);
  renderer->setParameter(vvRenderer::VV_LIGHTING, useHeadLight);
  renderer->setParameter(vvRenderer::VV_GPUPROXYGEO, ds->gpuproxygeo);
  renderer->setParameter(vvRenderer::VV_OFFSCREENBUFFER, useOffscreenBuffer);
  renderer->setParameter(vvRenderer::VV_IMG_PRECISION, bufferPrecision);
  static_cast<vvTexRend*>(renderer)->setShowBricks(showBricks);
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
  case 'B': ds->optionsMenuCallback(12); break;
  case 'b': ds->viewMenuCallback(0);  break;
  case 'c': ds->viewMenuCallback(10); break;
  case 'd': ds->mainMenuCallback(5);  break;
  case 'e': ds->mainMenuCallback(4);  break;
  case 'f': ds->viewMenuCallback(2);  break;
  case 'H': ds->optionsMenuCallback(5); break;
  case 'h': ds->optionsMenuCallback(6); break;
  case 'i': ds->optionsMenuCallback(0);  break;
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
  case 'q': ds->mainMenuCallback(13); break;
  case 'r': ds->mainMenuCallback(7);  break;
  case 'R': ds->mainMenuCallback(12); break;
  case 's': ds->animMenuCallback(4);  break;
  case 'S': ds->animMenuCallback(5);  break;
  case 't': ds->mainMenuCallback(11); break;
  case 'u': ds->viewMenuCallback(8);  break;
  case 'v': ds->viewMenuCallback(9);  break;
  case 'w': ds->viewMenuCallback(6);  break;
  case 'x': ds->optionsMenuCallback(2); break;
  case 'z': ds->viewMenuCallback(5);  break;
  case '<': ds->transferMenuCallback(11);  break;
  case '>': ds->transferMenuCallback(12);  break;
  case '[': ds->mainMenuCallback(98); break;
  case ']': ds->mainMenuCallback(99); break;
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

  switch(key)
  {
  case GLUT_KEY_LEFT:
    if (ds->roiEnabled)
    {
      probePos.e[0] -= delta;
      ds->renderer->setProbePosition(&probePos);
    }
    else
    {
      ds->pos.e[0] -= delta;
      ds->renderer->setPosition(&ds->pos);
    }
    break;
  case GLUT_KEY_RIGHT:
    if (ds->roiEnabled)
    {
      probePos.e[0] += delta;
      ds->renderer->setProbePosition(&probePos);
    }
    else
    {
      ds->pos.e[0] += delta;
      ds->renderer->setPosition(&ds->pos);
    }
    break;
  case GLUT_KEY_UP:
    if (ds->roiEnabled)
    {
      if (modifiers & GLUT_ACTIVE_SHIFT)
      {
        probePos.e[2] += delta;
      }
      else
      {
        probePos.e[1] += delta;
      }
      ds->renderer->setProbePosition(&probePos);
    }
    else
    {
      ds->pos.e[1] += delta;
      ds->renderer->setPosition(&ds->pos);
    }
    break;
  case GLUT_KEY_DOWN:
    if (ds->roiEnabled)
    {
      if (modifiers & GLUT_ACTIVE_SHIFT)
      {
        probePos.e[2] -= delta;
      }
      else
      {
        probePos.e[1] -= delta;
      }
      ds->renderer->setProbePosition(&probePos);
    }
    else
    {
      ds->pos.e[1] -= delta;
      ds->renderer->setPosition(&ds->pos);
    }
    break;
  default: break;
  }

  if (ds->remoteRendering)
  {
    ds->_renderMaster->setPosition(ds->pos);
  }
  glutPostRedisplay();
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
    performanceTest();
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

  vvVector3 probeSize;

  switch (item)
  {
  case 0:                                     // projection mode
    ds->setProjectionMode(!ds->perspectiveMode);
    if (ds->cudaRenderer)
    {
      delete ds->renderer;
      if (ds->perspectiveMode)
        ds->renderer = new vvCudaPer(ds->vd, ds->renderState);
      else
        ds->renderer = new vvCudaPar(ds->vd, ds->renderState);
    }
    else if (ds->softwareRenderer)
    {
      delete ds->renderer;
      if (ds->perspectiveMode)
        ds->renderer = new vvSoftPer(ds->vd, ds->renderState);
      else
        ds->renderer = new vvSoftPar(ds->vd, ds->renderState);
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
  case 12:                                    // toggle roi mode
    ds->roiEnabled = !ds->roiEnabled;
    ds->renderer->setROIEnable(ds->roiEnabled);

    if (ds->remoteRendering)
    {
      ds->_renderMaster->setROIEnabled(ds->roiEnabled);
    }
    printROIMessage();
    break;
  case 13:                                    // quit
    glutDestroyWindow(ds->window);
    delete ds;
    exit(0);
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
    ds->setSoftwareRenderer(false);
    ds->setCudaRenderer(false);
    ds->setRayRenderer(false);
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
  }
  else if (item==7)
  {
    cerr << "Switched to software shear-warp renderer" << endl;
    ds->setSoftwareRenderer(true);
    ds->setCudaRenderer(false);
    ds->setRenderer();
  }
  else if (item==8)
  {
    cerr << "Switched to CUDA shear-warp renderer" << endl;
    ds->setSoftwareRenderer(false);
    ds->setCudaRenderer(true);
    ds->setRenderer();
  }
  else if (item == 9)
  {
    cerr << "Switched to CUDA ray casting renderer" << endl;
    ds->setSoftwareRenderer(false);
    ds->setCudaRenderer(false);
    ds->setRayRenderer(true);
    ds->setRenderer();
  }
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
    ds->renderer->_renderState._quality = (ds->hqMode) ? ds->highQuality : ds->draftQuality;
    cerr << QUALITY_NAMES[ds->hqMode] << " quality: " <<
        ((ds->hqMode) ? ds->highQuality : ds->draftQuality) << endl;
    if (ds->remoteRendering)
    {
      ds->_renderMaster->adjustQuality(ds->renderer->_renderState._quality);
    }
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

  switch(item)
  {
  case 0:                                     // slice interpolation mode
    ds->interpolMode = !ds->interpolMode;
    ds->renderer->setParameter(vvRenderer::VV_SLICEINT, (ds->interpolMode) ? 1.0f : 0.0f);
    cerr << "Interpolation mode set to " << int(ds->interpolMode) << endl;
    if (ds->remoteRendering)
    {
      ds->_renderMaster->setInterpolation(ds->interpolMode);
    }
    break;
  case 1:
    ds->preintMode = !ds->preintMode;
    ds->renderer->setParameter(vvRenderer::VV_PREINT, (ds->preintMode) ? 1.0f : 0.0f);
    cerr << "Pre-integration set to " << int(ds->preintMode) << endl;
    break;
  case 2:                                     // min/maximum intensity projection
    ++ds->mipMode;
    if (ds->mipMode>2) ds->mipMode = 0;
    ds->renderer->_renderState._mipMode = ds->mipMode;
    cerr << "MIP mode set to " << ds->mipMode << endl;
    if (ds->remoteRendering)
    {
      ds->_renderMaster->setMipMode(ds->mipMode);
    }
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
    size.e[2] *= 1.05f;
    //ds->renderer->setVoxelSize(&size);
    cerr << "Z size set to " << size.e[2] << endl;
    break;
  case 9:                                     // decrease z size
    //ds->renderer->getVoxelSize(&size);
    size.e[2] *= 0.95f;
    //ds->renderer->setVoxelSize(&size);
    cerr << "Z size set to " << size.e[2] << endl;
    break;
  case 10:                                     // increase precision of visual
    if (ds->useOffscreenBuffer)
    {
      if (ds->bufferPrecision == 8)
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
    ds->renderer->setParameter(vvRenderer::VV_IMG_PRECISION, ds->bufferPrecision);
    break;
  case 11:                                    // increase precision of visual
    if (ds->useOffscreenBuffer)
    {
      if (ds->bufferPrecision == 32)
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
    ds->renderer->setParameter(vvRenderer::VV_IMG_PRECISION, ds->bufferPrecision);
    break;
  case 12:                                     // toggle showing of bricks
    {
      vvTexRend *rend = dynamic_cast<vvTexRend *>(ds->renderer);
      ds->showBricks = !ds->showBricks;
      rend->setShowBricks( ds->showBricks );
      cerr << (!ds->showBricks?"not ":"") << "showing bricks" << endl;
    }
    break;
  case 13:
    {
      vvTexRend *rend = dynamic_cast<vvTexRend *>(ds->renderer);
      int shader = rend->getCurrentShader()+1;
      rend->setCurrentShader(shader);
      cerr << "shader set to " << rend->getCurrentShader() << endl;
    }
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
  case 7:                                     // alpha: ascending
    ds->vd->tf.setDefaultAlpha(0, 0.0, 1.0);
    break;
  case 8:                                     // alpha: descending
    ds->vd->tf.setDefaultAlpha(1, 0.0, 1.0);
    break;
  case 9:                                     // alpha: opaque
    ds->vd->tf.setDefaultAlpha(2, 0.0, 1.0);
    break;
  case 10:                                    // alpha: display peak
  case 11:                                    // alpha: shift left peak
  case 12:                                    // alpha: shift right peak
    if(item == 11)
      peakPosX -= .05;
    else if(item == 12)
      peakPosX += .05;
    if (peakPosX < 0.0f) peakPosX += 1.0f;
    if (peakPosX > 1.0f) peakPosX -= 1.0f;
    cerr << "Peak position: " << peakPosX << endl;

    ds->vd->tf.deleteWidgets(vvTransFunc::TF_PYRAMID);
    ds->vd->tf.deleteWidgets(vvTransFunc::TF_BELL);
    ds->vd->tf.deleteWidgets(vvTransFunc::TF_CUSTOM);
    ds->vd->tf.deleteWidgets(vvTransFunc::TF_SKIP);
    ds->vd->tf._widgets.append(
          new vvTFPyramid(vvColor(1.f, 1.f, 1.f), false, 1.f, peakPosX, .2f, 0.f),
          vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    break;
  case 13:                                    // gamma red
  case 14:
    gamma = ds->renderer->getGamma(vvRenderer::VV_RED);
    gamma *= (item==13) ? 0.95f : 1.05f;
    ds->renderer->setGamma(vvRenderer::VV_RED, gamma);
    cerr << "gamma red = " << gamma << endl;
    break;
  case 15:                                    // gamma green
  case 16:
    gamma = ds->renderer->getGamma(vvRenderer::VV_GREEN);
    gamma *= (item==15) ? 0.95f : 1.05f;
    ds->renderer->setGamma(vvRenderer::VV_GREEN, gamma);
    cerr << "gamma green = " << gamma << endl;
    break;
  case 17:                                    // gamma blue
  case 18:
    gamma = ds->renderer->getGamma(vvRenderer::VV_BLUE);
    gamma *= (item==17) ? 0.95f : 1.05f;
    ds->renderer->setGamma(vvRenderer::VV_BLUE, gamma);
    cerr << "gamma blue = " << gamma << endl;
    break;
  case 19:                                    // channel 4 red
  case 20:
    chan4 = ds->renderer->getOpacityWeight(vvRenderer::VV_RED);
    chan4 *= (item==19) ? 0.95f : 1.05f;
    ds->renderer->setOpacityWeight(vvRenderer::VV_RED, chan4);
    cerr << "channel 4 red = " << chan4 << endl;
    break;
  case 21:                                    // channel 4 green
  case 22:
    chan4 = ds->renderer->getOpacityWeight(vvRenderer::VV_GREEN);
    chan4 *= (item==21) ? 0.95f : 1.05f;
    ds->renderer->setOpacityWeight(vvRenderer::VV_GREEN, chan4);
    cerr << "channel 4 green = " << chan4 << endl;
    break;
  case 23:                                    // channel 4 blue
  case 24:
    chan4 = ds->renderer->getOpacityWeight(vvRenderer::VV_BLUE);
    chan4 *= (item==23) ? 0.95f : 1.05f;
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
    ds->renderer->_renderState._boundaries = ds->boundariesMode;
    cerr << "Bounding box " << ds->onOff[ds->boundariesMode] << endl;
    if (ds->remoteRendering)
    {
      ds->_renderMaster->toggleBoundingBox();
    }
    break;
  case 1:                                     // axis orientation
    ds->orientationMode = !ds->orientationMode;
    ds->renderer->_renderState._orientation = !ds->renderer->_renderState._orientation;
    cerr << "Coordinate axes display " << ds->onOff[ds->orientationMode] << endl;
    break;
  case 2:                                     // frame rate
    ds->fpsMode = !ds->fpsMode;
    ds->renderer->_renderState._fpsDisplay = !ds->renderer->_renderState._fpsDisplay;
    cerr << "Frame rate display " << ds->onOff[ds->fpsMode] << endl;
    break;
  case 3:                                     // transfer function
    ds->paletteMode = !ds->paletteMode;
    ds->renderer->_renderState._palette = !ds->renderer->_renderState._palette;
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

//----------------------------------------------------------------------------
/** Do a performance test.
  Default behavior:
  The test resets the view position (but not the projection mode)
  and measures the time needed for a
  360 degrees rotation of the volume about its vertical axis.
  The image is drawn every 2 degrees.
  <br>
*/
void vvView::performanceTest()
{
  vvDebugMsg::msg(1, "vvView::performanceTest()");

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
                      test->getBrickDims()[0],
                      test->getBrickDims()[1],
                      test->getBrickDims()[2]);
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
        ds->ov->setProjection(vvObjView::ORTHO, 1.5f, -100.0, 100.0);
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
    ds->renderer->profileStop();
    //totalTime->stop();


    printProfilingResult(totalTime, framesRendered);

    delete totalTime;
  }
  delete testSuite;
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
  cerr << "Renderer geometry................................." << ds->currentGeom << endl;
  cerr << "Voxel type........................................" << ds->currentVoxels << endl;
  cerr << "Volume file name.................................." << ds->vd->getFilename() << endl;
  cerr << "Volume size [voxels].............................." << ds->vd->vox[0] << " x " << ds->vd->vox[1] << " x " << ds->vd->vox[2] << endl;
  cerr << "Output image size [pixels]........................" << viewport[2] << " x " << viewport[3] << endl;
  cerr << "Image quality....................................." << ds->renderer->_renderState._quality << endl;
  cerr << "Projection........................................" << projectMode[ds->perspectiveMode] << endl;
  cerr << "Interpolation mode................................" << interpolMode[ds->interpolMode] << endl;
  cerr << "Empty space leaping for bricks...................." << onOffMode[ds->emptySpaceLeapingMode] << endl;
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
  int rendererMenu, voxelMenu, optionsMenu, transferMenu, animMenu, viewMenu;

  vvDebugMsg::msg(1, "vvView::createMenus()");

  // Rendering geometry menu:
  rendererMenu = glutCreateMenu(rendererMenuCallback);
  glutAddMenuEntry("Auto select [0]", 0);
  if (vvTexRend::isSupported(vvTexRend::VV_SLICES))    glutAddMenuEntry("2D textures - slices [1]", 1);
  if (vvTexRend::isSupported(vvTexRend::VV_CUBIC2D))   glutAddMenuEntry("2D textures - cubic [2]", 2);
  if (vvTexRend::isSupported(vvTexRend::VV_VIEWPORT))  glutAddMenuEntry("3D textures - viewport aligned [3]", 3);
  if (vvTexRend::isSupported(vvTexRend::VV_SPHERICAL)) glutAddMenuEntry("3D textures - spherical [4]", 4);
  if (vvTexRend::isSupported(vvTexRend::VV_BRICKS))    glutAddMenuEntry("3D textures - bricked [5]", 5);
  if (vvTexRend::isSupported(vvTexRend::VV_BRICKS))    glutAddMenuEntry("Bricks - generate proxy geometry on GPU [6]", 6);
  glutAddMenuEntry("CPU Shear-warp [7]", 7);
  glutAddMenuEntry("GPU Shear-warp [8]", 8);
  glutAddMenuEntry("GPU Ray casting [9]", 9);
  glutAddMenuEntry("Decrease quality [-]", 98);
  glutAddMenuEntry("Increase quality [+]", 99);

  // Voxel menu:
  voxelMenu = glutCreateMenu(voxelMenuCallback);
  glutAddMenuEntry("Auto select", 0);
  glutAddMenuEntry("RGBA", 1);
  if (vvTexRend::isSupported(vvTexRend::VV_SGI_LUT)) glutAddMenuEntry("SGI look-up table", 2);
  if (vvTexRend::isSupported(vvTexRend::VV_PAL_TEX)) glutAddMenuEntry("Paletted textures", 3);
  if (vvTexRend::isSupported(vvTexRend::VV_TEX_SHD)) glutAddMenuEntry("Texture shader", 4);
  if (vvTexRend::isSupported(vvTexRend::VV_PIX_SHD)) glutAddMenuEntry("Cg pixel shader", 5);
  if (vvTexRend::isSupported(vvTexRend::VV_FRG_PRG)) glutAddMenuEntry("ARB fragment program", 6);

  // Renderer options menu:
  optionsMenu = glutCreateMenu(optionsMenuCallback);
  glutAddMenuEntry("Toggle slice interpolation [i]", 0);
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
  glutAddMenuEntry("Toggle offscreen buffering", 6);
  glutAddMenuEntry("Toggle head light", 7);
  glutAddMenuEntry("Increase z size [H]", 8);
  glutAddMenuEntry("Decrease z size [h]", 9);
  glutAddMenuEntry("Increase buffer precision", 10);
  glutAddMenuEntry("Decrease buffer precision", 11);
  glutAddMenuEntry("Show/hide bricks [B]", 12);

  // Transfer function menu:
  transferMenu = glutCreateMenu(transferMenuCallback);
  glutAddMenuEntry("Colors: bright", 0);
  glutAddMenuEntry("Colors: HSV", 1);
  glutAddMenuEntry("Colors: grayscale", 2);
  glutAddMenuEntry("Colors: white", 3);
  glutAddMenuEntry("Colors: red", 4);
  glutAddMenuEntry("Colors: green", 5);
  glutAddMenuEntry("Colors: blue", 6);
  glutAddMenuEntry("Alpha: ascending", 7);
  glutAddMenuEntry("Alpha: descending", 8);
  glutAddMenuEntry("Alpha: opaque", 9);
  glutAddMenuEntry("Alpha: single peak", 10);
  glutAddMenuEntry("Alpha: shift peak left [<]", 11);
  glutAddMenuEntry("Alpha: shift peak right [>]", 12);
  glutAddMenuEntry("Gamma: less red [1]", 13);
  glutAddMenuEntry("Gamma: more red [2]", 14);
  glutAddMenuEntry("Gamma: less green [3]", 15);
  glutAddMenuEntry("Gamma: more green [4]", 16);
  glutAddMenuEntry("Gamma: less blue [5]", 17);
  glutAddMenuEntry("Gamma: more blue [6]", 18);
  glutAddMenuEntry("Channel 4: less red", 19);
  glutAddMenuEntry("Channel 4: more red", 20);
  glutAddMenuEntry("Channel 4: less green", 21);
  glutAddMenuEntry("Channel 4: more green", 22);
  glutAddMenuEntry("Channel 4: less blue", 23);
  glutAddMenuEntry("Channel 4: more blue", 24);

  // Animation menu:
  animMenu = glutCreateMenu(animMenuCallback);
  glutAddMenuEntry("Next frame [n]", 0);
  glutAddMenuEntry("Previous frame [N]", 1);
  glutAddMenuEntry("Start/stop animation [a]", 2);
  glutAddMenuEntry("Rewind", 3);
  glutAddMenuEntry("Animation speed up [s]", 4);
  glutAddMenuEntry("Animation speed down [S]", 5);
  glutAddMenuEntry("Reset speed", 6);

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
  glutAddSubMenu("Viewing window", viewMenu);
  glutAddMenuEntry("Toggle perspective mode [p]", 0);
  glutAddMenuEntry("Toggle auto refinement [e]", 4);
  glutAddMenuEntry("Toggle rendering time display [d]", 5);
  glutAddMenuEntry("Reset object orientation", 7);
  glutAddMenuEntry("Toggle popup menu/zoom mode [m]", 8);
  glutAddMenuEntry("Save volume to file", 9);
  glutAddMenuEntry("Performance test [t]", 11);
  glutAddMenuEntry("Toggle region of intereset mode [R]", 12);
  glutAddMenuEntry("ROI size-- ['[']", 98);
  glutAddMenuEntry("ROI size++ [']']", 99);
  glutAddMenuEntry("Quit [q]", 13);

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
#ifndef __APPLE__
                                                  // create stereo context
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_STEREO | GLUT_DEPTH);
  if (!glutGet(GLUT_DISPLAY_MODE_POSSIBLE))
  {
    cerr << "Stereo mode not supported by display driver." << endl;
#endif
                                                  // create double buffering context
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
#ifndef __APPLE__
  }
  else
  {
    cerr << "Stereo mode found." << endl;
    activeStereoCapable = true;
  }
#endif
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
    ov->setProjection(vvObjView::ORTHO, 1.5f, -100.0, 100.0);
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
  const float width = 0.3f;
  const float height = 0.3f;
  const float depth = 0.3f;
  const float posX = 0.0f;
  const float posY = 0.0f;
  const float posZ = 0.0f;

  glBegin(GL_QUADS);

    glColor3f(1.0f, 1.0f, 1.0f);

    glNormal3f( 0.0f,  0.0f,  1.0f);
    glVertex3f(posX - width, posY - height, posZ + depth);
    glVertex3f(posX + width, posY - height, posZ + depth);
    glVertex3f(posX + width, posY + height, posZ + depth);
    glVertex3f(posX - width, posY + height, posZ + depth);

    glNormal3f( 0.0f,  0.0f, -1.0f);
    glVertex3f(posX + width, posY - height, posZ - depth);
    glVertex3f(posX - width, posY - height, posZ - depth);
    glVertex3f(posX - width, posY + height, posZ - depth);
    glVertex3f(posX + width, posY + height, posZ - depth);

    glNormal3f(-1.0f,  0.0f,  0.0f);
    glVertex3f(posX - width, posY - height, posZ - depth);
    glVertex3f(posX - width, posY - height, posZ + depth);
    glVertex3f(posX - width, posY + height, posZ + depth);
    glVertex3f(posX - width, posY + height, posZ - depth);

    glNormal3f( 1.0f,  0.0f,  0.0f);
    glVertex3f(posX + width, posY - height, posZ + depth);
    glVertex3f(posX + width, posY - height, posZ - depth);
    glVertex3f(posX + width, posY + height, posZ - depth);
    glVertex3f(posX + width, posY + height, posZ + depth);

    glNormal3f( 0.0f,  1.0f,  0.0f);
    glVertex3f(posX - width, posY + height, posZ + depth);
    glVertex3f(posX + width, posY + height, posZ + depth);
    glVertex3f(posX + width, posY + height, posZ - depth);
    glVertex3f(posX - width, posY + height, posZ - depth);

    glNormal3f( 0.0f, -1.0f,  0.0f);
    glVertex3f(posX - width, posY - height, -depth);
    glVertex3f(posX - width, posY - height,  depth);
    glVertex3f(posX + width, posY - height,  depth);
    glVertex3f(posX + width, posY - height, -depth);

  glEnd();
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
  cerr << "-s" << endl;
  cerr << " Renderer is a render slave and accepts assignments from a master renderer" << endl;
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
  cerr << " 4  = 3D Textures - Spherical" << endl;
  cerr << " 5  = 3D Textures - Bricks" << endl;
  cerr << " 7  = Shear-warp (CPU)" << endl;
  cerr << " 8  = Shear-warp (GPU)" << endl;
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
  cerr << "-dsp <host:display.screen>" << endl;
  cerr << "  Add x-org display for additional rendering context" << endl;
  cerr << endl;
  cerr << "-slave <url>[:port]" << endl;
  cerr << "  Add a slave renderer connected to over tcp ip" << endl;
  cerr << endl;
  cerr << "-slavefilename <path to file>" << endl;
  cerr << "  Path to a file where the slave can find its volume data" << endl;
  cerr << "  If this entry is -slavefilename n, the n'th slave will try to load this file" << endl;
  cerr << endl;
  cerr << "-redistributevoldata" << endl;
  cerr << "  Don't load slave volume data from file, it will be redistributed by the master to all slaves" << endl;
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
  cerr << "-perspective (-p)" << endl;
  cerr << " Use perspective projection mode" << endl;
  cerr << endl;
  cerr << "-boundaries (-b)" << endl;
  cerr << " Draw volume data set boundaries" << endl;
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
  cerr << "-transfunc (-t)" << endl;
  cerr << " Display transfer function color bar. Only works with 8 and 16 bit volumes" << endl;
  cerr << endl;
  cerr << "-testsuitefilename" << endl;
  cerr << " Specify a file with performance tests" << endl;
  cerr << endl;
  cerr << "-showbricks" << endl;
  cerr << " Show the brick outlines \\wo volume when brick renderer is used" << endl;
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
    else if (vvToolshed::strCompare(argv[arg], "-s")==0)
    {
      slaveMode = true;
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
        softwareRenderer = cudaRenderer = false;
      }
      else if(val == 7)
      {
        softwareRenderer = true;
        cudaRenderer = false;
      }
      else if(val == 8)
      {
        softwareRenderer = false;
        cudaRenderer = true;
      }
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
    else if (vvToolshed::strCompare(argv[arg], "-slave")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Slave unspecified." << endl;
        return false;
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
    else if (vvToolshed::strCompare(argv[arg], "-slavefilename")==0)
    {
      if ((++arg)>=argc)
      {
        cerr << "Slave file name unspecified" << endl;
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
    else if (vvToolshed::strCompare(argv[arg], "-redistributevoldata")==0)
    {
      redistributeVolData = true;
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
    else if (vvToolshed::strCompare(argv[arg], "-perspective")==0 ||
             vvToolshed::strCompare(argv[arg], "-p")==0)
    {
      perspectiveMode = true;
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
    else if (vvToolshed::strCompare(argv[arg], "-display")==0)
    {
      // handled by GLUT
      if ((++arg)>=argc)
      {
        cerr << "Display name unspecified" << endl;
        return false;
      }
    }
    else if (vvToolshed::strCompare(argv[arg], "-showbricks")==0)
    {
      showBricks = true;
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

  vvView* vview = new vvView();
  //vvDebugMsg::setDebugLevel(vvDebugMsg::NO_MESSAGES);
  int error = vview->run(argc, argv);
  delete vview;

#ifdef VV_DEBUG_MEMORY
  _CrtDumpMemoryLeaks();                         // display memory leaks, if any
#endif

  return error;
}


//============================================================================
// End of File
//============================================================================
