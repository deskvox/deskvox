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

// Glew:

// No circular dependencies between gl.h and glew.h
#ifndef GLEW_INCLUDED
#if !defined(__APPLE__)
#include <GL/glew.h>
#define GLEW_INCLUDED
#endif
#endif

#include <iostream>
#include <algorithm>
#include <limits.h>
#include <math.h>

#include "vvopengl.h"
#include "vvdynlib.h"

#if !defined(_WIN32) && !defined(__APPLE__)
#define HAVE_X11
#endif

// xlib:
#ifdef HAVE_X11
#include <GL/glx.h>
#include <X11/Xlib.h>
#endif

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvvecmath.h"
#include "vvdebugmsg.h"
#include "vvtoolshed.h"
#include "vvgltools.h"
#include "vvsphere.h"
#include "vvtexrend.h"
#include "vvstopwatch.h"
#include "vvoffscreenbuffer.h"
#include "vvprintgl.h"
#include "vvshaderfactory.h"

#ifdef __APPLE__
typedef struct pthead_barrierattr_t_ { int dummy; } pthread_barrierattr_t;
static int pthread_barrier_wait(pthread_barrier_t * /*barrier*/) { return 1; }
static int pthread_barrier_init(pthread_barrier_t * /*barrier*/,
                    const pthread_barrierattr_t * /*attr*/, unsigned /*count*/) { return 1; }
static int pthread_barrier_destroy(pthread_barrier_t * /*barrier*/) { return 1; }
#endif

// The following values will have to be approved empirically yet... .

// TODO: reasonable determination of these values.

// Redistribute bricks if rendering time deviation > MAX_DEVIATION %
#define MAX_DEVIATION 10.0f
// Don't redistribute bricks for peaks, but rather for constant deviation.
#define LIMIT 10
// Maximum individual deviation (otherwise the share will be adjusted).
#define MAX_IND_DEVIATION 0.001
#define INC 0.05

using namespace std;

//----------------------------------------------------------------------------
const int vvTexRend::NUM_PIXEL_SHADERS = 13;

struct ThreadArgs
{
  int threadId;                               ///< integer id of the thread

  // Algorithm specific.
  vvHalfSpace* halfSpace;
  std::vector<Brick*> sortedList;                ///< sorted list built up from the brick sets
  vvVector3 min;
  vvVector3 max;
  vvVector3 probeMin;
  vvVector3 probeMax;
  vvVector3 delta;
  vvVector3 farthest;
  vvVector3 normal;
  int numSlices;
  float texMin[3];
  int minSlice;
  int maxSlice;
  GLfloat* modelview;                         ///< the current GL_MODELVIEW matrix
  GLfloat* projection;                        ///< the current GL_PROJECTION matrix
  vvTexRend* renderer;                        ///< pointer to the calling instance. useful to use functions from the renderer class
  GLfloat* pixels;                            ///< after rendering each thread will read back its data to this array
  int width;                                  ///< viewport width to init the fbo with
  int height;                                 ///< viewport height to init the fbo with
  float lastRenderTime;                       ///< measured for dynamic load balancing
  float share;                                ///< ... of the volume managed by this thread. Adjustable for load balancing.
  bool transferFunctionChanged;

#ifdef HAVE_X11
  // Glx rendering specific.
  GLXContext glxContext;                      ///< the initial glx context
  Display* display;						      ///< a pointer to the current glx display
  Drawable drawable;
#endif

  // Gl state specific.
  GLuint* privateTexNames;
  int numTextures;
  GLuint pixLUTName;
  uchar* rgbaLUT;

  // Pthread specific.
  pthread_barrier_t* initBarrier;             ///< when this barrier is passed, the main rendering loop may be initially entered
  pthread_barrier_t* distributeBricksBarrier;
  pthread_barrier_t* distributedBricksBarrier;
  pthread_barrier_t* renderStartBarrier;      ///< when this barrier is passed, the main loop may resume rendering
  pthread_barrier_t* compositingBarrier;      ///< when this barrier is passed, compositing is done
  pthread_barrier_t* renderReadyBarrier;      ///< when this barrier is passed, all threads have completed rendering the current frame
  pthread_mutex_t* makeTextureMutex;          ///< mutex ensuring that textures for each thread are built up synchronized
};

void Brick::render(vvTexRend* renderer, const vvVector3& normal,
                   const vvVector3& farthest, const vvVector3& delta,
                   const vvVector3& probeMin, const vvVector3& probeMax,
                   GLuint*& texNames, vvShaderManager* isectShader, const bool setupEdges)
{
  vvVector3 dist = max;
  dist.sub(&min);

  vvVector3 texRange;
  vvVector3 texMin;

  glBindTexture(GL_TEXTURE_3D_EXT, texNames[index]);

  vvVector3 texPoint;
  texPoint.copy(&farthest);

  if(renderer->_proxyGeometryOnGpu)
  {
    glEnableClientState(GL_VERTEX_ARRAY);

    // Clip probe object to brick extends.
    vvVector3 minClipped;
    vvVector3 maxClipped;

    for (int i = 0; i < 3; ++i)
    {
      if (min[i] < probeMin[i])
      {
        minClipped[i] = probeMin[i];
      }
      else
      {
        minClipped[i] = min[i];
      }

      if (max[i] > probeMax[i])
      {
        maxClipped[i] = probeMax[i];
      }
      else
      {
        maxClipped[i] = max[i];
      }
      const float overlapNorm = (float)(brickTexelOverlap[i]) / (float)texels[i];
      texRange[i] = (1.0f - overlapNorm);
      texMin[i] = (1.0f / (2.0f * (float)(renderer->_renderState._brickTexelOverlap) * (float)texels[i]));
    }

    vvAABB box(minClipped, maxClipped);
    const vvVector3 (&verts)[8] = box.calcVertices();

    float minDot;
    float maxDot;
    const ushort idx = getFrontIndex(verts, texPoint, normal, minDot, maxDot);
    isectShader->setArrayParameter3f(0, "vertices", 0, verts[0].e[0], verts[0].e[1], verts[0].e[2]);
    if(setupEdges)
    {
      for (int i = 1; i < 8; ++i)
      {
        isectShader->setArrayParameter3f(0, "vertices", i, verts[i].e[0]-verts[0].e[0],
                                                           verts[i].e[1]-verts[0].e[1],
                                                           verts[i].e[2]-verts[0].e[2]);
      }
    }

    isectShader->setParameter4f(0, "brickMin", min[0], min[1], min[2], -texPoint.length());
    isectShader->setParameter3f(0, "brickDimInv", 1.0f/dist[0], 1.0f/dist[1], 1.0f/dist[2]);
    // Mind that textures overlap a little bit for correct interpolation at the borders.
    // Thus add that little difference.
    isectShader->setParameter3f(0, "texRange", texRange[0], texRange[1], texRange[2]);
    isectShader->setParameter3f(0, "texMin", texMin[0], texMin[1], texMin[2]);

    const float deltaInv = 1.0f / delta.length();

    const int startSlices = (int)ceilf(minDot * deltaInv);
    const int endSlices = (int)floorf(maxDot * deltaInv);

    isectShader->setParameter1i(0, "frontIndex", idx);

    const int primCount = (endSlices - startSlices) + 1;

    glVertexPointer(2, GL_INT, 0, &renderer->_vertArray[startSlices*12]);
    glMultiDrawElements(GL_POLYGON, &renderer->_elemCounts[0], GL_UNSIGNED_INT, (const GLvoid**)&renderer->_vertIndices[0], primCount);

    glDisableClientState(GL_VERTEX_ARRAY);
  }
  else // render proxy geometry on gpu? else then:
  {
    // Clip probe object to brick extends.
    vvVector3 minClipped;
    vvVector3 maxClipped;

    for (int i = 0; i < 3; ++i)
    {
      if (min[i] < probeMin[i])
      {
        minClipped[i] = probeMin[i];
      }
      else
      {
        minClipped[i] = min[i];
      }

      if (max[i] > probeMax[i])
      {
        maxClipped[i] = probeMax[i];
      }
      else
      {
        maxClipped[i] = max[i];
      }
      const float overlapNorm = (float)(brickTexelOverlap[i]) / (float)texels[i];
      texRange[i] = (1.0f - overlapNorm);
      texMin[i] = (1.0f / (2.0f * (float)(renderer->_renderState._brickTexelOverlap) * (float)texels[i]));
    }

    vvAABB box(minClipped, maxClipped);
    const vvVector3 (&verts)[8] = box.calcVertices();

    // Abuse getFrontIndex to calcuate minDot and maxDot.
    float minDot;
    float maxDot;
    getFrontIndex(verts, texPoint, normal, minDot, maxDot);

    const float deltaInv = 1.0f / delta.length();

    const int startSlices = (int)ceilf(minDot * deltaInv);
    const int endSlices = (int)floorf(maxDot * deltaInv);

    vvVector3 startPoint;
    startPoint.copy(farthest);
    vvVector3 add;
    add.copy(delta);
    add.scale(startSlices);
    startPoint.add(&add);

    for (int i = startSlices; i <= endSlices; ++i)
    {
      vvVector3 isect[6];
      const int isectCnt = isect->isectPlaneCuboid(&normal, &startPoint, &minClipped, &maxClipped);
      startPoint.add(&delta);

      if (isectCnt < 3) continue;                 // at least 3 intersections needed for drawing

      // Check volume section mode:
      if (renderer->minSlice != -1 && i < renderer->minSlice) continue;
      if (renderer->maxSlice != -1 && i > renderer->maxSlice) continue;

      // Put the intersecting 3 to 6 vertices in cyclic order to draw adjacent
      // and non-overlapping triangles:
      isect->cyclicSort(isectCnt, &normal);

      // Generate vertices in texture coordinates:
      vvVector3 texcoord[6];
      for (int j = 0; j < isectCnt; ++j)
      {
        for (int k = 0; k < 3; ++k)
        {
          texcoord[j][k] = (isect[j][k] - min.e[k]) / dist[k];
          texcoord[j][k] = texcoord[j][k] * texRange[k] + texMin[k];
        }
      }

      glBegin(GL_TRIANGLE_FAN);
      glColor4f(1.0, 1.0, 1.0, 1.0);
      glNormal3f(normal[0], normal[1], normal[2]);
      for (int j = 0; j < isectCnt; ++j)
      {
        // The following lines are the bottleneck of this method:
        glTexCoord3f(texcoord[j][0], texcoord[j][1], texcoord[j][2]);
        glVertex3f(isect[j][0], isect[j][1], isect[j][2]);
      }
      glEnd();
    }
  }
}

/** Get front index of the brick based upon the current modelview matrix.
  @param vertices  NOTE THE FOLLOWING: albeit vertices are constituents of
                   bricks, these are passed along with the function call in
                   order not to recalculate them. This is due to the fact that
                   AABBs actually don't store pointers to their eight verts,
                   but only to two corners.
  @param point     The point to calc the distance from. The distance of interest
                   is the one from point to the first and last vertex along
                   normal
  @param normal    The normal to calc the distance along.
  @param minDot    The minimum dot product of vector point-vertex and normal,
                   passed along for later calculations.
  @param maxDot    The maximum dot product of vector point-vertex and normal,
                   passed along for later calculations.
*/
ushort Brick::getFrontIndex(const vvVector3* vertices,
                            const vvVector3& point,
                            const vvVector3& normal,
                            float& minDot,
                            float& maxDot)
{

  // Get vertices with max and min distance to point along normal.
  maxDot = -FLT_MAX;
  minDot = FLT_MAX;
  ushort frontIndex;

  for (int i=0; i<8; i++)
  {
    vvVector3 v = vertices[i];
    v.sub(&point);
    const float dot = v.dot(&normal);
    if (dot > maxDot)
    {
      maxDot = dot;
      frontIndex = i;
    }

    if (dot < minDot)
    {
      minDot = dot;
    }
  }
  return frontIndex;
}

vvThreadVisitor::~vvThreadVisitor()
{
  delete[] _pixels;
  delete[] _frameBufferObjects;
  delete[] _imageSpaceTextures;
}

//----------------------------------------------------------------------------
/** Thread visitor visit method. Supplies logic to render results of worker
    thread.
  @param obj  node to render
*/
void vvThreadVisitor::visit(vvVisitable* obj)
{
  // This is rather specific: the visitor knows the thread id
  // of the bsp tree node, looks up the appropriate thread
  // and renders its data.

  vvHalfSpace* hs = dynamic_cast<vvHalfSpace*>(obj);

  glActiveTextureARB(GL_TEXTURE0_ARB);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, _imageSpaceTextures[hs->getId()]);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, _width, _height, 0, GL_RGBA, GL_FLOAT, _pixels[hs->getId()]);

  glBegin(GL_QUADS);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glNormal3f(0.0f, 0.0f, 1.0f);

    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(-1.0f, -1.0f, 0.0f);

    glTexCoord2f(1.0f, 0.0f);
    glVertex3f(1.0f, -1.0f, 0.0f);

    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(1.0f, 1.0f, 0.0f);

    glTexCoord2f(0.0f, 1.0f);
    glVertex3f(-1.0f, 1.0f, 0.0f);
  glEnd();
}

void vvThreadVisitor::setFrameBufferObjects(GLuint*& frameBufferObjects)
{
  _frameBufferObjects = frameBufferObjects;
}

void vvThreadVisitor::setDepthBuffers(GLuint*& depthBuffers)
{
  _depthBuffers = depthBuffers;
}

void vvThreadVisitor::setImageSpaceTextures(GLuint*& imageSpaceTextures)
{
  _imageSpaceTextures = imageSpaceTextures;
}

void vvThreadVisitor::setPixels(GLfloat**& pixels)
{
  _pixels = pixels;
}

void vvThreadVisitor::setWidth(const int width)
{
  _width = width;
}

void vvThreadVisitor::setHeight(const int height)
{
  _height = height;
}

//----------------------------------------------------------------------------
/** Constructor.
  @param vd  volume description
  @param m   render geometry (default: automatic)
*/
vvTexRend::vvTexRend(vvVolDesc* vd, vvRenderState renderState, GeometryType geom, VoxelType vox) : vvRenderer(vd, renderState)
{
  vvDebugMsg::msg(1, "vvTexRend::vvTexRend()");

  setDisplayNames(renderState._displayNames, renderState._numDisplays);
  dispatchThreadedGLXContexts();

  if (vvDebugMsg::isActive(1))
  {
#ifdef _WIN32
    cerr << "_WIN32 is defined" << endl;
#elif WIN32
    cerr << "_WIN32 is not defined, but should be if running under Windows" << endl;
#endif

#ifdef HAVE_CG
    cerr << "HAVE_CG is defined" << endl;
#else
    cerr << "Tip: define HAVE_CG for pixel shader support" << endl;
#endif

    cerr << "Compiler knows OpenGL versions: ";
#ifdef GL_VERSION_1_1
    cerr << "1.1";
#endif
#ifdef GL_VERSION_1_2
    cerr << ", 1.2";
#endif
#ifdef GL_VERSION_1_3
    cerr << ", 1.3";
#endif
#ifdef GL_VERSION_1_4
    cerr << ", 1.4";
#endif
#ifdef GL_VERSION_1_5
    cerr << ", 1.5";
#endif
#ifdef GL_VERSION_2_0
    cerr << ", 2.0";
#endif
    cerr << endl;
  }

  rendererType = TEXREND;
  texNames = NULL;
  _sliceOrientation = VV_VARIABLE;
  viewDir.zero();
  objDir.zero();
  minSlice = maxSlice = -1;
  rgbaTF  = new float[256 * 256 * 4];
  rgbaLUT = new uchar[256 * 256 * 4];
  preintTable = new uchar[getPreintTableSize()*getPreintTableSize()*4];
  preIntegration = false;
  usePreIntegration = false;
  textures = 0;
  opacityCorrection = true;
  interpolation = true;
  _bspTree = NULL;

  if (_renderState._useOffscreenBuffer)
  {
    _renderTarget = new vvOffscreenBuffer(_renderState._imageScale, VV_BYTE);
  }
  else
  {
    _renderTarget = new vvRenderTarget();
  }

  if (vvShaderFactory::isSupported(VV_CG_MANAGER))
  {
    _currentShader = vd->chan - 1;
  }
  else
  {
    _currentShader = 0;
  }
  _useOnlyOneBrick = false;
  _areBricksCreated = false;
  _lastFrame = -1;
  lutDistance = -1.0;
  _renderState._isROIChanged = true;

  // Find out which OpenGL extensions are supported:
#if defined(GL_VERSION_1_2) && defined(__APPLE__)
  extTex3d  = true;
  arbMltTex = true;
#else
  extTex3d  = vvGLTools::isGLextensionSupported("GL_EXT_texture3D");
  arbMltTex = vvGLTools::isGLextensionSupported("GL_ARB_multitexture");
#endif

  // If no Cg is available, the factory will return a NULL pointer.
  _isectShader = vvShaderFactory::provideShaderManager(VV_CG_MANAGER);
  _pixelShader = vvShaderFactory::provideShaderManager(VV_CG_MANAGER);

  extMinMax = vvGLTools::isGLextensionSupported("GL_EXT_blend_minmax");
  extBlendEquation = vvGLTools::isGLextensionSupported("GL_EXT_blend_equation");
  extColLUT = isSupported(VV_SGI_LUT);
  extPalTex = isSupported(VV_PAL_TEX);
  extTexShd = isSupported(VV_TEX_SHD);
  extPixShd = isSupported(VV_PIX_SHD);
  arbFrgPrg = isSupported(VV_FRG_PRG);
  _proxyGeometryOnGpu = extPixShd && arbFrgPrg && _isectShader;

  extNonPower2 = vvGLTools::isGLextensionSupported("GL_ARB_texture_non_power_of_two");

#if !defined(__APPLE__)
  // Init glew.
  glewInit();
#endif

  // Determine best rendering algorithm for current hardware:
  voxelType = findBestVoxelType(vox);
  geomType  = findBestGeometry(geom, voxelType);

  cerr << "Rendering algorithm: ";
  switch(geomType)
  {
    case VV_SLICES:    cerr << "VV_SLICES";  break;
    case VV_CUBIC2D:   cerr << "VV_CUBIC2D";   break;
    case VV_VIEWPORT:  cerr << "VV_VIEWPORT";  break;
    case VV_BRICKS:    cerr << "VV_BRICKS";  break;
    case VV_SPHERICAL: cerr << "VV_SPHERICAL"; break;
    default: assert(0); break;
  }
  cerr << ", ";
  switch(voxelType)
  {
    case VV_RGBA:    cerr << "VV_RGBA";    break;
    case VV_SGI_LUT: cerr << "VV_SGI_LUT"; break;
    case VV_PAL_TEX: cerr << "VV_PAL_TEX"; break;
    case VV_TEX_SHD: cerr << "VV_TEX_SHD"; break;
    case VV_PIX_SHD: cerr << "VV_PIX_SHD"; break;
    case VV_FRG_PRG: cerr << "VV_FRG_PRG"; break;
    default: assert(0); break;
  }
  cerr << endl;

  if (vvShaderFactory::isSupported(VV_CG_MANAGER))
  {
    if(geomType==VV_SLICES || geomType==VV_CUBIC2D)
    {
      _currentShader = 9;
    }
  }

  if (geomType == VV_BRICKS)
  {
    validateEmptySpaceLeaping();
  }

  if ((_numThreads == 0) && (voxelType==VV_TEX_SHD || voxelType==VV_PIX_SHD || voxelType==VV_FRG_PRG))
  {
    glGenTextures(1, &pixLUTName);
  }

  // Init fragment program:
  if((_numThreads == 0) && (voxelType == VV_FRG_PRG))
  {
    glGenProgramsARB(VV_FRAG_PROG_MAX, fragProgName);

    char fragProgString2D[] = "!!ARBfp1.0\n"
      "TEMP temp;\n"
      "TEX  temp, fragment.texcoord[0], texture[0], 2D;\n"
      "TEX  result.color, temp, texture[1], 2D;\n"
      "END\n";
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, fragProgName[VV_FRAG_PROG_2D]);
    glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB,
      GL_PROGRAM_FORMAT_ASCII_ARB,
      strlen(fragProgString2D),
      fragProgString2D);

    char fragProgString3D[] = "!!ARBfp1.0\n"
      "TEMP temp;\n"
      "TEX  temp, fragment.texcoord[0], texture[0], 3D;\n"
      "TEX  result.color, temp, texture[1], 2D;\n"
      "END\n";
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, fragProgName[VV_FRAG_PROG_3D]);
    glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB,
      GL_PROGRAM_FORMAT_ASCII_ARB,
      strlen(fragProgString3D),
      fragProgString3D);

    char fragProgStringPreint[] = "!!ARBfp1.0\n"
      "TEMP temp;\n"
      "TEX  temp.x, fragment.texcoord[0], texture[0], 3D;\n"
      "TEX  temp.y, fragment.texcoord[1], texture[0], 3D;\n"
      "TEX  result.color, temp, texture[1], 2D;\n"
      "END\n";
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, fragProgName[VV_FRAG_PROG_PREINT]);
    glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB,
      GL_PROGRAM_FORMAT_ASCII_ARB,
      strlen(fragProgStringPreint),
      fragProgStringPreint);
  }

  if (voxelType==VV_PIX_SHD)
  {
    if (!initPixelShaders(_pixelShader))
    {
      voxelType = VV_RGBA;
    }
  }

  textures = 0;

  switch(voxelType)
  {
    case VV_FRG_PRG:
      texelsize=1;
      internalTexFormat = GL_LUMINANCE;
      texFormat = GL_LUMINANCE;
      break;
    case VV_PAL_TEX:
      texelsize=1;
      internalTexFormat = GL_COLOR_INDEX8_EXT;
      texFormat = GL_COLOR_INDEX;
      break;
    case VV_TEX_SHD:
    case VV_PIX_SHD:
      texelsize=4;
      internalTexFormat = GL_RGBA;
      texFormat = GL_RGBA;
      break;
    case VV_SGI_LUT:
      texelsize=2;
      internalTexFormat = GL_LUMINANCE_ALPHA;
      texFormat = GL_LUMINANCE_ALPHA;
      break;
    case VV_RGBA:
      internalTexFormat = GL_RGBA;
      texFormat = GL_RGBA;
      texelsize=4;
      break;
    default:
      assert(0);
      break;
  }

  if ((geomType == VV_BRICKS) && _renderState._computeBrickSize)
  {
    _areBricksCreated = false;
    computeBrickSize();
  }

  if (_numThreads == 0)
  {
    if (voxelType != VV_RGBA)
    {
      makeTextures(texNames, &textures, pixLUTName, rgbaLUT);      // we only have to do this once for non-RGBA textures
    }
    updateTransferFunction(pixLUTName, rgbaLUT);
  }
  else
  {
    // Here initialization is done. I.e. the worker threads may start rendering as soon as the
    // brick list is built up properly. Waiting for a properly sorted brick list is assured
    // by another barrier.
    pthread_barrier_wait(&_initBarrier);

    // Each thread will make its own share of textures and texture ids. Don't distribute
    // the bricks before this is completed.
    pthread_barrier_wait(&_distributeBricksBarrier);

    vvVector3 probePosObj;
    vvVector3 probeSizeObj;
    vvVector3 probeMin, probeMax;

    // Build global transfer function once in order to build up _nonemptyList.
    updateTransferFunction(pixLUTName, rgbaLUT);
    calcProbeDims(probePosObj, probeSizeObj, probeMin, probeMax);
    getBricksInProbe(probePosObj, probeSizeObj);
    distributeBricks();
    _somethingChanged = true;
    pthread_barrier_wait(&_distributedBricksBarrier);
  }

  if (_proxyGeometryOnGpu && (_numThreads == 0))
  {
    // Remember that if you bypass the fixed function pipeline, you have to
    // supply matrix state, normals, colors etc. on your own. So if a vertex
    // shader for isect tests is bound, at least a simple frag program is
    // necessary that passes the color through.
    if (voxelType == VV_RGBA)
    {
      initIntersectionShader(_isectShader, _pixelShader);
    }
    else
    {
      initIntersectionShader(_isectShader);
    }
    setupIntersectionParameters(_isectShader);
  }
}

//----------------------------------------------------------------------------
/// Destructor
vvTexRend::~vvTexRend()
{
  vvDebugMsg::msg(1, "vvTexRend::~vvTexRend()");

  delete _renderTarget;

  if (voxelType==VV_FRG_PRG)
  {
    glDeleteProgramsARB(3, fragProgName);
  }
  if (voxelType==VV_FRG_PRG || voxelType==VV_TEX_SHD || voxelType==VV_PIX_SHD)
  {
    glDeleteTextures(1, &pixLUTName);
  }

  if (_numThreads == 0)
  {
    removeTextures();
  }

  delete[] rgbaTF;
  delete[] rgbaLUT;


  if (_proxyGeometryOnGpu)
  {
    glDisableClientState(GL_VERTEX_ARRAY);
  }
  delete _isectShader;
  delete _pixelShader;

  delete[] preintTable;


  for(std::vector<BrickList>::iterator frame = _brickList.begin();
      frame != _brickList.end();
      ++frame)
  {
    for(BrickList::iterator brick = frame->begin(); brick != frame->end(); ++brick)
      delete *brick;
  }

  if (_numThreads > 0)
  {
    // Finally join the threads.
    for (int i = 0; i < _numThreads; ++i)
    {
      void* exitStatus;
      pthread_join(_threads[i], &exitStatus);
    }

    for (int i = 0; i < _numThreads; ++i)
    {
      delete[] _threadData[i].pixels;
      delete[] _threadData[i].rgbaLUT;
    }

    // Cleanup locking specific stuff.
    pthread_barrier_destroy(&_initBarrier);
    pthread_barrier_destroy(&_distributeBricksBarrier);
    pthread_barrier_destroy(&_distributedBricksBarrier);
    pthread_barrier_destroy(&_renderStartBarrier);
    pthread_mutex_destroy(&_makeTextureMutex);

    // Clean up arrays.
    delete[] _threads;
    delete[] _threadData;
    delete[] _frameBufferObjects;
    delete[] _depthBuffers;
    delete[] _imageSpaceTextures;
    delete[] _displayNames;
    delete[] _screens;

    delete _bspTree;
  }
}

//----------------------------------------------------------------------------
/** Chooses the best rendering geometry depending on the graphics hardware's
  capabilities.
  @param geom desired geometry
*/
vvTexRend::GeometryType vvTexRend::findBestGeometry(vvTexRend::GeometryType geom, vvTexRend::VoxelType vox)
{
  (void)vox;
  vvDebugMsg::msg(1, "vvTexRend::findBestGeometry()");

  if (geom==VV_AUTO)
  {
    if (extTex3d) return VV_BRICKS;
    else return VV_SLICES;
  }
  else
  {
    if (!extTex3d && (geom==VV_VIEWPORT || geom==VV_SPHERICAL || geom==VV_BRICKS))
    {
      return VV_SLICES;
    }
    else return geom;
  }
}

//----------------------------------------------------------------------------
/// Chooses the best voxel type depending on the graphics hardware's
/// capabilities.
vvTexRend::VoxelType vvTexRend::findBestVoxelType(vvTexRend::VoxelType vox)
{
  vvDebugMsg::msg(1, "vvTexRend::findBestVoxelType()");

  if (vox==VV_BEST)
  {
    if (vd->chan==1)
    {
      // TODO: make threads compatible with arb fragment program.
      // Until then, Cg pixel shaders are prefered over arbFrgPrg.
      if (_numThreads == 0)
      {
        if (arbFrgPrg) return VV_FRG_PRG;
        else if (extPixShd) return VV_PIX_SHD;
        else if (extTexShd) return VV_TEX_SHD;
        else if (extPalTex) return VV_PAL_TEX;
        else if (extColLUT) return VV_SGI_LUT;
      }
      else
      {
        if (extPixShd) return VV_PIX_SHD;
        else if (arbFrgPrg) return VV_FRG_PRG;
        else if (extTexShd) return VV_TEX_SHD;
        else if (extPalTex) return VV_PAL_TEX;
        else if (extColLUT) return VV_SGI_LUT;
      }
    }
    else
    {
      if (extPixShd) return VV_PIX_SHD;
    }
    return VV_RGBA;
  }
  else
  {
    switch(vox)
    {
      case VV_PIX_SHD: if (extPixShd) return VV_PIX_SHD;
      case VV_FRG_PRG: if (arbFrgPrg && vd->chan==1) return VV_FRG_PRG;
      case VV_TEX_SHD: if (extTexShd && vd->chan==1) return VV_TEX_SHD;
      case VV_PAL_TEX: if (extPalTex && vd->chan==1) return VV_PAL_TEX;
      case VV_SGI_LUT: if (extColLUT && vd->chan==1) return VV_SGI_LUT;
      default: return VV_RGBA;
    }
  }
}

//----------------------------------------------------------------------------
/// Remove all textures from texture memory.
void vvTexRend::removeTextures()
{
  vvDebugMsg::msg(1, "vvTexRend::removeTextures()");

  if (textures>0)
  {
    glDeleteTextures(textures, texNames);
    delete[] texNames;
    texNames = NULL;
    textures = 0;
  }
}

//----------------------------------------------------------------------------
/// Generate textures for all rendering modes.
vvTexRend::ErrorType vvTexRend::makeTextures(GLuint*& privateTexNames, int* numTextures,
                                             GLuint& lutName, uchar*& lutData)
{
  static bool first = true;
  ErrorType err = OK;

  vvDebugMsg::msg(2, "vvTexRend::makeTextures()");

  // Compute texture dimensions (must be power of 2):
  texels[0] = vvToolshed::getTextureSize(vd->vox[0]);
  texels[1] = vvToolshed::getTextureSize(vd->vox[1]);
  texels[2] = vvToolshed::getTextureSize(vd->vox[2]);

  switch (geomType)
  {
    case VV_SLICES:  err=makeTextures2D(1); updateTextures2D(1, 0, 10, 20, 15, 10, 5); break;
    case VV_CUBIC2D: err=makeTextures2D(3); updateTextures2D(3, 0, 10, 20, 15, 10, 5); break;
    // Threads will make their own copies of the textures as well as their own gl ids.
    case VV_BRICKS:  err=makeTextureBricks(privateTexNames, numTextures, lutData); break;
    default: updateTextures3D(0, 0, 0, texels[0], texels[1], texels[2], true); break;
  }
  vvGLTools::printGLError("vvTexRend::makeTextures");

  if (voxelType==VV_PIX_SHD || voxelType==VV_FRG_PRG || voxelType==VV_TEX_SHD)
  {
    //if (first)
    {
      makeLUTTexture(lutName, lutData);               // FIXME: works only once, then generates OpenGL error
      first = false;
    }
  }
  return err;
}

//----------------------------------------------------------------------------
/// Generate texture for look-up table.
void vvTexRend::makeLUTTexture(GLuint& lutName, uchar* lutData)
{
  int size[3];

  vvGLTools::printGLError("enter makeLUTTexture");
  if(voxelType!=VV_PIX_SHD)
     glActiveTextureARB(GL_TEXTURE1_ARB);
  getLUTSize(size);
  glBindTexture(GL_TEXTURE_2D, lutName);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size[0], size[1], 0,
    GL_RGBA, GL_UNSIGNED_BYTE, lutData);
  if(voxelType!=VV_PIX_SHD)
     glActiveTextureARB(GL_TEXTURE0_ARB);
  vvGLTools::printGLError("leave makeLUTTexture");
}

//----------------------------------------------------------------------------
/// Generate 2D textures for cubic 2D mode.
vvTexRend::ErrorType vvTexRend::makeTextures2D(int axes)
{
  GLint glWidth;                                  // return value from OpenGL call
  uchar* rgbaSlice[3];                            // RGBA slice data for texture memory for each principal axis
  int rawVal[4];                                  // raw values for R,G,B,A
  int rawSliceSize;                               // number of bytes in a slice of the raw data array
  int rawLineSize;                                // number of bytes in a row of the raw data array
  int texSize[3];                                 // size of a 2D texture in bytes for each principal axis
  int frames;                                     // sequence timesteps
  uchar* raw;                                     // raw volume data
  int texIndex=0;                                 // index of current texture
  int texSliceIndex;                              // index of current voxel in texture
  int tw[3], th[3];                               // current texture width and height for each principal axis
  int rw[3], rh[3], rs[3];                        // raw data width, height, slices for each principal axis
  int rawStart[3];                                // starting offset into raw data, for each principal axis
  int rawStepW[3];                                // raw data step size for texture row, for each principal axis
  int rawStepH[3];                                // raw data step size for texture column, for each principal axis
  int rawStepS[3];                                // raw data step size for texture slices, for each principal axis
  uchar* rawVoxel;                                // current raw data voxel
  float fval;
  int i, s, w, h, f, c, alpha;
  bool accommodated = true;                       // false if a texture cannot be accommodated in TRAM
  ErrorType err = OK;

  vvDebugMsg::msg(1, "vvTexRend::makeTextures2D()");

  assert(axes==1 || axes==3);

  removeTextures();                               // first remove previously generated textures from TRAM

  frames = vd->frames;
  
  // Determine total number of textures:
  if (axes==1)
  {
    textures = vd->vox[2] * frames;
  }
  else
  {
    textures = (vd->vox[0] + vd->vox[1] + vd->vox[2]) * frames;
  }

  vvDebugMsg::msg(1, "Total number of 2D textures:    ", textures);
  vvDebugMsg::msg(1, "Total size of 2D textures [KB]: ",
    frames * axes * texels[0] * texels[1] * texels[2] * texelsize / 1024);

  // Generate texture names:
  assert(texNames==NULL);
  texNames = new GLuint[textures];
  glGenTextures(textures, texNames);

  // Initialize texture sizes:
  th[1] = tw[2] = texels[0];
  tw[0] = th[2] = texels[1];
  tw[1] = th[0] = texels[2];
  for (i=3-axes; i<3; ++i)
  {
    texSize[i] = tw[i] * th[i] * texelsize;
  }

  // Initialize raw data sizes:
  rs[0] = rh[1] = rw[2] = vd->vox[0];
  rw[0] = rs[1] = rh[2] = vd->vox[1];
  rh[0] = rw[1] = rs[2] = vd->vox[2];
  rawLineSize  = vd->vox[0] * vd->getBPV();
  rawSliceSize = vd->getSliceBytes();

  // Initialize raw data access info:
  rawStart[0] = (vd->vox[2]) * rawSliceSize - vd->getBPV();
  rawStepW[0] = -rawLineSize;
  rawStepH[0] = -rawSliceSize;
  rawStepS[0] = -vd->getBPV();
  rawStart[1] = (vd->vox[2] - 1) * rawSliceSize;
  rawStepW[1] = -rawSliceSize;
  rawStepH[1] = vd->getBPV();
  rawStepS[1] = rawLineSize;
  rawStart[2] = (vd->vox[1] - 1) * rawLineSize;;
  rawStepW[2] = vd->getBPV();
  rawStepH[2] = -rawLineSize;
  rawStepS[2] = rawSliceSize;

  // Generate texture data arrays:
  for (i=3-axes; i<3; ++i)
  {
    rgbaSlice[i] = new uchar[texSize[i]];
  }

  // Generate texture data:
  for (f=0; f<frames; ++f)
  {
    raw = vd->getRaw(f);                          // points to beginning of frame in raw data
    for (i=3-axes; i<3; ++i)                      // generate textures for each principal axis
    {
      memset(rgbaSlice[i], 0, texSize[i]);        // initialize with 0's for invisible empty regions

      // Generate texture contents:
      for (s=0; s<rs[i]; ++s)                     // loop thru texture and raw data slices
      {
        for (h=0; h<rh[i]; ++h)                   // loop thru raw data rows
        {
          // Set voxel to starting position in raw data array:
          rawVoxel = raw + rawStart[i] + s * rawStepS[i] + h * rawStepH[i];

          for (w=0; w<rw[i]; ++w)                 // loop thru raw data columns
          {
            texSliceIndex = texelsize * (w + h * tw[i]);
            if (vd->chan==1 && (vd->bpc==1 || vd->bpc==2 || vd->bpc==4))
            {
              if (vd->bpc == 1)                   // 8 bit voxels
              {
                rawVal[0] = int(*rawVoxel);
              }
              else if (vd->bpc == 2)              // 16 bit voxels
              {
                rawVal[0] = (int(*rawVoxel) << 8) | int(*(rawVoxel+1));
                rawVal[0] >>= 4;                  // make 12 bit LUT index
              }
              else                                // float voxels
              {
                fval = *((float*)(rawVoxel));
                rawVal[0] = vd->mapFloat2Int(fval);
              }
              switch(voxelType)
              {
                case VV_SGI_LUT:
                  rgbaSlice[i][texSliceIndex] = rgbaSlice[i][texSliceIndex+1] = (uchar)rawVal[0];
                  break;
                case VV_PAL_TEX:
                case VV_FRG_PRG:
                  rgbaSlice[i][texSliceIndex] = (uchar)rawVal[0];
                  break;
                case VV_TEX_SHD:
                  for (c=0; c<4; ++c)
                  {
                    rgbaSlice[i][texSliceIndex + c]   = (uchar)rawVal[0];
                  }
                  break;
                case VV_PIX_SHD:
                  rgbaSlice[i][texSliceIndex] = (uchar)rawVal[0];
                  break;
                case VV_RGBA:
                  for (c=0; c<4; ++c)
                  {
                    rgbaSlice[i][texSliceIndex + c] = rgbaLUT[rawVal[0] * 4 + c];
                  }
                  break;
                default: assert(0); break;
              }
            }
            else if (vd->bpc==1)                  // for up to 4 channels
            {
              //XXX: das in Ordnung bringen fuer 2D-Texturen mit LUT
              // Fetch component values from memory:
              for (c=0; c<ts_min(vd->chan,4); ++c)
              {
                rawVal[c] = *(rawVoxel + c);
              }

              // Copy color components:
              for (c=0; c<ts_min(vd->chan,3); ++c)
              {
                rgbaSlice[i][texSliceIndex + c] = (uchar)rawVal[c];
              }

              // Alpha channel:
              if (vd->chan>=4)                    // RGBA?
              {
                rgbaSlice[i][texSliceIndex + 3] = rgbaLUT[rawVal[3] * 4 + 3];
              }
              else                                // compute alpha from color components
              {
                alpha = 0;
                for (c=0; c<vd->chan; ++c)
                {
                  // Alpha: mean of sum of RGB conversion table results:
                  alpha += (int)rgbaLUT[rawVal[c] * 4 + c];
                }
                rgbaSlice[i][texSliceIndex + 3] = (uchar)(alpha / vd->chan);
              }
            }
            rawVoxel += rawStepW[i];
          }
        }
        glBindTexture(GL_TEXTURE_2D, texNames[texIndex]);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

        // Only load texture if it can be accommodated:
        glTexImage2D(GL_PROXY_TEXTURE_2D, 0, internalTexFormat,
          tw[i], th[i], 0, texFormat, GL_UNSIGNED_BYTE, NULL);
        glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &glWidth);
        if (glWidth!=0)
        {
          glTexImage2D(GL_TEXTURE_2D, 0, internalTexFormat,
            tw[i], th[i], 0, texFormat, GL_UNSIGNED_BYTE, rgbaSlice[i]);
        }
        else
        {
          accommodated = false;
        }
        ++texIndex;
      }
    }
  }

  if (accommodated==false)
    cerr << "Insufficient texture memory for" << (axes==3?" cubic ":" ") << "2D textures." << endl;
  err = TRAM_ERROR;

  assert(texIndex == textures);
  for (i=3-axes; i<3; ++i)
    delete[] rgbaSlice[i];
  return err;
}

vvTexRend::ErrorType vvTexRend::makeEmptyBricks()
{
  // TODO: separate generation of brick hulls from uploading brick textures.
  return OK;
}

vvTexRend::ErrorType vvTexRend::makeTextureBricks(GLuint*& privateTexNames, int* numTextures, uchar*& lutData)
{
  ErrorType err = OK;
  GLint glWidth;                                  // return value from OpenGL call
  uchar* texData;                                 // data for texture memory
  uchar* raw;                                     // raw volume data
  vvVector3 voxSize;                              // size of a voxel
  vvVector3 halfBrick;                            // middle of the current brick in texels
  vvVector3 halfVolume;                           // middle of the volume in voxel
  float fval;                                     // floating point voxel value
  int rawVal[4];                                  // raw values for R,G,B,A
  int frames;
  int tmpTexels[3];                               // number of texels in each dimension for current brick
  int texSize;                                    // texture size in bytes
  int sliceSize;                                  // number of voxels in a raw slice
  int startOffset[3];                             // offset to first voxel of current brick
  int texOffset;                                  // currently processed texel [texels]
  int texLineOffset;                              // index into currently processed line of texture data array [texels]
  int rawSliceOffset;                             // slice offset in raw data
  int heightOffset, srcIndex;
  int bx, by, bz, x, y, s, f, c, alpha, texIndex;
  bool accommodated = true;                       // false if a texture cannot be accommodated in TRAM
  Brick* currBrick;

  if (!extTex3d) return NO3DTEX;

  if (_renderState._brickSize == 0)
  {
    vvDebugMsg::msg(1, "3D Texture brick size unknown");
    return err;
  }

  if (_numThreads == 0)
  {
    // TODO: make removeTextures() compatible with "privateTexNames"
    removeTextures();
  }

  frames = vd->frames;

  for(std::vector<BrickList>::iterator frame = _brickList.begin();
      frame != _brickList.end();
      ++frame)
  {
    for(BrickList::iterator brick = frame->begin(); brick != frame->end(); ++brick)
      delete *brick;
    frame->clear();
  }
  _brickList.clear();

  // compute number of texels / per brick (should be of power 2)
  texels[0] = vvToolshed::getTextureSize(_renderState._brickSize[0]);
  texels[1] = vvToolshed::getTextureSize(_renderState._brickSize[1]);
  texels[2] = vvToolshed::getTextureSize(_renderState._brickSize[2]);

  // compute number of bricks
  if ((_useOnlyOneBrick) ||
    ((texels[0] == vd->vox[0]) && (texels[1] == vd->vox[1]) && (texels[2] == vd->vox[2])))
    _numBricks[0] = _numBricks[1] = _numBricks[2] = 1;

  else
  {
    _numBricks[0] = (int) ceil((float) (vd->vox[0]) / (float) (_renderState._brickSize[0]));
    _numBricks[1] = (int) ceil((float) (vd->vox[1]) / (float) (_renderState._brickSize[1]));
    _numBricks[2] = (int) ceil((float) (vd->vox[2]) / (float) (_renderState._brickSize[2]));
  }

  // number of textures needed
  *numTextures = frames * _numBricks[0] * _numBricks[1] * _numBricks[2];

  texSize = texels[0] * texels[1] * texels[2] * texelsize;

  vvDebugMsg::msg(1, "3D Texture (bricking) width     = ", texels[0]);
  vvDebugMsg::msg(1, "3D Texture (bricking) height    = ", texels[1]);
  vvDebugMsg::msg(1, "3D Texture (bricking) depth     = ", texels[2]);
  vvDebugMsg::msg(1, "3D Texture (bricking) size (KB) = ", texSize / 1024);

  texData = new uchar[texSize];

  sliceSize = vd->getSliceBytes();

  vvDebugMsg::msg(2, "Creating texture names. # of names: ", *numTextures);
  privateTexNames = new GLuint[*numTextures];
  glGenTextures(*numTextures, privateTexNames);

  // generate textures contents:
  vvDebugMsg::msg(2, "Transferring textures to TRAM. Total size [KB]: ",
    *numTextures * texSize / 1024);

  // helper variables
  voxSize = vd->getSize();
  voxSize[0] /= (vd->vox[0]-1);
  voxSize[1] /= (vd->vox[1]-1);
  voxSize[2] /= (vd->vox[2]-1);

  halfBrick.set(float(texels[0]-_renderState._brickTexelOverlap),
                float(texels[1]-_renderState._brickTexelOverlap),
                float(texels[2]-_renderState._brickTexelOverlap));
  halfBrick.scale(0.5);

  halfVolume.set(float(vd->vox[0]), float(vd->vox[1]), float(vd->vox[2]));
  halfVolume.sub(1.0);
  halfVolume.scale(0.5);

  _brickList.resize(frames);
  for (f = 0; f < frames; f++)
  {
    raw = vd->getRaw(f);

    for (bx = 0; bx < _numBricks[0]; bx++)
      for (by = 0; by < _numBricks[1]; by++)
        for (bz = 0; bz < _numBricks[2]; bz++)
        {
          // Memorize the max and min scalar values in the volume. These are stored
          // to perform empty space leaping later on.
          int minValue = INT_MAX;
          int maxValue = INT_MIN;

          startOffset[0] = bx * _renderState._brickSize[0];
          startOffset[1] = by * _renderState._brickSize[1];
          startOffset[2] = bz * _renderState._brickSize[2];

          // Guarantee that startOffset[i] + brickSize[i] won't exceed actual size of the volume.
          if ((startOffset[0] + _renderState._brickSize[0]) >= vd->vox[0])
            tmpTexels[0] = vvToolshed::getTextureSize(vd->vox[0] - startOffset[0]);
          else
            tmpTexels[0] = texels[0];
          if ((startOffset[1] + _renderState._brickSize[1]) >= vd->vox[1])
            tmpTexels[1] = vvToolshed::getTextureSize(vd->vox[1] - startOffset[1]);
          else
            tmpTexels[1] = texels[1];
          if ((startOffset[2] + _renderState._brickSize[2]) >= vd->vox[2])
            tmpTexels[2] = vvToolshed::getTextureSize(vd->vox[2] - startOffset[2]);
          else
            tmpTexels[2] = texels[2];

          memset(texData, 0, texSize);

          // Essentially: for s :=  startOffset[2] to (startOffset[2] + texels[2]) do
          for (s = startOffset[2]; (s < (startOffset[2] + tmpTexels[2])) && (s < vd->vox[2]); s++)
          {
            if (s < 0) continue;
            rawSliceOffset = (vd->vox[2] - s - 1) * sliceSize;
            // Essentially: for y :=  startOffset[1] to (startOffset[1] + texels[1]) do
            for (y = startOffset[1]; (y < (startOffset[1] + tmpTexels[1])) && (y < vd->vox[1]); y++)
            {
              if (y < 0) continue;
              heightOffset = (vd->vox[1] - y - 1) * vd->vox[0] * vd->bpc * vd->chan;
              texLineOffset = (y - startOffset[1]) * tmpTexels[0] + (s - startOffset[2]) * tmpTexels[0] * tmpTexels[1];
              if (vd->chan == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
              {
                // Essentially: for x :=  startOffset[0] to (startOffset[0] + texels[0]) do
                for (x = startOffset[0]; (x < (startOffset[0] + tmpTexels[0])) && (x < vd->vox[0]); x++)
                {
                  if (x < 0) continue;
                  srcIndex = vd->bpc * x + rawSliceOffset + heightOffset;
                  if (vd->bpc == 1) rawVal[0] = (int) raw[srcIndex];
                  else if (vd->bpc == 2)
                  {
                    rawVal[0] = ((int) raw[srcIndex] << 8) | (int) raw[srcIndex + 1];
                    rawVal[0] >>= 4;
                  }
                  else  // vd->bpc == 4
                  {
                    fval = *((float*)(raw + srcIndex));      // fetch floating point data value
                    rawVal[0] = vd->mapFloat2Int(fval);
                  }

                  // Store min and max for empty space leaping.
                  if (rawVal[0] > maxValue)
                  {
                    maxValue = rawVal[0];
                  }
                  if (rawVal[0] < minValue)
                  {
                    minValue = rawVal[0];
                  }

                  texOffset = (x - startOffset[0]) + texLineOffset;
                  switch (voxelType)
                  {
                    case VV_SGI_LUT:
                      texData[2*texOffset] = texData[2*texOffset + 1] = (uchar) rawVal[0];
                      break;
                    case VV_PAL_TEX:
                    case VV_FRG_PRG:
                      texData[texOffset] = (uchar) rawVal[0];
                      break;
                    case VV_TEX_SHD:
                      for (c = 0; c < 4; c++)
                        texData[4 * texOffset + c] = (uchar) rawVal[0];
                      break;
                    case VV_PIX_SHD:
                      texData[4 * texOffset] = (uchar) rawVal[0];
                      break;
                    case VV_RGBA:
                      for (c = 0; c < 4; c++)
                        texData[4 * texOffset + c] = lutData[rawVal[0] * 4 + c];
                      break;
                    default:
                      assert(0);
                      break;
                  }
                }
              }
              else if (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4)
              {
                if (voxelType == VV_RGBA || voxelType == VV_PIX_SHD)
                {
                  for (x = startOffset[0]; (x < (startOffset[0] + tmpTexels[0])) && (x < vd->vox[0]); x++)
                  {
                    if (x < 0) continue;

                    texOffset = (x - startOffset[0]) + texLineOffset;

                    // fetch component values from memory:
                    for (c = 0; c < ts_min(vd->chan, 4); c++)
                    {
                      srcIndex = rawSliceOffset + heightOffset + vd->bpc * (x * vd->chan + c);
                      if (vd->bpc == 1)
                        rawVal[c] = (int) raw[srcIndex];
                      else if (vd->bpc == 2)
                      {
                        rawVal[c] = ((int) raw[srcIndex] << 8) | (int) raw[srcIndex + 1];
                        rawVal[c] >>= 4;
                      }
                     else  // vd->bpc==4
                      {
                        fval = *((float*) (raw + srcIndex));
                        rawVal[c] = vd->mapFloat2Int(fval);
                      }
                    }

                    // TODO: init empty-space leaping minValue and maxValue for multiple channels as well.


                    // copy color components:
                    for (c = 0; c < ts_min(vd->chan, 3); c++)
                    {
                      texData[4 * texOffset + c] = (uchar) rawVal[c];
                    }
                
                    // alpha channel
                    if (vd->chan >= 4)  // RGBA
                    {
                      if (voxelType == VV_RGBA)
                        texData[4 * texOffset + 3] = lutData[rawVal[3] * 4 + 3];
                      else
                        texData[4 * texOffset + 3] = (uchar) rawVal[3];
                    }
                    else if(vd->chan > 0) // compute alpha from color components
                    {
                      alpha = 0;
                      for (c = 0; c < vd->chan; c++)
                      {
                        // alpha: mean of sum of RGB conversion table results:
                        alpha += (int) lutData[rawVal[c] * 4 + c];
                      }
                      texData[4 * texOffset + 3] = (uchar) (alpha / vd->chan);
                    }
                    else
                    {
                      texData[4 * texOffset + 3] = 0;
                    }
                  }
                }
              }
              else cerr << "Cannot create texture: unsupported voxel format (3)." << endl;
            } // startoffset[1] to startoffset[1]+tmpTexels[1]
          } // startoffset[2] to startoffset[2]+tmpTexels[2]

          texIndex = (f * _numBricks[0] * _numBricks[1] * _numBricks[2]) + (bx * _numBricks[2] * _numBricks[1])
            + (by * _numBricks[2]) + bz;

          currBrick = new Brick();

          currBrick->minValue = minValue;
          currBrick->maxValue = maxValue;
          currBrick->visible = true;

          int bs[3];
          bs[0] = _renderState._brickSize[0];
          bs[1] = _renderState._brickSize[1];
          bs[2] = _renderState._brickSize[2];
          if (_useOnlyOneBrick)
          {
            bs[0] += _renderState._brickTexelOverlap;
            bs[1] += _renderState._brickTexelOverlap;
            bs[2] += _renderState._brickTexelOverlap;
          }

          bool atBorder = false;
          for (int d = 0; d < 3; ++d)
          {
            currBrick->brickTexelOverlap[d] = _renderState._brickTexelOverlap;
            const float maxObj = (startOffset[d] + bs[d]) * vd->dist[d] * vd->_scale;
            if (maxObj > vd->getSize()[d])
            {
              atBorder = true;
              currBrick->brickTexelOverlap[d] = 0;
            }
          }
          currBrick->atBorder= atBorder;

          currBrick->index = texIndex;
          currBrick->pos.set(vd->pos[0] + voxSize[0] * (startOffset[0] + halfBrick[0] - halfVolume[0]),
            vd->pos[1] + voxSize[1] * (startOffset[1] + halfBrick[1] - halfVolume[1]),
            vd->pos[2] + voxSize[2] * (startOffset[2] + halfBrick[2] - halfVolume[2]));
          currBrick->min.set(vd->pos[0] + voxSize[0] * (startOffset[0] - halfVolume[0]),
            vd->pos[1] + voxSize[1] * (startOffset[1] - halfVolume[1]),
            vd->pos[2] + voxSize[2] * (startOffset[2] - halfVolume[2]));
          currBrick->max.set(vd->pos[0] + voxSize[0] * (startOffset[0] + (tmpTexels[0] - currBrick->brickTexelOverlap[0]) - halfVolume[0]),
            vd->pos[1] + voxSize[1] * (startOffset[1] + (tmpTexels[1] - currBrick->brickTexelOverlap[1]) - halfVolume[1]),
            vd->pos[2] + voxSize[2] * (startOffset[2] + (tmpTexels[2] - currBrick->brickTexelOverlap[2]) - halfVolume[2]));

          for (int d = 0; d < 3; ++d)
          {
            if (currBrick->max[d] > vd->getSize()[d])
            {
              currBrick->max[d] = vd->getSize()[d];
            }
          }

          currBrick->texels[0] = tmpTexels[0];
          currBrick->texels[1] = tmpTexels[1];
          currBrick->texels[2] = tmpTexels[2];
          currBrick->startOffset[0] = startOffset[0];
          currBrick->startOffset[1] = startOffset[1];
          currBrick->startOffset[2] = startOffset[2];

          _brickList[f].push_back(currBrick);

          glBindTexture(GL_TEXTURE_3D_EXT, privateTexNames[texIndex]);

          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
          glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
          glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
          glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
          glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP);
          glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP);
          glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP);

          glTexImage3D(GL_PROXY_TEXTURE_3D_EXT, 0, internalTexFormat,
            tmpTexels[0], tmpTexels[1], tmpTexels[2], 0, texFormat, GL_UNSIGNED_BYTE, NULL);

          glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D_EXT, 0, GL_TEXTURE_WIDTH, &glWidth);
          if (glWidth != 0)
          {
            glTexImage3D(GL_TEXTURE_3D_EXT, 0, internalTexFormat, tmpTexels[0], tmpTexels[1], tmpTexels[2], 0,
              texFormat, GL_UNSIGNED_BYTE, texData);
          }
          else accommodated = false;
        } // # foreach (numBricks[i])
  } // # frames

  if (!accommodated)
  {
    cerr << "Insufficient texture memory for 3D textures." << endl;
    err = TRAM_ERROR;
  }

  delete[] texData;
  _areBricksCreated = true;

  return err;
}

vvTexRend::ErrorType vvTexRend::setDisplayNames(const char** displayNames, const unsigned int numNames)
{
  ErrorType err = OK;

  // Displays specified via _renderState?
  if ((numNames <= 0) || (displayNames == NULL))
  {
    _numThreads = 0;
    return NO_DISPLAYS_SPECIFIED;
  }

  bool hostSeen;
  int i;
  size_t j;

  _numThreads = numNames;

  _threads = new pthread_t[_numThreads];
  _threadData = new ThreadArgs[_numThreads];
  _displayNames = new const char*[_numThreads];
  _screens = new unsigned int[_numThreads];

  for (i = 0; i < _numThreads; ++i)
  {
    _displayNames[i] = displayNames[i];
    hostSeen = false;

    // Parse the display name string for host, display and screen.
    for (j = 0; j < strlen(_displayNames[i]); ++j)
    {
      if (_displayNames[i][j] == ':')
      {
        hostSeen = true;
      }

      // . could also be part of the host name ==> ensure that host was parsed.
      if (hostSeen && (_displayNames[i][j] == '.'))
      {
        ++j;
        _screens[i] = vvToolshed::parseNextUint32(_displayNames[i], j);
      }
    }
  }

  return err;
}

vvTexRend::ErrorType vvTexRend::dispatchThreadedGLXContexts()
{
#ifndef HAVE_X11
  return UNSUPPORTED;
#else
  ErrorType err = OK;
  int i;
  int attrList[] = { GLX_RGBA , GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8, None};

  for (i = 0; i < _numThreads; ++i)
  {
    _threadData[i].display = XOpenDisplay(_displayNames[i]);
    Drawable parent = RootWindow(_threadData[i].display, _screens[i]);

    XVisualInfo* vi = glXChooseVisual(_threadData[i].display,
                                      DefaultScreen(_threadData[i].display),
                                      attrList);

    XSetWindowAttributes wa = { 0 };
    wa.colormap = XCreateColormap(_threadData[i].display, parent, vi->visual, AllocNone );
    wa.background_pixmap = None;
    wa.border_pixel = 0;

    wa.event_mask =  (StructureNotifyMask | ExposureMask | KeyPressMask | KeyReleaseMask
                      | EnterWindowMask | LeaveWindowMask | FocusChangeMask | ButtonMotionMask
                      | PointerMotionMask | ButtonPressMask | ButtonReleaseMask);
    _threadData[i].glxContext = glXCreateContext(_threadData[i].display, vi, NULL, GL_TRUE);
    _threadData[i].drawable = XCreateWindow(_threadData[i].display, parent, 0, 0, 512, 512, 0,
                                            vi->depth, InputOutput, vi->visual,
                                            CWBackPixmap|CWBorderPixel|CWEventMask|CWColormap, &wa );
  }

  // If smth changed, the bricks will be distributed among the threads before rendering.
  // Obviously, initially something changed.
  _somethingChanged = true;

  // When this barrier is unblocked (by the main thread), initialization is done. NUM_THREADS + 1,
  // since every worker thread, as well as the main thread during sequential execution will
  // call pthread_barrier_wait on _initBarrier exactly once.
  pthread_barrier_init(&_initBarrier, NULL, _numThreads + 1);

  // First a barrier is passed that ensures that each thread built its textures (_distributeBricksBarrier).
  // The following barrier (_distributedBricksBarrier) is passed right after each thread
  // was assigned the bricks it is responsible for.
  pthread_barrier_init(&_distributeBricksBarrier, NULL, _numThreads + 1);
  pthread_barrier_init(&_distributedBricksBarrier, NULL, _numThreads + 1);

  // This barrier will be waited for by every worker thread and additionally once by the main
  // thread for every frame.
  pthread_barrier_init(&_renderStartBarrier, NULL, _numThreads + 1);

  // This mutex is used to ensure that each thread will built up its 3D texture one after another.
  pthread_mutex_init(&_makeTextureMutex, NULL);

  // Frame buffer object for render to texture functionality.
  _frameBufferObjects = new GLuint[_numThreads];

  // Depth buffers for the frame buffer objects.
  _depthBuffers = new GLuint[_numThreads];

  // One texture per context, blended during compositing.
  _imageSpaceTextures = new GLuint[_numThreads];

  vvGLTools::Viewport viewport = vvGLTools::getViewport();
  for (i = 0; i < _numThreads; ++i)
  {
    _threadData[i].threadId = i;
    _threadData[i].renderer = this;
    _threadData[i].initBarrier = &_initBarrier;
    _threadData[i].distributeBricksBarrier = &_distributeBricksBarrier;
    _threadData[i].distributedBricksBarrier = &_distributedBricksBarrier;
    _threadData[i].renderStartBarrier = &_renderStartBarrier;
    _threadData[i].makeTextureMutex = &_makeTextureMutex;
    // Start solution for load balancing: every thread renders 1/n of the volume.
    // During load balancing, the share will (probably) be adjusted.
    _threadData[i].share = 1.0f / static_cast<float>(_numThreads);
    _threadData[i].width = viewport.values[2];
    _threadData[i].height = viewport.values[3];
    _threadData[i].pixels = new GLfloat[viewport.values[2] * viewport.values[3] * 4];
    _threadData[i].transferFunctionChanged = false;
    glGenTextures(1, &_imageSpaceTextures[i]);
    _threadData[i].rgbaLUT = new uchar[256 * 256 * 4];
    XMapWindow(_threadData[i].display, _threadData[i].drawable);
    XFlush(_threadData[i].display);
  }

  // Dispatch the threads. This has to be done in a separate loop since the first operation
  // the callback function will perform is making its gl context current. This would most
  // probably interfer with the context creation performed in the loop above.
  for (i = 0; i < _numThreads; ++i)
  {
    pthread_create(&_threads[i], NULL, threadFuncTexBricks, (void*)&_threadData[i]);
  }
  return err;
#endif
}

/*!
 *
 */
vvTexRend::ErrorType vvTexRend::distributeBricks()
{
  ErrorType err = OK;

  int i;

  // A new game... .
  _deviationExceedCnt = 0;

  float* part = new float[_numThreads];
  for (i = 0; i < _numThreads; ++i)
  {
    part[i] = _threadData[i].share;
  }

  delete _bspTree;
  _bspTree = new vvBspTree(part, _numThreads, &_insideList);
  // The visitor (supplies rendering logic) must have knowledge
  // about the texture ids of the compositing polygons.
  // Thus provide a pointer to these.
  vvThreadVisitor* visitor = new vvThreadVisitor;
  visitor->setFrameBufferObjects(_frameBufferObjects);
  visitor->setImageSpaceTextures(_imageSpaceTextures);

  // Provide the visitor with the pixel data of each thread either.
  GLfloat** pixels = new GLfloat*[_numThreads];
  for (i = 0; i < _numThreads; ++i)
  {
    pixels[i] = _threadData[i].pixels;
  }
  visitor->setPixels(pixels);

  vvGLTools::Viewport viewport = vvGLTools::getViewport();
  visitor->setWidth(viewport.values[2]);
  visitor->setHeight(viewport.values[3]);

  _bspTree->setVisitor(visitor);

  delete[] part;

  i = 0;
  for(std::vector<vvHalfSpace *>::const_iterator it = _bspTree->getLeafs()->begin();
      it != _bspTree->getLeafs()->end();
      ++it)
  {
    _threadData[i].halfSpace = *it;
    _threadData[i].halfSpace->setId(_threadData[i].threadId);
    ++i;
    if (i >= _numThreads) break;
  }

  _somethingChanged = false;

  return err;
}

void vvTexRend::updateBrickGeom()
{
  int c, f;
  Brick* tmp;
  vvVector3 voxSize;
  vvVector3 halfBrick;
  vvVector3 halfVolume;

  for (f = 0; f < _brickList.size(); f++)
  {
    // help variables
    voxSize = vd->getSize();
    voxSize[0] /= (vd->vox[0]-1);
    voxSize[1] /= (vd->vox[1]-1);
    voxSize[2] /= (vd->vox[2]-1);

    halfBrick.set(float(texels[0]-1), float(texels[1]-1), float(texels[2]-1));
    halfBrick.scale(0.5);

    halfVolume.set(float(vd->vox[0]), float(vd->vox[1]), float(vd->vox[2]));
    halfVolume.sub(1.0);
    halfVolume.scale(0.5);

    for (c = 0; c < _brickList[f].size(); c++)
    {
      tmp = _brickList[f][c];
      tmp->pos.set(vd->pos[0] + voxSize[0] * (tmp->startOffset[0] + halfBrick[0] - halfVolume[0]),
        vd->pos[1] + voxSize[1] * (tmp->startOffset[1] + halfBrick[1] - halfVolume[1]),
        vd->pos[2] + voxSize[2] * (tmp->startOffset[2] + halfBrick[2] - halfVolume[2]));
      tmp->min.set(vd->pos[0] + voxSize[0] * (tmp->startOffset[0] - halfVolume[0]),
        vd->pos[1] + voxSize[1] * (tmp->startOffset[1] - halfVolume[1]),
        vd->pos[2] + voxSize[2] * (tmp->startOffset[2] - halfVolume[2]));
      tmp->max.set(vd->pos[0] + voxSize[0] * (tmp->startOffset[0] + (tmp->texels[0] - 1) - halfVolume[0]),
        vd->pos[1] + voxSize[1] * (tmp->startOffset[1] + (tmp->texels[1] - 1) - halfVolume[1]),
        vd->pos[2] + voxSize[2] * (tmp->startOffset[2] + (tmp->texels[2] - 1) - halfVolume[2]));
    }
  }
}

void vvTexRend::setShowBricks(bool flag)
{
  _renderState._showBricks = flag;
}

bool vvTexRend::getShowBricks()
{
  return _renderState._showBricks;
}

void vvTexRend::setComputeBrickSize(const bool flag)
{
  _renderState._computeBrickSize = flag;
  if (_renderState._computeBrickSize)
  {
    computeBrickSize();
    if(!_areBricksCreated)
    {
      if (_numThreads > 0)
      {
        for (int i = 0; i < _numThreads; ++i)
        {
          makeTextures(_threadData[i].privateTexNames, &_threadData[i].numTextures,
                       _threadData[i].pixLUTName, _threadData[i].rgbaLUT);
        }
      }
      else
      {
        makeTextures(texNames, &textures, pixLUTName, rgbaLUT);
      }
    }
  }
}

bool vvTexRend::getComputeBrickSize()
{
  return _renderState._computeBrickSize;
}

void vvTexRend::setBrickSize(int newSize)
{
  vvDebugMsg::msg(3, "vvRenderer::setBricksize()");
  _renderState._brickSize[0] = _renderState._brickSize[1] = _renderState._brickSize[2] = newSize-1;
  _useOnlyOneBrick = false;

  if (_numThreads > 0)
  {
    for (int i = 0; i < _numThreads; ++i)
    {
      makeTextures(_threadData[i].privateTexNames, &_threadData[i].numTextures,
                   _threadData[i].pixLUTName, _threadData[i].rgbaLUT);
    }
  }
  else
  {
    makeTextures(texNames, &textures, pixLUTName, rgbaLUT);
  }
}

int vvTexRend::getBrickSize()
{
  vvDebugMsg::msg(3, "vvRenderer::getBricksize()");
  return _renderState._brickSize[0]+1;
}

void vvTexRend::setTexMemorySize(int newSize)
{
  if (_renderState._texMemorySize == newSize)
    return;

  _renderState._texMemorySize = newSize;
  if (_renderState._computeBrickSize)
  {
    computeBrickSize();

    if(!_areBricksCreated)
    {
      if (_numThreads > 0)
      {
        for (int i = 0; i < _numThreads; ++i)
        {
          makeTextures(_threadData[i].privateTexNames, &_threadData[i].numTextures,
                       _threadData[i].pixLUTName, _threadData[i].rgbaLUT);
        }
      }
      else
      {
        makeTextures(texNames, &textures, pixLUTName, rgbaLUT);
      }
    }
  }
}

int vvTexRend::getTexMemorySize()
{
  return _renderState._texMemorySize;
}

void vvTexRend::computeBrickSize()
{
  vvVector3 probeSize;
  int newBrickSize[3];

  int texMemorySize = _renderState._texMemorySize;
  if (texMemorySize == 0)
  {
     vvDebugMsg::msg(1, "vvTexRend::computeBrickSize(): unknown texture memory size, assuming 32 M");
     texMemorySize = 32;
  }

  if (texMemorySize == 0)
  {
    _renderState._brickSize[0] = _renderState._brickSize[1] = _renderState._brickSize[2] = 0;
    return;
  }

  const int maxBrickSize[3] = { 64, 64, 64 };

  // this has been tested to be optimal on a Quadro FX570m
  _useOnlyOneBrick = true;
  for(int i=0; i<3; ++i)
  {
    newBrickSize[i] = vvToolshed::getTextureSize(vd->vox[i]);
    if(newBrickSize[i] > maxBrickSize[i])
    {
      newBrickSize[i] = maxBrickSize[i];
      _useOnlyOneBrick = false;
    }
  }

  if(_useOnlyOneBrick)
  {
    setROIEnable(false);
  }
  else
  {
    probeSize[0] = 2 * (newBrickSize[0]-_renderState._brickTexelOverlap) / (float) vd->vox[0];
    probeSize[1] = 2 * (newBrickSize[1]-_renderState._brickTexelOverlap) / (float) vd->vox[1];
    probeSize[2] = 2 * (newBrickSize[2]-_renderState._brickTexelOverlap) / (float) vd->vox[2];

    setProbeSize(&probeSize);
    //setROIEnable(true);
  }
  if (newBrickSize[0]-_renderState._brickTexelOverlap != _renderState._brickSize[0]
      || newBrickSize[1]-_renderState._brickTexelOverlap != _renderState._brickSize[1]
      || newBrickSize[2]-_renderState._brickTexelOverlap != _renderState._brickSize[2]
      || !_areBricksCreated)
  {
    _renderState._brickSize[0] = newBrickSize[0]-_renderState._brickTexelOverlap;
    _renderState._brickSize[1] = newBrickSize[1]-_renderState._brickTexelOverlap;
    _renderState._brickSize[2] = newBrickSize[2]-_renderState._brickTexelOverlap;
    _areBricksCreated = false;
  }
}


//----------------------------------------------------------------------------
/// Update transfer function from volume description.
void vvTexRend::updateTransferFunction()
{
  if (_numThreads == 0)
  {
    updateTransferFunction(pixLUTName, rgbaLUT);
  }
  else
  {
    for (int i = 0; i < _numThreads; ++i)
    {
      _threadData[i].transferFunctionChanged = true;
    }
  }
}

void vvTexRend::updateTransferFunction(GLuint& lutName, uchar*& lutData)
{
  int size[3];

  vvDebugMsg::msg(1, "vvTexRend::updateTransferFunction()");
  if (preIntegration &&
      arbMltTex && 
      geomType==VV_VIEWPORT && 
      !(_renderState._clipMode && (_renderState._clipSingleSlice || _renderState._clipOpaque)) &&
      (voxelType==VV_FRG_PRG || (voxelType==VV_PIX_SHD && (_currentShader==0 || _currentShader==11))))
  {
    usePreIntegration = true;
    if(_currentShader==0)
      _currentShader = 11;
  }
  else
  {
    usePreIntegration = false;
    if(_currentShader==11)
      _currentShader = 0;
  }

  // Generate arrays from pins:
  getLUTSize(size);
  vd->computeTFTexture(size[0], size[1], size[2], rgbaTF);

  if(!instantClassification())
    updateLUT(1.0f, lutName, lutData);                                // generate color/alpha lookup table
  else
    lutDistance = -1.;                              // invalidate LUT

  // No space leaping per thread, but only once. Otherwise brick list setup would collide.
  const bool calledByWorkerThread = (lutName != pixLUTName);
  if (!calledByWorkerThread)
  {
    // Each bricks visible flag was initially set to true.
    // If empty-space leaping isn't active, all bricks are
    // visible by default.
    if (_renderState._emptySpaceLeaping)
    {
      int nbricks = 0, nvis=0;
      _nonemptyList.clear();
      _nonemptyList.resize(_brickList.size());
      // Determine visibility of each single brick in all frames
      for (int frame = 0; frame < _brickList.size(); ++frame)
      {
         for (BrickList::iterator it = _brickList[frame].begin(); it != _brickList[frame].end(); ++it)
         {
            Brick *tmp = *it;
            nbricks++;

            // If max intensity projection, make all bricks visible.
            if (_renderState._mipMode > 0)
            {
               _nonemptyList[frame].push_back(tmp);
               nvis++;
            }
            else
            {
               for (int i = tmp->minValue; i <= tmp->maxValue; ++i)
               {
                  if(rgbaTF[i * 4 + 3] > 0.)
                  {
                     _nonemptyList[frame].push_back(tmp);
                     nvis++;
                     break;
                  }
               }
            }
         }
      }
    }
    else
    {
      _nonemptyList = _brickList;
    }
  }
  else
  {
    _nonemptyList = _brickList;
  }

  _renderState._isROIChanged = true; // have to update list of visible bricks
}

//----------------------------------------------------------------------------
// see parent in vvRenderer
void vvTexRend::updateVolumeData()
{
  vvRenderer::updateVolumeData();
  if (_renderState._computeBrickSize)
  {
    _areBricksCreated = false;
    computeBrickSize();
  }

  if (_numThreads > 0)
  {
    for (int i = 0; i < _numThreads; ++i)
    {
      makeTextures(_threadData[i].privateTexNames, &_threadData[i].numTextures,
                   _threadData[i].pixLUTName, _threadData[i].rgbaLUT);
    }
  }
  else
  {
    makeTextures(texNames, &textures, pixLUTName, rgbaLUT);
  }
}

//----------------------------------------------------------------------------
void vvTexRend::updateVolumeData(int offsetX, int offsetY, int offsetZ,
  int sizeX, int sizeY, int sizeZ)
{
  switch (geomType)
  {
    case VV_VIEWPORT:
      updateTextures3D(offsetX, offsetY, offsetZ, sizeX, sizeY, sizeZ, false);
      break;
    case VV_BRICKS:
      updateTextureBricks(offsetX, offsetY, offsetZ, sizeX, sizeY, sizeZ);
      break;
    case VV_SLICES:
      updateTextures2D(1, offsetX, offsetY, offsetZ, sizeX, sizeY, sizeZ);
      break;
    case VV_CUBIC2D:
      updateTextures2D(3, offsetX, offsetY, offsetZ, sizeX, sizeY, sizeZ);
      break;
    default:
      // nothing to do
      break;
  }
}

//----------------------------------------------------------------------------
/**
   Method to create a new 3D texture or update parts of an existing 3D texture.
   @param offsetX, offsetY, offsetZ: lower left corner of texture
   @param sizeX, sizeY, sizeZ: size of texture
   @param newTex: true: create a new texture
                  false: update an existing texture
*/
vvTexRend::ErrorType vvTexRend::updateTextures3D(int offsetX, int offsetY, int offsetZ,
int sizeX, int sizeY, int sizeZ, bool newTex)
{
  ErrorType err;
  int frames;
  int texSize;
  int sliceSize;
  int rawSliceOffset;
  int heightOffset;
  int texLineOffset;
  int srcIndex;
  int texOffset=0;
  int rawVal[4];
  int alpha;
  int c, f, s, x, y;
  float fval;
  unsigned char* raw;
  unsigned char* texData;
  bool accommodated = true;
  GLint glWidth;

  vvDebugMsg::msg(1, "vvTexRend::updateTextures3D()");

  if (!extTex3d) return NO3DTEX;

  frames = vd->frames;

  texSize = sizeX * sizeY * sizeZ * texelsize;

  vvDebugMsg::msg(1, "3D Texture width     = ", sizeX);
  vvDebugMsg::msg(1, "3D Texture height    = ", sizeY);
  vvDebugMsg::msg(1, "3D Texture depth     = ", sizeZ);
  vvDebugMsg::msg(1, "3D Texture size (KB) = ", texSize / 1024);

  texData = new uchar[texSize];
  memset(texData, 0, texSize);

  sliceSize = vd->getSliceBytes();

  if (newTex)
  {
    vvDebugMsg::msg(2, "Creating texture names. # of names: ", frames);

    removeTextures();
    textures  = frames;
    delete[] texNames;
    texNames = new GLuint[textures];
    glGenTextures(frames, texNames);
  }

  vvDebugMsg::msg(2, "Transferring textures to TRAM. Total size [KB]: ",
    frames * texSize / 1024);

  // Generate sub texture contents:
  for (f = 0; f < frames; f++)
  {
    raw = vd->getRaw(f);
    for (s = offsetZ; s < (offsetZ + sizeZ); s++)
    {
      rawSliceOffset = (vd->vox[2] - min(s,vd->vox[2]-1) - 1) * sliceSize;
      for (y = offsetY; y < (offsetY + sizeY); y++)
      {
        heightOffset = (vd->vox[1] - min(y,vd->vox[1]-1) - 1) * vd->vox[0] * vd->bpc * vd->chan;
        texLineOffset = (y - offsetY) * sizeX + (s - offsetZ) * sizeX * sizeY;
        if (vd->chan == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
        {
          for (x = offsetX; x < (offsetX + sizeX); x++)
          {
            srcIndex = vd->bpc * min(x,vd->vox[0]-1) + rawSliceOffset + heightOffset;
            if (vd->bpc == 1) rawVal[0] = int(raw[srcIndex]);
            else if (vd->bpc == 2)
            {
              rawVal[0] = ((int) raw[srcIndex] << 8) | (int) raw[srcIndex + 1];
              rawVal[0] >>= 4;
            }
            else // vd->bpc==4: convert floating point to 8bit value
            {
              fval = *((float*)(raw + srcIndex));      // fetch floating point data value
              rawVal[0] = vd->mapFloat2Int(fval);
            }
            texOffset = (x - offsetX) + texLineOffset;
            switch(voxelType)
            {
              case VV_SGI_LUT:
                texData[2 * texOffset] = texData[2 * texOffset + 1] = (uchar) rawVal[0];
                break;
              case VV_PAL_TEX:
              case VV_FRG_PRG:
                texData[texOffset] = (uchar) rawVal[0];
                break;
              case VV_TEX_SHD:
                for (c = 0; c < 4; c++)
                {
                  texData[4 * texOffset + c] = (uchar) rawVal[0];
                }
                break;
              case VV_PIX_SHD:
                texData[4 * texOffset] = (uchar) rawVal[0];
                break;
              case VV_RGBA:
                for (c = 0; c < 4; c++)
                {
                  texData[4 * texOffset + c] = rgbaLUT[rawVal[0] * 4 + c];
                }
                break;
              default:
                assert(0);
                break;
            }
          }
        }
        else if (vd->bpc==1 || vd->bpc==2 || vd->bpc==4)
        {
          if (voxelType == VV_RGBA || voxelType == VV_PIX_SHD)
          {
            for (x = offsetX; x < (offsetX + sizeX); x++)
            {
              texOffset = (x - offsetX) + texLineOffset;
              for (c = 0; c < ts_min(vd->chan,4); c++)
              {
                srcIndex = rawSliceOffset + heightOffset + vd->bpc * (x * vd->chan + c);
                if (vd->bpc == 1)
                  rawVal[c] = (int) raw[srcIndex];
                else if (vd->bpc == 2)
                {
                  rawVal[c] = ((int) raw[srcIndex] << 8) | (int) raw[srcIndex + 1];
                  rawVal[c] >>= 4;
                }
                else  // vd->bpc == 4
                {
                  fval = *((float*)(raw + srcIndex));      // fetch floating point data value
                  rawVal[c] = vd->mapFloat2Int(fval);
                }
              }

              // Copy color components:
              for (c = 0; c < ts_min(vd->chan, 3); c++)
              {
                texData[4 * texOffset + c] = (uchar) rawVal[c];
              }
            }

            // Alpha channel:
            if (vd->chan >= 4)
            {
              texData[4 * texOffset + 3] = (uchar)rawVal[3];
            }
            else
            {
              alpha = 0;
              for (c = 0; c < vd->chan; c++)
              {
                // Alpha: mean of sum of RGB conversion table results:
                alpha += (int) rgbaLUT[rawVal[c] * 4 + c];
              }
              texData[4 * texOffset + 3] = (uchar) (alpha / vd->chan);
            }
          }
        }
        else cerr << "Cannot create texture: unsupported voxel format (3)." << endl;
      }
    }

    if (geomType == VV_SPHERICAL)
    {
      // Set edge values to 0 for spheric textures, because textures
      // may exceed texel volume:
      for (s = offsetZ; s < (offsetZ + sizeZ); ++s)
      {
        for (y = offsetY; y < (offsetY + sizeY); ++y)
        {
          for (x = offsetX; x < (offsetX + sizeX); ++x)
          {
            if ((s == 0) || (s==vd->vox[2]-1) ||
              (y == 0) || (y==vd->vox[1]-1) ||
              (x == 0) || (x==vd->vox[0]-1))
            {
              texOffset = x + y * texels[0] + s * texels[0] * texels[1];
              for(int i=0; i<texelsize; i++)
                texData[texelsize*texOffset+i] = 0;
            }
          }
        }
      }
    }

    if (newTex)
    {
      glBindTexture(GL_TEXTURE_3D_EXT, texNames[f]);
      glPixelStorei(GL_UNPACK_ALIGNMENT,1);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP);

      glTexImage3D(GL_PROXY_TEXTURE_3D_EXT, 0, internalTexFormat,
        texels[0], texels[1], texels[2], 0, texFormat, GL_UNSIGNED_BYTE, NULL);
      glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D_EXT, 0, GL_TEXTURE_WIDTH, &glWidth);
      if (glWidth!=0)
      {
        glTexImage3D(GL_TEXTURE_3D_EXT, 0, internalTexFormat, texels[0], texels[1], texels[2], 0,
          texFormat, GL_UNSIGNED_BYTE, texData);
      }
      else
        accommodated = false;
    }
    else
    {
      glBindTexture(GL_TEXTURE_3D_EXT, texNames[f]);
      glTexSubImage3DEXT(GL_TEXTURE_3D_EXT, 0, offsetX, offsetY, offsetZ,
        sizeX, sizeY, sizeZ, texFormat, GL_UNSIGNED_BYTE, texData);
    }
  }

  if (newTex && (accommodated == false))
  {
    cerr << "Insufficient texture memory for 3D texture(s)." << endl;
    err = TRAM_ERROR;
  }

  delete[] texData;
  return OK;
}

vvTexRend::ErrorType vvTexRend::updateTextures2D(int axes, int offsetX, int offsetY, int offsetZ,
  int sizeX, int sizeY, int sizeZ)
{
  int rawVal[4];
  int rawSliceSize;
  int rawLineSize;
  int texSize[3];
  int frames;
  int texIndex = 0;
  int texSliceIndex;
  int texW[3], texH[3];
  int tw[3], th[3];
  int rw[3], rh[3], rs[3];
  int sw[3], sh[3], ss[3];
  int rawStart[3];
  int rawStepW[3];
  int rawStepH[3];
  int rawStepS[3];
  uchar* rgbaSlice[3];
  uchar* raw;
  uchar* rawVoxel;
  float fval;
  int i, s, w, h, f, c, alpha;

  assert(axes == 1 || axes == 3);

  frames = vd->frames;

  ss[0] = vd->vox[0] - offsetX - sizeX;
  sw[0] = offsetY;
  sh[0] = offsetZ;

  texW[0] = offsetY;
  texH[0] = offsetZ;

  ss[1] = vd->vox[1] - offsetY - sizeY;
  sh[1] = offsetZ;
  sw[1] = offsetX;

  texW[1] = offsetZ;
  texH[1] = offsetX;

  ss[2] = vd->vox[2] - offsetZ - sizeZ;
  sw[2] = offsetX;
  sh[2] = offsetY;

  texW[2] = offsetX;
  texH[2] = offsetY;

  rs[0] = ts_clamp(vd->vox[0] - offsetX, 0, vd->vox[0]);
  rw[0] = ts_clamp(offsetY + sizeY, 0, vd->vox[1]);
  rh[0] = ts_clamp(offsetZ + sizeZ, 0, vd->vox[2]);

  rs[1] = ts_clamp(vd->vox[1] - offsetY, 0, vd->vox[1]);
  rh[1] = ts_clamp(offsetZ + sizeZ, 0, vd->vox[2]);
  rw[1] = ts_clamp(offsetX + sizeX, 0, vd->vox[0]);

  rs[2] = ts_clamp(vd->vox[2] - offsetZ, 0, vd->vox[2]);
  rw[2] = ts_clamp(offsetX + sizeX, 0, vd->vox[0]);
  rh[2] = ts_clamp(offsetY + sizeY, 0, vd->vox[1]);

  rawLineSize  = vd->vox[0] * vd->getBPV();
  rawSliceSize = vd->getSliceBytes();

  // initialize raw data access info
  rawStart[0] = (vd->vox[2]) * rawSliceSize - vd->getBPV();
  rawStepW[0] = -rawLineSize;
  rawStepH[0] = -rawSliceSize;
  rawStepS[0] = -vd->getBPV();
  rawStart[1] = (vd->vox[2] - 1) * rawSliceSize;
  rawStepW[1] = -rawSliceSize;
  rawStepH[1] = vd->getBPV();
  rawStepS[1] = rawLineSize;
  rawStart[2] = (vd->vox[1] - 1) * rawLineSize;;
  rawStepW[2] = vd->getBPV();
  rawStepH[2] = -rawLineSize;
  rawStepS[2] = rawSliceSize;

  // initialize size of sub region
  th[1] = tw[2] = sizeX;
  tw[0] = th[2] = sizeY;
  tw[1] = th[0] = sizeZ;

  for (i = 3-axes; i < 3; i++)
  {
    texSize[i] = tw[i] * th[i] * texelsize;
  }

  // generate texture data arrays
  for (i=3-axes; i<3; ++i)
  {
    rgbaSlice[i] = new uchar[texSize[i]];
  }

  // generate texture data
  for (f = 0; f < frames; f++)
  {
    raw = vd->getRaw(f);

    for (i = 3-axes; i < 3; i++)
    {
      memset(rgbaSlice[i], 0, texSize[i]);

      if (axes == 1)
      {
        texIndex = f * vd->vox[i] + ss[i];
      }
      else
      {
        texIndex = f * (vd->vox[0] + vd->vox[1] + vd->vox[2]);
        if (i == 0)
          texIndex += ss[0];
        else if (i == 1)
          texIndex += vd->vox[0] + ss[1];
        else
          texIndex += vd->vox[0] + vd->vox[1] + ss[2];
      }

      // generate texture contents
      for (s = ss[i]; s < rs[i]; s++)
      {
        if (s < 0) continue;

        for (h = sh[i]; h < rh[i]; h++)
        {
          if (h < 0) continue;

          // set voxel to starting position in raw data array
          rawVoxel = raw + rawStart[i] + s * rawStepS[i] + h * rawStepH[i] + sw[i];

          for (w = sw[i]; w < rw[i]; w++)
          {
            if (w < 0) continue;

            texSliceIndex = texelsize * ((w - sw[i]) + (h - sh[i]) * tw[i]);

            if (vd->chan == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
            {
              if (vd->bpc == 1)
                rawVal[0] = int(*rawVoxel);
              else if (vd->bpc == 2)
              {
                rawVal[0] = (int(*rawVoxel) << 8) | int(*(rawVoxel+1));
                rawVal[0] >>= 4;
              }
              else
              {
                fval = *((float*)(rawVoxel));
                rawVal[0] = vd->mapFloat2Int(fval);
              }

              for (c = 0; c < texelsize; c++)
                rgbaSlice[i][texSliceIndex + c] = rgbaLUT[rawVal[0] * 4 + c];
            }
            else if (vd->bpc == 1)
            {
              // fetch component values from memory
              for (c = 0; c < ts_min(vd->chan,4); c++)
                rawVal[c] = *(rawVoxel + c);

              // copy color components
              for (c = 0; c < ts_min(vd->chan,3); c++)
                rgbaSlice[i][texSliceIndex + c] = (uchar) rawVal[c];

              // alpha channel
              if (vd->chan >= 4)
                rgbaSlice[i][texSliceIndex + 3] = rgbaLUT[rawVal[3] * 4 + 3];
              else
              {
                alpha = 0;
                for (c = 0; c < vd->chan; c++)
                  // alpha: mean of sum of RGB conversion table results
                  alpha += (int)rgbaLUT[rawVal[c] * 4 + c];

                rgbaSlice[i][texSliceIndex + 3] = (uchar)(alpha / vd->chan);
              }
            }

            rawVoxel += rawStepW[i];
          }
        }

        glBindTexture(GL_TEXTURE_2D, texNames[texIndex]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, texW[i], texH[i], tw[i], th[i],
          texFormat, GL_UNSIGNED_BYTE, rgbaSlice[i]);

        ++texIndex;
      }
    }
  }

  for (i = 3-axes; i < 3; i++)
  {
    delete[] rgbaSlice[i];
  }

  return OK;
}

vvTexRend::ErrorType vvTexRend::updateTextureBricks(int offsetX, int offsetY, int offsetZ,
int sizeX, int sizeY, int sizeZ)
{
  int frames;
  int texSize;
  int rawSliceOffset;
  int heightOffset;
  int texLineOffset;
  int srcIndex;
  int texOffset;
  int sliceSize;
  int rawVal[4];
  int alpha;
  int startOffset[3], endOffset[3];
  int start[3], end[3], size[3];
  int c, f, s, x, y;
  float fval;
  unsigned char* raw;
  unsigned char* texData = 0;

  if (!extTex3d) return NO3DTEX;

  frames = vd->frames;
  texSize = texels[0] * texels[1] * texels[2] * texelsize;
  texData = new uchar[texSize];
  sliceSize = vd->getSliceBytes();

  for (f = 0; f < frames; f++)
  {
    raw = vd->getRaw(f);

    for(BrickList::iterator brick = _brickList[f].begin(); brick != _brickList[f].end(); ++brick)
    {
      startOffset[0] = (*brick)->startOffset[0];
      startOffset[1] = (*brick)->startOffset[1];
      startOffset[2] = (*brick)->startOffset[2];

      endOffset[0] = startOffset[0] + _renderState._brickSize[0];
      endOffset[1] = startOffset[1] + _renderState._brickSize[1];
      endOffset[2] = startOffset[2] + _renderState._brickSize[2];

      endOffset[0] = ts_clamp(endOffset[0], 0, vd->vox[0] - 1);
      endOffset[1] = ts_clamp(endOffset[1], 0, vd->vox[1] - 1);
      endOffset[2] = ts_clamp(endOffset[2], 0, vd->vox[2] - 1);

      if ((offsetX > endOffset[0]) || ((offsetX + sizeX - 1) < startOffset[0]))
      {
        continue;
      }
      else if (offsetX >= startOffset[0])
      {
        start[0] = offsetX;

        if ((offsetX + sizeX - 1) <= endOffset[0])
        {
          end[0] = offsetX + sizeX - 1;
          size[0]= sizeX;
        }
        else
        {
          end[0] = endOffset[0];
          size[0] = end[0] - start[0] + 1;
        }
      }
      else
      {
        start[0] = startOffset[0];

        if ((offsetX + sizeX - 1) <= endOffset[0]) end[0] = offsetX + sizeX - 1;
        else end[0] = endOffset[0];

        size[0] = end[0] - start[0] + 1;
      }

      if ((offsetY > endOffset[1]) || ((offsetY + sizeY - 1) < startOffset[1]))
      {
        continue;
      }
      else if (offsetY >= startOffset[1])
      {
        start[1] = offsetY;

        if ((offsetY + sizeY - 1) <= endOffset[1])
        {
          end[1] = offsetY + sizeY - 1;
          size[1]= sizeY;
        }
        else
        {
          end[1] = endOffset[1];
          size[1] = end[1] - start[1] + 1;
        }
      }
      else
      {
        start[1] = startOffset[1];

        if ((offsetY + sizeY - 1) <= endOffset[1])
          end[1] = offsetY + sizeY - 1;
        else
          end[1] = endOffset[1];

        size[1] = end[1] - start[1] + 1;
      }

      if ((offsetZ > endOffset[2]) ||
        ((offsetZ + sizeZ - 1) < startOffset[2]))
      {
        continue;
      }
      else if (offsetZ >= startOffset[2])
      {
        start[2] = offsetZ;

        if ((offsetZ + sizeZ - 1) <= endOffset[2])
        {
          end[2] = offsetZ + sizeZ - 1;
          size[2]= sizeZ;
        }
        else
        {
          end[2] = endOffset[2];
          size[2] = end[2] - start[2] + 1;
        }
      }
      else
      {
        start[2] = startOffset[2];

        if ((offsetZ + sizeZ - 1) <= endOffset[2])
          end[2] = offsetZ + sizeZ - 1;
        else
          end[2] = endOffset[2];

        size[2] = end[2] - start[2] + 1;
      }

      texSize = size[0] * size[1] * size[2] * texelsize;
      if (texData != 0)
        delete[] texData;
      texData = new unsigned char[texSize];

      memset(texData, 0, texSize);

      for (s = start[2]; s <= end[2]; s++)
      {
        rawSliceOffset = (vd->vox[2] - s - 1) * sliceSize;

        for (y = start[1]; y <= end[1]; y++)
        {
          if (y < 0) continue;

          heightOffset = (vd->vox[1] - y - 1) * vd->vox[0] * vd->bpc * vd->chan;

          texLineOffset = (y - start[1]) * size[0] + (s - start[2]) * size[0] * size[1];

          if (vd->chan == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
          {
            for (x = start[0]; x <= end[0]; x++)
            {
              if (x < 0) continue;

              srcIndex = vd->bpc * x + rawSliceOffset + heightOffset;
              if (vd->bpc == 1)
                rawVal[0] = (int) raw[srcIndex];
              else if (vd->bpc == 2)
              {
                rawVal[0] = ((int) raw[srcIndex] << 8) | (int) raw[srcIndex + 1];
                rawVal[0] >>= 4;
              }
              else
              {
                fval = *((float*) (raw + srcIndex));
                rawVal[0] = vd->mapFloat2Int(fval);
              }

              texOffset = (x - start[0]) + texLineOffset;

              switch (voxelType)
              {
                case VV_SGI_LUT:
                  texData[2*texOffset] = texData[2*texOffset + 1] = (uchar) rawVal[0];
                  break;
                case VV_PAL_TEX:
                case VV_FRG_PRG:
                  texData[texOffset] = (uchar) rawVal[0];
                  break;
                case VV_TEX_SHD:
                  for (c = 0; c < 4; c++)
                    texData[4 * texOffset + c] = (uchar) rawVal[0];
                  break;
                case VV_PIX_SHD:
                  texData[4 * texOffset] = (uchar) rawVal[0];
                  break;
                case VV_RGBA:
                  for (c = 0; c < 4; c++)
                    texData[4 * texOffset + c] = rgbaLUT[rawVal[0] * 4 + c];
                  break;
                default:
                  assert(0);
                  break;
              }
            }
          }
          else if (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4)
          {
            if (voxelType == VV_RGBA || voxelType == VV_PIX_SHD)
            {
              for (x = start[0]; x <= end[0]; x++)
              {
                if (x < 0) continue;

                texOffset = (x - start[0]) + texLineOffset;

                // fetch component values from memory:
                for (c = 0; c < ts_min(vd->chan, 4); c++)
                {
                  srcIndex = rawSliceOffset + heightOffset + vd->bpc * (x * vd->chan + c);
                  if (vd->bpc == 1)
                    rawVal[c] = (int) raw[srcIndex];
                  else if (vd->bpc == 2)
                  {
                    rawVal[c] = ((int) raw[srcIndex] << 8) | (int) raw[srcIndex + 1];
                    rawVal[c] >>= 4;
                  }
                  else
                  {
                    fval = *((float*) (raw + srcIndex));
                    rawVal[c] = vd->mapFloat2Int(fval);
                  }
                }

                // copy color components:
                for (c = 0; c < ts_min(vd->chan, 3); c++)
                  texData[4 * texOffset + c] = (uchar) rawVal[c];

                // alpha channel
                if (vd->chan >= 4)
                  // RGBA
                {
                  if (voxelType == VV_RGBA)
                    texData[4 * texOffset + 3] = rgbaLUT[rawVal[3] * 4 + 3];
                  else
                    texData[4 * texOffset + 3] = (uchar) rawVal[3];
                }
                else
                  // compute alpha from color components
                {
                  alpha = 0;
                  for (c = 0; c < vd->chan; c++)
                    // alpha: mean of sum of RGB conversion table results:
                    alpha += (int) rgbaLUT[rawVal[c] * 4 + c];

                  texData[4 * texOffset + 3] = (uchar) (alpha / vd->chan);
                }
              }
            }
          }
          else cerr << "Cannot create texture: unsupported voxel format (3)." << endl;
        }
      }

      //memset(texData, 255, texSize);

      glBindTexture(GL_TEXTURE_3D_EXT, texNames[(*brick)->index]);

      glTexSubImage3DEXT(GL_TEXTURE_3D_EXT, 0, start[0] - startOffset[0], start[1] - startOffset[1], start[2] - startOffset[2],
        size[0], size[1], size[2], texFormat, GL_UNSIGNED_BYTE, texData);
    }
  }

  delete[] texData;
  return OK;
}

//----------------------------------------------------------------------------
/// Set GL environment for texture rendering.
void vvTexRend::setGLenvironment()
{
  vvDebugMsg::msg(3, "vvTexRend::setGLenvironment()");

  // Save current GL state:
  glPushAttrib(GL_COLOR_BUFFER_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT
               | GL_ENABLE_BIT | GL_TEXTURE_BIT | GL_TRANSFORM_BIT);

  // Set new GL state:
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);                           // default depth function
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_BLEND);

  if (_numThreads > 0)
  {
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  }
  else
  {
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  }
  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glDepthMask(GL_FALSE);

  switch (voxelType)
  {
    case VV_SGI_LUT:
      glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
      break;
    case VV_PAL_TEX:
      glDisable(GL_SHARED_TEXTURE_PALETTE_EXT);
      break;
    default: break;
  }
  switch (_renderState._mipMode)
  {
    // alpha compositing
    case 0: glBlendEquation(GL_FUNC_ADD); break;
    case 1: glBlendEquation(GL_MAX); break;   // maximum intensity projection
    case 2: glBlendEquation(GL_MIN); break;   // minimum intensity projection
    default: break;
  }
  vvDebugMsg::msg(3, "vvTexRend::setGLenvironment() done");
}

//----------------------------------------------------------------------------
/// Unset GL environment for texture rendering.
void vvTexRend::unsetGLenvironment()
{
  vvDebugMsg::msg(3, "vvTexRend::unsetGLenvironment()");

  glPopAttrib();

  vvDebugMsg::msg(3, "vvTexRend::unsetGLenvironment() done");
}

//----------------------------------------------------------------------------
void vvTexRend::enableLUTMode(vvShaderManager* pixelShader, GLuint& lutName)
{
  switch(voxelType)
  {
    case VV_FRG_PRG:
      enableFragProg(lutName);
      break;
    case VV_TEX_SHD:
      enableNVShaders();
      break;
    case VV_PIX_SHD:
      enablePixelShaders(pixelShader, lutName);
      break;
    case VV_SGI_LUT:
      glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
      break;
    case VV_PAL_TEX:
      glEnable(GL_SHARED_TEXTURE_PALETTE_EXT);
      break;
    default:
      // nothing to do
      break;
  }
}

//----------------------------------------------------------------------------
void vvTexRend::disableLUTMode(vvShaderManager* pixelShader)
{
  switch(voxelType)
  {
    case VV_FRG_PRG:
      disableFragProg();
      break;
    case VV_TEX_SHD:
      disableNVShaders();
      break;
    case VV_PIX_SHD:
      disablePixelShaders(pixelShader);
      break;
    case VV_SGI_LUT:
      if (glsTexColTable==(uchar)true) glEnable(GL_TEXTURE_COLOR_TABLE_SGI);
      else glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
      break;
    case VV_PAL_TEX:
      if (glsSharedTexPal==(uchar)true) glEnable(GL_SHARED_TEXTURE_PALETTE_EXT);
      else glDisable(GL_SHARED_TEXTURE_PALETTE_EXT);
      break;
    default:
      // nothing to do
      break;
  }
}

//----------------------------------------------------------------------------
/** Render a volume entirely if probeSize=0 or a cubic sub-volume of size probeSize.
  @param mv        model-view matrix
*/
void vvTexRend::renderTex3DPlanar(vvMatrix* mv)
{
  vvMatrix invMV;                                 // inverse of model-view matrix
  vvMatrix pm;                                    // OpenGL projection matrix
  vvVector3 size, size2;                          // full and half object sizes
  vvVector3 isect[6];                             // intersection points, maximum of 6 allowed when intersecting a plane and a volume [object space]
  vvVector3 texcoord[12];                         // intersection points in texture coordinate space [0..1]
  vvVector3 farthest;                             // volume vertex farthest from the viewer
  vvVector3 delta;                                // distance vector between textures [object space]
  vvVector3 normal;                               // normal vector of textures
  vvVector3 temp;                                 // temporary vector
  vvVector3 origin;                               // origin (0|0|0) transformed to object space
  vvVector3 eye;                                  // user's eye position [object space]
  vvVector3 normClipPoint;                        // normalized point on clipping plane
  vvVector3 clipPosObj;                           // clipping plane position in object space w/o position
  vvVector3 probePosObj;                          // probe midpoint [object space]
  vvVector3 probeSizeObj;                         // probe size [object space]
  vvVector3 probeTexels;                          // number of texels in each probe dimension
  vvVector3 probeMin, probeMax;                   // probe min and max coordinates [object space]
  vvVector3 texSize;                              // size of 3D texture [object space]
  vvVector3 pos;                                  // volume location
  float     maxDist;                              // maximum length of texture drawing path
  int       i;                                    // general counter
  int       numSlices;

  vvDebugMsg::msg(3, "vvTexRend::renderTex3DPlanar()");

  if (!extTex3d) return;                          // needs 3D texturing extension

  // Determine texture object dimensions and half object size as a shortcut:
  size.copy(vd->getSize());
  for (i=0; i<3; ++i)
  {
    texSize.e[i] = size.e[i] * (float)texels[i] / (float)vd->vox[i];
    size2.e[i]   = 0.5f * size.e[i];
  }
  pos.copy(&vd->pos);

  // Calculate inverted modelview matrix:
  invMV.copy(mv);
  invMV.invert();

  // Find eye position:
  getEyePosition(&eye);
  eye.multiply(&invMV);

  if (_renderState._isROIUsed)
  {
    // Convert probe midpoint coordinates to object space w/o position:
    probePosObj.copy(&_renderState._roiPos);
    probePosObj.sub(&pos);                        // eliminate object position from probe position

    // Compute probe min/max coordinates in object space:
    for (i=0; i<3; ++i)
    {
      probeMin[i] = probePosObj[i] - (_renderState._roiSize[i] * size[i]) * 0.5f;
      probeMax[i] = probePosObj[i] + (_renderState._roiSize[i] * size[i]) * 0.5f;
    }

    // Constrain probe boundaries to volume data area:
    for (i=0; i<3; ++i)
    {
      if (probeMin[i] > size2[i] || probeMax[i] < -size2[i])
      {
        vvDebugMsg::msg(3, "probe outside of volume");
        return;
      }
      if (probeMin[i] < -size2[i]) probeMin[i] = -size2[i];
      if (probeMax[i] >  size2[i]) probeMax[i] =  size2[i];
      probePosObj[i] = (probeMax[i] + probeMin[i]) *0.5;
    }

    // Compute probe edge lengths:
    for (i=0; i<3; ++i)
      probeSizeObj[i] = probeMax[i] - probeMin[i];
  }
  else                                            // probe mode off
  {
    probeSizeObj.copy(&size);
    probeMin.set(-size2[0], -size2[1], -size2[2]);
    probeMax.copy(&size2);
    probePosObj.zero();
  }

  // Initialize texture counters
  if (_renderState._roiSize[0])
  {
    probeTexels.zero();
    for (i=0; i<3; ++i)
    {
      probeTexels[i] = texels[i] * probeSizeObj[i] / texSize.e[i];
    }
  }
  else                                            // probe mode off
  {
    probeTexels.set((float)vd->vox[0], (float)vd->vox[1], (float)vd->vox[2]);
  }

  // Get projection matrix:
  getProjectionMatrix(&pm);
  bool isOrtho = pm.isProjOrtho();

  // Compute normal vector of textures using the following strategy:
  // For orthographic projections or if viewDir is (0|0|0) use
  // (0|0|1) as the normal vector.
  // Otherwise use objDir as the normal.
  // Exception: if user's eye is inside object and probe mode is off,
  // then use viewDir as the normal.
  if (_sliceOrientation==VV_CLIPPLANE ||
     (_sliceOrientation==VV_VARIABLE && _renderState._clipMode))
  {
    normal.copy(&_renderState._clipNormal);
  }
  else if(_sliceOrientation==VV_VIEWPLANE)
  {
    normal.set(0.0f, 0.0f, 1.0f);                 // (0|0|1) is normal on projection plane
    vvMatrix invPM;
    invPM.copy(&pm);
    invPM.invert();
    normal.multiply(&invPM);
    normal.multiply(&invMV);
    normal.negate();
  }
  else if (_sliceOrientation==VV_ORTHO ||
          (_sliceOrientation==VV_VARIABLE &&
          (isOrtho || (viewDir.e[0]==0.0f && viewDir.e[1]==0.0f && viewDir.e[2]==0.0f))))
  {
    // Draw slices parallel to projection plane:
    normal.set(0.0f, 0.0f, 1.0f);                 // (0|0|1) is normal on projection plane
    normal.multiply(&invMV);
    origin.zero();
    origin.multiply(&invMV);
    normal.sub(&origin);
  }
  else if (_sliceOrientation==VV_VIEWDIR || 
          (_sliceOrientation==VV_VARIABLE && (!_renderState._isROIUsed && isInVolume(&eye))))
  {
    // Draw slices perpendicular to viewing direction:
    normal.copy(&viewDir);
    normal.negate();                              // viewDir points away from user, the normal should point towards them
  }
  else
  {
    // Draw slices perpendicular to line eye->object:
    normal.copy(&objDir);
    normal.negate();
  }

  // Compute distance vector between textures:
  normal.normalize();
  // compute number of slices to draw
  float depth = fabs(normal[0]*probeSizeObj[0]) + fabs(normal[1]*probeSizeObj[1]) + fabs(normal[2]*probeSizeObj[2]);
  int minDistanceInd = 0;
  if(probeSizeObj[1]/probeTexels[1] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
    minDistanceInd=1;
  if(probeSizeObj[2]/probeTexels[2] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
    minDistanceInd=2;
  float voxelDistance = probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd];

  float sliceDistance = voxelDistance / _renderState._quality;
  if(_renderState._isROIUsed && _renderState._quality < 2.0)
  {
    // draw at least twice as many slices as there are samples in the probe depth.
    sliceDistance = voxelDistance / 2.0f;
  }
  numSlices = 2*(int)ceilf(depth/sliceDistance*.5f);

  if (numSlices < 1)                              // make sure that at least one slice is drawn
    numSlices = 1;

  vvDebugMsg::msg(3, "Number of textures rendered: ", numSlices);

  // Use alpha correction in indexed mode: adapt alpha values to number of textures:
  if (instantClassification())
  {
    float thickness = sliceDistance/voxelDistance;

    // just tolerate slice distance differences imposed on us
    // by trying to keep the number of slices constant
    if(lutDistance/thickness < 0.88 || thickness/lutDistance < 0.88)
    {
      updateLUT(thickness, pixLUTName, rgbaLUT);
    }
  }

  delta.copy(&normal);
  delta.scale(sliceDistance);

  // Compute farthest point to draw texture at:
  farthest.copy(&delta);
  farthest.scale((float)(numSlices - 1) * -0.5f);
  farthest.add(&probePosObj);

  if (_renderState._clipMode)                     // clipping plane present?
  {
    // Adjust numSlices and set farthest point so that textures are only
    // drawn up to the clipPoint. (Delta may not be changed
    // due to the automatic opacity correction.)
    // First find point on clipping plane which is on a line perpendicular
    // to clipping plane and which traverses the origin:
    temp.copy(&delta);
    temp.scale(-0.5f);
    farthest.add(&temp);                          // add a half delta to farthest
    clipPosObj.copy(&_renderState._clipPoint);
    clipPosObj.sub(&pos);
    temp.copy(&probePosObj);
    temp.add(&normal);
    normClipPoint.isectPlaneLine(&normal, &clipPosObj, &probePosObj, &temp);
    maxDist = farthest.distance(&normClipPoint);
    numSlices = (int)(maxDist / delta.length()) + 1;
    temp.copy(&delta);
    temp.scale((float)(1 - numSlices));
    farthest.copy(&normClipPoint);
    farthest.add(&temp);
    if (_renderState._clipSingleSlice)
    {
      // Compute slice position:
      temp.copy(&delta);
      temp.scale((float)(numSlices-1));
      farthest.add(&temp);
      numSlices = 1;

      // Make slice opaque if possible:
      if (instantClassification())
      {
        updateLUT(0.0f, pixLUTName, rgbaLUT);
      }
    }
  }

  // Translate object by its position:
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  //glTranslatef(pos.e[0], pos.e[1], pos.e[2]);

  vvVector3 texPoint;                             // arbitrary point on current texture
  int isectCnt;                                   // intersection counter
  int j,k;                                        // counters
  int drawn = 0;                                  // counter for drawn textures
  vvVector3 deltahalf;
  deltahalf.copy(&delta);
  deltahalf.scale(0.5f);

  // Relative viewing position
  vvVector3 releye;
  releye.copy(&eye);
  releye.sub(&pos);

  // Volume render a 3D texture:
  enableTexture(GL_TEXTURE_3D_EXT);
  glBindTexture(GL_TEXTURE_3D_EXT, texNames[vd->getCurrentFrame()]);
  texPoint.copy(&farthest);
  for (i=0; i<numSlices; ++i)                     // loop thru all drawn textures
  {
    // Search for intersections between texture plane (defined by texPoint and
    // normal) and texture object (0..1):
    isectCnt = isect->isectPlaneCuboid(&normal, &texPoint, &probeMin, &probeMax);

    texPoint.add(&delta);

    if (isectCnt<3) continue;                     // at least 3 intersections needed for drawing

    // Check volume section mode:
    if (minSlice != -1 && i < minSlice) continue;
    if (maxSlice != -1 && i > maxSlice) continue;

    // Put the intersecting 3 to 6 vertices in cyclic order to draw adjacent
    // and non-overlapping triangles:
    isect->cyclicSort(isectCnt, &normal);

    // Generate vertices in texture coordinates:
    if(usePreIntegration)
    {
      for (j=0; j<isectCnt; ++j)
      {
        vvVector3 front, back;

        if(isOrtho)
        {
          back.copy(&isect[j]);
          back.sub(&deltahalf);
        }
        else
        {
          vvVector3 v;
          v.copy(&isect[j]);
          v.sub(&deltahalf);
          back.isectPlaneLine(&normal, &v, &releye, &isect[j]);
        }

        if(isOrtho)
        {
          front.copy(&isect[j]);
          front.add(&deltahalf);
        }
        else
        {
          vvVector3 v;
          v.copy(&isect[j]);
          v.add(&deltahalf);
          front.isectPlaneLine(&normal, &v, &releye, &isect[j]);
        }

        for (k=0; k<3; ++k)
        {
          texcoord[j][k] = (back[k] + size2.e[k]) / size.e[k];
          texcoord[j][k] = texcoord[j][k] * (texMax[k] - texMin[k]) + texMin[k];

          texcoord[j+6][k] = (front[k] + size2.e[k]) / size.e[k];
          texcoord[j+6][k] = texcoord[j+6][k] * (texMax[k] - texMin[k]) + texMin[k];
        }
      }
    }
    else
    {
      for (j=0; j<isectCnt; ++j)
      {
        for (k=0; k<3; ++k)
        {
          texcoord[j][k] = (isect[j][k] + size2.e[k]) / size.e[k];
          texcoord[j][k] = texcoord[j][k] * (texMax[k] - texMin[k]) + texMin[k];
        }
      }
    }

    glBegin(GL_TRIANGLE_FAN);
    glColor4f(1.0, 1.0, 1.0, 1.0);
    glNormal3f(normal[0], normal[1], normal[2]);
    ++drawn;
    for (j=0; j<isectCnt; ++j)
    {
      // The following lines are the bottleneck of this method:
      if(usePreIntegration)
      {
        glMultiTexCoord3fARB(GL_TEXTURE0_ARB, texcoord[j][0], texcoord[j][1], texcoord[j][2]);
        glMultiTexCoord3fARB(GL_TEXTURE1_ARB, texcoord[j+6][0], texcoord[j+6][1], texcoord[j+6][2]);
      }
      else
      {
        glTexCoord3f(texcoord[j][0], texcoord[j][1], texcoord[j][2]);
      }

      glVertex3f(isect[j][0], isect[j][1], isect[j][2]);
    }
    glEnd();
  }
  vvDebugMsg::msg(3, "Number of textures drawn: ", drawn);
  disableTexture(GL_TEXTURE_3D_EXT);

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

void vvTexRend::renderTexBricks(vvMatrix* mv)
{
  vvMatrix invMV;                                 // inverse of model-view matrix
  vvMatrix pm;                                    // OpenGL projection matrix
  vvVector3 farthest;                             // volume vertex farthest from the viewer
  vvVector3 delta;                                // distance vector between textures [object space]
  vvVector3 normal;                               // normal vector of textures
  vvVector3 origin;                               // origin (0|0|0) transformed to object space
  vvVector3 eye;                                  // user's eye position [object space]
  vvVector3 pos;                                  // volume location
  vvVector3 probePosObj;                          // probe midpoint [object space]
  vvVector3 probeSizeObj;                         // probe size [object space]
  vvVector3 probeMin, probeMax;                   // probe min and max coordinates [object space]
  vvVector3 min, max;                             // min and max pos of current brick (cut with probe)
  vvVector3 texMin;                               // minimum texture coordinate
  int       numSlices;                            // number of texture slices along diagonal
  int       i;                                    // general counters
  int       discarded;                            // count discarded bricks due to empty-space leaping

  vvDebugMsg::msg(3, "vvTexRend::renderTexBricks()");

  // needs 3D texturing extension
  if (!extTex3d) return;

  if (_brickList.empty()) return;

  pos.copy(&vd->pos);

  // Calculate inverted modelview matrix:
  invMV.copy(mv);
  invMV.invert();

  // Find eye position:
  getEyePosition(&eye);
  eye.multiply(&invMV);

  calcProbeDims(probePosObj, probeSizeObj, probeMin, probeMax);

  // Compute length of probe diagonal [object space]:
  const float diagonal = sqrtf(probeSizeObj[0] * probeSizeObj[0] +
                               probeSizeObj[1] * probeSizeObj[1] +
                               probeSizeObj[2] * probeSizeObj[2]);

  const float diagonalVoxels = sqrtf(float(vd->vox[0] * vd->vox[0] +
                                           vd->vox[1] * vd->vox[1] +
                                           vd->vox[2] * vd->vox[2]));
  numSlices = int(_renderState._quality * diagonalVoxels);
  if (numSlices < 1) numSlices = 1;               // make sure that at least one slice is drawn

  vvDebugMsg::msg(3, "Number of texture slices rendered: ", numSlices);

  if (_numThreads == 0)
  {
    // Use alpha correction in indexed mode: adapt alpha values to number of textures:
    if (instantClassification())
    {
      const float thickness = diagonalVoxels / float(numSlices);
      if(lutDistance/thickness < 0.88 || thickness/lutDistance < 0.88)
      {
        updateLUT(thickness, pixLUTName, rgbaLUT);
      }
    }
  }

  // Get projection matrix:
  getProjectionMatrix(&pm);
  const bool isOrtho = pm.isProjOrtho();

  // Compute normal vector of textures using the following strategy:
  // For orthographic projections or if viewDir is (0|0|0) use
  // (0|0|1) as the normal vector.
  // Otherwise use objDir as the normal.
  // Exception: if user's eye is inside object and probe mode is off,
  // then use viewDir as the normal.
  if (_renderState._clipMode)
  {
    normal.copy(&_renderState._clipNormal);
  }
  else if (isOrtho || (viewDir.e[0] == 0.0f && viewDir.e[1] == 0.0f && viewDir.e[2] == 0.0f))
  {
    // Draw slices parallel to projection plane:
    normal.set(0.0f, 0.0f, 1.0f);                 // (0|0|1) is normal on projection plane
    normal.multiply(&invMV);
    origin.zero();
    origin.multiply(&invMV);
    normal.sub(&origin);
  }
  else if (!_renderState._isROIUsed && isInVolume(&eye))
  {
    // Draw slices perpendicular to viewing direction:
    normal.copy(&viewDir);
    normal.negate();                              // viewDir points away from user, the normal should point towards them
  }
  else
  {
    // Draw slices perpendicular to line eye->object:
    normal.copy(&objDir);
    normal.negate();
  }

  normal.normalize();

  // Local illumination based on blinn-phong shading.
  if (voxelType == VV_PIX_SHD && _currentShader == 12)
  {
    // Currently the light is exactly at the current viewing position.
    // TODO: arbitrary (not necessarily frame dependend) lighting position.
    vvVector3 L;
    L.copy(&normal);
    L.negate();

    // Viewing vector.
    vvVector3 V(0, 1, 0);
    V.copy(&normal);
    V.negate();

    // Half way vector.
    vvVector3 H;
    H.copy(L);
    H.add(&V);
    H.normalize();
    _pixelShader->setParameter3f(_currentShader, "L", L[0], L[1], L[2]);
    _pixelShader->setParameter3f(_currentShader, "H", H[0], H[1], H[2]);
  }

  delta.copy(&normal);
  delta.scale(diagonal / ((float)numSlices));

  // Compute farthest point to draw texture at:
  farthest.copy(&delta);
  farthest.scale((float)(numSlices - 1) * -0.5f);
  if(!_proxyGeometryOnGpu)
    farthest.add(&probePosObj);

  vvVector3 temp, clipPosObj, normClipPoint;
  float maxDist;

  if (_renderState._clipMode)                     // clipping plane present?
  {
    // Adjust numSlices and set farthest point so that textures are only
    // drawn up to the clipPoint. (Delta may not be changed
    // due to the automatic opacity correction.)
    // First find point on clipping plane which is on a line perpendicular
    // to clipping plane and which traverses the origin:
    temp.copy(&delta);
    temp.scale(-0.5f);
    farthest.add(&temp);                          // add a half delta to farthest
    clipPosObj.copy(&_renderState._clipPoint);
    clipPosObj.sub(&pos);
    temp.copy(&probePosObj);
    temp.add(&normal);
    normClipPoint.isectPlaneLine(&normal, &clipPosObj, &probePosObj, &temp);
    maxDist = farthest.distance(&normClipPoint);
    numSlices = (int)(maxDist / delta.length()) + 1;
    temp.copy(&delta);
    temp.scale((float)(1 - numSlices));
    farthest.copy(&normClipPoint);
    farthest.add(&temp);
    if (_renderState._clipSingleSlice)
    {
      // Compute slice position:
      delta.scale((float)(numSlices-1));
      farthest.add(&delta);
      numSlices = 1;

      // Make slice opaque if possible:
      if (instantClassification() && (_numThreads == 0))
      {
        updateLUT(1.0f, pixLUTName, rgbaLUT);
      }
    }
  }

  initVertArray(numSlices);

  getBricksInProbe(probePosObj, probeSizeObj);

  markBricksInFrustum();

  if (_numThreads > 0)
  {
    for (i = 0; i < _numThreads; ++i)
    {
      sortBrickList(i, eye, normal, isOrtho);
    }

    if (_somethingChanged)
    {
      distributeBricks();
    }
  }
  else
  {
    sortBrickList(-1, eye, normal, isOrtho);
  }

  // Translate object by its position:
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();

  if (_numThreads > 0)
  {
    GLfloat* modelview = new GLfloat[16];
    glGetFloatv(GL_MODELVIEW_MATRIX , modelview);

    GLfloat* projection = new GLfloat[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projection);

    // Make sure that the _threadData array is only changed again (due to the
    // fact that the main thread already wants to sort the brick list again for
    // the next frame) until all workers have finished rendering the current
    // frame. Of course, this barrier has to be reinitialized frame per frame.
    pthread_barrier_init(&_renderReadyBarrier, NULL, _numThreads + 1);

    pthread_barrier_init(&_compositingBarrier, NULL, _numThreads + 1);

    for (i = 0; i < _numThreads; ++i)
    {
      _threadData[i].probeMin = vvVector3(&probeMin);
      _threadData[i].probeMax = vvVector3(&probeMax);
      _threadData[i].min = vvVector3(&min);
      _threadData[i].max = vvVector3(&max);
      _threadData[i].delta = vvVector3(&delta);
      _threadData[i].farthest = vvVector3(&farthest);
      _threadData[i].normal = vvVector3(&normal);
      _threadData[i].numSlices = numSlices;
      _threadData[i].texMin[0] = texMin[0];
      _threadData[i].texMin[1] = texMin[1];
      _threadData[i].texMin[2] = texMin[2];
      _threadData[i].minSlice = minSlice;
      _threadData[i].maxSlice = maxSlice;

      _threadData[i].modelview = modelview;
      _threadData[i].projection = projection;

      _threadData[i].renderReadyBarrier = &_renderReadyBarrier;
      _threadData[i].compositingBarrier = &_compositingBarrier;
    }

    // The sorted brick lists are distributed among the threads. All other data specific
    // to the workers is assigned either. Now broadcast a signal telling the worker threads'
    // rendering loops to resume.
    pthread_barrier_wait(&_renderStartBarrier);

    // Resume sequential operations only after all worker threads have finished rendering.
    pthread_barrier_wait(&_renderReadyBarrier);

    // Do compositing.
    pthread_barrier_wait(&_compositingBarrier);

    delete[] projection;
    delete[] modelview;

    // Destroy this barrier to resume execution. The barrier will be reinitialized with
    // the next frame.
    pthread_barrier_destroy(&_renderReadyBarrier);

    // Blend the images from the worker threads onto a 2D-texture.

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    // Orthographic projection.
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    // Fix the proxy quad for the frame buffer texture.
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Setup compositing.
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    // Traverse the bsp tree. As a side effect,
    // the textured proxy geometry will be
    // rendered by the bsp tree's visitor.
    _bspTree->traverse(eye);

    //performLoadBalancing();

    vvGLTools::printGLError("orthotexture");
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    pthread_barrier_destroy(&_compositingBarrier);
  }
  else
  {
    // Volume render a 3D texture:
    enableTexture(GL_TEXTURE_3D_EXT);

    // If the render target is of base class type, nothing
    // will happen here. Offscreen buffers e.g. need to
    // cleanup the content from the last rendering step.
    _renderTarget->clearBuffer();

    if(_proxyGeometryOnGpu)
    {
      // Per frame parameters.
      _isectShader->setModelViewProj(0, "modelViewProj");

      _isectShader->setParameter1f(0, "delta", delta.length());
      _isectShader->setParameter3f(0, "planeNormal", normal[0], normal[1], normal[2]);
    }

    // Count empty-space leaped bricks.
    discarded = 0;

    bool lastInsideProbe = false; // don't update brick edge vectors if they didn't change
    for(BrickList::iterator it = _sortedList.begin(); it != _sortedList.end(); ++it)
    {
      if (!(*it)->visible)
      {
        ++discarded;
      }
      else
      {
        (*it)->render(this, normal, farthest, delta, probeMin, probeMax,
                    texNames,
                    _isectShader, !((*it)->insideProbe && lastInsideProbe));
        lastInsideProbe = (*it)->insideProbe;
      }
    }

    vvDebugMsg::msg(3, "Bricks discarded: ", discarded);
  }

  disableTexture(GL_TEXTURE_3D_EXT);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

/** Partially renders the volume data set. Only that part is rendered that is
 *  represented by the provided bricks. Method is intended to be used as
 *  callback function for posix thread's create method.
 */
void* vvTexRend::threadFuncTexBricks(void* threadargs)
{
#ifndef HAVE_X11
  return NULL;
#else
  ThreadArgs* data = reinterpret_cast<ThreadArgs*>(threadargs);

  if (!glXMakeCurrent(data->display, data->drawable, data->glxContext))
  {
    vvDebugMsg::msg(3, "Couldn't make glx context current");
  }
  else
  {
    vvShaderManager* pixelShader = vvShaderFactory::provideShaderManager(VV_CG_MANAGER);

    vvStopwatch* stopwatch;

    /////////////////////////////////////////////////////////
    // Wait until sequential initialization is done.
    /////////////////////////////////////////////////////////

    // Not until all worker threads as well as the primarcgIsectProgram, cgIsectProfiley thread have called
    // pthread_barrier_wait exactly once, parallel operations resume. Note:
    // The main thread (as not occupying its role as a worker thread) will
    // call pthread_barrier_wait on initBarrier right at the time when initialization
    // is performed, i.e. on the sequential path of the program rather than in
    // a parallel callback.
    pthread_barrier_wait(data->initBarrier);

#if !defined(__APPLE__)
    // Init glew.
    glewInit();
#endif

    /////////////////////////////////////////////////////////
    // Make textures.
    /////////////////////////////////////////////////////////
    pthread_mutex_lock(data->makeTextureMutex);
    glGenTextures(1, &data->pixLUTName);
    data->renderer->initPixelShaders(pixelShader);
    data->renderer->_areBricksCreated = false;
    // Generate a set of gl tex names private to this thread.
    // This solution differs from the one chosen for sequential
    // execution where the texNames array is a member.
    data->renderer->makeTextures(data->privateTexNames, &data->numTextures,
                                 data->pixLUTName, data->rgbaLUT);
    pthread_mutex_unlock(data->makeTextureMutex);

    // Now that the textures are built, the bricks may be distributed
    // by the main thread.
    pthread_barrier_wait(data->distributeBricksBarrier);
    // For the main thread, this will take some time... .
    pthread_barrier_wait(data->distributedBricksBarrier);

    /////////////////////////////////////////////////////////
    // Setup an appropriate GL state.
    /////////////////////////////////////////////////////////
    data->renderer->setGLenvironment();

    vvShaderManager* isectShader = vvShaderFactory::provideShaderManager(VV_CG_MANAGER);

    if(data->renderer->_proxyGeometryOnGpu)
    {
      data->renderer->initIntersectionShader(isectShader);
      data->renderer->setupIntersectionParameters(isectShader);
      data->renderer->enableIntersectionShader(isectShader);
    }

    GLuint tex;

    // Init framebuffer objects.
    glGenFramebuffersEXT(1, &data->renderer->_frameBufferObjects[data->threadId]);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, data->renderer->_frameBufferObjects[data->threadId]);
    glGenRenderbuffersEXT(1, &data->renderer->_depthBuffers[data->threadId]);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, data->renderer->_depthBuffers[data->threadId]);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT32, data->width, data->height);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                                 GL_RENDERBUFFER_EXT, data->renderer->_depthBuffers[data->threadId]);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, data->width, data->height, 0, GL_RGBA, GL_FLOAT, NULL);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D,
                              tex, 0);
    GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    switch (status)
    {
    case GL_FRAMEBUFFER_COMPLETE_EXT:
      std::cerr << "Framebuffer complete" << std::endl;
      break;
    case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
      std::cerr << "Framebuffer unsupported" << std::endl;
      break;
    default:
      std::cerr << "Unknown error while initializing framebuffer objects" << std::endl;
      break;
    }

    // Init stop watch.
    stopwatch = new vvStopwatch();

    // TODO: identify if method is called from an ordinary worker thread or
    // the main thread. If the latter is the case, ensure that the environment
    // is reset properly after rendering.

    // Main render loop that is suspended while the sequential program flow is executing
    // and resumes when the texture bricks are sorted in back to front order and the
    // thread specific data is supplied.
    while (1)
    {
      // Don't start rendering until the bricks are sorted in back to front order
      // and appropriatly distributed among the respective worker threads. The main
      // thread will issue an alert if this is the case.
      pthread_barrier_wait(data->renderStartBarrier);
      data->renderer->enableLUTMode(pixelShader, data->pixLUTName);

      // Use alpha correction in indexed mode: adapt alpha values to number of textures:
      if (data->renderer->instantClassification())
      {
        const float diagonalVoxels = sqrtf(float(data->renderer->vd->vox[0] * data->renderer->vd->vox[0] +
          data->renderer->vd->vox[1] * data->renderer->vd->vox[1] +
          data->renderer->vd->vox[2] * data->renderer->vd->vox[2]));
          data->renderer->updateLUT(diagonalVoxels / float(data->numSlices),
                                    data->pixLUTName, data->rgbaLUT);
      }
      /////////////////////////////////////////////////////////
      // Start rendering.
      /////////////////////////////////////////////////////////
      glEnable(GL_TEXTURE_3D_EXT);

      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, data->renderer->_frameBufferObjects[data->threadId]);
      glClearColor(0.0, 0.0, 0.0, 0.0);
      glClear(GL_COLOR_BUFFER_BIT);

      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, data->width, data->height);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glMultMatrixf(data->projection);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glMultMatrixf(data->modelview);

      stopwatch->start();

      if(data->renderer->_proxyGeometryOnGpu)
      {
        // Per frame parameters.
        isectShader->setModelViewProj(0, "modelViewProj");

        isectShader->setParameter1f(0, "delta", data->delta.length());
        isectShader->setParameter3f(0, "planeNormal", data->normal[0], data->normal[1], data->normal[2]);
      }

      for(BrickList::iterator it = data->sortedList.begin(); it != data->sortedList.end(); ++it)
      {
        Brick *tmp = *it;
        tmp->render(data->renderer, data->normal,
                    data->farthest, data->delta,
                    data->probeMin, data->probeMax,
                    data->privateTexNames, isectShader, true);
      }
      glDisable(GL_TEXTURE_3D_EXT);
      data->renderer->disableLUTMode(pixelShader);

      // When all worker threads and the main thread have waited once (per frame) on
      // this barrier, all workers have finished rendering this frame and the main
      // thread may resume.
      pthread_barrier_wait(data->renderReadyBarrier);

      // Blend the images to one single image.

      // This call switches the currently readable buffer to the fbo offscreen buffer.
      glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
      glReadPixels(0, 0, data->width, data->height, GL_RGBA, GL_FLOAT, data->pixels);
      glPopAttrib();
      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

#if 1
      // Output to screen.
      glBindFramebufferEXT(GL_READ_FRAMEBUFFER_EXT, data->renderer->_frameBufferObjects[data->threadId]);
      glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT, 0);
      glBlitFramebufferEXT(0, 0, data->width, data->height,
                           0, 0, data->width, data->height,
                           GL_COLOR_BUFFER_BIT, GL_LINEAR);
      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
      glFlush();
#endif

      // Store the time to render. Based upon this, dynamic load balancing will be performed.
      data->lastRenderTime = stopwatch->getTime();

      pthread_barrier_wait(data->compositingBarrier);

      // Finally update the transfer function in a synchronous fashion.
      if (data->transferFunctionChanged)
      {
        data->renderer->updateTransferFunction(data->pixLUTName, data->rgbaLUT);
      }
    }

    // Exited render loop - perform cleanup.
    //data->texRend->removeTextures(privateTexNames);
    delete isectShader;
    delete pixelShader;
    delete stopwatch;
  }
  pthread_exit(NULL);
#endif
}

/** Partially renders the volume data set. Only renders the outlines!
 *  Method is intended to be used as callback function for posix thread's
 *  create method.
 */
void* vvTexRend::threadFuncBricks(void* threadargs)
{
#ifndef HAVE_X11
  return NULL;
#else
  ThreadArgs* data = reinterpret_cast<ThreadArgs*>(threadargs);

  if (!glXMakeCurrent(data->display, data->drawable, data->glxContext))
  {
    vvDebugMsg::msg(3, "Couldn't make glx context current");
  }
  else
  {
    vvStopwatch* stopwatch;

    /////////////////////////////////////////////////////////
    // Wait until sequential initialization is done.
    /////////////////////////////////////////////////////////

    // Not until all worker threads as well as the primary thread have called
    // pthread_barrier_wait exactly once, parallel operations resume. Note:
    // The main thread (as not occupying its role as a worker thread) will
    // call pthread_barrier_wait on initBarrier right at the time when initialization
    // is performed, i.e. on the sequential path of the program rather than in
    // a parallel callback.
    pthread_barrier_wait(data->initBarrier);

    /////////////////////////////////////////////////////////
    // Make textures.
    /////////////////////////////////////////////////////////
    pthread_mutex_lock(data->makeTextureMutex);

    data->renderer->texelsize=4;
    data->renderer->internalTexFormat = GL_RGBA;
    data->renderer->texFormat = GL_RGBA;
    data->renderer->_areBricksCreated = false;
    data->renderer->computeBrickSize();
    data->renderer->updateTransferFunction(data->pixLUTName, data->rgbaLUT);
    // Generate a set of gl tex names private to this thread.
    // This solution differs from the one chosen for sequential
    // execution where the texNames array is a member.
    data->renderer->makeTextures(data->privateTexNames, &data->numTextures,
                                 data->pixLUTName, data->rgbaLUT);
    pthread_mutex_unlock(data->makeTextureMutex);

    // Now that the textures are built, the bricks may be distributed
    // by the main thread.
    pthread_barrier_wait(data->distributeBricksBarrier);
    // For the main thread, this will take some time... .
    pthread_barrier_wait(data->distributedBricksBarrier);

    /////////////////////////////////////////////////////////
    // Setup an appropriate GL state.
    /////////////////////////////////////////////////////////
    data->renderer->setGLenvironment();

    // Init stop watch.
    stopwatch = new vvStopwatch();

    // TODO: identify if method is called from an ordinary worker thread or
    // the main thread. If the latter is the case, ensure that the environment
    // is reset properly after rendering.

    // Main render loop that is suspended while the sequential program flow is executing
    // and resumes when the texture bricks are sorted in back to front order and the
    // thread specific data is supplied.
    while (1)
    {
      // Don't start rendering until the bricks are sorted in back to front order
      // and appropriatly distributed among the respective worker threads. The main
      // thread will issue an alert if this is the case.
      pthread_barrier_wait(data->renderStartBarrier);

      // Use alpha correction in indexed mode: adapt alpha values to number of textures:
      if (data->renderer->instantClassification())
      {
        float diagonalVoxels;
        diagonalVoxels = sqrtf(float(data->renderer->vd->vox[0] * data->renderer->vd->vox[0] +
          data->renderer->vd->vox[1] * data->renderer->vd->vox[1] +
          data->renderer->vd->vox[2] * data->renderer->vd->vox[2]));
        data->renderer->updateLUT(diagonalVoxels / float(data->numSlices),
                                  data->pixLUTName, data->rgbaLUT);
      }

      /////////////////////////////////////////////////////////
      // Start rendering.
      /////////////////////////////////////////////////////////
      glEnable(GL_TEXTURE_3D_EXT);
      int drawn;

      glClearColor(0.0, 0.0, 0.0, 0.0);
      glClear(GL_COLOR_BUFFER_BIT);
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glMultMatrixf(data->projection);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glMultMatrixf(data->modelview);

      stopwatch->start();

      for(std::vector<vvConvexObj *>::iterator it = data->halfSpace->getObjects()->begin();
          it != data->halfSpace->getObjects()->end();
          ++it)
      {
        Brick *tmp = dynamic_cast<Brick *>(*it);
        drawn = 0;

        for (int i = 0; i < 3; i++)
        {
          if (tmp->min.e[i] < data->probeMin.e[i])
            data->min.e[i] = data->probeMin.e[i];
          else
            data->min.e[i] = tmp->min.e[i];

          if (tmp->max.e[i] > data->probeMax.e[i])
            data->max.e[i] = data->probeMax.e[i];
          else
            data->max.e[i] = tmp->max.e[i];
        }

        glBegin(GL_LINES);
        glColor4f(1.0, 1.0, 1.0, 1.0);
        glVertex3f(data->min[0], data->min[1], data->min[2]);
          glVertex3f(data->max[0], data->min[1], data->min[2]);

          glVertex3f(data->min[0], data->min[1], data->min[2]);
          glVertex3f(data->min[0], data->max[1], data->min[2]);

          glVertex3f(data->min[0], data->min[1], data->min[2]);
          glVertex3f(data->min[0], data->min[1], data->max[2]);

          glVertex3f(data->max[0], data->max[1], data->max[2]);
          glVertex3f(data->min[0], data->max[1], data->max[2]);

          glVertex3f(data->max[0], data->max[1], data->max[2]);
          glVertex3f(data->max[0], data->min[1], data->max[2]);

          glVertex3f(data->max[0], data->max[1], data->max[2]);
          glVertex3f(data->max[0], data->max[1], data->min[2]);

          glVertex3f(data->max[0], data->min[1], data->min[2]);
          glVertex3f(data->max[0], data->max[1], data->min[2]);

          glVertex3f(data->max[0], data->min[1], data->min[2]);
          glVertex3f(data->max[0], data->min[1], data->max[2]);

          glVertex3f(data->min[0], data->max[1], data->min[2]);
          glVertex3f(data->max[0], data->max[1], data->min[2]);

          glVertex3f(data->min[0], data->max[1], data->min[2]);
          glVertex3f(data->min[0], data->max[1], data->max[2]);

          glVertex3f(data->min[0], data->min[1], data->max[2]);
          glVertex3f(data->max[0], data->min[1], data->max[2]);

          glVertex3f(data->min[0], data->min[1], data->max[2]);
          glVertex3f(data->min[0], data->max[1], data->max[2]);
        glEnd();
        glFlush();
        vvDebugMsg::msg(3, "Number of textures drawn: ", drawn);
      }

      // When all worker threads and the main thread have waited once (per frame) on
      // this barrier, all workers have finished rendering this frame and the main
      // thread may resume.
      pthread_barrier_wait(data->renderReadyBarrier);

      // Blend the images to one single image.
      glReadPixels(0, 0, data->width, data->height, GL_RGBA, GL_UNSIGNED_BYTE, data->pixels);

      // Store the time to render. Based upon this, dynamic load balancing will be performed.
      data->lastRenderTime = stopwatch->getTime();

      pthread_barrier_wait(data->compositingBarrier);
    }

    // Exited render loop - perform cleanup.
    //data->texRend->removeTextures(privateTexNames);

    delete stopwatch;
  }
  pthread_exit(NULL);
#endif
}

/** Renders the outline of the bricks, but not the volume data set.
 */
void vvTexRend::renderBricks(vvMatrix* mv)
{
  vvMatrix invMV;                                 // inverse of model-view matrix
  vvMatrix pm;                                    // OpenGL projection matrix
  vvVector3 farthest;                             // volume vertex farthest from the viewer
  vvVector3 delta;                                // distance vector between textures [object space]
  vvVector3 normal;                               // normal vector of textures
  vvVector3 origin;                               // origin (0|0|0) transformed to object space
  vvVector3 eye;                                  // user's eye position [object space]
  vvVector3 pos;                                  // volume location
  vvVector3 probePosObj;                          // probe midpoint [object space]
  vvVector3 probeSizeObj;                         // probe size [object space]
  vvVector3 probeMin, probeMax;                   // probe min and max coordinates [object space]
  vvVector3 min, max;                             // min and max pos of current brick (cut with probe)
  vvVector3 texMin;                               // minimum texture coordinate
  float     diagonal;                             // probe diagonal [object space]
  int       numSlices;                            // number of texture slices along diagonal
  int       drawn;                                // counter for drawn textures
  int       i;                                    // general counters

  vvDebugMsg::msg(3, "vvTexRend::renderTexBricks()");

  // needs 3D texturing extension
  if (!extTex3d) return;

  if (_brickList.empty()) return;

  pos.copy(&vd->pos);

  // Calculate inverted modelview matrix:
  invMV.copy(mv);
  invMV.invert();

  // Find eye position:
  getEyePosition(&eye);
  eye.multiply(&invMV);

  calcProbeDims(probePosObj, probeSizeObj, probeMin, probeMax);

  // Compute length of probe diagonal [object space]:
  diagonal = sqrtf(
    probeSizeObj[0] * probeSizeObj[0] +
    probeSizeObj[1] * probeSizeObj[1] +
    probeSizeObj[2] * probeSizeObj[2]);

  numSlices = int(_renderState._quality * 100.0f);
  if (numSlices < 1) numSlices = 1;               // make sure that at least one slice is drawn

  vvDebugMsg::msg(3, "Number of textures rendered per brick: ", numSlices);


  if (_numThreads == 0)
  {
    // Use alpha correction in indexed mode: adapt alpha values to number of textures:
    if (instantClassification())
    {
      float diagonalVoxels;
      diagonalVoxels = sqrtf(float(vd->vox[0] * vd->vox[0] +
        vd->vox[1] * vd->vox[1] +
        vd->vox[2] * vd->vox[2]));
      updateLUT(diagonalVoxels / float(numSlices), pixLUTName, rgbaLUT);
    }
  }

  // Get projection matrix:
  getProjectionMatrix(&pm);
  bool isOrtho = pm.isProjOrtho();

  // Compute normal vector of textures using the following strategy:
  // For orthographic projections or if viewDir is (0|0|0) use
  // (0|0|1) as the normal vector.
  // Otherwise use objDir as the normal.
  // Exception: if user's eye is inside object and probe mode is off,
  // then use viewDir as the normal.
  if (_renderState._clipMode)
  {
    normal.copy(&_renderState._clipNormal);
  }
  else if (isOrtho || (viewDir.e[0] == 0.0f && viewDir.e[1] == 0.0f && viewDir.e[2] == 0.0f))
  {
    // Draw slices parallel to projection plane:
    normal.set(0.0f, 0.0f, 1.0f);                 // (0|0|1) is normal on projection plane
    normal.multiply(&invMV);
    origin.zero();
    origin.multiply(&invMV);
    normal.sub(&origin);
  }
  else if (!_renderState._isROIUsed && isInVolume(&eye))
  {
    // Draw slices perpendicular to viewing direction:
    normal.copy(&viewDir);
    normal.negate();                              // viewDir points away from user, the normal should point towards them
  }
  else
  {
    // Draw slices perpendicular to line eye->object:
    normal.copy(&objDir);
    normal.negate();
  }

  normal.normalize();
  delta.copy(&normal);
  delta.scale(diagonal / ((float)numSlices));

  // Compute farthest point to draw texture at:
  farthest.copy(&delta);
  farthest.scale((float)(numSlices - 1) * -0.5f);
  farthest.add(&probePosObj);

  vvVector3 temp, clipPosObj, normClipPoint;
  float maxDist;

  if (_renderState._clipMode)                     // clipping plane present?
  {
    // Adjust numSlices and set farthest point so that textures are only
    // drawn up to the clipPoint. (Delta may not be changed
    // due to the automatic opacity correction.)
    // First find point on clipping plane which is on a line perpendicular
    // to clipping plane and which traverses the origin:
    temp.copy(&delta);
    temp.scale(-0.5f);
    farthest.add(&temp);                          // add a half delta to farthest
    clipPosObj.copy(&_renderState._clipPoint);
    clipPosObj.sub(&pos);
    temp.copy(&probePosObj);
    temp.add(&normal);
    normClipPoint.isectPlaneLine(&normal, &clipPosObj, &probePosObj, &temp);
    maxDist = farthest.distance(&normClipPoint);
    numSlices = (int)(maxDist / delta.length()) + 1;
    temp.copy(&delta);
    temp.scale((float)(1 - numSlices));
    farthest.copy(&normClipPoint);
    farthest.add(&temp);
    if (_renderState._clipSingleSlice)
    {
      // Compute slice position:
      delta.scale((float)(numSlices-1));
      farthest.add(&delta);
      numSlices = 1;
    }
  }

  getBricksInProbe(probePosObj, probeSizeObj);

  markBricksInFrustum();

  if (_numThreads > 0)
  {
    for (i = 0; i < _numThreads; ++i)
    {
      sortBrickList(i, eye, normal, isOrtho);
    }

    if (_somethingChanged)
    {
      distributeBricks();
    }
  }
  else
  {
    sortBrickList(-1, eye, normal, isOrtho);
  }

  // Translate object by its position:
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  //glTranslatef(pos.e[0], pos.e[1], pos.e[2]);

  if (_numThreads > 0)
  {
    GLfloat* modelview = new GLfloat[16];
    glGetFloatv(GL_MODELVIEW_MATRIX , modelview);

    GLfloat* projection = new GLfloat[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projection);

    // Make sure that the _threadData array is only changed again (due to the
    // fact that the main thread already wants to sort the brick list again for
    // the next frame) until all workers have finished rendering the current
    // frame. Of course, this barrier has to be reinitialized frame per frame.
    pthread_barrier_init(&_renderReadyBarrier, NULL, _numThreads + 1);

    pthread_barrier_init(&_compositingBarrier, NULL, _numThreads + 1);

    for (i = 0; i < _numThreads; ++i)
    {
      _threadData[i].probeMin = vvVector3(&probeMin);
      _threadData[i].probeMax = vvVector3(&probeMax);
      _threadData[i].min = vvVector3(&min);
      _threadData[i].max = vvVector3(&max);
      _threadData[i].delta = vvVector3(&delta);
      _threadData[i].farthest = vvVector3(&farthest);
      _threadData[i].normal = vvVector3(&normal);
      _threadData[i].numSlices = numSlices;
      _threadData[i].texMin[0] = texMin[0];
      _threadData[i].texMin[1] = texMin[1];
      _threadData[i].texMin[2] = texMin[2];
      _threadData[i].minSlice = minSlice;
      _threadData[i].maxSlice = maxSlice;

      _threadData[i].modelview = modelview;
      _threadData[i].projection = projection;

      _threadData[i].renderReadyBarrier = &_renderReadyBarrier;
      _threadData[i].compositingBarrier = &_compositingBarrier;
    }

    // The sorted brick lists are distributed among the threads. All other data specific
    // to the workers is assigned either. Now broadcast a signal telling the worker threads'
    // rendering loops to resume.
    pthread_barrier_wait(&_renderStartBarrier);

    // Resume sequential operations only after all worker threads have finished rendering.
    pthread_barrier_wait(&_renderReadyBarrier);

    // Do compositing.
    pthread_barrier_wait(&_compositingBarrier);

    delete[] projection;
    delete[] modelview;

    // Destroy this barrier to resume execution. The barrier will be reinitialized with
    // the next frame.
    pthread_barrier_destroy(&_renderReadyBarrier);

    // Blend the images from the worker threads onto a 2D-texture.

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    // Orthographic projection.
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    // Fix the proxy quad for the frame buffer texture.
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Setup compositing.
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    // Traverse the bsp tree. As a side effect,
    // the textured proxy geometry will be
    // rendered by the bsp tree's visitor.
    _bspTree->traverse(eye);

    performLoadBalancing();

    vvGLTools::printGLError("orthotexture");
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    pthread_barrier_destroy(&_compositingBarrier);
  }
  else
  {
    // Disable pixlut shader, brick are uniformly colored.
    if (voxelType == VV_PIX_SHD)
    {
      _pixelShader->disableShader(_currentShader);
    }
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_LINES);
    glColor4f(1.0, 1.0, 1.0, 1.0);
    for(BrickList::iterator it = _sortedList.begin(); it != _sortedList.end(); ++it)
    {
      Brick *tmp = *it;
      drawn = 0;

      for (i = 0; i < 3; i++)
      {
        if (tmp->min.e[i] < probeMin.e[i])
          min.e[i] = probeMin.e[i];
        else
          min.e[i] = tmp->min.e[i];

        if (tmp->max.e[i] > probeMax.e[i])
          max.e[i] = probeMax.e[i];
        else
          max.e[i] = tmp->max.e[i];
      }

      glVertex3f(min[0], min[1], min[2]);
      glVertex3f(max[0], min[1], min[2]);

      glVertex3f(min[0], min[1], min[2]);
      glVertex3f(min[0], max[1], min[2]);

      glVertex3f(min[0], min[1], min[2]);
      glVertex3f(min[0], min[1], max[2]);

      glVertex3f(max[0], max[1], max[2]);
      glVertex3f(min[0], max[1], max[2]);

      glVertex3f(max[0], max[1], max[2]);
      glVertex3f(max[0], min[1], max[2]);

      glVertex3f(max[0], max[1], max[2]);
      glVertex3f(max[0], max[1], min[2]);

      glVertex3f(max[0], min[1], min[2]);
      glVertex3f(max[0], max[1], min[2]);

      glVertex3f(max[0], min[1], min[2]);
      glVertex3f(max[0], min[1], max[2]);

      glVertex3f(min[0], max[1], min[2]);
      glVertex3f(max[0], max[1], min[2]);

      glVertex3f(min[0], max[1], min[2]);
      glVertex3f(min[0], max[1], max[2]);

      glVertex3f(min[0], min[1], max[2]);
      glVertex3f(max[0], min[1], max[2]);

      glVertex3f(min[0], min[1], max[2]);
      glVertex3f(min[0], max[1], max[2]);

      vvDebugMsg::msg(3, "Number of textures drawn: ", drawn);
    }
    glEnd();
  }
  disableTexture(GL_TEXTURE_3D_EXT);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

void vvTexRend::updateFrustum()
{
  float pm[16];
  float mvm[16];
  vvMatrix proj, modelview, clip;

  // Get the current projection matrix from OpenGL
  glGetFloatv(GL_PROJECTION_MATRIX, pm);
  proj.getGL(pm);

  // Get the current modelview matrix from OpenGL
  glGetFloatv(GL_MODELVIEW_MATRIX, mvm);
  modelview.getGL(mvm);

  clip = proj;
  clip.multiplyPre(&modelview);

  // extract the planes of the viewing frustum

  // left plane
  _frustum[0].set(clip.e[3][0]+clip.e[0][0], clip.e[3][1]+clip.e[0][1],
    clip.e[3][2]+clip.e[0][2], clip.e[3][3]+clip.e[0][3]);
  // right plane
  _frustum[1].set(clip.e[3][0]-clip.e[0][0], clip.e[3][1]-clip.e[0][1],
    clip.e[3][2]-clip.e[0][2], clip.e[3][3]-clip.e[0][3]);
  // top plane
  _frustum[2].set(clip.e[3][0]-clip.e[1][0], clip.e[3][1]-clip.e[1][1],
    clip.e[3][2]-clip.e[1][2], clip.e[3][3]-clip.e[1][3]);
  // bottom plane
  _frustum[3].set(clip.e[3][0]+clip.e[1][0], clip.e[3][1]+clip.e[1][1],
    clip.e[3][2]+clip.e[1][2], clip.e[3][3]+clip.e[1][3]);
  // near plane
  _frustum[4].set(clip.e[3][0]+clip.e[2][0], clip.e[3][1]+clip.e[2][1],
    clip.e[3][2]+clip.e[2][2], clip.e[3][3]+clip.e[2][3]);
  // far plane
  _frustum[5].set(clip.e[3][0]-clip.e[2][0], clip.e[3][1]-clip.e[2][1],
    clip.e[3][2]-clip.e[2][2], clip.e[3][3]-clip.e[2][3]);
}

bool vvTexRend::insideFrustum(const vvVector3 &min, const vvVector3 &max)
{
  vvVector3 pv, normal;

  // get p-vertex (that's the farthest vertex in the direction of the normal plane
  for (int i = 0; i < 6; i++)
  {
    normal.set(_frustum[i][0], _frustum[i][1], _frustum[i][2]);

    for(int j = 0; j < 8; ++j)
    {
      for(int c = 0; c < 3; ++c)
        pv[c] = (j & (1<<c)) ? min[c] : max[c];

      if ((pv.dot(&normal) + _frustum[i][3]) < 0)
      {
        return false;
      }
    }
  }

  return true;
}

bool vvTexRend::intersectsFrustum(const vvVector3 &min, const vvVector3 &max)
{
  vvVector3 pv, normal;

  // get p-vertex (that's the farthest vertex in the direction of the normal plane
  for (int i = 0; i < 6; i++)
  {
    if (_frustum[i][0] > 0.0)
      pv[0] = max[0];
    else
      pv[0] = min[0];
    if (_frustum[i][1] > 0.0)
      pv[1] = max[1];
    else
      pv[1] = min[1];
    if (_frustum[i][2] > 0.0)
      pv[2] = max[2];
    else
      pv[2] = min[2];

    normal.set(_frustum[i][0], _frustum[i][1], _frustum[i][2]);

    if ((pv.dot(&normal) + _frustum[i][3]) < 0)
    {
      return false;
    }
  }

  return true;
}

bool vvTexRend::testBrickVisibility(Brick* brick)
{
  return intersectsFrustum(brick->min, brick->max);
}

bool vvTexRend::testBrickVisibility(Brick* brick, const vvMatrix& mvpMat)
{
  //sample the brick at many point and test them all for visibility
  const float numSteps = 3;
  const float xStep = (brick->max[0] - brick->min[0]) / (numSteps - 1.0f);
  const float yStep = (brick->max[1] - brick->min[1]) / (numSteps - 1.0f);
  const float zStep = (brick->max[2] - brick->min[2]) / (numSteps - 1.0f);
  for(int i = 0; i < numSteps; i++)
  {
    float x = brick->min.e[0] + xStep * i;
    for(int j = 0; j < numSteps; j++)
    {
      float y = brick->min.e[1] + yStep * j;
      for(int k = 0; k < numSteps; k++)
      {
        float z = brick->min.e[2] + zStep * k;
        vvVector3 clipPnt(x, y, z);
        clipPnt.multiply(&mvpMat);

        //test if this point falls within screen space
        if(clipPnt.e[0] >= -1.0 && clipPnt.e[0] <= 1.0 &&
          clipPnt.e[1] >= -1.0 && clipPnt.e[1] <= 1.0)
        {
          return true;
        }
      }
    }
  }

  return false;
}

void vvTexRend::calcProbeDims(vvVector3& probePosObj, vvVector3& probeSizeObj, vvVector3& probeMin, vvVector3& probeMax)
{
  vvVector3 size, size2;                          // full and half object sizes
  vvVector3 pos;                                  // volume location
  vvVector3 maxSize;                              // probe edge length [object space]
  int i;

  pos.copy(&vd->pos);

  // Determine texture object dimensions and half object size as a shortcut:
  size.copy(vd->getSize());
  size2 = size;
  size2.scale(0.5);

  if (_renderState._isROIUsed)
  {
    // Convert probe midpoint coordinates to object space w/o position:
    probePosObj.copy(&_renderState._roiPos);
    probePosObj.sub(&pos);                        // eliminate object position from probe position

    // Compute probe min/max coordinates in object space:
    maxSize[0] = _renderState._roiSize[0] * size2[0];
    maxSize[1] = _renderState._roiSize[1] * size2[1];
    maxSize[2] = _renderState._roiSize[2] * size2[2];

    probeMin = probePosObj;
    probeMin.sub(&maxSize);
    probeMax = probePosObj;
    probeMax.add(&maxSize);

    // Constrain probe boundaries to volume data area:
    for (i = 0; i < 3; ++i)
    {
      if (probeMin[i] > size2[i] || probeMax[i] < -size2[i])
      {
        vvDebugMsg::msg(3, "probe outside of volume");
        return;
      }
      if (probeMin[i] < -size2[i]) probeMin[i] = -size2[i];
      if (probeMax[i] >  size2[i]) probeMax[i] =  size2[i];
      probePosObj[i] = (probeMax[i] + probeMin[i]) / 2.0f;
    }

    // Compute probe edge lengths:
    probeSizeObj = probeMax;
    probeSizeObj.sub(&probeMin);
  }
  else                                            // probe mode off
  {
    probeSizeObj.copy(&size);
    probePosObj.copy(&vd->pos);
    probeMin = probePosObj-size2;
    probeMax = probePosObj+size2;
  }
}

void vvTexRend::markBricksInFrustum()
{
  vvVector3 probePosObj, probeSizeObj, probeMin, probeMax;
  calcProbeDims(probePosObj, probeSizeObj, probeMin, probeMax);

  updateFrustum();

  const bool inside = insideFrustum(probeMin, probeMax);
  const bool outside = !intersectsFrustum(probeMin, probeMax);
  
  for(BrickList::iterator it = _sortedList.begin(); it != _sortedList.end(); ++it)
    if(inside || outside)
      (*it)->visible = inside;
    else
      (*it)->visible = testBrickVisibility( *it );
}

void vvTexRend::getBricksInProbe(vvVector3 pos, vvVector3 size)
{
  if(!_renderState._isROIChanged && vd->getCurrentFrame() == _lastFrame)
    return;

  _lastFrame = vd->getCurrentFrame();
  _renderState._isROIChanged = false;

  _insideList.clear();

  vvVector3 tmpVec = size;
  tmpVec.scale(0.5);

  const vvVector3 min = pos - tmpVec;
  const vvVector3 max = pos + tmpVec;

  int countVisible = 0, countInvisible = 0;

  int frame = vd->getCurrentFrame();
  for(BrickList::iterator it = _nonemptyList[frame].begin(); it != _nonemptyList[frame].end(); ++it)
  {
    Brick *tmp = *it;
    if ((tmp->min.e[0] <= max.e[0]) && (tmp->max.e[0] >= min.e[0]) &&
      (tmp->min.e[1] <= max.e[1]) && (tmp->max.e[1] >= min.e[1]) &&
      (tmp->min.e[2] <= max.e[2]) && (tmp->max.e[2] >= min.e[2]))
    {
      _insideList.push_back(tmp);
      if ((tmp->min.e[0] >= min.e[0]) && (tmp->max.e[0] <= max.e[0]) &&
        (tmp->min.e[1] >= min.e[1]) && (tmp->max.e[1] <= max.e[1]) &&
        (tmp->min.e[2] >= min.e[2]) && (tmp->max.e[2] <= max.e[2]))
        tmp->insideProbe = !tmp->atBorder;
      else
        tmp->insideProbe = false;
      ++countVisible;
    }
    else
    {
      ++countInvisible;
    }
  }

  _sortedList.clear();
  for(std::vector<vvConvexObj *>::iterator it = _insideList.begin(); it != _insideList.end(); ++it)
     _sortedList.push_back(static_cast<Brick *>(*it));
}

struct BrickCompare
{
  bool operator()(Brick *a, Brick *b)
  {
    return a->dist > b->dist;
  }
};

void vvTexRend::sortBrickList(const int threadId, const vvVector3& pos, const vvVector3& normal, const bool isOrtho)
{
  if (_numThreads > 0)
  {
    _threadData[threadId].sortedList.clear();
    for(std::vector<vvConvexObj *>::iterator it = _threadData[threadId].halfSpace->getObjects()->begin();
        it != _threadData[threadId].halfSpace->getObjects()->end();
        ++it)
    {
      Brick *brick = static_cast<Brick*>(*it);
      if (isOrtho)
        brick->dist = -brick->pos.dot(&normal);
      else
        brick->dist = (brick->pos - pos).length();

      _threadData[threadId].sortedList.push_back(brick);
    }
    std::sort(_threadData[threadId].sortedList.begin(), _threadData[threadId].sortedList.end(), BrickCompare());
  }
  else
  {
    for(BrickList::iterator it = _sortedList.begin(); it != _sortedList.end(); ++it)
    {
      if (isOrtho)
        (*it)->dist = -(*it)->pos.dot(&normal);
      else
        (*it)->dist = ((*it)->pos - pos).length();
    }

    std::sort(_sortedList.begin(), _sortedList.end(), BrickCompare());
  }
}

void vvTexRend::performLoadBalancing()
{
  float* idealDistribution;
  float* actualDistribution;
  float deviation;
  float tmp;

  // Only redistribute if the deviation of the rendering times
  // exceeds a certain limit.
  float expectedValue = 0;
  for (int i = 0; i < _numThreads; ++i)
  {
    expectedValue += _threadData[i].lastRenderTime;
  }
  expectedValue /= static_cast<float>(_numThreads);

  // An ideal distribution is one where each rendering time for each thread
  // equals the expected value exactly.
  idealDistribution = new float[_numThreads];

  // This is (not necessarily) true in reality... .
  actualDistribution = new float[_numThreads];

  // Build both arrays.
  for (int i = 0; i < _numThreads; ++i)
  {
    idealDistribution[i] = expectedValue;
    actualDistribution[i] = _threadData[i].lastRenderTime;
  }

  deviation = vvToolshed::meanAbsError(idealDistribution, actualDistribution, _numThreads);

  // Normalize the deviation (to percent).
  deviation *= 100.0f;
  deviation /= expectedValue;

  // Only reorder if
  // a) a given deviation was exceeded
  // b) this happened at least x times
  if (deviation > MAX_DEVIATION)
  {
    if (_deviationExceedCnt > LIMIT)
    {
      tmp = 0.0f;

      // Rearrange the share for each thread.
      for (int i = 0; i < _numThreads; ++i)
      {
        float cmp = _threadData[i].lastRenderTime - expectedValue;
        if (cmp < MAX_IND_DEVIATION)
        {
          if ((_threadData[i].share + INC) <= 1.0f)
          {
            _threadData[i].share += INC;
          }
        }

        if (cmp > MAX_IND_DEVIATION)
        {
          if ((_threadData[i].share + INC) >= 0.0f)
          {
            _threadData[i].share -= INC;
          }
        }
        tmp += _threadData[i].share;
      }

      // Normalize partitioning.
      for (int i = 0; i < _numThreads; ++i)
      {
        _threadData[i].share /= tmp;
      }

      // Something changed ==> before rendering the next time,
      // the bricks will be redistributed.
      _somethingChanged = true;
    }
    else
    {
      ++_deviationExceedCnt;
    }
  }
  delete[] idealDistribution;
  delete[] actualDistribution;
}

//----------------------------------------------------------------------------
/** Render the volume using a 3D texture (needs 3D texturing extension).
  Spherical slices are surrounding the observer.
  @param view       model-view matrix
*/
void vvTexRend::renderTex3DSpherical(vvMatrix* view)
{
  int i, k;
  int ix, iy, iz;
  float  spacing;                                 // texture spacing
  float  maxDist = 0.0;
  float  minDist = 0.0;
  vvVector3 texSize;                              // size of 3D texture [object space]
  vvVector3 texSize2;                             // half size of 3D texture [object space]
  vvVector3 volumeVertices[8];
  vvVector3 size;
  vvMatrix invView;
  int numShells;

  vvDebugMsg::msg(3, "vvTexRend::renderTex3DSpherical()");

  if (!extTex3d) return;

  numShells = int(_renderState._quality * 100.0f);
  if (numShells < 1)                              // make sure that at least one shell is drawn
    numShells = 1;

  // Determine texture object dimensions:
  size.copy(vd->getSize());
  for (i=0; i<3; ++i)
  {
    texSize.e[i]  = size.e[i] * (float)texels[i] / (float)vd->vox[i];
    texSize2.e[i] = 0.5f * texSize.e[i];
  }

  invView.copy(view);
  invView.invert();

  // generates the vertices of the cube (volume) in world coordinates
  int vertexIdx = 0;
  for (ix=0; ix<2; ++ix)
    for (iy=0; iy<2; ++iy)
      for (iz=0; iz<2; ++iz)
      {
        volumeVertices[vertexIdx].e[0] = (float)ix;
        volumeVertices[vertexIdx].e[1] = (float)iy;
        volumeVertices[vertexIdx].e[2] = (float)iz;
    // transfers vertices to world coordinates:
        for (k=0; k<3; ++k)
          volumeVertices[vertexIdx].e[k] =
            (volumeVertices[vertexIdx].e[k] * 2.0f - 1.0f) * texSize2.e[k];
        volumeVertices[vertexIdx].multiply(view);
        vertexIdx++;
  }

  // Determine maximal and minimal distance of the volume from the eyepoint:
  maxDist = minDist = volumeVertices[0].length();
  for (i = 1; i<7; i++)
  {
    float dist = volumeVertices[i].length();
    if (dist > maxDist)  maxDist = dist;
    if (dist < minDist)  minDist = dist;
  }

  maxDist *= 1.4f;
  minDist *= 0.5f;

  // transfer the eyepoint to the object coordinates of the volume
  // to check whether the camera is inside the volume:
  vvVector3 eye(0.0,0.0,0.0);
  eye.multiply(&invView);
  int inside = 1;
  for (k=0; k<3; ++k)
  {
    if (eye.e[k] < -texSize2.e[k] || eye.e[k] > texSize2.e[k])
      inside = 0;
  }
  if (inside != 0)
    minDist = 0.0f;

  // Determine texture spacing:
  spacing = (maxDist-minDist) / (float)(numShells-1);

  if (instantClassification())
  {
    float diagonalVoxels = sqrtf(float(vd->vox[0] * vd->vox[0] +
      vd->vox[1] * vd->vox[1] +
      vd->vox[2] * vd->vox[2]));
    updateLUT(diagonalVoxels / numShells, pixLUTName, rgbaLUT);
  }

  vvSphere shell;
  shell.subdivide();
  shell.subdivide();
  shell.setVolumeDim(&texSize);
  shell.setViewMatrix(view);
  float offset[3];
  for (k=0; k<3; ++k)
  {
    offset[k] = -(0.5f - (texMin[k] + texMax[k]) / 2.0f);
  }
  shell.setTextureOffset(offset);

  float  radius;

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // Volume render a 3D texture:
  enableTexture(GL_TEXTURE_3D_EXT);
  glBindTexture(GL_TEXTURE_3D_EXT, texNames[0]);

  // Enable clipping plane if appropriate:
  if (_renderState._clipMode) activateClippingPlane();

  radius = maxDist;
  for (i=0; i<numShells; ++i)                     // loop thru all drawn textures
  {
    shell.setRadius(radius);
    shell.calculateTexCoords();
    shell.performCulling();
    glColor4f(1.0, 1.0, 1.0, 1.0);
    shell.render();
    radius -= spacing;
  }
  deactivateClippingPlane();
  disableTexture(GL_TEXTURE_3D_EXT);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

//----------------------------------------------------------------------------
/** Render the volume using 2D textures (OpenGL 1.1 compatible).
  @param zz  z coordinate of transformed z base vector
*/
void vvTexRend::renderTex2DSlices(float zz)
{
  vvVector3 normal;                               // normal vector for slices
  vvVector3 size, size2;                          // full and half texture sizes
  vvVector3 pos;                                  // object location
  float     texSpacing;                           // spacing for texture coordinates
  float     zPos;                                 // texture z position
  float     texStep;                              // step between texture indices
  float     texIndex;                             // current texture index
  int       numTextures;                          // number of textures drawn
  int       i;

  vvDebugMsg::msg(3, "vvTexRend::renderTex2DSlices()");

  // Translate object by its position:
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  pos.copy(&vd->pos);
  //glTranslatef(pos.e[0], pos.e[1], pos.e[2]);

  // Enable clipping plane if appropriate:
  if (_renderState._clipMode) activateClippingPlane();

  // Generate half object size as shortcut:
  size.copy(vd->getSize());
  size2.e[0] = 0.5f * size.e[0];
  size2.e[1] = 0.5f * size.e[1];
  size2.e[2] = 0.5f * size.e[2];

  numTextures = int(_renderState._quality * 100.0f);
  if (numTextures < 1) numTextures = 1;

  normal.set(0.0f, 0.0f, 1.0f);
  zPos = -size2.e[2];
  if (numTextures>1)                              // prevent division by zero
  {
    texSpacing = size.e[2] / (float)(numTextures - 1);
    texStep = (float)(vd->vox[2] - 1) / (float)(numTextures - 1);
  }
  else
  {
    texSpacing = 0.0f;
    texStep = 0.0f;
    zPos = 0.0f;
  }

  // Offset for current time step:
  texIndex = float(vd->getCurrentFrame()) * float(vd->vox[2]);
  
  if (zz>0.0f)                                    // draw textures back to front?
  {
    texIndex += float(vd->vox[2] - 1);
    texStep  = -texStep;
  }
  else                                            // draw textures front to back
  {
    zPos        = -zPos;
    texSpacing  = -texSpacing;
    normal.e[2] = -normal.e[2];
  }

  if (instantClassification())
  {
    float diagVoxels = sqrtf(float(vd->vox[0]*vd->vox[0]
      + vd->vox[1]*vd->vox[1]
      + vd->vox[2]*vd->vox[2]));
    updateLUT(diagVoxels/numTextures, pixLUTName, rgbaLUT);
  }

  // Volume rendering with multiple 2D textures:
  enableTexture(GL_TEXTURE_2D);

  for (i=0; i<numTextures; ++i)
  {
    glBindTexture(GL_TEXTURE_2D, texNames[vvToolshed::round(texIndex)]);

    if(voxelType==VV_PAL_TEX)
    {
      int size[3];
      getLUTSize(size);
      glColorTableEXT(GL_TEXTURE_2D, GL_RGBA,
        size[0], GL_RGBA, GL_UNSIGNED_BYTE, rgbaLUT);
    }

    glBegin(GL_QUADS);
      glColor4f(1.0, 1.0, 1.0, 1.0);
      glNormal3f(normal.e[0], normal.e[1], normal.e[2]);
      glTexCoord2f(texMin[0], texMax[1]); glVertex3f(-size2.e[0],  size2.e[1], zPos);
      glTexCoord2f(texMin[0], texMin[1]); glVertex3f(-size2.e[0], -size2.e[1], zPos);
      glTexCoord2f(texMax[0], texMin[1]); glVertex3f( size2.e[0], -size2.e[1], zPos);
      glTexCoord2f(texMax[0], texMax[1]); glVertex3f( size2.e[0],  size2.e[1], zPos);
    glEnd();

    zPos += texSpacing;
    texIndex += texStep;
  }

  disableTexture(GL_TEXTURE_2D);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  deactivateClippingPlane();
  vvDebugMsg::msg(3, "Number of textures stored: ", vd->vox[2]);
  vvDebugMsg::msg(3, "Number of textures drawn:  ", numTextures);
}

//----------------------------------------------------------------------------
/** Render the volume using 2D textures, switching to the optimum
    texture set to prevent holes.
  @param principal  principal viewing axis
  @param zx,zy,zz   z coordinates of transformed base vectors
*/
void vvTexRend::renderTex2DCubic(AxisType principal, float zx, float zy, float zz)
{
  vvVector3 normal;                               // normal vector for slices
  vvVector3 texTL, texTR, texBL, texBR;           // texture coordinates (T=top etc.)
  vvVector3 objTL, objTR, objBL, objBR;           // object coordinates in world space
  vvVector3 texSpacing;                           // distance between textures
  vvVector3 pos;                                  // object location
  vvVector3 size, size2;                          // full and half object sizes
  float  texStep;                                 // step size for texture names
  float  texIndex;                                // textures index
  int    numTextures;                             // number of textures drawn
  int    frameTextures;                           // number of textures per frame
  int    i;

  vvDebugMsg::msg(3, "vvTexRend::renderTex2DCubic()");

  // Translate object by its position:
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  pos.copy(&vd->pos);
  glTranslatef(pos.e[0], pos.e[1], pos.e[2]);

  // Enable clipping plane if appropriate:
  if (_renderState._clipMode) activateClippingPlane();

  // Initialize texture parameters:
  numTextures = int(_renderState._quality * 100.0f);
  frameTextures = vd->vox[0] + vd->vox[1] + vd->vox[2];
  if (numTextures < 2)  numTextures = 2;          // make sure that at least one slice is drawn to prevent division by zero

  // Generate half object size as a shortcut:
  size.copy(vd->getSize());
  size2.copy(size);
  size2.scale(0.5f);

  // Initialize parameters upon principal viewing direction:
  switch (principal)
  {
    case X_AXIS:                                  // zx>0 -> draw left to right
      // Coordinate system:
      //     z
      //     |__y
      //   x/
      objTL.set(-size2[0],-size2[1], size2[2]);
      objTR.set(-size2[0], size2[1], size2[2]);
      objBL.set(-size2[0],-size2[1],-size2[2]);
      objBR.set(-size2[0], size2[1],-size2[2]);

      texTL.set(texMin[1], texMax[2], 0.0f);
      texTR.set(texMax[1], texMax[2], 0.0f);
      texBL.set(texMin[1], texMin[2], 0.0f);
      texBR.set(texMax[1], texMin[2], 0.0f);

      texSpacing.set(size.e[0] / float(numTextures - 1), 0.0f, 0.0f);
      texStep = -1.0f * float(vd->vox[0] - 1) / float(numTextures - 1);
      normal.set(1.0f, 0.0f, 0.0f);
      texIndex = float(vd->getCurrentFrame() * frameTextures);
      if (zx<0)                                   // reverse order? draw right to left
      {
        normal.e[0]     = -normal.e[0];
        objTL.e[0]      = objTR.e[0] = objBL.e[0] = objBR.e[0] = size2[0];
        texSpacing.e[0] = -texSpacing.e[0];
        texStep         = -texStep;
      }
      else
      {
        texIndex += float(vd->vox[0] - 1);
      }
      break;

    case Y_AXIS:                                  // zy>0 -> draw bottom to top
      // Coordinate system:
      //     x
      //     |__z
      //   y/
      objTL.set( size2[0],-size2[1],-size2[2]);
      objTR.set( size2[0],-size2[1], size2[2]);
      objBL.set(-size2[0],-size2[1],-size2[2]);
      objBR.set(-size2[0],-size2[1], size2[2]);

      texTL.set(texMin[2], texMax[0], 0.0f);
      texTR.set(texMax[2], texMax[0], 0.0f);
      texBL.set(texMin[2], texMin[0], 0.0f);
      texBR.set(texMax[2], texMin[0], 0.0f);

      texSpacing.set(0.0f, size.e[1] / float(numTextures - 1), 0.0f);
      texStep = -1.0f * float(vd->vox[1] - 1) / float(numTextures - 1);
      normal.set(0.0f, 1.0f, 0.0f);
      texIndex = float(vd->getCurrentFrame() * frameTextures + vd->vox[0]);
      if (zy<0)                                   // reverse order? draw top to bottom
      {
        normal.e[1]     = -normal.e[1];
        objTL.e[1]      = objTR.e[1] = objBL.e[1] = objBR.e[1] = size2[1];
        texSpacing.e[1] = -texSpacing.e[1];
        texStep         = -texStep;
      }
      else
      {
        texIndex += float(vd->vox[1] - 1);
      }
      break;

    case Z_AXIS:                                  // zz>0 -> draw back to front
    default:
      // Coordinate system:
      //     y
      //     |__x
      //   z/
      objTL.set(-size2[0], size2[1],-size2[2]);
      objTR.set( size2[0], size2[1],-size2[2]);
      objBL.set(-size2[0],-size2[1],-size2[2]);
      objBR.set( size2[0],-size2[1],-size2[2]);

      texTL.set(texMin[0], texMax[1], 0.0f);
      texTR.set(texMax[0], texMax[1], 0.0f);
      texBL.set(texMin[0], texMin[1], 0.0f);
      texBR.set(texMax[0], texMin[1], 0.0f);

      texSpacing.set(0.0f, 0.0f, size.e[2] / float(numTextures - 1));
      normal.set(0.0f, 0.0f, 1.0f);
      texStep = -1.0f * float(vd->vox[2] - 1) / float(numTextures - 1);
      texIndex = float(vd->getCurrentFrame() * frameTextures + vd->vox[0] + vd->vox[1]);
      if (zz<0)                                   // reverse order? draw front to back
      {
        normal.e[2]     = -normal.e[2];
        objTL.e[2]      = objTR.e[2] = objBL.e[2] = objBR.e[2] = size2[2];
        texSpacing.e[2] = -texSpacing.e[2];
        texStep         = -texStep;
      }
      else                                        // draw back to front
      {
        texIndex += float(vd->vox[2] - 1);
      }
      break;
  }

  if (instantClassification())
  {
    float diagVoxels = sqrtf(float(vd->vox[0]*vd->vox[0]
      + vd->vox[1]*vd->vox[1]
      + vd->vox[2]*vd->vox[2]));
    updateLUT(diagVoxels/numTextures, pixLUTName, rgbaLUT);
  }

  // Volume render a 2D texture:
  enableTexture(GL_TEXTURE_2D);
  for (i=0; i<numTextures; ++i)
  {
    glBindTexture(GL_TEXTURE_2D, texNames[vvToolshed::round(texIndex)]);

    if(voxelType==VV_PAL_TEX)
    {
      int size[3];
      getLUTSize(size);
      glColorTableEXT(GL_TEXTURE_2D, GL_RGBA,
        size[0], GL_RGBA, GL_UNSIGNED_BYTE, rgbaLUT);
    }

    glBegin(GL_QUADS);
    glColor4f(1.0, 1.0, 1.0, 1.0);
    glNormal3f(normal.e[0], normal.e[1], normal.e[2]);
    glTexCoord2f(texTL.e[0], texTL.e[1]); glVertex3f(objTL.e[0], objTL.e[1], objTL.e[2]);
    glTexCoord2f(texBL.e[0], texBL.e[1]); glVertex3f(objBL.e[0], objBL.e[1], objBL.e[2]);
    glTexCoord2f(texBR.e[0], texBR.e[1]); glVertex3f(objBR.e[0], objBR.e[1], objBR.e[2]);
    glTexCoord2f(texTR.e[0], texTR.e[1]); glVertex3f(objTR.e[0], objTR.e[1], objTR.e[2]);
    glEnd();
    objTL.add(&texSpacing);
    objBL.add(&texSpacing);
    objBR.add(&texSpacing);
    objTR.add(&texSpacing);

    texIndex += texStep;
  }
  disableTexture(GL_TEXTURE_2D);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  deactivateClippingPlane();
}

//----------------------------------------------------------------------------
/** Render the volume onto currently selected drawBuffer.
 Viewport size in world coordinates is -1.0 .. +1.0 in both x and y direction
*/
void vvTexRend::renderVolumeGL()
{
  static vvStopwatch sw;                          // stop watch for performance measurements
  vvMatrix mv;                                    // current modelview matrix
  vvVector3 probeSizeObj;                         // probe size [object space]
  vvVector3 size;                                 // volume size [world coordinates]
  float zx, zy, zz;                               // base vector z coordinates
  vvRenderer::AxisType principal;                 // principal viewing axis with mv transformation applied
  int i;

  vvDebugMsg::msg(3, "vvTexRend::renderVolumeGL()");

  sw.start();

  // Reroute output to alternative render target.
  _renderTarget->initForRender();

  size.copy(vd->getSize());

  // Draw boundary lines (must be done before setGLenvironment()):
  if (_renderState._boundaries)
  {
    drawBoundingBox(&size, &vd->pos, _renderState._boundColor);
  }
  if (_renderState._isROIUsed)
  {
    probeSizeObj.set(size[0] * _renderState._roiSize[0], size[1] * _renderState._roiSize[1], size[2] * _renderState._roiSize[2]);
    drawBoundingBox(&probeSizeObj, &_renderState._roiPos, _renderState._probeColor);
  }
  if (_renderState._clipMode && _renderState._clipPerimeter)
  {
    drawPlanePerimeter(&size, &vd->pos, &_renderState._clipPoint, &_renderState._clipNormal, _renderState._clipColor);
  }

  setGLenvironment();

  if (_numThreads == 0)
  {
    if ((geomType == VV_BRICKS) && (!_renderState._showBricks) && _proxyGeometryOnGpu)
    {
      if (voxelType == VV_RGBA)
      {
        // Activate passthrough pixel shader as well.
        enableIntersectionShader(_isectShader, _pixelShader);
      }
      else
      {
        enableIntersectionShader(_isectShader);
      }
    }
  }

  // Determine texture object extensions:
  for (i=0; i<3; ++i)
  {
    texMin[i] = 0.5f / (float)texels[i];
    texMax[i] = (float)vd->vox[i] / (float)texels[i] - texMin[i];
  }

  // Find principle viewing direction:

  // Get OpenGL modelview matrix:
  getModelviewMatrix(&mv);

  if (_numThreads == 0)
  {
    if (geomType != VV_BRICKS || !_renderState._showBricks)
    {
      enableLUTMode(_pixelShader, pixLUTName);
    }
  }

  switch (geomType)
  {
    default:
    case VV_SLICES:
      getPrincipalViewingAxis(mv, zx, zy, zz);renderTex2DSlices(zz);
      break;
    case VV_CUBIC2D:
      principal = getPrincipalViewingAxis(mv, zx, zy, zz);
      renderTex2DCubic(principal, zx, zy, zz); break;
    case VV_SPHERICAL: renderTex3DSpherical(&mv); break;
    case VV_VIEWPORT:  renderTex3DPlanar(&mv); break;
    case VV_BRICKS:
      if (_renderState._showBricks)
        renderBricks(&mv);
      else
        renderTexBricks(&mv);
      break;
  }

  if (_numThreads == 0)
  {
    disableLUTMode(_pixelShader);
    unsetGLenvironment();
    if (voxelType == VV_RGBA)
    {
      disableIntersectionShader(_isectShader, _pixelShader);
    }
    else
    {
      disableIntersectionShader(_isectShader);
    }
  }
  else
  {
    unsetGLenvironment();
  }
  vvRenderer::renderVolumeGL();

  glFinish();                                     // make sure rendering is done to measure correct time

  // Write output of alternative render target. Depending on the type of render target,
  // output can be redirected to the screen or for example an image file.
  _renderTarget->writeBack();
  _lastRenderTime = sw.getTime();

  vvDebugMsg::msg(3, "vvTexRend::renderVolumeGL() done");
}

//----------------------------------------------------------------------------
/** Activate the previously set clipping plane.
    Clipping plane parameters have to be set with setClippingPlane().
*/
void vvTexRend::activateClippingPlane()
{
  GLdouble planeEq[4];                            // plane equation
  vvVector3 clipNormal2;                          // clipping normal pointing to opposite direction
  float thickness;                                // thickness of single slice clipping plane

  vvDebugMsg::msg(3, "vvTexRend::activateClippingPlane()");

  // Generate OpenGL compatible clipping plane parameters:
  // normal points into oppisite direction
  planeEq[0] = -_renderState._clipNormal[0];
  planeEq[1] = -_renderState._clipNormal[1];
  planeEq[2] = -_renderState._clipNormal[2];
  planeEq[3] = _renderState._clipNormal.dot(&_renderState._clipPoint);
  glClipPlane(GL_CLIP_PLANE0, planeEq);
  glEnable(GL_CLIP_PLANE0);

  // Generate second clipping plane in single slice mode:
  if (_renderState._clipSingleSlice)
  {
    thickness = vd->_scale * vd->dist[0] * vd->vox[0] / 100.0f;
    clipNormal2.copy(&_renderState._clipNormal);
    clipNormal2.negate();
    planeEq[0] = -clipNormal2[0];
    planeEq[1] = -clipNormal2[1];
    planeEq[2] = -clipNormal2[2];
    planeEq[3] = clipNormal2.dot(&_renderState._clipPoint) + thickness;
    glClipPlane(GL_CLIP_PLANE1, planeEq);
    glEnable(GL_CLIP_PLANE1);
  }
}

//----------------------------------------------------------------------------
/** Deactivate the clipping plane.
 */
void vvTexRend::deactivateClippingPlane()
{
  vvDebugMsg::msg(3, "vvTexRend::deactivateClippingPlane()");
  glDisable(GL_CLIP_PLANE0);
  if (_renderState._clipSingleSlice) glDisable(GL_CLIP_PLANE1);
}

//----------------------------------------------------------------------------
/** Set number of lights in the scene.
  Fixed material characteristics are used with each setting.
  @param numLights  number of lights in scene (0=ambient light only)
*/
void vvTexRend::setNumLights(int numLights)
{
  float ambient[]  = {0.5f, 0.5f, 0.5f, 1.0f};
  float pos0[] = {0.0f, 10.0f, 10.0f, 0.0f};
  float pos1[] = {0.0f, -10.0f, -10.0f, 0.0f};

  vvDebugMsg::msg(1, "vvTexRend::setNumLights()");

  // Generate light source 1:
  glEnable(GL_LIGHT0);
  glLightfv(GL_LIGHT0, GL_POSITION, pos0);
  glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);

  // Generate light source 2:
  glEnable(GL_LIGHT1);
  glLightfv(GL_LIGHT1, GL_POSITION, pos1);
  glLightfv(GL_LIGHT1, GL_AMBIENT, ambient);

  // At least 2 lights:
  if (numLights >= 2)
    glEnable(GL_LIGHT1);
  else
    glDisable(GL_LIGHT1);

  // At least one light:
  if (numLights >= 1)
    glEnable(GL_LIGHT0);
  else                                            // no lights selected
    glDisable(GL_LIGHT0);
}

//----------------------------------------------------------------------------
/// @return true if classification is done in no time
bool vvTexRend::instantClassification()
{
  vvDebugMsg::msg(3, "vvTexRend::instantClassification()");
  if (voxelType == VV_RGBA) return false;
  else return true;
}

//----------------------------------------------------------------------------
/// Returns the number of entries in the RGBA lookup table.
int vvTexRend::getLUTSize(int* size)
{
  int x, y, z;

  vvDebugMsg::msg(3, "vvTexRend::getLUTSize()");
  if (vd->bpc==2 && voxelType==VV_SGI_LUT)
  {
    x = 4096;
    y = z = 1;
  }
  else if (vvShaderFactory::isSupported(VV_CG_MANAGER)
    && _currentShader==8 && voxelType==VV_PIX_SHD)
  {
    x = y = getPreintTableSize();
    z = 1;
  }
  else
  {
    x = 256;
    if (vd->chan == 2)
    {
       y = x;
       z = 1;
    }
    else
       y = z = 1;
  }
  if (size)
  {
    size[0] = x;
    size[1] = y;
    size[2] = z;
  }
  return x * y * z;
}

//----------------------------------------------------------------------------
/// Returns the size (width and height) of the pre-integration lookup table.
int vvTexRend::getPreintTableSize()
{
  vvDebugMsg::msg(1, "vvTexRend::getPreintTableSize()");
  return 256;
}

//----------------------------------------------------------------------------
/** Update the color/alpha look-up table.
 Note: glColorTableSGI can have a maximum width of 1024 RGBA entries on IR2 graphics!
 @param dist  slice distance relative to 3D texture sample point distance
              (1.0 for original distance, 0.0 for all opaque).
*/
void vvTexRend::updateLUT(const float dist, GLuint& lutName, uchar*& lutData)
{
  vvDebugMsg::msg(3, "Generating texture LUT. Slice distance = ", dist);

  float corr[4];                                  // gamma/alpha corrected RGBA values [0..1]
  int lutSize[3];                                 // number of entries in the RGBA lookup table
  int i,c;
  int total=0;
  lutDistance = dist;

  if(usePreIntegration)
  {
    vd->tf.makePreintLUTOptimized(getPreintTableSize(), preintTable, dist);
  }
  else
  {
    total = getLUTSize(lutSize);
    for (i=0; i<total; ++i)
    {
      // Gamma correction:
      if (_renderState._gammaCorrection)
      {
        corr[0] = gammaCorrect(rgbaTF[i * 4],     VV_RED);
        corr[1] = gammaCorrect(rgbaTF[i * 4 + 1], VV_GREEN);
        corr[2] = gammaCorrect(rgbaTF[i * 4 + 2], VV_BLUE);
        corr[3] = gammaCorrect(rgbaTF[i * 4 + 3], VV_ALPHA);
      }
      else
      {
        corr[0] = rgbaTF[i * 4];
        corr[1] = rgbaTF[i * 4 + 1];
        corr[2] = rgbaTF[i * 4 + 2];
        corr[3] = rgbaTF[i * 4 + 3];
      }

      // Opacity correction:
                                                  // for 0 distance draw opaque slices
      if (dist<=0.0 || (_renderState._clipMode && _renderState._clipOpaque)) corr[3] = 1.0f;
      else if (opacityCorrection) corr[3] = 1.0f - powf(1.0f - corr[3], dist);

      // Convert float to uchar and copy to rgbaLUT array:
      for (c=0; c<4; ++c)
      {
        lutData[i * 4 + c] = uchar(corr[c] * 255.0f);
      }
    }
  }

  // Copy LUT to graphics card:
  vvGLTools::printGLError("enter updateLUT()");
  switch (voxelType)
  {
    case VV_RGBA:
      if (_numThreads == 0)
      {
        makeTextures(texNames, &textures,
                     lutName, lutData);// this mode doesn't use a hardware LUT, so every voxel has to be updated
      }
      break;
    case VV_SGI_LUT:
      glColorTableSGI(GL_TEXTURE_COLOR_TABLE_SGI, GL_RGBA,
          lutSize[0], GL_RGBA, GL_UNSIGNED_BYTE, rgbaLUT);
      break;
    case VV_PAL_TEX:
      // Load color LUT for pre-classification:
      assert(total==256);
      glColorTableEXT(GL_SHARED_TEXTURE_PALETTE_EXT, GL_RGBA8,
          lutSize[0], GL_RGBA, GL_UNSIGNED_BYTE, rgbaLUT);
      break;
    case VV_PIX_SHD:
    case VV_TEX_SHD:
    case VV_FRG_PRG:
      if(voxelType!=VV_PIX_SHD)
        glActiveTextureARB(GL_TEXTURE1_ARB);
      glBindTexture(GL_TEXTURE_2D, lutName);
      if(usePreIntegration)
      {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, getPreintTableSize(), getPreintTableSize(), 0,
            GL_RGBA, GL_UNSIGNED_BYTE, preintTable);
      }
      else
      {
         glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, lutSize[0], lutSize[1], 0,
               GL_RGBA, GL_UNSIGNED_BYTE, lutData);
      }
      if(voxelType!=VV_PIX_SHD)
        glActiveTextureARB(GL_TEXTURE0_ARB);
      break;
    default: assert(0); break;
  }
  vvGLTools::printGLError("leave updateLUT()");
}

//----------------------------------------------------------------------------
/** Set user's viewing direction.
  This information is needed to correctly orientate the texture slices
  in 3D texturing mode if the user is inside the volume.
  @param vd  viewing direction in object coordinates
*/
void vvTexRend::setViewingDirection(const vvVector3* vd)
{
  vvDebugMsg::msg(3, "vvTexRend::setViewingDirection()");
  viewDir.copy(vd);
}

//----------------------------------------------------------------------------
/** Set the direction from the viewer to the object.
  This information is needed to correctly orientate the texture slices
  in 3D texturing mode if the viewer is outside of the volume.
  @param vd  object direction in object coordinates
*/
void vvTexRend::setObjectDirection(const vvVector3* vd)
{
  vvDebugMsg::msg(3, "vvTexRend::setObjectDirection()");
  objDir.copy(vd);
}

//----------------------------------------------------------------------------
// see parent
void vvTexRend::setParameter(ParameterType param, float newValue, char*)
{
  bool newInterpol;

  vvDebugMsg::msg(3, "vvTexRend::setParameter()");
  switch (param)
  {
    case vvRenderer::VV_SLICEINT:
      newInterpol = (newValue == 0.0f) ? false : true;
      if (interpolation!=newInterpol)
      {
        interpolation = newInterpol;
        if (_numThreads > 0)
        {
          for (int i = 0; i < _numThreads; ++i)
          {
            makeTextures(_threadData[i].privateTexNames, &_threadData[i].numTextures,
                         _threadData[i].pixLUTName, _threadData[i].rgbaLUT);
            updateTransferFunction(_threadData[i].pixLUTName, _threadData[i].rgbaLUT);
          }
        }
        else
        {
          makeTextures(texNames, &textures, pixLUTName, rgbaLUT);
          updateTransferFunction(pixLUTName, rgbaLUT);
        }
      }
      break;
    case vvRenderer::VV_MIN_SLICE:
      minSlice = int(newValue);
      break;
    case vvRenderer::VV_MAX_SLICE:
      maxSlice = int(newValue);
      break;
    case vvRenderer::VV_OPCORR:
      opacityCorrection = (newValue==0.0f) ? false : true;
      break;
    case vvRenderer::VV_SLICEORIENT:
      _sliceOrientation = SliceOrientation(int(newValue));
      break;
    case vvRenderer::VV_PREINT:
      preIntegration = (newValue == 0.0f) ? false : true;
      updateTransferFunction(pixLUTName, rgbaLUT);
      break;
    case vvRenderer::VV_BINNING:
      if (newValue==0.0f) vd->_binning = vvVolDesc::LINEAR;
      else if (newValue==1.0f) vd->_binning = vvVolDesc::ISO_DATA;
      else vd->_binning = vvVolDesc::OPACITY;
      break;
    case vvRenderer::VV_GPUPROXYGEO:
      if (newValue == 0.0f)
      {
        _proxyGeometryOnGpu = false;
      }
      else
      {
        _proxyGeometryOnGpu = _isectShader;
      }
      break;
    case vvRenderer::VV_LEAPEMPTY:
      _renderState._emptySpaceLeaping = (newValue == 0.0f) ? false : true;
      // Maybe a tf type was chosen which is incompatible with empty space leaping.
      validateEmptySpaceLeaping();
      updateTransferFunction(pixLUTName, rgbaLUT);
      break;
    case vvRenderer::VV_OFFSCREENBUFFER:
      _renderState._useOffscreenBuffer = (newValue == 0.0f) ? false : true;
      if (_renderState._useOffscreenBuffer)
      {
        if (dynamic_cast<vvOffscreenBuffer*>(_renderTarget) == NULL)
        {
          delete _renderTarget;
          _renderTarget = new vvOffscreenBuffer(_renderState._imageScale, _renderState._imagePrecision);
        }
      }
      else
      {
        delete _renderTarget;
        _renderTarget = new vvRenderTarget();
      }
      break;
    case vvRenderer::VV_IMG_SCALE:
      _renderState._imageScale = newValue;
      if (_renderState._useOffscreenBuffer)
      {
        if (dynamic_cast<vvOffscreenBuffer*>(_renderTarget) != NULL)
        {
          dynamic_cast<vvOffscreenBuffer*>(_renderTarget)->setScale(newValue);
        }
        else
        {
          delete _renderTarget;
          _renderTarget = new vvOffscreenBuffer(_renderState._imageScale, _renderState._imagePrecision);
        }
      }
      break;
    case vvRenderer::VV_IMG_PRECISION:
      if (_renderState._useOffscreenBuffer)
      {
        if (int(newValue) <= 8)
        {
          if (dynamic_cast<vvOffscreenBuffer*>(_renderTarget) != NULL)
          {
            dynamic_cast<vvOffscreenBuffer*>(_renderTarget)->setPrecision(VV_BYTE);
          }
          else
          {
            delete _renderTarget;
            _renderTarget = new vvOffscreenBuffer(_renderState._imageScale, VV_BYTE);
          }
          break;
        }

        if (int(newValue) >= 32)
        {
          if (dynamic_cast<vvOffscreenBuffer*>(_renderTarget) != NULL)
          {
            dynamic_cast<vvOffscreenBuffer*>(_renderTarget)->setPrecision(VV_FLOAT);
          }
          else
          {
            delete _renderTarget;
            _renderTarget = new vvOffscreenBuffer(_renderState._imageScale, VV_FLOAT);
          }
          break;
        }
      }
    default:
      vvRenderer::setParameter(param, newValue);
      break;
  }
}

//----------------------------------------------------------------------------
// see parent for comments
float vvTexRend::getParameter(ParameterType param, char*)
{
  vvDebugMsg::msg(3, "vvTexRend::getParameter()");

  switch (param)
  {
    case vvRenderer::VV_SLICEINT:
      return (interpolation) ? 1.0f : 0.0f;
    case vvRenderer::VV_MIN_SLICE:
      return float(minSlice);
    case vvRenderer::VV_MAX_SLICE:
      return float(maxSlice);
    case vvRenderer::VV_SLICEORIENT:
      return float(_sliceOrientation);
    case vvRenderer::VV_PREINT:
      return float(preIntegration);
    case vvRenderer::VV_BINNING:
      return float(vd->_binning);
    default: return vvRenderer::getParameter(param);
  }
}

//----------------------------------------------------------------------------
/** Get information on hardware support for rendering modes.
  This routine cannot guarantee that a requested method works on any
  dataset, but it will at least work for datasets which fit into
  texture memory.
  @param geom geometry type to get information about
  @return true if the requested rendering type is supported by
    the system's graphics hardware.
*/
bool vvTexRend::isSupported(GeometryType geom)
{
  vvDebugMsg::msg(3, "vvTexRend::isSupported(0)");

  switch (geom)
  {
    case VV_AUTO:
    case VV_SLICES:
    case VV_CUBIC2D:
      return true;

    case VV_VIEWPORT:
    case VV_BRICKS:
    case VV_SPHERICAL:
#if defined(__APPLE__) && defined(GL_VERSION_1_2)
      return true;
#else
      return vvGLTools::isGLextensionSupported("GL_EXT_texture3D");
#endif
    default:
      return false;
  }
}

//----------------------------------------------------------------------------
/** Get information on hardware support for rendering modes.
  @param geom voxel type to get information about
  @return true if the requested voxel type is supported by
    the system's graphics hardware.
*/
bool vvTexRend::isSupported(VoxelType voxel)
{
  vvDebugMsg::msg(3, "vvTexRend::isSupported(1)");

  switch(voxel)
  {
    case VV_BEST:
    case VV_RGBA:
      return true;
    case VV_SGI_LUT:
      return vvGLTools::isGLextensionSupported("GL_SGI_texture_color_table");
    case VV_PAL_TEX:
      return vvGLTools::isGLextensionSupported("GL_EXT_paletted_texture");
    case VV_TEX_SHD:
      return vvGLTools::isGLextensionSupported("GL_ARB_multitexture") &&
        vvGLTools::isGLextensionSupported("GL_NV_texture_shader") &&
        vvGLTools::isGLextensionSupported("GL_NV_texture_shader2") &&
        vvGLTools::isGLextensionSupported("GL_ARB_texture_env_combine") &&
        vvGLTools::isGLextensionSupported("GL_NV_register_combiners") &&
        vvGLTools::isGLextensionSupported("GL_NV_register_combiners2");
    case VV_PIX_SHD:
      if (vvShaderFactory::isSupported(VV_CG_MANAGER))
      {
        return vvGLTools::isGLextensionSupported("GL_ARB_fragment_program");
      }
      else
      {
        return false;
      }
    case VV_FRG_PRG:
      return vvGLTools::isGLextensionSupported("GL_ARB_fragment_program");
    default: return false;
  }
}

//----------------------------------------------------------------------------
/** Return true if a feature is supported.
 */
bool vvTexRend::isSupported(FeatureType feature)
{
  vvDebugMsg::msg(3, "vvTexRend::isSupported()");
  switch(feature)
  {
    case VV_MIP: return true;
    default: assert(0); break;
  }
  return false;
}

//----------------------------------------------------------------------------
/** Return the currently used rendering geometry.
  This is expecially useful if VV_AUTO was passed in the constructor.
*/
vvTexRend::GeometryType vvTexRend::getGeomType()
{
  vvDebugMsg::msg(3, "vvTexRend::getGeomType()");
  return geomType;
}

//----------------------------------------------------------------------------
/** Return the currently used voxel type.
  This is expecially useful if VV_AUTO was passed in the constructor.
*/
vvTexRend::VoxelType vvTexRend::getVoxelType()
{
  vvDebugMsg::msg(3, "vvTexRend::getVoxelType()");
  return voxelType;
}

//----------------------------------------------------------------------------
/** Return the currently used pixel shader [0..numShaders-1].
 */
int vvTexRend::getCurrentShader()
{
  vvDebugMsg::msg(3, "vvTexRend::getCurrentShader()");
  return _currentShader;
}

//----------------------------------------------------------------------------
/** Set the currently used pixel shader [0..numShaders-1].
 */
void vvTexRend::setCurrentShader(int shader)
{
  vvDebugMsg::msg(3, "vvTexRend::setCurrentShader()");
  if(shader >= NUM_PIXEL_SHADERS || shader < 0)
     shader = 0;
  _currentShader = shader;
}

//----------------------------------------------------------------------------
/// inherited from vvRenderer, only valid for planar textures
void vvTexRend::renderQualityDisplay()
{
  int numSlices = int(_renderState._quality * 100.0f);
  vvPrintGL* printGL = new vvPrintGL();
  printGL->print(-0.9f, 0.9f, "Textures: %d", numSlices);
  delete printGL;
}

//----------------------------------------------------------------------------
void vvTexRend::enableTexture(GLenum target)
{
  if (voxelType==VV_TEX_SHD)
  {
    glTexEnvi(GL_TEXTURE_SHADER_NV, GL_SHADER_OPERATION_NV, target);
  }
  else
  {
    glEnable(target);
  }
}

//----------------------------------------------------------------------------
void vvTexRend::disableTexture(GLenum target)
{
  if (voxelType==VV_TEX_SHD)
  {
    glTexEnvi(GL_TEXTURE_SHADER_NV, GL_SHADER_OPERATION_NV, GL_NONE);
  }
  else
  {
    glDisable(target);
  }
}

//----------------------------------------------------------------------------
void vvTexRend::enableNVShaders()
{
  glEnable(GL_TEXTURE_SHADER_NV);

  glActiveTextureARB(GL_TEXTURE0_ARB);          // the volume data
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  vvGLTools::printGLError("Texture unit 0");

  glActiveTextureARB(GL_TEXTURE1_ARB);
  glBindTexture(GL_TEXTURE_2D, pixLUTName);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexEnvi(GL_TEXTURE_SHADER_NV, GL_SHADER_OPERATION_NV, GL_DEPENDENT_AR_TEXTURE_2D_NV);
  glTexEnvi(GL_TEXTURE_SHADER_NV, GL_PREVIOUS_TEXTURE_INPUT_NV, GL_TEXTURE0_ARB);
  vvGLTools::printGLError("Texture unit 1");

  glActiveTextureARB(GL_TEXTURE2_ARB);
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_TEXTURE_3D);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_NONE);
  glTexEnvi(GL_TEXTURE_SHADER_NV, GL_SHADER_OPERATION_NV, GL_NONE);

  glActiveTextureARB(GL_TEXTURE3_ARB);
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_TEXTURE_3D);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_NONE);
  glTexEnvi(GL_TEXTURE_SHADER_NV, GL_SHADER_OPERATION_NV, GL_NONE);

  glActiveTextureARB(GL_TEXTURE0_ARB);
}

//----------------------------------------------------------------------------
void vvTexRend::disableNVShaders()
{
  glDisable(GL_TEXTURE_SHADER_NV);
  glActiveTextureARB(GL_TEXTURE1_ARB);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexEnvi(GL_TEXTURE_SHADER_NV, GL_SHADER_OPERATION_NV, GL_TEXTURE_2D);
  glActiveTextureARB(GL_TEXTURE0_ARB);
}

//----------------------------------------------------------------------------
void vvTexRend::enableFragProg(GLuint& lutName)
{
  glActiveTextureARB(GL_TEXTURE1_ARB);
  glBindTexture(GL_TEXTURE_2D, lutName);
  glActiveTextureARB(GL_TEXTURE0_ARB);

  glEnable(GL_FRAGMENT_PROGRAM_ARB);
  switch(geomType)
  {
    case VV_CUBIC2D:
    case VV_SLICES:
      glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, fragProgName[VV_FRAG_PROG_2D]);
      break;
    case VV_VIEWPORT:
    case VV_SPHERICAL:
    case VV_BRICKS:
      if(usePreIntegration)
      {
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, fragProgName[VV_FRAG_PROG_PREINT]);
      }
      else
      {
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, fragProgName[VV_FRAG_PROG_3D]);
      }
      break;
    default:
      vvDebugMsg::msg(1, "vvTexRend::enableFragProg(): unknown method used\n");
      break;
  }
}

//----------------------------------------------------------------------------
void vvTexRend::disableFragProg()
{
  glDisable(GL_FRAGMENT_PROGRAM_ARB);
}

//----------------------------------------------------------------------------
void vvTexRend::enablePixelShaders(vvShaderManager* pixelShader, GLuint& lutName)
{
  if(VV_PIX_SHD == voxelType)
  {
    // Load, enable, and bind fragment shader:
    pixelShader->enableShader(_currentShader);

    const int MAX_PARAMETERS = 5;
    const char* parameterNames[MAX_PARAMETERS];
    vvShaderParameterType parameterTypes[MAX_PARAMETERS];
    void* values[MAX_PARAMETERS];

    int parameterCount = 0;

    // Set fragment program parameters:
    if (_currentShader != 4)                          // pixLUT, doesn't work with grayscale shader
    {
      glBindTexture(GL_TEXTURE_2D, lutName);

      // TODO: cgGLEnableTextureParameter.
      parameterNames[parameterCount] = "pixLUT";
      parameterTypes[parameterCount] = VV_SHD_TEXTURE_ID;
      values[parameterCount] = (void*)lutName;
      parameterCount++;
    }

    if (_currentShader == 3 || _currentShader == 7)   // chan4color
    {
      parameterNames[parameterCount] = "chan4color";
      parameterTypes[parameterCount] = VV_SHD_VEC3;
      values[parameterCount] = _channel4Color;
      parameterCount++;
    }

    if (_currentShader > 4 && _currentShader < 8)     // opWeights
    {
      parameterNames[parameterCount] = "opWeights";
      parameterTypes[parameterCount] = VV_SHD_VEC4;
      values[parameterCount] = _opacityWeights;
      parameterCount++;
    }

    if (_currentShader == 12)                         // L and H vectors for local illumination
    {
      parameterNames[parameterCount] = "L";
      parameterTypes[parameterCount] = VV_SHD_VEC3;
      values[parameterCount] = NULL;
      parameterCount++;

      parameterNames[parameterCount] = "H";
      parameterTypes[parameterCount] = VV_SHD_VEC3;
      values[parameterCount] = NULL;
      parameterCount++;
    }

    // Copy data to arrays on heap.
    const char** names = new const char*[parameterCount];
    vvShaderParameterType* types = new vvShaderParameterType[parameterCount];

    for (int i = 0; i < parameterCount; ++i)
    {
      names[i] = parameterNames[i];
      types[i] = parameterTypes[i];
    }

    pixelShader->initParameters(_currentShader, names, types, parameterCount);

    // Set the values.
    GLuint tmpGLuint;
    float* tmp3;
    float* tmp4;
    for (int i = 0; i < parameterCount; ++i)
    {
      switch (types[i])
      {
      case VV_SHD_TEXTURE_ID:
        tmpGLuint = (unsigned long)values[i];
        pixelShader->setParameterTexId(_currentShader, names[i], tmpGLuint);
        pixelShader->enableTexture(_currentShader, names[i]);
        break;
      case VV_SHD_VEC3:
        tmp3 = (float*)values[i];
        if (tmp3 != NULL)
        {
          pixelShader->setParameter3f(_currentShader, names[i], tmp3[0], tmp3[1], tmp3[2]);
        }
        break;
      case VV_SHD_VEC4:
        tmp4 = (float*)values[i];
        if (tmp4 != NULL)
        {
          pixelShader->setParameter4f(_currentShader, names[i], tmp4[0], tmp4[1], tmp4[2], tmp4[3]);
        }
        break;
      default:
        break;
      }
    }
  }
}

//----------------------------------------------------------------------------
void vvTexRend::disablePixelShaders(vvShaderManager* pixelShader)
{
  if(voxelType == VV_PIX_SHD)
  {
    if ((_currentShader != 4) && (pixelShader->parametersInitialized(_currentShader)))
    {
      pixelShader->disableTexture(_currentShader, "pixLUT");
    }
    pixelShader->disableShader(_currentShader);
  }
}

//----------------------------------------------------------------------------
void vvTexRend::enableIntersectionShader(vvShaderManager* isectShader, vvShaderManager* pixelShader)
{
  if(_proxyGeometryOnGpu)
  {
    isectShader->enableShader(0);
    if (pixelShader != NULL)
    {
      // Activate passthrough shader.
      pixelShader->enableShader(0);
    }
  }
}

//----------------------------------------------------------------------------
void vvTexRend::disableIntersectionShader(vvShaderManager* isectShader, vvShaderManager* pixelShader)
{
  if(_proxyGeometryOnGpu)
  {
    isectShader->disableShader(0);
    if (pixelShader != NULL)
    {
      // Deactivate the passthrough shader.
      pixelShader->disableShader(0);
    }
  }
}

//----------------------------------------------------------------------------
/** @return true if initialization successful
 */
bool vvTexRend::initPixelShaders(vvShaderManager* pixelShader)
{
  const char* shaderFileName = "vv_shader";
  const char* shaderExt = ".cg";
  const char* unixShaderDir = NULL;
  char* shaderFile = NULL;
  char* shaderPath = NULL;
  int i;

  cerr << "enable PIX called"<< endl;

  pixelShader->printCompatibilityInfo();

  // Specify shader path:
  cerr << "Searching for shader files..." << endl;

  unixShaderDir = pixelShader->getShaderDir();
  cerr << "Using shader path: " << unixShaderDir << endl;

  // Load shader files:
  for (i=0; i<NUM_PIXEL_SHADERS; ++i)
  {
    shaderFile = new char[strlen(shaderFileName) + 2 + strlen(shaderExt) + 1];
    sprintf(shaderFile, "%s%02d%s", shaderFileName, i+1, shaderExt);

    // Load Vertex Shader From File:
    // FIXME: why don't relative paths work under Linux?
    shaderPath = new char[strlen(unixShaderDir) + 1 + strlen(shaderFile) + 1];
#ifdef _WIN32
    sprintf(shaderPath, "%s\\%s", unixShaderDir, shaderFile);
#else
    sprintf(shaderPath, "%s/%s", unixShaderDir, shaderFile);
#endif

    cerr << "Loading shader file: " << shaderPath << endl;

    pixelShader->loadShader(shaderPath, vvShaderManager::VV_FRAG_SHD);

    delete[] shaderFile;
    delete[] shaderPath;
  }

  cerr << "Fragment programs ready." << endl;

  return true;
}

//----------------------------------------------------------------------------
/** @return true if initialization successful
 */
bool vvTexRend::initIntersectionShader(vvShaderManager* isectShader, vvShaderManager* pixelShader)
{
  const char* shaderFileName = "vv_intersection.cg";
  const char* unixShaderDir = NULL;
  char* shaderFile = NULL;
  char* shaderPath = NULL;

  shaderFile = new char[strlen(shaderFileName) + 1];
  sprintf(shaderFile, "%s", shaderFileName);

  unixShaderDir = isectShader->getShaderDir();

  shaderPath = new char[strlen(unixShaderDir) + 1 + strlen(shaderFile) + 1];
#ifdef _WIN32
  sprintf(shaderPath, "%s\\%s", unixShaderDir, shaderFile);
#else
  sprintf(shaderPath, "%s/%s", unixShaderDir, shaderFile);
#endif

  isectShader->loadShader(shaderPath, vvShaderManager::VV_VERT_SHD);

  if (pixelShader != NULL)
  {
    const char* passthroughFile = "vv_shader00.cg";
    char* passthroughShader = new char[strlen(unixShaderDir) + 1 + strlen(passthroughFile) + 1];;
    sprintf(passthroughShader, "%s/%s", unixShaderDir, passthroughFile);std::cout << passthroughShader << std::endl;
    pixelShader->loadShader(passthroughShader, vvShaderManager::VV_FRAG_SHD);
    delete[] passthroughShader;
  }

  delete[] shaderFile;
  delete[] shaderPath;

  setupIntersectionParameters(isectShader);

  return true;
}

void vvTexRend::setupIntersectionParameters(vvShaderManager* isectShader)
{
  int parameterCount = 12;
  const char** parameterNames = new const char*[parameterCount];
  vvShaderParameterType* parameterTypes = new vvShaderParameterType[parameterCount];
  parameterNames[0] = "sequence";       parameterTypes[0] = VV_SHD_ARRAY;
  parameterNames[1] = "v1";             parameterTypes[1] = VV_SHD_ARRAY;
  parameterNames[2] = "v2";             parameterTypes[2] = VV_SHD_ARRAY;

  parameterNames[3] = "brickMin";       parameterTypes[3] = VV_SHD_VEC4;
  parameterNames[4] = "brickDimInv";    parameterTypes[4] = VV_SHD_VEC3;
  parameterNames[5] = "texRange";       parameterTypes[5] = VV_SHD_VEC3;
  parameterNames[6] = "texMin";         parameterTypes[6] = VV_SHD_VEC3;
  parameterNames[7] = "modelViewProj";  parameterTypes[7] = VV_SHD_ARRAY;
  parameterNames[8] = "delta";          parameterTypes[8] = VV_SHD_SCALAR;
  parameterNames[9] = "planeNormal";    parameterTypes[9] = VV_SHD_VEC3;
  parameterNames[10] = "frontIndex";     parameterTypes[10] = VV_SHD_SCALAR;
  parameterNames[11] = "vertices";       parameterTypes[11] = VV_SHD_ARRAY;

  isectShader->initParameters(0, parameterNames, parameterTypes, parameterCount);

  // Global scope, values will never be changed.

  isectShader->setArrayParameter1i(0, "sequence", 0, 0);
  isectShader->setArrayParameter1i(0, "sequence", 1, 1);
  isectShader->setArrayParameter1i(0, "sequence", 2, 2);
  isectShader->setArrayParameter1i(0, "sequence", 3, 3);
  isectShader->setArrayParameter1i(0, "sequence", 4, 4);
  isectShader->setArrayParameter1i(0, "sequence", 5, 5);
  isectShader->setArrayParameter1i(0, "sequence", 6, 6);
  isectShader->setArrayParameter1i(0, "sequence", 7, 7);

  isectShader->setArrayParameter1i(0, "sequence", 8, 1);
  isectShader->setArrayParameter1i(0, "sequence", 9, 2);
  isectShader->setArrayParameter1i(0, "sequence", 10, 3);
  isectShader->setArrayParameter1i(0, "sequence", 11, 0);
  isectShader->setArrayParameter1i(0, "sequence", 12, 7);
  isectShader->setArrayParameter1i(0, "sequence", 13, 4);
  isectShader->setArrayParameter1i(0, "sequence", 14, 5);
  isectShader->setArrayParameter1i(0, "sequence", 15, 6);

  isectShader->setArrayParameter1i(0, "sequence", 16, 2);
  isectShader->setArrayParameter1i(0, "sequence", 17, 7);
  isectShader->setArrayParameter1i(0, "sequence", 18, 6);
  isectShader->setArrayParameter1i(0, "sequence", 19, 3);
  isectShader->setArrayParameter1i(0, "sequence", 20, 4);
  isectShader->setArrayParameter1i(0, "sequence", 21, 1);
  isectShader->setArrayParameter1i(0, "sequence", 22, 0);
  isectShader->setArrayParameter1i(0, "sequence", 23, 5);

  isectShader->setArrayParameter1i(0, "sequence", 24, 3);
  isectShader->setArrayParameter1i(0, "sequence", 25, 6);
  isectShader->setArrayParameter1i(0, "sequence", 26, 5);
  isectShader->setArrayParameter1i(0, "sequence", 27, 0);
  isectShader->setArrayParameter1i(0, "sequence", 28, 7);
  isectShader->setArrayParameter1i(0, "sequence", 29, 2);
  isectShader->setArrayParameter1i(0, "sequence", 30, 1);
  isectShader->setArrayParameter1i(0, "sequence", 31, 4);

  isectShader->setArrayParameter1i(0, "sequence", 32, 4);
  isectShader->setArrayParameter1i(0, "sequence", 33, 5);
  isectShader->setArrayParameter1i(0, "sequence", 34, 6);
  isectShader->setArrayParameter1i(0, "sequence", 35, 7);
  isectShader->setArrayParameter1i(0, "sequence", 36, 0);
  isectShader->setArrayParameter1i(0, "sequence", 37, 1);
  isectShader->setArrayParameter1i(0, "sequence", 38, 2);
  isectShader->setArrayParameter1i(0, "sequence", 39, 3);

  isectShader->setArrayParameter1i(0, "sequence", 40, 5);
  isectShader->setArrayParameter1i(0, "sequence", 41, 0);
  isectShader->setArrayParameter1i(0, "sequence", 42, 3);
  isectShader->setArrayParameter1i(0, "sequence", 43, 6);
  isectShader->setArrayParameter1i(0, "sequence", 44, 1);
  isectShader->setArrayParameter1i(0, "sequence", 45, 4);
  isectShader->setArrayParameter1i(0, "sequence", 46, 7);
  isectShader->setArrayParameter1i(0, "sequence", 47, 2);

  isectShader->setArrayParameter1i(0, "sequence", 48, 6);
  isectShader->setArrayParameter1i(0, "sequence", 49, 7);
  isectShader->setArrayParameter1i(0, "sequence", 50, 4);
  isectShader->setArrayParameter1i(0, "sequence", 51, 5);
  isectShader->setArrayParameter1i(0, "sequence", 52, 2);
  isectShader->setArrayParameter1i(0, "sequence", 53, 3);
  isectShader->setArrayParameter1i(0, "sequence", 54, 0);
  isectShader->setArrayParameter1i(0, "sequence", 55, 1);

  isectShader->setArrayParameter1i(0, "sequence", 56, 7);
  isectShader->setArrayParameter1i(0, "sequence", 57, 6);
  isectShader->setArrayParameter1i(0, "sequence", 58, 3);
  isectShader->setArrayParameter1i(0, "sequence", 59, 2);
  isectShader->setArrayParameter1i(0, "sequence", 60, 5);
  isectShader->setArrayParameter1i(0, "sequence", 61, 4);
  isectShader->setArrayParameter1i(0, "sequence", 62, 1);
  isectShader->setArrayParameter1i(0, "sequence", 63, 0);

  // P0.
  isectShader->setArrayParameter1i(0, "v1", 0, 0); isectShader->setArrayParameter1i(0, "v2", 0, 1);
  isectShader->setArrayParameter1i(0, "v1", 1, 1); isectShader->setArrayParameter1i(0, "v2", 1, 2);
  isectShader->setArrayParameter1i(0, "v1", 2, 2); isectShader->setArrayParameter1i(0, "v2", 2, 7);

  // P2.
  isectShader->setArrayParameter1i(0, "v1", 8, 0); isectShader->setArrayParameter1i(0, "v2", 8, 5);
  isectShader->setArrayParameter1i(0, "v1", 9, 5); isectShader->setArrayParameter1i(0, "v2", 9, 4);
  isectShader->setArrayParameter1i(0, "v1", 10, 4); isectShader->setArrayParameter1i(0, "v2", 10, 7);

  // P4.
  isectShader->setArrayParameter1i(0, "v1", 16, 0); isectShader->setArrayParameter1i(0, "v2", 16, 3);
  isectShader->setArrayParameter1i(0, "v1", 17, 3); isectShader->setArrayParameter1i(0, "v2", 17, 6);
  isectShader->setArrayParameter1i(0, "v1", 18, 6); isectShader->setArrayParameter1i(0, "v2", 18, 7);

  // P1.
  // First intersect the dotted edge.
  isectShader->setArrayParameter1i(0, "v1", 4, 1); isectShader->setArrayParameter1i(0, "v2", 4, 4);
  // Then intersect the path.
  isectShader->setArrayParameter1i(0, "v1", 5, 0); isectShader->setArrayParameter1i(0, "v2", 5, 1);
  isectShader->setArrayParameter1i(0, "v1", 6, 1); isectShader->setArrayParameter1i(0, "v2", 6, 2);
  isectShader->setArrayParameter1i(0, "v1", 7, 2); isectShader->setArrayParameter1i(0, "v2", 7, 7);

  // P3.
  // First intersect the dotted edge.
  isectShader->setArrayParameter1i(0, "v1", 12, 5); isectShader->setArrayParameter1i(0, "v2", 12, 6);
  // Then intersect the path.
  isectShader->setArrayParameter1i(0, "v1", 13, 0); isectShader->setArrayParameter1i(0, "v2", 13, 5);
  isectShader->setArrayParameter1i(0, "v1", 14, 5); isectShader->setArrayParameter1i(0, "v2", 14, 4);
  isectShader->setArrayParameter1i(0, "v1", 15, 4); isectShader->setArrayParameter1i(0, "v2", 15, 7);

  // P5.
  // First intersect the dotted edge.
  isectShader->setArrayParameter1i(0, "v1", 20, 3); isectShader->setArrayParameter1i(0, "v2", 20, 2);
  // Then intersect the path.
  isectShader->setArrayParameter1i(0, "v1", 21, 0); isectShader->setArrayParameter1i(0, "v2", 21, 3);
  isectShader->setArrayParameter1i(0, "v1", 22, 3); isectShader->setArrayParameter1i(0, "v2", 22, 6);
  isectShader->setArrayParameter1i(0, "v1", 23, 6); isectShader->setArrayParameter1i(0, "v2", 23, 7);
}

//----------------------------------------------------------------------------
void vvTexRend::printLUT()
{
  int i,c;
  int lutEntries[3];
  int total;

  total = getLUTSize(lutEntries);
  for (i=0; i<total; ++i)
  {
    cerr << "#" << i << ": ";
    for (c=0; c<4; ++c)
    {
      cerr << int(rgbaLUT[i * 4 + c]);
      if (c<3) cerr << ", ";
    }
    cerr << endl;
  }
}

unsigned char* vvTexRend::getHeightFieldData(float points[4][3], int& width, int& height)
{
  GLint viewport[4];
  unsigned char *pixels, *data, *result=NULL;
  int numPixels;
  int i, j, k;
  int x, y, c;
  int index;
  float sizeX, sizeY;
  vvVector3 size, size2;
  vvVector3 texcoord[4];

  std::cerr << "getHeightFieldData" << endl;

  glGetIntegerv(GL_VIEWPORT, viewport);

  width = int(ceil(getManhattenDist(points[0], points[1])));
  height = int(ceil(getManhattenDist(points[0], points[3])));

  numPixels = width * height;
  pixels = new unsigned char[4*numPixels];

  glReadPixels(viewport[0], viewport[1], width, height,
    GL_RGBA, GL_UNSIGNED_BYTE, pixels);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();

  size.copy(vd->getSize());
  for (i = 0; i < 3; ++i)
    size2.e[i]   = 0.5f * size.e[i];

  for (j = 0; j < 4; j++)
    for (k = 0; k < 3; k++)
  {
    texcoord[j][k] = (points[j][k] + size2.e[k]) / size.e[k];
    texcoord[j][k] = texcoord[j][k] * (texMax[k] - texMin[k]) + texMin[k];
  }

  enableTexture(GL_TEXTURE_3D_EXT);
  glBindTexture(GL_TEXTURE_3D_EXT, texNames[vd->getCurrentFrame()]);

  if (glIsTexture(texNames[vd->getCurrentFrame()]))
    std::cerr << "true" << endl;
  else
    std::cerr << "false" << endl;

  sizeX = 2.0f * float(width)  / float(viewport[2] - 1);
  sizeY = 2.0f * float(height) / float(viewport[3] - 1);

  std::cerr << "SizeX: " << sizeX << endl;
  std::cerr << "SizeY: " << sizeY << endl;
  std::cerr << "Viewport[2]: " << viewport[2] << endl;
  std::cerr << "Viewport[3]: " << viewport[3] << endl;

  std::cerr << "TexCoord1: " << texcoord[0][0] << " " << texcoord[0][1] << " " << texcoord[0][2] << endl;
  std::cerr << "TexCoord2: " << texcoord[1][0] << " " << texcoord[1][1] << " " << texcoord[1][2] << endl;
  std::cerr << "TexCoord3: " << texcoord[2][0] << " " << texcoord[2][1] << " " << texcoord[2][2] << endl;
  std::cerr << "TexCoord4: " << texcoord[3][0] << " " << texcoord[3][1] << " " << texcoord[3][2] << endl;

  glBegin(GL_QUADS);
  glTexCoord3f(texcoord[0][0], texcoord[0][1], texcoord[0][2]);
  glVertex3f(-1.0, -1.0, -1.0);
  glTexCoord3f(texcoord[1][0], texcoord[1][1], texcoord[1][2]);
  glVertex3f(sizeX, -1.0, -1.0);
  glTexCoord3f(texcoord[2][0], texcoord[2][1], texcoord[2][2]);
  glVertex3f(sizeX, sizeY, -1.0);
  glTexCoord3f(texcoord[3][0], texcoord[3][1], texcoord[3][2]);
  glVertex3f(-1.0, sizeY, -1.0);
  glEnd();

  glFinish();
  glReadBuffer(GL_BACK);

  data = new unsigned char[texelsize * numPixels];
  memset(data, 0, texelsize * numPixels);
  glReadPixels(viewport[0], viewport[1], width, height,
    GL_RGB, GL_UNSIGNED_BYTE, data);

  std::cerr << "data read" << endl;

  if (vd->chan == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
  {
    result = new unsigned char[numPixels];
    for (y = 0; y < height; y++)
      for (x = 0; x < width; x++)
    {
      index = y * width + x;
      switch (voxelType)
      {
        case VV_SGI_LUT:
          result[index] = data[2*index];
          break;
        case VV_PAL_TEX:
        case VV_FRG_PRG:
          result[index] = data[index];
          break;
        case VV_TEX_SHD:
        case VV_PIX_SHD:
          result[index] = data[4*index];
          break;
        case VV_RGBA:
          assert(0);
          break;
        default:
          assert(0);
          break;
      }
      std::cerr << "Result: " << index << " " << (int) (result[index]) << endl;
    }
  }
  else if (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4)
  {
    if ((voxelType == VV_RGBA) || (voxelType == VV_PIX_SHD))
    {
      result = new unsigned char[vd->chan * numPixels];

      for (y = 0; y < height; y++)
        for (x = 0; x < width; x++)
      {
        index = (y * width + x) * vd->chan;
        for (c = 0; c < vd->chan; c++)
        {
          result[index + c] = data[index + c];
          std::cerr << "Result: " << index+c << " " << (int) (result[index+c]) << endl;
        }
      }
    }
    else
      assert(0);
  }

  std::cerr << "result read" << endl;

  disableTexture(GL_TEXTURE_3D_EXT);

  glPopMatrix();

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

  return result;
}

float vvTexRend::getManhattenDist(float p1[3], float p2[3])
{
  float dist = 0;
  int i;

  for (i=0; i<3; ++i)
  {
    dist += float(fabs(p1[i] - p2[i])) / float(vd->getSize()[i] * vd->vox[i]);
  }

  std::cerr << "Manhattan Distance: " << dist << endl;

  return dist;
}

void vvTexRend::initVertArray(const int numSlices)
{
  if(_elemCounts.size() >= numSlices)
    return;

  _elemCounts.resize(numSlices);
  _vertIndices.resize(numSlices);
  _vertIndicesAll.resize(numSlices*6);
  _vertArray.resize(numSlices*12);

  int idxIterator = 0;
  int vertIterator = 0;
  for (int i = 0; i < numSlices; ++i)
  {
    _elemCounts[i] = 6;
    _vertIndices[i] = &_vertIndicesAll[i*6];
    for (int j = 0; j < 6; ++j)
    {
      _vertIndices[i][j] = idxIterator;
      ++idxIterator;

      _vertArray[vertIterator] = j;
      ++vertIterator;
      _vertArray[vertIterator] = i;
      ++vertIterator;
    }
  }
}

void vvTexRend::validateEmptySpaceLeaping()
{
  // Only do empty space leaping for ordinary transfer functions
  if (_renderState._emptySpaceLeaping == true)
  {
    _renderState._emptySpaceLeaping &= (voxelType != VV_PIX_SHD) || (_currentShader == 0) || (_currentShader == 12);
    _renderState._emptySpaceLeaping &= (voxelType != VV_RGBA);
  }
}

//============================================================================
// End of File
//============================================================================
