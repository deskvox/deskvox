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

#include "vvglew.h"

#include <iostream>
#include <algorithm>
#include <limits.h>
#include <math.h>

#include "vvopengl.h"
#include "vvdynlib.h"

#include "vvx11.h"

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvbsptreevisitors.h"
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

// The following values will have to be approved empirically yet... .

// TODO: reasonable determination of these values.

// Redistribute bricks if rendering time deviation > MAX_DEVIATION %
const float MAX_DEVIATION = 10.0f;
// Don't redistribute bricks for peaks, but rather for constant deviation.
const int LIMIT = 10;
// Maximum individual deviation (otherwise the share will be adjusted).
const float MAX_IND_DEVIATION = 0.001f;
const float INC = 0.05f;

const int MAX_VIEWPORT_WIDTH = 4800;
const int MAX_VIEWPORT_HEIGHT = 1200;

using namespace std;

//----------------------------------------------------------------------------
const int vvTexRend::NUM_PIXEL_SHADERS = 13;

struct ThreadArgs
{
  int threadId;                               ///< integer id of the thread
  bool active;

  // Algorithm specific.
  vvHalfSpace* halfSpace;
  std::vector<BrickList> brickList;
  std::vector<BrickList> nonemptyList;
  BrickList sortedList;                       ///< sorted list built up from the brick sets
  BrickList insideList;
  vvVector3 probeMin;
  vvVector3 probeMax;
  vvVector3 probePosObj;
  vvVector3 probeSizeObj;
  vvVector3 delta;
  vvVector3 farthest;
  vvVector3 normal;
  vvVector3 eye;
  bool isOrtho;
  int numSlices;
  GLfloat* modelview;                         ///< the current GL_MODELVIEW matrix
  GLfloat* projection;                        ///< the current GL_PROJECTION matrix
  vvTexRend* renderer;                        ///< pointer to the calling instance. useful to use functions from the renderer class
  GLfloat* pixels;                            ///< after rendering each thread will read back its data to this array
  int width;                                  ///< viewport width to init the offscreen buffer with
  int height;                                 ///< viewport height to init the offscreen buffer with
  float lastRenderTime;                       ///< measured for dynamic load balancing
  float share;                                ///< ... of the volume managed by this thread. Adjustable for load balancing.
  bool brickDataChanged;
  bool transferFunctionChanged;
  int lastFrame;                              ///< last rendered animation frame

#ifdef HAVE_X11
  // Glx rendering specific.
  GLXContext glxContext;                      ///< the initial glx context
  Display* display;                           ///< a pointer to the current glx display
  Drawable drawable;
#elif defined _WIN32
  HGLRC wglContext;
  HWND window;
  HDC deviceContext;
#endif

  // Gl state specific.
  GLuint* privateTexNames;
  int numTextures;
  GLuint pixLUTName;
  uchar* rgbaLUT;
};

//----------------------------------------------------------------------------
/** Constructor.
  @param vd                      volume description
  @param renderState             object describing the render state
  @param geom                    render geometry (default: automatic)
  @param vox                     voxel type (default: best)
  @param displayNames            names of x-displays (host:display.screen) for multi-gpu rendering
  @param numDisplays             # displays for multi-gpu rendering
  @param multiGpuBufferPrecision precision of the offscreen buffer used for multi-gpu rendering
*/
vvTexRend::vvTexRend(vvVolDesc* vd, vvRenderState renderState, GeometryType geom, VoxelType vox,
                     std::vector<BrickList>* bricks,
                     const char** displayNames, const int numDisplays,
                     const BufferPrecision multiGpuBufferPrecision)
  : vvRenderer(vd, renderState)
{
  vvDebugMsg::msg(1, "vvTexRend::vvTexRend()");

  setDisplayNames(displayNames, numDisplays);
  _multiGpuBufferPrecision = multiGpuBufferPrecision;

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
#ifdef GL_VERSION_2_1
    cerr << ", 2.1";
#endif
#ifdef GL_VERSION_3_0
    cerr << ", 3.0";
#endif
#ifdef GL_VERSION_3_1
    cerr << ", 3.1";
#endif
#ifdef GL_VERSION_3_2
    cerr << ", 3.2";
#endif
#ifdef GL_VERSION_3_3
    cerr << ", 3.3";
#endif
#ifdef GL_VERSION_4_0
    cerr << ", 4.0";
#endif
#ifdef GL_VERSION_4_1
    cerr << ", 4.1";
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
  _measureRenderTime = false;
  interpolation = true;
  _bspTree = NULL;
  _numSlaveNodes = 0;
  _aabbMask = NULL;
  _isSlave = false;

  if (_renderState._useOffscreenBuffer)
  {
    _renderTarget = new vvOffscreenBuffer(_renderState._imageScale, VV_BYTE);
    if (_renderState._opaqueGeometryPresent)
    {
      dynamic_cast<vvOffscreenBuffer*>(_renderTarget)->setPreserveDepthBuffer(true);
    }
  }
  else
  {
    _renderTarget = new vvRenderTarget();
  }

  if (vvShaderFactory::isSupported(VV_CG_MANAGER))
  {
    _currentShader = vd->chan - 1;
    _previousShader = _currentShader;
  }
  else
  {
    _currentShader = 0;
    _previousShader = 0;
  }
  _useOnlyOneBrick = false;
  if (bricks != NULL)
  {
    _brickList = *bricks;
    calcAABBMask();
    _areEmptyBricksCreated = true;
  }
  else
  {
    _areEmptyBricksCreated = false;
  }
  _areBricksCreated = false;
  _lastFrame = -1;
  lutDistance = -1.0;
  _renderState._isROIChanged = true;

  // Find out which OpenGL extensions are supported:
  extTex3d  = vvGLTools::isGLextensionSupported("GL_EXT_texture3D") || vvGLTools::isGLVersionSupported(1,2,1);
  arbMltTex = vvGLTools::isGLextensionSupported("GL_ARB_multitexture") || vvGLTools::isGLVersionSupported(1,3,0);

  _isectShader = vvShaderFactory::provideShaderManager(VV_GLSL_MANAGER);
  _pixelShader = vvShaderFactory::provideShaderManager(VV_CG_MANAGER);

  extMinMax = vvGLTools::isGLextensionSupported("GL_EXT_blend_minmax") || vvGLTools::isGLVersionSupported(1,4,0);
  extBlendEquation = vvGLTools::isGLextensionSupported("GL_EXT_blend_equation") || vvGLTools::isGLVersionSupported(1,1,0);
  extColLUT  = isSupported(VV_SGI_LUT);
  extPalTex  = isSupported(VV_PAL_TEX);
  extTexShd  = isSupported(VV_TEX_SHD);
  extPixShd  = isSupported(VV_PIX_SHD);
  arbFrgPrg  = isSupported(VV_FRG_PRG);
  extGlslShd = isSupported(VV_GLSL_SHD);
  _proxyGeometryOnGpuSupported = extGlslShd && arbFrgPrg && _isectShader;
  _proxyGeometryOnGpu = _proxyGeometryOnGpuSupported;

  extNonPower2 = vvGLTools::isGLextensionSupported("GL_ARB_texture_non_power_of_two") || vvGLTools::isGLVersionSupported(2,0,0);

  // Init glew.
  glewInit();

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
  if (geomType == VV_BRICKS)
  {
    cerr << ", proxy geometry is generated on the ";
    if (_proxyGeometryOnGpu)
    {
      cerr << "GPU";
    }
    else
    {
      cerr << "CPU";
    }
  }
  cerr << endl;

  if ((_numThreads > 0) && (geomType != VV_BRICKS))
  {
    cerr << "No multi-gpu support for the chosen render algorithm. Falling back to single-gpu mode" << endl;
    _numThreads = 0;
  }

  if (_numThreads > 0)
  {
    dispatchThreadedGLXContexts();
    dispatchThreadedWGLContexts();
  }

  if (vvShaderFactory::isSupported(VV_CG_MANAGER))
  {
    if(geomType==VV_SLICES || geomType==VV_CUBIC2D)
    {
      _currentShader = get2DTextureShader();
    }
  }

  if (geomType == VV_BRICKS)
  {
    validateEmptySpaceLeaping();
  }

  if ((_usedThreads == 0) && (voxelType==VV_TEX_SHD || voxelType==VV_PIX_SHD || voxelType==VV_FRG_PRG))
  {
    glGenTextures(1, &pixLUTName);
  }

  if (_usedThreads == 0)
  {
    initPostClassificationStage(_pixelShader, fragProgName);
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

  if (_usedThreads == 0)
  {
    if (voxelType != VV_RGBA)
    {
      makeTextures(pixLUTName, rgbaLUT);      // we only have to do this once for non-RGBA textures
    }
    updateTransferFunction(pixLUTName, rgbaLUT);
  }
  else
  {
    if (vvDebugMsg::getDebugLevel() > 1)
    {
      generateDebugColors();
    }

    // Each thread will make its own share of textures and texture ids. Don't distribute
    // the bricks before this is completed.
    pthread_barrier_wait(&_distributeBricksBarrier);

    vvVector3 probePosObj;
    vvVector3 probeSizeObj;
    vvVector3 probeMin, probeMax;

    // Build global transfer function once in order to build up _nonemptyList.
    updateTransferFunction(pixLUTName, rgbaLUT);
    calcProbeDims(probePosObj, probeSizeObj, probeMin, probeMax);
    getBricksInProbe(_nonemptyList, _insideList, _sortedList, probePosObj, probeSizeObj, _renderState._isROIChanged);
    distributeBricks();
    pthread_barrier_wait(&_distributedBricksBarrier);
  }

  if (_proxyGeometryOnGpu && (_usedThreads == 0))
  {
    // Remember that if you bypass the fixed function pipeline, you have to
    // supply matrix state, normals, colors etc. on your own. So if a vertex
    // shader for isect tests is bound, at least a simple frag program is
    // necessary that passes the color through.
    if (voxelType == VV_RGBA)
    {
      _proxyGeometryOnGpuSupported = initIntersectionShader(_isectShader, _pixelShader);
    }
    else
    {
      _proxyGeometryOnGpuSupported = initIntersectionShader(_isectShader);
    }
    if (_proxyGeometryOnGpuSupported)
      setupIntersectionParameters(_isectShader);
    else
      _proxyGeometryOnGpu = false;
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

  if (_usedThreads == 0)
  {
    removeTextures(texNames, &textures);
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

  if (_usedThreads > 0)
  {
    // Finally join the threads.
    _terminateThreads = true;
    pthread_barrier_wait(&_renderStartBarrier);
    for (unsigned int i = 0; i < _usedThreads; ++i)
    {
      void* exitStatus;
      pthread_join(_threads[i], &exitStatus);
    }

    for (unsigned int i = 0; i < _usedThreads; ++i)
    {
      delete[] _threadData[i].pixels;
      delete[] _threadData[i].rgbaLUT;
    }

    // Cleanup locking specific stuff.
    pthread_barrier_destroy(&_distributeBricksBarrier);
    pthread_barrier_destroy(&_distributedBricksBarrier);
    pthread_barrier_destroy(&_renderStartBarrier);

    // Clean up arrays.
    delete[] _threads;
    delete[] _threadData;
    delete[] _displayNames;
    delete[] _screens;

    delete _visitor;
    delete _bspTree;
  }
}

//----------------------------------------------------------------------------
/** Chooses the best rendering geometry depending on the graphics hardware's
  capabilities.
  @param geom desired geometry
*/
vvTexRend::GeometryType vvTexRend::findBestGeometry(const vvTexRend::GeometryType geom,
                                                    const vvTexRend::VoxelType vox) const
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
vvTexRend::VoxelType vvTexRend::findBestVoxelType(const vvTexRend::VoxelType vox) const
{
  vvDebugMsg::msg(1, "vvTexRend::findBestVoxelType()");

  if (vox==VV_BEST)
  {
    if (vd->chan==1)
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
void vvTexRend::removeTextures(GLuint*& privateTexNames, int* numTextures) const
{
  vvDebugMsg::msg(1, "vvTexRend::removeTextures()");

  if ((*numTextures > 0) && (privateTexNames != NULL))
  {
    glDeleteTextures(*numTextures, privateTexNames);
    delete[] privateTexNames;
    privateTexNames = NULL;
    numTextures = 0;
  }
}

//----------------------------------------------------------------------------
/// Generate textures for all rendering modes.
vvTexRend::ErrorType vvTexRend::makeTextures(const GLuint& lutName, uchar*& lutData)
{
  static bool first = true;
  ErrorType err = OK;

  vvDebugMsg::msg(2, "vvTexRend::makeTextures()");

  if (vd->vox[0] == 0 || vd->vox[1] == 0 || vd->vox[2] == 0)
    return err;

  if (geomType != VV_BRICKS)
  {
    // Compute texture dimensions (must be power of 2):
    texels[0] = vvToolshed::getTextureSize(vd->vox[0]);
    texels[1] = vvToolshed::getTextureSize(vd->vox[1]);
    texels[2] = vvToolshed::getTextureSize(vd->vox[2]);
  }
  else
  {
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
  }

  switch (geomType)
  {
    case VV_SLICES:  err=makeTextures2D(1); updateTextures2D(1, 0, 10, 20, 15, 10, 5); break;
    case VV_CUBIC2D: err=makeTextures2D(3); updateTextures2D(3, 0, 10, 20, 15, 10, 5); break;
    // Threads will make their own copies of the textures as well as their own gl ids.
    case VV_BRICKS:
      if (!_areEmptyBricksCreated)
      {
        err = makeEmptyBricks();
      }
      // If in threaded mode, each thread will upload its texture data on its own.
      if ((err == OK) && (_usedThreads == 0))
      {
        err = makeTextureBricks(texNames, &textures, lutData, _brickList, _areBricksCreated);
      }
      break;
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
void vvTexRend::makeLUTTexture(const GLuint& lutName, uchar* lutData) const
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
vvTexRend::ErrorType vvTexRend::makeTextures2D(const int axes)
{
  GLint glWidth;                                  // return value from OpenGL call
  uchar* rgbaSlice[3];                            // RGBA slice data for texture memory for each principal axis
  int rawVal[4];                                  // raw values for R,G,B,A
  int rawSliceSize;                               // number of bytes in a slice of the raw data array
  int rawLineSize;                                // number of bytes in a row of the raw data array
  int texSize[3];                                 // size of a 2D texture in bytes for each principal axis
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
  bool accommodated = true;                       // false if a texture cannot be accommodated in TRAM
  ErrorType err = OK;

  vvDebugMsg::msg(1, "vvTexRend::makeTextures2D()");

  assert(axes==1 || axes==3);

  removeTextures(texNames, &textures);            // first remove previously generated textures from TRAM

  const int frames = vd->frames;
  
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
  for (int i=3-axes; i<3; ++i)
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
  for (int i=3-axes; i<3; ++i)
  {
    rgbaSlice[i] = new uchar[texSize[i]];
  }

  // Generate texture data:
  for (int f=0; f<frames; ++f)
  {
    raw = vd->getRaw(f);                          // points to beginning of frame in raw data
    for (int i=3-axes; i<3; ++i)                      // generate textures for each principal axis
    {
      memset(rgbaSlice[i], 0, texSize[i]);        // initialize with 0's for invisible empty regions

      // Generate texture contents:
      for (int s=0; s<rs[i]; ++s)                 // loop thru texture and raw data slices
      {
        for (int h=0; h<rh[i]; ++h)               // loop thru raw data rows
        {
          // Set voxel to starting position in raw data array:
          rawVoxel = raw + rawStart[i] + s * rawStepS[i] + h * rawStepH[i];

          for (int w=0; w<rw[i]; ++w)             // loop thru raw data columns
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
                const float fval = *((float*)(rawVoxel));
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
                  for (int c=0; c<4; ++c)
                  {
                    rgbaSlice[i][texSliceIndex + c]   = (uchar)rawVal[0];
                  }
                  break;
                case VV_PIX_SHD:
                  rgbaSlice[i][texSliceIndex] = (uchar)rawVal[0];
                  break;
                case VV_RGBA:
                  for (int c=0; c<4; ++c)
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
              for (int c=0; c<ts_min(vd->chan,4); ++c)
              {
                rawVal[c] = *(rawVoxel + c);
              }

              // Copy color components:
              for (int c=0; c<ts_min(vd->chan,3); ++c)
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
                int alpha = 0;
                for (int c=0; c<vd->chan; ++c)
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
  for (int i=3-axes; i<3; ++i)
    delete[] rgbaSlice[i];
  return err;
}

vvTexRend::ErrorType vvTexRend::makeEmptyBricks()
{
  ErrorType err = OK;
  int tmpTexels[3];                               // number of texels in each dimension for current brick

  if (!extTex3d) return NO3DTEX;

  if (_renderState._brickSize == 0)
  {
    vvDebugMsg::msg(1, "3D Texture brick size unknown");
    return TEX_SIZE_UNKNOWN;
  }

  const int frames = vd->frames;

  for(std::vector<BrickList>::iterator frame = _brickList.begin();
      frame != _brickList.end();
      ++frame)
  {
    for(BrickList::iterator brick = frame->begin(); brick != frame->end(); ++brick)
      delete *brick;
    frame->clear();
  }
  _brickList.clear();

  const int texSize = texels[0] * texels[1] * texels[2] * texelsize;

  vvDebugMsg::msg(1, "3D Texture (bricking) width     = ", texels[0]);
  vvDebugMsg::msg(1, "3D Texture (bricking) height    = ", texels[1]);
  vvDebugMsg::msg(1, "3D Texture (bricking) depth     = ", texels[2]);
  vvDebugMsg::msg(1, "3D Texture (bricking) size (KB) = ", texSize / 1024);

  // helper variables
  const vvVector3 voxSize(vd->getSize()[0] / (vd->vox[0] - 1),
                          vd->getSize()[1] / (vd->vox[1] - 1),
                          vd->getSize()[2] / (vd->vox[2] - 1));

  const vvVector3 halfBrick(float(texels[0]-_renderState._brickTexelOverlap) * 0.5f,
                            float(texels[1]-_renderState._brickTexelOverlap) * 0.5f,
                            float(texels[2]-_renderState._brickTexelOverlap) * 0.5f);

  const vvVector3 halfVolume(float(vd->vox[0] - 1) * 0.5f,
                             float(vd->vox[1] - 1) * 0.5f,
                             float(vd->vox[2] - 1) * 0.5f);

  _brickList.resize(frames);
  for (int f = 0; f < frames; f++)
  {
    for (int bx = 0; bx < _numBricks[0]; bx++)
      for (int by = 0; by < _numBricks[1]; by++)
        for (int bz = 0; bz < _numBricks[2]; bz++)
        {
          // offset to first voxel of current brick
          const int startOffset[3] = { bx * _renderState._brickSize[0],
                                       by * _renderState._brickSize[1],
                                       bz * _renderState._brickSize[2] };

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

          vvBrick* currBrick = new vvBrick();
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

          int brickTexelOverlap[3];
          for (int d = 0; d < 3; ++d)
          {
            brickTexelOverlap[d] = _renderState._brickTexelOverlap;
            const float maxObj = (startOffset[d] + bs[d]) * vd->dist[d] * vd->_scale;
            if (maxObj > vd->getSize()[d])
            {
              brickTexelOverlap[d] = 0;
            }
          }

          currBrick->pos.set(voxSize[0] * (startOffset[0] + halfBrick[0] - halfVolume[0]),
            voxSize[1] * (startOffset[1] + halfBrick[1] - halfVolume[1]),
            voxSize[2] * (startOffset[2] + halfBrick[2] - halfVolume[2]));
          currBrick->min.set(voxSize[0] * (startOffset[0] - halfVolume[0]),
            voxSize[1] * (startOffset[1] - halfVolume[1]),
            voxSize[2] * (startOffset[2] - halfVolume[2]));
          currBrick->max.set(voxSize[0] * (startOffset[0] + (tmpTexels[0] - brickTexelOverlap[0]) - halfVolume[0]),
            voxSize[1] * (startOffset[1] + (tmpTexels[1] - brickTexelOverlap[1]) - halfVolume[1]),
            voxSize[2] * (startOffset[2] + (tmpTexels[2] - brickTexelOverlap[2]) - halfVolume[2]));

          for (int d = 0; d < 3; ++d)
          {
            if (currBrick->max[d] > vd->getSize()[d])
            {
              currBrick->max[d] = vd->getSize()[d];
            }

            currBrick->texels[d] = tmpTexels[d];
            currBrick->startOffset[d] = startOffset[d];
            const float overlapNorm = (float)(brickTexelOverlap[d]) / (float)tmpTexels[d];
            currBrick->texRange[d] = (1.0f - overlapNorm);
            currBrick->texMin[d] = (1.0f / (2.0f * (float)(_renderState._brickTexelOverlap) * (float)tmpTexels[d]));
          }

          const int texIndex = (f * _numBricks[0] * _numBricks[1] * _numBricks[2]) + (bx * _numBricks[2] * _numBricks[1])
            + (by * _numBricks[2]) + bz;
          currBrick->index = texIndex;

          _brickList[f].push_back(currBrick);
        } // # foreach (numBricks[i])
  } // # frames

  _areEmptyBricksCreated = true;

  return err;
}

vvTexRend::ErrorType vvTexRend::makeTextureBricks(GLuint*& privateTexNames, int* numTextures, uchar*& lutData,
                                                  std::vector<BrickList>& brickList, bool& areBricksCreated) const
{
  ErrorType err = OK;
  int rawVal[4];                                  // raw values for R,G,B,A
  bool accommodated = true;                       // false if a texture cannot be accommodated in TRAM

  removeTextures(privateTexNames, numTextures);

  const int frames = vd->frames;

  const int texSize = texels[0] * texels[1] * texels[2] * texelsize;
  uchar* texData = new uchar[texSize];

  // number of textures needed
  *numTextures = frames * _numBricks[0] * _numBricks[1] * _numBricks[2];

  privateTexNames = new GLuint[*numTextures];
  glGenTextures(*numTextures, privateTexNames);

  // generate textures contents:
  vvDebugMsg::msg(2, "Transferring textures to TRAM. Total size [KB]: ",
    *numTextures * texSize / 1024);

  const bool bpcValid = (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4);
  const bool oneChannel = ((vd->chan == 1) && bpcValid);
  const int rawSliceSize = vd->getSliceBytes();

  int f = 0;
  for(std::vector<BrickList>::iterator frame = brickList.begin();
      frame != brickList.end();
      ++frame)
  {
    const uchar* raw = vd->getRaw(f);
    ++f;

    for(BrickList::iterator b = frame->begin(); b != frame->end(); ++b)
    {
      vvBrick* currBrick = *b;
      // offset to first voxel of current brick
      const int startOffset[3] = { currBrick->startOffset[0],
                                   currBrick->startOffset[1],
                                   currBrick->startOffset[2] };

      const int tmpTexels[3] = { currBrick->texels[0],
                                 currBrick->texels[1],
                                 currBrick->texels[2] };

      // Memorize the max and min scalar values in the volume. These are stored
      // to perform empty space leaping later on.
      int minValue = INT_MAX;
      int maxValue = -INT_MAX;

      memset(texData, 0, texSize);

      // Essentially: for s :=  startOffset[2] to (startOffset[2] + texels[2]) do
      for (int s = startOffset[2]; (s < (startOffset[2] + tmpTexels[2])) && (s < vd->vox[2]); s++)
      {
        if (s < 0) continue;
        const int rawSliceOffset = (vd->vox[2] - s - 1) * rawSliceSize;
        // Essentially: for y :=  startOffset[1] to (startOffset[1] + texels[1]) do
        for (int y = startOffset[1]; (y < (startOffset[1] + tmpTexels[1])) && (y < vd->vox[1]); y++)
        {
          if (y < 0) continue;
          const int heightOffset = (vd->vox[1] - y - 1) * vd->vox[0] * vd->bpc * vd->chan;
          const int texLineOffset = (y - startOffset[1]) * tmpTexels[0] + (s - startOffset[2]) * tmpTexels[0] * tmpTexels[1];
          if (oneChannel)
          {
            // Essentially: for x :=  startOffset[0] to (startOffset[0] + texels[0]) do
            for (int x = startOffset[0]; (x < (startOffset[0] + tmpTexels[0])) && (x < vd->vox[0]); x++)
            {
              if (x < 0) continue;
              const int srcIndex = vd->bpc * x + rawSliceOffset + heightOffset;
              if (vd->bpc == 1) rawVal[0] = (int) raw[srcIndex];
              else if (vd->bpc == 2)
              {
                rawVal[0] = ((int) raw[srcIndex] << 8) | (int) raw[srcIndex + 1];
                rawVal[0] >>= 4;
              }
              else  // vd->bpc == 4
              {
                const float fval = *((float*)(raw + srcIndex));      // fetch floating point data value
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

              const int texOffset = (x - startOffset[0]) + texLineOffset;
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
                  for (int c = 0; c < 4; c++)
                  {
                    texData[4 * texOffset + c] = (uchar) rawVal[0];
                  }
                  break;
                case VV_PIX_SHD:
                  texData[4 * texOffset] = (uchar) rawVal[0];
                  break;
                case VV_RGBA:
                  for (int c = 0; c < 4; c++)
                  {
                    texData[4 * texOffset + c] = lutData[rawVal[0] * 4 + c];
                  }
                  break;
                default:
                  assert(0);
                  break;
              }
            }
          }
          else if (bpcValid)
          {
            if (voxelType == VV_RGBA || voxelType == VV_PIX_SHD)
            {
              for (int x = startOffset[0]; (x < (startOffset[0] + tmpTexels[0])) && (x < vd->vox[0]); x++)
              {
                if (x < 0) continue;

                const int texOffset = (x - startOffset[0]) + texLineOffset;

                // fetch component values from memory:
                for (int c = 0; c < ts_min(vd->chan, 4); c++)
                {
                  int srcIndex = rawSliceOffset + heightOffset + vd->bpc * (x * vd->chan + c);
                  if (vd->bpc == 1)
                    rawVal[c] = (int) raw[srcIndex];
                  else if (vd->bpc == 2)
                  {
                    rawVal[c] = ((int) raw[srcIndex] << 8) | (int) raw[srcIndex + 1];
                    rawVal[c] >>= 4;
                  }
                 else  // vd->bpc==4
                  {
                    const float fval = *((float*) (raw + srcIndex));
                    rawVal[c] = vd->mapFloat2Int(fval);
                  }
                }

                // TODO: init empty-space leaping minValue and maxValue for multiple channels as well.


                // copy color components:
                for (int c = 0; c < ts_min(vd->chan, 3); c++)
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
                  int alpha = 0;
                  for (int c = 0; c < vd->chan; c++)
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

      currBrick->minValue = minValue;
      currBrick->maxValue = maxValue;
      currBrick->visible = true;

      accommodated = currBrick->upload3DTexture(privateTexNames[currBrick->index], texData,
                                                texFormat, internalTexFormat,
                                                interpolation);
      if(!accommodated)
         break;
    } // # foreach (numBricks[i])
    if(!accommodated)
       break;
  } // # frames

  if (!accommodated)
  {
    cerr << "Insufficient texture memory for 3D textures." << endl;
    err = TRAM_ERROR;
  }

  delete[] texData;
  areBricksCreated = true;

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
  _numThreads = numNames;

  _threads = new pthread_t[_numThreads];
  _threadData = new ThreadArgs[_numThreads];
  _displayNames = new const char*[_numThreads];
  _screens = new unsigned int[_numThreads];

  bool hostSeen;
  for (unsigned int i = 0; i < _numThreads; ++i)
  {
    _displayNames[i] = displayNames[i];
    hostSeen = false;

    // Parse the display name string for host, display and screen.
    for (size_t j = 0; j < strlen(_displayNames[i]); ++j)
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
      else
      {
        // Default to screen 0.
        _screens[i] = 0;
      }
    }
  }

  return err;
}

#ifdef _WIN32
LRESULT CALLBACK WndProcedure(HWND hWnd, UINT Msg, WPARAM wParam, LPARAM lParam)
{
  switch(Msg)
  {
  case WM_DESTROY:
    PostQuitMessage(WM_QUIT);
    break;
  default:
    return DefWindowProc(hWnd, Msg, wParam, lParam);
  }
  return 0;
}
#endif

vvTexRend::ErrorType vvTexRend::dispatchThreadedWGLContexts()
{
#if !defined(_WIN32)
  return UNSUPPORTED;
#else
  ErrorType err = OK;

  DISPLAY_DEVICE dispDev;
  DEVMODE devMode;
  dispDev.cb = sizeof(DISPLAY_DEVICE);

  for (unsigned int i = 0; i < _numThreads; ++i)
  {
    if (EnumDisplayDevices(NULL, i, &dispDev, NULL))
    {
      EnumDisplaySettings(dispDev.DeviceName, ENUM_CURRENT_SETTINGS, &devMode);
    }

    HINSTANCE hInstance = GetModuleHandle(0);
    WNDCLASSEX WndClsEx;
    ZeroMemory(&WndClsEx, sizeof(WNDCLASSEX));


    LPCTSTR ClsName = L"Virvo Multi-GPU Renderer";
    LPCTSTR WndName = L"Debug Window";

    WndClsEx.cbSize        = sizeof(WNDCLASSEX);
    WndClsEx.style         = CS_HREDRAW | CS_VREDRAW;
    WndClsEx.lpfnWndProc   = WndProcedure;
    WndClsEx.cbClsExtra    = 0;
    WndClsEx.cbWndExtra    = 0;
    WndClsEx.hIcon         = LoadIcon(NULL, IDI_APPLICATION);
    WndClsEx.hCursor       = LoadCursor(NULL, IDC_ARROW);
    WndClsEx.lpszMenuName  = NULL;
    WndClsEx.lpszClassName = ClsName;
    WndClsEx.hInstance     = hInstance;
    WndClsEx.hbrBackground = 0;
    WndClsEx.hIconSm       = LoadIcon(NULL, IDI_APPLICATION);
    ATOM WindowAtom = RegisterClassEx(&WndClsEx);

    PIXELFORMATDESCRIPTOR pfd;
    ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 16;
    pfd.iLayerType = PFD_MAIN_PLANE;

    HWND hWnd = CreateWindow(ClsName, WndName, WS_OVERLAPPEDWINDOW,
                             CW_USEDEFAULT, CW_USEDEFAULT, 512, 512,
                             NULL, NULL, hInstance, NULL);

    if (!hWnd)
    {
      cerr << "Couldn't create window on display: " << i << endl;
    }
    else
    {
      _threadData[i].window = hWnd;

      //if (vvDebugMsg::getDebugLevel() > 0)
      {
        ShowWindow(hWnd, SW_SHOWNORMAL);
      }
      _threadData[i].deviceContext = GetDC(hWnd);
      int pixelFormat = ChoosePixelFormat(_threadData[i].deviceContext, &pfd);
      if (pixelFormat != 0)
      {
        SetPixelFormat(_threadData[i].deviceContext, pixelFormat, &pfd);
        _threadData[i].wglContext = wglCreateContext(_threadData[i].deviceContext);
      }
    }
  }

  err = dispatchThreads();

  return err;
#endif
}

vvTexRend::ErrorType vvTexRend::dispatchThreadedGLXContexts()
{
#ifndef HAVE_X11
  return UNSUPPORTED;
#else
  ErrorType err = OK;
  int attrList[] = { GLX_RGBA , GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8, None};

  int slaveWindowWidth = 1;
  int slaveWindowHeight = 1;
  if (vvDebugMsg::getDebugLevel() > 0)
  {
    slaveWindowWidth = 512;
    slaveWindowHeight = 512;
  }
  int unresolvedDisplays = 0;
  for (int i = 0; i < _numThreads; ++i)
  {
    _threadData[i].display = XOpenDisplay(_displayNames[i]);

    if (_threadData[i].display != NULL)
    {
      _threadData[i].active = true;
      const Drawable parent = RootWindow(_threadData[i].display, _screens[i]);

      XVisualInfo* vi = glXChooseVisual(_threadData[i].display,
                                        DefaultScreen(_threadData[i].display),
                                        attrList);

      XSetWindowAttributes wa = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
      wa.colormap = XCreateColormap(_threadData[i].display, parent, vi->visual, AllocNone );
      wa.background_pixmap = None;
      wa.border_pixel = 0;

      if (vvDebugMsg::getDebugLevel() == 0)
      {
        wa.override_redirect = true;
      }
      else
      {
        wa.override_redirect = false;
      }

      _threadData[i].glxContext = glXCreateContext(_threadData[i].display, vi, NULL, GL_TRUE);
      _threadData[i].drawable = XCreateWindow(_threadData[i].display, parent, 0, 0, slaveWindowWidth, slaveWindowHeight, 0,
                                              vi->depth, InputOutput, vi->visual,
                                              CWBackPixmap|CWBorderPixel|CWEventMask|CWColormap|CWOverrideRedirect, &wa );
    }
    else
    {
      _threadData[i].active = false;
      cerr << "Couldn't open display: " << _displayNames[i] << endl;
      ++unresolvedDisplays;
    }
  }

  if (unresolvedDisplays >= _numThreads)
  {
    _numThreads = 0;
    cerr << "Falling back to none-threaded rendering mode" << endl;
    return NO_DISPLAYS_OPENED;
  }
  _usedThreads = _numThreads - unresolvedDisplays;

  err = dispatchThreads();

  return err;
#endif
}

vvTexRend::ErrorType vvTexRend::dispatchThreads()
{
  ErrorType err = OK;

  const vvGLTools::Viewport viewport = vvGLTools::getViewport();
  for (unsigned int i = 0; i < _numThreads; ++i)
  {
    if (_threadData[i].active)
    {
      _threadData[i].threadId = i;
      _threadData[i].renderer = this;
      // Start solution for load balancing: every thread renders 1/n of the volume.
      // During load balancing, the share will (probably) be adjusted.
      _threadData[i].share = 1.0f / static_cast<float>(_usedThreads);
      _threadData[i].width = viewport[2];
      _threadData[i].height = viewport[3];
      _threadData[i].pixels = new GLfloat[MAX_VIEWPORT_WIDTH * MAX_VIEWPORT_HEIGHT * 4];
      _threadData[i].brickDataChanged = false;
      _threadData[i].transferFunctionChanged = false;
      _threadData[i].lastFrame = -1;
      _threadData[i].rgbaLUT = new uchar[256 * 256 * 4];
      _threadData[i].privateTexNames = NULL;
#ifdef HAVE_X11
      XMapWindow(_threadData[i].display, _threadData[i].drawable);
      XFlush(_threadData[i].display);
#endif
    }
  }

  _visitor = NULL;

  // Only set to true by destructor to join threads.
  _terminateThreads = false;

  // If smth changed, the bricks will be distributed among the threads before rendering.
  // Obviously, initially something changed.
  _somethingChanged = true;

  // First a barrier is passed that ensures that each thread built its textures (_distributeBricksBarrier).
  // The following barrier (_distributedBricksBarrier) is passed right after each thread
  // was assigned the bricks it is responsible for.
  pthread_barrier_init(&_distributeBricksBarrier, NULL, _usedThreads + 1);
  pthread_barrier_init(&_distributedBricksBarrier, NULL, _usedThreads + 1);

  // This barrier will be waited for by every worker thread and additionally once by the main
  // thread for every frame.
  pthread_barrier_init(&_renderStartBarrier, NULL, _usedThreads + 1);

  // Container for offscreen buffers. The threads will initialize
  // their buffers themselves when their respective gl context is
  // bound, thus the buffers are set to 0 initially.
  _offscreenBuffers = new vvOffscreenBuffer*[_numThreads];
  for (unsigned int i = 0; i < _numThreads; ++i)
  {
    _offscreenBuffers[i] = NULL;
  }

  // Dispatch the threads. This has to be done in a separate loop since the first operation
  // the callback function will perform is making its gl context current. This would most
  // probably interfer with the context creation performed in the loop above.
  for (unsigned int i = 0; i < _numThreads; ++i)
  {
    if (_threadData[i].active)
    {
      pthread_create(&_threads[i], NULL, threadFuncTexBricks, (void*)&_threadData[i]);
    }
  }

  return err;
}

/*!
 *
 */
vvTexRend::ErrorType vvTexRend::distributeBricks()
{
  ErrorType err = OK;

  // A new game... .
  _deviationExceedCnt = 0;

  float* part = new float[_usedThreads];
  int p = 0;
  for (unsigned int i = 0; i < _numThreads; ++i)
  {
    if (_threadData[i].active)
    {
      part[p] = _threadData[i].share;
      ++p;
    }
  }

  delete _bspTree;
  _bspTree = new vvBspTree(part, _usedThreads, _brickList);
  // The visitor (supplies rendering logic) must have knowledge
  // about the texture ids of the compositing polygons.
  // Thus provide a pointer to these.
  delete _visitor;
  _visitor = new vvThreadVisitor();
  _visitor->setOffscreenBuffers(_offscreenBuffers, _usedThreads);

  // Provide the visitor with the pixel data of each thread either.
  GLfloat** pixels = new GLfloat*[_usedThreads];
  for (unsigned int i = 0; i < _numThreads; ++i)
  {
    if (_threadData[i].active)
    {
      pixels[i] = _threadData[i].pixels;
    }
  }
  _visitor->setPixels(pixels);

  const vvGLTools::Viewport viewport = vvGLTools::getViewport();
  _visitor->setWidth(viewport.values[2]);
  _visitor->setHeight(viewport.values[3]);

  _bspTree->setVisitor(_visitor);

  delete[] part;

  unsigned int i = 0;
  std::vector<vvHalfSpace *>::const_iterator it = _bspTree->getLeafs()->begin();
  while (true)
  {
    if (_threadData[i].active)
    {
      _threadData[i].halfSpace = *it;
      _threadData[i].halfSpace->setId(_threadData[i].threadId);

      _threadData[i].brickList.clear();

      for (int f=0; f<_threadData[i].halfSpace->getBrickList().size(); ++f)
      {
        _threadData[i].brickList.push_back(BrickList());
        for (BrickList::iterator brickIt = _threadData[i].halfSpace->getBrickList()[f].begin();
             brickIt != _threadData[i].halfSpace->getBrickList()[f].end(); ++brickIt)
        {
          _threadData[i].brickList[f].push_back(*brickIt);
        }
      }
      ++it;
    }
    ++i;
    if (i >= _numThreads) break;
  }

  _somethingChanged = false;

  return err;
}

void vvTexRend::updateBrickGeom()
{
  vvBrick* tmp;
  vvVector3 voxSize;
  vvVector3 halfBrick;
  vvVector3 halfVolume;

  for (size_t f = 0; f < _brickList.size(); ++f)
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

    for (size_t c = 0; c < _brickList[f].size(); ++c)
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

void vvTexRend::setShowBricks(const bool flag)
{
  _renderState._showBricks = flag;
}

bool vvTexRend::getShowBricks() const
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
        for (unsigned int i = 0; i < _numThreads; ++i)
        {
          if (_threadData[i].active)
          {
            _threadData[i].brickDataChanged = true;
          }
        }
      }
      else
      {
        makeTextures(pixLUTName, rgbaLUT);
      }
    }
  }
}

bool vvTexRend::getComputeBrickSize() const
{
  return _renderState._computeBrickSize;
}

void vvTexRend::setBrickSize(const int newSize)
{
  vvDebugMsg::msg(3, "vvRenderer::setBricksize()");
  _renderState._brickSize[0] = _renderState._brickSize[1] = _renderState._brickSize[2] = newSize-1;
  _useOnlyOneBrick = false;

  if (_numThreads > 0)
  {
    for (unsigned int i = 0; i < _numThreads; ++i)
    {
      if (_threadData[i].active)
      {
        _threadData[i].brickDataChanged = true;
      }
    }
  }
  else
  {
    makeTextures(pixLUTName, rgbaLUT);
  }
}

int vvTexRend::getBrickSize() const
{
  vvDebugMsg::msg(3, "vvRenderer::getBricksize()");
  return _renderState._brickSize[0]+1;
}

void vvTexRend::setTexMemorySize(const int newSize)
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
        for (unsigned int i = 0; i < _numThreads; ++i)
        {
          if (_threadData[i].active)
          {
            _threadData[i].brickDataChanged = true;
          }
        }
      }
      else
      {
        makeTextures(pixLUTName, rgbaLUT);
      }
    }
  }
}

int vvTexRend::getTexMemorySize() const
{
  return _renderState._texMemorySize;
}

vvBspTree* vvTexRend::getBspTree() const
{
  return _bspTree;
}

//----------------------------------------------------------------------------
/** A mask telling which portion of the loaded volume data this
    renderer instance is responsible for. Used to calculate the
    projected screen rect.
    TODO: load only the relevant portion of the volume data.
 */
void vvTexRend::setAABBMask(vvAABB* aabbMask)
{
  delete aabbMask;
  _aabbMask = aabbMask;
}

vvAABB* vvTexRend::getAABBMask() const
{
  return _aabbMask;
}

vvAABB vvTexRend::getProbedMask() const
{
  vvAABB result = vvAABB(*_aabbMask);

  vvVector3 probePosObj;
  vvVector3 probeSizeObj;
  vvVector3 probeMin, probeMax;

  calcProbeDims(probePosObj, probeSizeObj, probeMin, probeMax);
  vvAABB probeBox = vvAABB(probeMin, probeMax);

  result.intersect(&probeBox);

  return result;
}

void vvTexRend::setIsSlave(const bool isSlave)
{
  _isSlave = isSlave;
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

  _useOnlyOneBrick = true;
  for(int i=0; i<3; ++i)
  {
    newBrickSize[i] = vvToolshed::getTextureSize(vd->vox[i]);
    if(newBrickSize[i] > _renderState._maxBrickSize[i])
    {
      newBrickSize[i] = _renderState._maxBrickSize[i];
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
    for (unsigned int i = 0; i < _numThreads; ++i)
    {
      if (_threadData[i].active)
      {
        _threadData[i].transferFunctionChanged = true;
      }
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
  const bool calledByWorkerThread = (lutName != pixLUTName); // FIXME: pixLUTName only initialised for certain voxel types
  if (!calledByWorkerThread)
  {
    fillNonemptyList(_nonemptyList, _brickList);
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
    _areEmptyBricksCreated = false;
    _areBricksCreated = false;
    computeBrickSize();
  }

  if (_numThreads > 0)
  {
    for (unsigned int i = 0; i < _numThreads; ++i)
    {
      if (_threadData[i].active)
      {
        _threadData[i].brickDataChanged = true;
      }
    }
  }
  else
  {
    makeTextures(pixLUTName, rgbaLUT);
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

void vvTexRend::fillNonemptyList(std::vector<BrickList>& nonemptyList, std::vector<BrickList>& brickList) const
{
  // Each bricks visible flag was initially set to true.
  // If empty-space leaping isn't active, all bricks are
  // visible by default.
  if (_renderState._emptySpaceLeaping)
  {
    int nbricks = 0, nvis=0;
    nonemptyList.clear();
    nonemptyList.resize(_brickList.size());
    // Determine visibility of each single brick in all frames
    for (size_t frame = 0; frame < brickList.size(); ++frame)
    {
      for (BrickList::iterator it = brickList[frame].begin(); it != brickList[frame].end(); ++it)
      {
        vvBrick *tmp = *it;
        nbricks++;

        // If max intensity projection, make all bricks visible.
        if (_renderState._mipMode > 0)
        {
          nonemptyList[frame].push_back(tmp);
          nvis++;
        }
        else
        {
          for (int i = tmp->minValue; i <= tmp->maxValue; ++i)
          {
            if(rgbaTF[i * 4 + 3] > 0.)
            {
              nonemptyList[frame].push_back(tmp);
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
    nonemptyList = brickList;
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
vvTexRend::ErrorType vvTexRend::updateTextures3D(const int offsetX, const int offsetY, const int offsetZ,
                                                 const int sizeX, const int sizeY, const int sizeZ, const bool newTex)
{
  ErrorType err;
  int srcIndex;
  int texOffset=0;
  int rawVal[4];
  unsigned char* raw;
  unsigned char* texData;
  bool accommodated = true;
  GLint glWidth;

  vvDebugMsg::msg(1, "vvTexRend::updateTextures3D()");

  if (!extTex3d) return NO3DTEX;

  const int texSize = sizeX * sizeY * sizeZ * texelsize;

  vvDebugMsg::msg(1, "3D Texture width     = ", sizeX);
  vvDebugMsg::msg(1, "3D Texture height    = ", sizeY);
  vvDebugMsg::msg(1, "3D Texture depth     = ", sizeZ);
  vvDebugMsg::msg(1, "3D Texture size (KB) = ", texSize / 1024);

  texData = new uchar[texSize];
  memset(texData, 0, texSize);

  const int sliceSize = vd->getSliceBytes();

  if (newTex)
  {
    vvDebugMsg::msg(2, "Creating texture names. # of names: ", vd->frames);

    removeTextures(texNames, &textures);
    textures  = vd->frames;
    delete[] texNames;
    texNames = new GLuint[textures];
    glGenTextures(vd->frames, texNames);
  }

  vvDebugMsg::msg(2, "Transferring textures to TRAM. Total size [KB]: ",
    vd->frames * texSize / 1024);

  // Generate sub texture contents:
  for (int f = 0; f < vd->frames; f++)
  {
    raw = vd->getRaw(f);
    for (int s = offsetZ; s < (offsetZ + sizeZ); s++)
    {
      const int rawSliceOffset = (vd->vox[2] - min(s,vd->vox[2]-1) - 1) * sliceSize;
      for (int y = offsetY; y < (offsetY + sizeY); y++)
      {
        const int heightOffset = (vd->vox[1] - min(y,vd->vox[1]-1) - 1) * vd->vox[0] * vd->bpc * vd->chan;
        const int texLineOffset = (y - offsetY) * sizeX + (s - offsetZ) * sizeX * sizeY;
        if (vd->chan == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
        {
          for (int x = offsetX; x < (offsetX + sizeX); x++)
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
              const float fval = *((float*)(raw + srcIndex));      // fetch floating point data value
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
                for (int c = 0; c < 4; c++)
                {
                  texData[4 * texOffset + c] = (uchar) rawVal[0];
                }
                break;
              case VV_PIX_SHD:
                texData[4 * texOffset] = (uchar) rawVal[0];
                break;
              case VV_RGBA:
                for (int c = 0; c < 4; c++)
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
            for (int x = offsetX; x < (offsetX + sizeX); x++)
            {
              texOffset = (x - offsetX) + texLineOffset;
              for (int c = 0; c < ts_min(vd->chan,4); c++)
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
                  const float fval = *((float*)(raw + srcIndex));      // fetch floating point data value
                  rawVal[c] = vd->mapFloat2Int(fval);
                }
              }

              // Copy color components:
              for (int c = 0; c < ts_min(vd->chan, 3); c++)
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
              int alpha = 0;
              for (int c = 0; c < vd->chan; c++)
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
      // Set edge values and values beyond edges to 0 for spheric textures,
      // because textures may exceed texel volume:
      for (int s = offsetZ; s < (offsetZ + sizeZ); ++s)
      {
        for (int y = offsetY; y < (offsetY + sizeY); ++y)
        {
          for (int x = offsetX; x < (offsetX + sizeX); ++x)
          {
            if ((s == 0) || (s>=vd->vox[2]-1) ||
                (y == 0) || (y>=vd->vox[1]-1) ||
                (x == 0) || (x>=vd->vox[0]-1))
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

vvTexRend::ErrorType vvTexRend::updateTextures2D(const int axes,
                                                 const int offsetX, const int offsetY, const int offsetZ,
                                                 const int sizeX, const int sizeY, const int sizeZ)
{
  int rawVal[4];
  int rawSliceSize;
  int rawLineSize;
  int texSize[3];
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

  assert(axes == 1 || axes == 3);

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

  for (int i = 3-axes; i < 3; i++)
  {
    texSize[i] = tw[i] * th[i] * texelsize;
  }

  // generate texture data arrays
  for (int i=3-axes; i<3; ++i)
  {
    rgbaSlice[i] = new uchar[texSize[i]];
  }

  // generate texture data
  for (int f = 0; f < vd->frames; f++)
  {
    raw = vd->getRaw(f);

    for (int i = 3-axes; i < 3; i++)
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
      for (int s = ss[i]; s < rs[i]; s++)
      {
        if (s < 0) continue;

        for (int h = sh[i]; h < rh[i]; h++)
        {
          if (h < 0) continue;

          // set voxel to starting position in raw data array
          rawVoxel = raw + rawStart[i] + s * rawStepS[i] + h * rawStepH[i] + sw[i];

          for (int w = sw[i]; w < rw[i]; w++)
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
                const float fval = *((float*)(rawVoxel));
                rawVal[0] = vd->mapFloat2Int(fval);
              }

              for (int c = 0; c < texelsize; c++)
                rgbaSlice[i][texSliceIndex + c] = rgbaLUT[rawVal[0] * 4 + c];
            }
            else if (vd->bpc == 1)
            {
              // fetch component values from memory
              for (int c = 0; c < ts_min(vd->chan,4); c++)
                rawVal[c] = *(rawVoxel + c);

              // copy color components
              for (int c = 0; c < ts_min(vd->chan,3); c++)
                rgbaSlice[i][texSliceIndex + c] = (uchar) rawVal[c];

              // alpha channel
              if (vd->chan >= 4)
                rgbaSlice[i][texSliceIndex + 3] = rgbaLUT[rawVal[3] * 4 + 3];
              else
              {
                int alpha = 0;
                for (int c = 0; c < vd->chan; c++)
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

  for (int i = 3-axes; i < 3; i++)
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
/// Do things that need to be done before virvo sets the rendering specific gl environment.
void vvTexRend::beforeSetGLenvironment() const
{
  const vvVector3 size(vd->getSize());            // volume size [world coordinates]

  // Draw boundary lines (must be done before setGLenvironment()):
  if (_renderState._boundaries)
  {
    drawBoundingBox(&size, &vd->pos, _renderState._boundColor);
  }
  if (_renderState._isROIUsed)
  {
    const vvVector3 probeSizeObj(size[0] * _renderState._roiSize[0],
                                 size[1] * _renderState._roiSize[1],
                                 size[2] * _renderState._roiSize[2]);
    drawBoundingBox(&probeSizeObj, &_renderState._roiPos, _renderState._probeColor);
  }
  if (_renderState._clipMode && _renderState._clipPerimeter)
  {
    drawPlanePerimeter(&size, &vd->pos, &_renderState._clipPoint, &_renderState._clipNormal, _renderState._clipColor);
  }
}

//----------------------------------------------------------------------------
/// Set GL environment for texture rendering.
void vvTexRend::setGLenvironment() const
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

  if ((_usedThreads > 0) || (_isSlave) || (_renderState._opaqueGeometryPresent))
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
void vvTexRend::unsetGLenvironment() const
{
  vvDebugMsg::msg(3, "vvTexRend::unsetGLenvironment()");

  glPopAttrib();

  vvDebugMsg::msg(3, "vvTexRend::unsetGLenvironment() done");
}

//----------------------------------------------------------------------------
void vvTexRend::enableLUTMode(vvShaderManager* pixelShader, GLuint& lutName,
                              GLuint progName[VV_FRAG_PROG_MAX])
{
  switch(voxelType)
  {
    case VV_FRG_PRG:
      enableFragProg(lutName, progName);
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
void vvTexRend::disableLUTMode(vvShaderManager* pixelShader) const
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
      probePosObj[i] = (probeMax[i] + probeMin[i]) *0.5f;
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

  getObjNormal(normal, origin, eye, invMV, isOrtho);
  evaluateLocalIllumination(_pixelShader, normal);

  // compute number of slices to draw
  float depth = fabs(normal[0]*probeSizeObj[0]) + fabs(normal[1]*probeSizeObj[1]) + fabs(normal[2]*probeSizeObj[2]);
  int minDistanceInd = 0;
  if(probeSizeObj[1]/probeTexels[1] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
    minDistanceInd=1;
  if(probeSizeObj[2]/probeTexels[2] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
    minDistanceInd=2;
  float voxelDistance = probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd];

  const float quality = calcQualityAndScaleImage();

  float sliceDistance = voxelDistance / quality;
  if(_renderState._isROIUsed && _renderState._quality < 2.0)
  {
    // draw at least twice as many slices as there are samples in the probe depth.
    sliceDistance = voxelDistance * 0.5f;
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
  glTranslatef(pos[0], pos[1], pos[2]);

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

void vvTexRend::renderTexBricks(const vvMatrix* mv)
{
  vvMatrix pm;                                    // OpenGL projection matrix
  vvVector3 farthest;                             // volume vertex farthest from the viewer
  vvVector3 delta;                                // distance vector between textures [object space]
  vvVector3 normal;                               // normal vector of textures
  vvVector3 origin;                               // origin (0|0|0) transformed to object space
  vvVector3 eye;                                  // user's eye position [object space]
  vvVector3 probePosObj;                          // probe midpoint [object space]
  vvVector3 probeSizeObj;                         // probe size [object space]
  vvVector3 probeMin, probeMax;                   // probe min and max coordinates [object space]

  vvDebugMsg::msg(3, "vvTexRend::renderTexBricks()");

  // needs 3D texturing extension
  if (!extTex3d) return;

  if (_brickList.empty()) return;

  // Calculate inverted modelview matrix:
  vvMatrix invMV(mv);
  invMV.invert();

  // Find eye position:
  getEyePosition(&eye);
  eye.multiply(&invMV);

  calcProbeDims(probePosObj, probeSizeObj, probeMin, probeMax);

  vvVector3 clippedProbeSizeObj;
  clippedProbeSizeObj.copy(&probeSizeObj);
  for (int i=0; i<3; ++i)
  {
    if (clippedProbeSizeObj[i] < vd->getSize()[i])
    {
      clippedProbeSizeObj[i] = vd->getSize()[i];
    }
  }

  // Compute length of probe diagonal [object space]:
  const float diagonal = sqrtf(clippedProbeSizeObj[0] * clippedProbeSizeObj[0] +
                               clippedProbeSizeObj[1] * clippedProbeSizeObj[1] +
                               clippedProbeSizeObj[2] * clippedProbeSizeObj[2]);

  const float diagonalVoxels = sqrtf(float(vd->vox[0] * vd->vox[0] +
                                           vd->vox[1] * vd->vox[1] +
                                           vd->vox[2] * vd->vox[2]));

  const float quality = calcQualityAndScaleImage();

  // make sure that at least one slice is drawn.
  // <> deceives msvc so that it won't use the windows.h max macro.
  int numSlices = ::max<>(1, static_cast<int>(quality * diagonalVoxels));

  vvDebugMsg::msg(3, "Number of texture slices rendered: ", numSlices);

  // Get projection matrix:
  getProjectionMatrix(&pm);
  const bool isOrtho = pm.isProjOrtho();

  getObjNormal(normal, origin, eye, invMV, isOrtho);
  if (_usedThreads == 0)
  {
    evaluateLocalIllumination(_pixelShader, normal);

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

  delta.copy(&normal);
  delta.scale(diagonal / ((float)numSlices));

  // Compute farthest point to draw texture at:
  farthest.copy(&delta);
  farthest.scale((float)(numSlices - 1) * -0.5f);

  if (_renderState._clipMode)                     // clipping plane present?
  {
    // Adjust numSlices and set farthest point so that textures are only
    // drawn up to the clipPoint. (Delta may not be changed
    // due to the automatic opacity correction.)
    // First find point on clipping plane which is on a line perpendicular
    // to clipping plane and which traverses the origin:
    vvVector3 temp(delta);
    temp.scale(-0.5f);
    farthest.add(&temp);                          // add a half delta to farthest
    vvVector3 clipPosObj(_renderState._clipPoint);
    clipPosObj.sub(&vd->pos);
    temp.copy(&probePosObj);
    temp.add(&normal);
    vvVector3 normClipPoint;
    normClipPoint.isectPlaneLine(&normal, &clipPosObj, &probePosObj, &temp);
    const float maxDist = farthest.distance(&normClipPoint);
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
      if (instantClassification() && (_usedThreads == 0))
      {
        updateLUT(1.0f, pixLUTName, rgbaLUT);
      }
    }
  }

#ifndef ISECT_GLSL_INST
  initVertArray(numSlices);
#endif

  if (_usedThreads == 0)
  {
    getBricksInProbe(_nonemptyList, _insideList, _sortedList, probePosObj, probeSizeObj, _renderState._isROIChanged);
  }

  markBricksInFrustum(probeMin, probeMax);

  // Translate object by its position:
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glTranslatef(vd->pos[0], vd->pos[1], vd->pos[2]);

  if (_usedThreads > 0)
  {
    if (_somethingChanged)
    {
      distributeBricks();
    }

    GLfloat modelview[16];
    glGetFloatv(GL_MODELVIEW_MATRIX , modelview);

    GLfloat projection[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projection);

    const vvGLTools::Viewport viewport = vvGLTools::getViewport();

    pthread_barrier_init(&_compositingBarrier, NULL, _usedThreads + 1);

    int leaf = 0;
    for (unsigned int i = 0; i < _numThreads; ++i)
    {
      if (_threadData[i].active)
      {
        // Clip the probe using the plane of the half space.
        vvVector3 probeMinI(&probeMin);
        vvVector3 probeMaxI(&probeMax);
        vvVector3 probePosObjI(&probePosObj);
        vvVector3 probeSizeObjI(&probeSizeObj);
        _bspTree->getLeafs()->at(leaf)->clipProbe(probeMinI, probeMaxI, probePosObjI, probeSizeObjI);
        _threadData[i].probeMin = vvVector3(&probeMinI);
        _threadData[i].probeMax = vvVector3(&probeMaxI);
        _threadData[i].probePosObj = vvVector3(&probePosObjI);
        _threadData[i].probeSizeObj = vvVector3(&probeSizeObjI);
        _threadData[i].delta = vvVector3(&delta);
        _threadData[i].farthest = vvVector3(&farthest);
        _threadData[i].normal = vvVector3(&normal);
        _threadData[i].eye = vvVector3(&eye);
        _threadData[i].isOrtho = isOrtho;
        _threadData[i].numSlices = numSlices;

        _threadData[i].modelview = modelview;
        _threadData[i].projection = projection;
        _threadData[i].width = viewport[2];
        _threadData[i].height = viewport[3];

        ++leaf;
      }
    }

    // The sorted brick lists are distributed among the threads. All other data specific
    // to the workers is assigned either. Now broadcast a signal telling the worker threads'
    // rendering loops to resume.
    pthread_barrier_wait(&_renderStartBarrier);

    // Do compositing.
    pthread_barrier_wait(&_compositingBarrier);

    // Blend the images from the worker threads onto a 2D-texture.

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
    sortBrickList(_sortedList, eye, normal, isOrtho);

    if(_proxyGeometryOnGpu)
    {
      // Per frame parameters.
      _isectShader->setParameter1f(0, ISECT_SHADER_DELTA, delta.length());
      _isectShader->setParameter3f(0, ISECT_SHADER_PLANENORMAL, normal[0], normal[1], normal[2]);
    }

    if (_renderState._showBricks)
    {
      // Debugging mode: render the brick outlines and deactivate shaders,
      // lighting and texturing.
      glDisable(GL_TEXTURE_3D_EXT);
      glDisable(GL_LIGHTING);
      disablePixelShaders(_pixelShader);
      disableFragProg();
      for(BrickList::iterator it = _sortedList.begin(); it != _sortedList.end(); ++it)
      {
        (*it)->renderOutlines(probeMin, probeMax);
      }
    }
    else
    {
      if (_proxyGeometryOnGpu)
      {
        glEnableClientState(GL_VERTEX_ARRAY);
#ifdef ISECT_GLSL_INST
#ifdef ISECT_GLSL_GEO
        const GLint vertArray[6] = { 0, 0,
                                     3, 0,
                                     6, 0 };

#else
        const GLint vertArray[12] = { 0, 0,
                                      4, 0,
                                      8, 0,
                                      12, 0,
                                      16, 0,
                                      20, 0 };
#endif
        glVertexPointer(2, GL_INT, 0, vertArray);
#endif
      }
      for(BrickList::iterator it = _sortedList.begin(); it != _sortedList.end(); ++it)
      {
        (*it)->render(this, normal, farthest, delta, probeMin, probeMax,
                     texNames,
                     _isectShader);
      }
      if (_proxyGeometryOnGpu)
      {
        glDisableClientState(GL_VERTEX_ARRAY);
      }
    }

    vvDebugMsg::msg(3, "Bricks discarded: ",
                    static_cast<int>(_brickList[vd->getCurrentFrame()].size() - _sortedList.size()));
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
  ThreadArgs* data = reinterpret_cast<ThreadArgs*>(threadargs);

#ifdef HAVE_X11
  if (!glXMakeCurrent(data->display, data->drawable, data->glxContext))
#elif defined _WIN32
  if (!wglMakeCurrent(data->deviceContext, data->wglContext))
#else
  if (0)
#endif
  {
    vvDebugMsg::msg(3, "Couldn't make OpenGL context current");
  }
  else
  {
    vvShaderManager* pixelShader = vvShaderFactory::provideShaderManager(VV_CG_MANAGER);
    GLuint fragProgName[VV_FRAG_PROG_MAX];

    vvStopwatch* stopwatch;

    // Init glew.
    glewInit();

    glGenTextures(1, &data->pixLUTName);
    data->renderer->updateTransferFunction(data->pixLUTName, data->rgbaLUT);
    data->renderer->initPostClassificationStage(pixelShader, fragProgName);

    data->renderer->makeTextures(data->pixLUTName, data->rgbaLUT);

    // Now that the textures are built, the bricks may be distributed
    // by the main thread.
    pthread_barrier_wait(&data->renderer->_distributeBricksBarrier);
    // For the main thread, this will take some time... .
    pthread_barrier_wait(&data->renderer->_distributedBricksBarrier);
    bool areBricksCreated;
    data->renderer->makeTextureBricks(data->privateTexNames, &data->numTextures,
                                      data->rgbaLUT, data->brickList, areBricksCreated);

    /////////////////////////////////////////////////////////
    // Setup an appropriate GL state.
    /////////////////////////////////////////////////////////
    data->renderer->setGLenvironment();

    vvShaderManager* isectShader = vvShaderFactory::provideShaderManager(VV_GLSL_MANAGER);

    if(data->renderer->_proxyGeometryOnGpu)
    {
      if (data->renderer->voxelType != VV_RGBA)
      {
        data->renderer->initIntersectionShader(isectShader);
      }
      else
      {
        data->renderer->initIntersectionShader(isectShader, pixelShader);
      }
      data->renderer->setupIntersectionParameters(isectShader);
      data->renderer->enableIntersectionShader(isectShader);
    }

    data->renderer->_offscreenBuffers[data->threadId] = new vvOffscreenBuffer(1.0f, data->renderer->_multiGpuBufferPrecision);

    // Init framebuffer objects.
    data->renderer->_offscreenBuffers[data->threadId]->initForRender();

    // Init stop watch.
    stopwatch = new vvStopwatch();

    bool roiChanged = true;

    // TODO: identify if method is called from an ordinary worker thread or
    // the main thread. If the latter is the case, ensure that the environment
    // is reset properly after rendering.

    // Main render loop that is suspended while the sequential program flow is executing
    // and resumes when the texture bricks are sorted in back to front order and the
    // thread specific data is supplied.
    while (1)
    {
      data->renderer->fillNonemptyList(data->nonemptyList, data->brickList);

      // Don't start rendering until the bricks are sorted in back to front order
      // and appropriatly distributed among the respective worker threads. The main
      // thread will issue an alert if this is the case.
      pthread_barrier_wait(&data->renderer->_renderStartBarrier);

      // Break out of loop if dtor was called.
      if (data->renderer->_terminateThreads)
      {
        vvDebugMsg::msg(3, "Thread exiting rendering loop: ", data->threadId);
        break;
      }

      data->renderer->_offscreenBuffers[data->threadId]->resize(data->width, data->height);
      data->renderer->enableLUTMode(pixelShader, data->pixLUTName, fragProgName);
      data->renderer->evaluateLocalIllumination(pixelShader, data->normal);

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

      data->renderer->_offscreenBuffers[data->threadId]->bindFramebuffer();

      // Individual per thread color to visualize the screen rect
      // occupied by the the content rendered by this thread.
      if (vvDebugMsg::getDebugLevel() > 1)
      {
        glClearColor(data->renderer->_debugColors[data->threadId][0],
                     data->renderer->_debugColors[data->threadId][1],
                     data->renderer->_debugColors[data->threadId][2], 0.0);
      }
      else
      {
        glClearColor(0.0, 0.0, 0.0, 0.0);
      }
      glClear(GL_COLOR_BUFFER_BIT);
      data->renderer->setGLenvironment();

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
        data->renderer->enableIntersectionShader(isectShader);

        // Per frame parameters.
        isectShader->setParameter1f(0, "delta", data->delta.length());
        isectShader->setParameter3f(0, "planeNormal", data->normal[0], data->normal[1], data->normal[2]);
      }

      data->renderer->getBricksInProbe(data->nonemptyList, data->insideList, data->sortedList,
                                       data->probePosObj, data->probeSizeObj, roiChanged, data->threadId);
      data->renderer->sortBrickList(data->sortedList, data->eye, data->normal, data->isOrtho);

      if (data->renderer->_renderState._showBricks)
      {
        // Debugging mode: render the brick outlines and deactivate shaders,
        // lighting and texturing.
        glDisable(GL_TEXTURE_3D_EXT);
        glDisable(GL_LIGHTING);
        data->renderer->disableIntersectionShader(isectShader);
        data->renderer->disablePixelShaders(pixelShader);
        data->renderer->disableFragProg();
        for(BrickList::iterator it = data->sortedList.begin(); it != data->sortedList.end(); ++it)
        {
          (*it)->renderOutlines(data->probeMin, data->probeMax);
        }
      }
      else
      {
        if (data->renderer->_proxyGeometryOnGpu)
        {
          glEnableClientState(GL_VERTEX_ARRAY);
        }
        for(BrickList::iterator it = data->sortedList.begin(); it != data->sortedList.end(); ++it)
        {
          vvBrick *tmp = *it;
          tmp->render(data->renderer, data->normal,
                      data->farthest, data->delta,
                      data->probeMin, data->probeMax,
                      data->privateTexNames, isectShader);
        }
        if (data->renderer->_proxyGeometryOnGpu)
        {
          glDisableClientState(GL_VERTEX_ARRAY);
        }
      }

      vvDebugMsg::msg(3, "Bricks discarded: ",
                      static_cast<int>(data->brickList[data->renderer->vd->getCurrentFrame()].size()
                                       - data->sortedList.size()));

      glDisable(GL_TEXTURE_3D_EXT);
      data->renderer->disableLUTMode(pixelShader);

      // Blend the images to one single image.

      // Get the screen rect. Be sure to do this here, while the fbo's transformation
      // matrices are still applied. The boolean parameter tells the half space to
      // recalculate the screen rect as well as storing it for later use.
      vvRect* screenRect = data->halfSpace->getProjectedScreenRect(&data->probeMin, &data->probeMax, true);

      // This call switches the currently readable buffer to the fbo offscreen buffer.
      glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
      glReadPixels(screenRect->x, screenRect->y, screenRect->width, screenRect->height, GL_RGBA, GL_FLOAT, data->pixels);
      glPopAttrib();
      data->renderer->unsetGLenvironment();
      data->renderer->_offscreenBuffers[data->threadId]->unbindFramebuffer();

      // Output to screen in debug mode.
      if (vvDebugMsg::getDebugLevel() > 0)
      {
        data->renderer->_offscreenBuffers[data->threadId]->writeBack(data->width, data->height);
        glFlush();
      }

      // Store the time to render. Based upon this, dynamic load balancing will be performed.
      data->lastRenderTime = stopwatch->getTime();

      pthread_barrier_wait(&data->renderer->_compositingBarrier);

      if (data->renderer->_proxyGeometryOnGpu)
      {
        if (data->renderer->voxelType != VV_RGBA)
        {
          data->renderer->disableIntersectionShader(isectShader);
        }
        else
        {
          data->renderer->disableIntersectionShader(isectShader, pixelShader);
        }
      }

      // Rebuild textures synchronized.
      if (data->brickDataChanged)
      {
        data->renderer->makeTextureBricks(data->privateTexNames, &data->numTextures,
                                          data->rgbaLUT, data->brickList, areBricksCreated);
        data->renderer->fillNonemptyList(data->nonemptyList, data->brickList);
        data->brickDataChanged = false;
      }

      // Finally pdate the transfer function in a synchronous fashion.
      if (data->transferFunctionChanged)
      {
        data->renderer->updateTransferFunction(data->pixLUTName, data->rgbaLUT);
        data->renderer->fillNonemptyList(data->nonemptyList, data->brickList);
        data->transferFunctionChanged = false;
        // TODO: reorganize getBricksInProbe so that this hack is no longer necessary.
        roiChanged = true;
      }
    }

    // Exited render loop - perform cleanup.
    if (data->renderer->voxelType==VV_FRG_PRG)
    {
      glDeleteProgramsARB(3, fragProgName);
    }

    data->renderer->removeTextures(data->privateTexNames, &data->numTextures);
    delete isectShader;
    delete pixelShader;
    delete stopwatch;
#ifdef HAVE_X11
    XCloseDisplay(data->display);
#endif
  }
  pthread_exit(NULL);
#ifndef HAVE_X11
  return NULL;
#endif
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

bool vvTexRend::insideFrustum(const vvVector3 &min, const vvVector3 &max) const
{
  vvVector3 pv;

  // get p-vertex (that's the farthest vertex in the direction of the normal plane
  for (int i = 0; i < 6; i++)
  {
    const vvVector3 normal(_frustum[i][0], _frustum[i][1], _frustum[i][2]);

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

bool vvTexRend::intersectsFrustum(const vvVector3 &min, const vvVector3 &max) const
{
  vvVector3 pv;

  // get p-vertex (that's the farthest vertex in the direction of the normal plane
  for (int i = 0; i < 6; ++i)
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

    const vvVector3 normal(_frustum[i][0], _frustum[i][1], _frustum[i][2]);

    if ((pv.dot(&normal) + _frustum[i][3]) < 0)
    {
      return false;
    }
  }

  return true;
}

bool vvTexRend::testBrickVisibility(const vvBrick* brick) const
{
  return intersectsFrustum(brick->min, brick->max);
}

bool vvTexRend::testBrickVisibility(const vvBrick* brick, const vvMatrix& mvpMat) const
{
  //sample the brick at many point and test them all for visibility
  const float numSteps = 3;
  const float divisorInv = 1.0f / (numSteps - 1.0f);
  const float xStep = (brick->max[0] - brick->min[0]) * divisorInv;
  const float yStep = (brick->max[1] - brick->min[1]) * divisorInv;
  const float zStep = (brick->max[2] - brick->min[2]) * divisorInv;
  for(int i = 0; i < numSteps; i++)
  {
    const float x = brick->min.e[0] + xStep * i;
    for(int j = 0; j < numSteps; j++)
    {
      const float y = brick->min.e[1] + yStep * j;
      for(int k = 0; k < numSteps; k++)
      {
        const float z = brick->min.e[2] + zStep * k;
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

void vvTexRend::calcProbeDims(vvVector3& probePosObj, vvVector3& probeSizeObj, vvVector3& probeMin, vvVector3& probeMax) const
{
  // Determine texture object dimensions and half object size as a shortcut:
  const vvVector3 size(vd->getSize());
  const vvVector3 size2 = size * 0.5f;

  if (_renderState._isROIUsed)
  {
    // Convert probe midpoint coordinates to object space w/o position:
    probePosObj = _renderState._roiPos;

    // Compute probe min/max coordinates in object space:
    const vvVector3 maxSize = _renderState._roiSize * size2;

    probeMin = probePosObj - maxSize;
    probeMax = probePosObj + maxSize;

    // Constrain probe boundaries to volume data area:
    for (int i = 0; i < 3; ++i)
    {
      if (probeMin[i] > size2[i] || probeMax[i] < -size2[i])
      {
        vvDebugMsg::msg(3, "probe outside of volume");
        return;
      }
      if (probeMin[i] < -size2[i]) probeMin[i] = -size2[i];
      if (probeMax[i] >  size2[i]) probeMax[i] =  size2[i];
      probePosObj[i] = (probeMax[i] + probeMin[i]) * 0.5f;
    }

    // Compute probe edge lengths:
    probeSizeObj = probeMax - probeMin;
  }
  else                                            // probe mode off
  {
    probeSizeObj.copy(&size);
    probeMin = -size2;
    probeMax = size2;
  }
}

void vvTexRend::calcAABBMask()
{
  vvVector3 min = vvVector3( FLT_MAX,  FLT_MAX,  FLT_MAX);
  vvVector3 max = vvVector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

  for (std::vector<BrickList>::const_iterator it1 = _brickList.begin();
       it1 != _brickList.end(); ++it1)
  {
    for (BrickList::const_iterator it2 = (*it1).begin();
         it2 != (*it1).end(); ++it2)
    {
      vvBrick* brick = (*it2);

      for (int i = 0; i < 3; ++i)
      {
        if (brick->min[i] < min[i])
        {
          min[i] = brick->min[i];
        }

        if (brick->max[i] > max[i])
        {
          max[i] = brick->max[i];
        }
      }
    }
  }
  delete _aabbMask;
  _aabbMask = new vvAABB(min, max);
}

void vvTexRend::markBricksInFrustum(const vvVector3& probeMin, const vvVector3& probeMax)
{
  updateFrustum();

  const bool inside = insideFrustum(probeMin, probeMax);
  const bool outside = !intersectsFrustum(probeMin, probeMax);

  if (inside || outside)
  {
    for (BrickList::iterator it = _sortedList.begin(); it != _sortedList.end(); ++it)
    {
      (*it)->visible = inside;
    }
  }
  else
  {
    for (BrickList::iterator it = _sortedList.begin(); it != _sortedList.end(); ++it)
    {
      (*it)->visible = testBrickVisibility( *it );
    }
  }
}

void vvTexRend::getBricksInProbe(std::vector<BrickList>& nonemptyList, BrickList& insideList, BrickList& sortedList,
                                 const vvVector3 pos, const vvVector3 size, bool& roiChanged, const int threadId)
{
  if (threadId == -1)
  {
    // Single gpu mode.
    if(!roiChanged && vd->getCurrentFrame() == _lastFrame)
      return;
    _lastFrame = vd->getCurrentFrame();
  }
  else
  {
    // Multi gpu mode.
    if(!roiChanged && vd->getCurrentFrame() == _threadData[threadId].lastFrame)
      return;
    _threadData[threadId].lastFrame = vd->getCurrentFrame();
  }
  roiChanged = false;

  insideList.clear();

  const vvVector3 tmpVec = size * 0.5f;
  const vvVector3 min = pos - tmpVec;
  const vvVector3 max = pos + tmpVec;

  int countVisible = 0, countInvisible = 0;

  const int frame = vd->getCurrentFrame();
  for(BrickList::iterator it = nonemptyList[frame].begin(); it != nonemptyList[frame].end(); ++it)
  {
    vvBrick *tmp = *it;
    if ((tmp->min.e[0] <= max.e[0]) && (tmp->max.e[0] >= min.e[0]) &&
      (tmp->min.e[1] <= max.e[1]) && (tmp->max.e[1] >= min.e[1]) &&
      (tmp->min.e[2] <= max.e[2]) && (tmp->max.e[2] >= min.e[2]))
    {
      insideList.push_back(tmp);
      if ((tmp->min.e[0] >= min.e[0]) && (tmp->max.e[0] <= max.e[0]) &&
        (tmp->min.e[1] >= min.e[1]) && (tmp->max.e[1] <= max.e[1]) &&
        (tmp->min.e[2] >= min.e[2]) && (tmp->max.e[2] <= max.e[2]))
        tmp->insideProbe = true;
      else
        tmp->insideProbe = false;
      ++countVisible;
    }
    else
    {
      ++countInvisible;
    }
  }

  sortedList.clear();
  for(std::vector<vvBrick*>::iterator it = insideList.begin(); it != insideList.end(); ++it)
     sortedList.push_back(static_cast<vvBrick *>(*it));
}

void vvTexRend::sortBrickList(std::vector<vvBrick*>& list, const vvVector3& eye, const vvVector3& normal, const bool isOrtho)
{
  if (isOrtho)
  {
    for(std::vector<vvBrick*>::iterator it = list.begin(); it != list.end(); ++it)
    {
      (*it)->dist = -(*it)->pos.dot(&normal);
    }
  }
  else
  {
    for(std::vector<vvBrick*>::iterator it = list.begin(); it != list.end(); ++it)
    {
      (*it)->dist = ((*it)->pos + vd->pos - eye).length();
    }
  }
  std::sort(list.begin(), list.end(), vvBrick::Compare());
}

void vvTexRend::performLoadBalancing()
{
  // Only redistribute if the deviation of the rendering times
  // exceeds a certain limit.
  float expectedValue = 0;
  for (unsigned int i = 0; i < _numThreads; ++i)
  {
    if (_threadData[i].active)
    {
      expectedValue += _threadData[i].lastRenderTime;
    }
  }
  expectedValue /= static_cast<float>(_numThreads);

  // An ideal distribution is one where each rendering time for each thread
  // equals the expected value exactly.
  float* idealDistribution = new float[_numThreads];

  // This is (not necessarily) true in reality... .
  float* actualDistribution = new float[_numThreads];

  // Build both arrays.
  for (unsigned int i = 0; i < _numThreads; ++i)
  {
    if (_threadData[i].active)
    {
      idealDistribution[i] = expectedValue;
      actualDistribution[i] = _threadData[i].lastRenderTime;
    }
  }

  // Normalized deviation (percent).
  const float deviation = (vvToolshed::meanAbsError(idealDistribution, actualDistribution, _numThreads) * 100.0f)
                          / expectedValue;

  // Only reorder if
  // a) a given deviation was exceeded
  // b) this happened at least x times
  float tmp;
  if (deviation > MAX_DEVIATION)
  {
    if (_deviationExceedCnt > LIMIT)
    {
      tmp = 0.0f;

      // Rearrange the share for each thread.
      for (unsigned int i = 0; i < _numThreads; ++i)
      {
        if (_threadData[i].active)
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
      }

      // Normalize partitioning.
      for (unsigned int i = 0; i < _numThreads; ++i)
      {
        if (_threadData[i].active)
        {
          _threadData[i].share /= tmp;
        }
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
  float  maxDist = 0.0;
  float  minDist = 0.0;
  vvVector3 texSize;                              // size of 3D texture [object space]
  vvVector3 texSize2;                             // half size of 3D texture [object space]
  vvVector3 volumeVertices[8];
  vvMatrix invView;

  vvDebugMsg::msg(3, "vvTexRend::renderTex3DSpherical()");

  if (!extTex3d) return;

  // make sure that at least one shell is drawn
  const int numShells = max(1, static_cast<int>(_renderState._quality * 100.0f));

  // Determine texture object dimensions:
  const vvVector3 size(vd->getSize());
  for (int i=0; i<3; ++i)
  {
    texSize.e[i]  = size.e[i] * (float)texels[i] / (float)vd->vox[i];
    texSize2.e[i] = 0.5f * texSize.e[i];
  }

  invView.copy(view);
  invView.invert();

  // generates the vertices of the cube (volume) in world coordinates
  int vertexIdx = 0;
  for (int ix=0; ix<2; ++ix)
    for (int iy=0; iy<2; ++iy)
      for (int iz=0; iz<2; ++iz)
      {
        volumeVertices[vertexIdx].e[0] = (float)ix;
        volumeVertices[vertexIdx].e[1] = (float)iy;
        volumeVertices[vertexIdx].e[2] = (float)iz;
        // transfers vertices to world coordinates:
        for (int k=0; k<3; ++k)
          volumeVertices[vertexIdx].e[k] =
            (volumeVertices[vertexIdx].e[k] * 2.0f - 1.0f) * texSize2.e[k];
        volumeVertices[vertexIdx].multiply(view);
        vertexIdx++;
  }

  // Determine maximal and minimal distance of the volume from the eyepoint:
  maxDist = minDist = volumeVertices[0].length();
  for (int i = 1; i<7; i++)
  {
    const float dist = volumeVertices[i].length();
    if (dist > maxDist)  maxDist = dist;
    if (dist < minDist)  minDist = dist;
  }

  maxDist *= 1.4f;
  minDist *= 0.5f;

  // transfer the eyepoint to the object coordinates of the volume
  // to check whether the camera is inside the volume:
  vvVector3 eye(0.0, 0.0, 0.0);
  eye.multiply(&invView);
  bool inside = true;
  for (int k=0; k<3; ++k)
  {
    if (eye.e[k] < -texSize2.e[k] || eye.e[k] > texSize2.e[k])
      inside = false;
  }
  if (inside)
    minDist = 0.0f;

  // Determine texture spacing:
  const float spacing = (maxDist-minDist) / (float)(numShells-1);

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
  for (int k=0; k<3; ++k)
  {
    offset[k] = -(0.5f - (texMin[k] + texMax[k]) * 0.5f);
  }
  shell.setTextureOffset(offset);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glTranslatef(vd->pos[0], vd->pos[1], vd->pos[2]);

  // Volume render a 3D texture:
  enableTexture(GL_TEXTURE_3D_EXT);
  glBindTexture(GL_TEXTURE_3D_EXT, texNames[0]);

  // Enable clipping plane if appropriate:
  if (_renderState._clipMode) activateClippingPlane();

  float radius = maxDist;
  for (int i=0; i<numShells; ++i)                     // loop thru all drawn textures
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

  vvDebugMsg::msg(3, "vvTexRend::renderTex2DSlices()");

  // Translate object by its position:
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  pos.copy(&vd->pos);
  glTranslatef(pos.e[0], pos.e[1], pos.e[2]);

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

  for (int i=0; i<numTextures; ++i)
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

void vvTexRend::generateDebugColors()
{
  _debugColors[0] = vvColor(0.5f, 0.7f, 0.4f);
  _debugColors[1] = vvColor(0.7f, 0.5f, 0.0f);
  _debugColors[2] = vvColor(0.0f, 0.5f, 0.7f);
  _debugColors[3] = vvColor(0.7f, 0.0f, 0.5f);
  _debugColors[4] = vvColor(1.0f, 1.0f, 0.0f);
  _debugColors[5] = vvColor(0.0f, 1.0f, 0.0f);
  _debugColors[6] = vvColor(0.0f, 0.0f, 1.0f);
  _debugColors[7] = vvColor(1.0f, 0.0f, 0.0f);
}

//----------------------------------------------------------------------------
/** Render the volume onto currently selected drawBuffer.
 Viewport size in world coordinates is -1.0 .. +1.0 in both x and y direction
*/
void vvTexRend::renderVolumeGL()
{
  static vvStopwatch sw;                          // stop watch for performance measurements
  vvMatrix mv;                                    // current modelview matrix
  float zx, zy, zz;                               // base vector z coordinates

  vvDebugMsg::msg(3, "vvTexRend::renderVolumeGL()");

  if (vd->vox[0] * vd->vox[1] * vd->vox[2] == 0)
    return;

  if (_measureRenderTime)
  {
    sw.start();
  }

  beforeSetGLenvironment();

  // Reroute output to alternative render target.
  _renderTarget->initForRender();

  setGLenvironment();

  // If the render target is of base class type, nothing
  // will happen here. Offscreen buffers e.g. need to
  // cleanup the content from the last rendering step.
  _renderTarget->clearBuffer();

  if (_usedThreads == 0)
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
  for (int i = 0; i < 3; ++i)
  {
    texMin[i] = 0.5f / (float)texels[i];
    texMax[i] = (float)vd->vox[i] / (float)texels[i] - texMin[i];
  }

  // Get OpenGL modelview matrix:
  getModelviewMatrix(&mv);

  if (_usedThreads == 0)
  {
    if (geomType != VV_BRICKS || !_renderState._showBricks)
    {
      enableLUTMode(_pixelShader, pixLUTName, fragProgName);
    }
  }

  switch (geomType)
  {
    default:
    case VV_SLICES:
      getPrincipalViewingAxis(mv, zx, zy, zz);renderTex2DSlices(zz);
      break;
    case VV_CUBIC2D:
      {
        const AxisType at = getPrincipalViewingAxis(mv, zx, zy, zz);
        renderTex2DCubic(at, zx, zy, zz);
      }
      break;
    case VV_SPHERICAL: renderTex3DSpherical(&mv); break;
    case VV_VIEWPORT:  renderTex3DPlanar(&mv); break;
    case VV_BRICKS:
        renderTexBricks(&mv);
      break;
  }

  if (_usedThreads == 0)
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

  // Write output of alternative render target. Depending on the type of render target,
  // output can be redirected to the screen or for example an image file.
  _renderTarget->writeBack();

  if (_measureRenderTime)
  {
    // Make sure rendering is done to measure correct time.
    // Since this operation is costly, only do it if necessary.
    glFinish();
    _lastRenderTime = sw.getTime();
  }

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
    thickness = vd->_scale * vd->dist[0] * (vd->vox[0] * 0.01f);
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
void vvTexRend::setNumLights(const int numLights)
{
  const float ambient[]  = {0.5f, 0.5f, 0.5f, 1.0f};
  const float pos0[] = {0.0f, 10.0f, 10.0f, 0.0f};
  const float pos1[] = {0.0f, -10.0f, -10.0f, 0.0f};

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
bool vvTexRend::instantClassification() const
{
  vvDebugMsg::msg(3, "vvTexRend::instantClassification()");
  return (voxelType != VV_RGBA);
}

//----------------------------------------------------------------------------
/// Returns the number of entries in the RGBA lookup table.
int vvTexRend::getLUTSize(int* size) const
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
int vvTexRend::getPreintTableSize() const
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
  int total=0;
  lutDistance = dist;

  if(usePreIntegration)
  {
    vd->tf.makePreintLUTCorrect(getPreintTableSize(), preintTable, dist);
  }
  else
  {
    total = getLUTSize(lutSize);
    for (int i=0; i<total; ++i)
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
      for (int c=0; c<4; ++c)
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
      if (_usedThreads == 0)
      {
        makeTextures(lutName, lutData);// this mode doesn't use a hardware LUT, so every voxel has to be updated
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
void vvTexRend::setObjectDirection(const vvVector3* od)
{
  vvDebugMsg::msg(3, "vvTexRend::setObjectDirection()");
  objDir.copy(od);
}

//----------------------------------------------------------------------------
// see parent
void vvTexRend::setParameter(const ParameterType param, const float newValue, char*)
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
          for (unsigned int i = 0; i < _numThreads; ++i)
          {
            if (_threadData[i].active)
            {
              _threadData[i].brickDataChanged = true;
              _threadData[i].transferFunctionChanged = true;
            }
          }
        }
        else
        {
          makeTextures(pixLUTName, rgbaLUT);
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
        _proxyGeometryOnGpu = _isectShader && _proxyGeometryOnGpuSupported;
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
          if (_renderState._opaqueGeometryPresent)
          {
            dynamic_cast<vvOffscreenBuffer*>(_renderTarget)->setPreserveDepthBuffer(true);
          }
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
        else if ((int(newValue) > 8) && (int(newValue) < 32))
        {
          if (dynamic_cast<vvOffscreenBuffer*>(_renderTarget) != NULL)
          {
            dynamic_cast<vvOffscreenBuffer*>(_renderTarget)->setPrecision(VV_SHORT);
          }
          else
          {
            delete _renderTarget;
            _renderTarget = new vvOffscreenBuffer(_renderState._imageScale, VV_SHORT);
          }
          break;
        }
        else if (int(newValue) >= 32)
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
      break;
    case vvRenderer::VV_LIGHTING:
      if (static_cast<bool>(newValue) == true)
      {
        _previousShader = _currentShader;
        _currentShader = getLocalIlluminationShader();
      }
      else
      {
        _currentShader = _previousShader;
      }
      break;
    case vvRenderer::VV_MEASURETIME:
      _measureRenderTime = static_cast<bool>(newValue);
      break;
    default:
      vvRenderer::setParameter(param, newValue);
      break;
  }
}

//----------------------------------------------------------------------------
// see parent for comments
float vvTexRend::getParameter(const ParameterType param, char*) const
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
bool vvTexRend::isSupported(const GeometryType geom)
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
      return vvGLTools::isGLextensionSupported("GL_EXT_texture3D") || vvGLTools::isGLVersionSupported(1,2,0);
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
bool vvTexRend::isSupported(const VoxelType voxel)
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
      return (vvGLTools::isGLextensionSupported("GL_ARB_multitexture") || vvGLTools::isGLVersionSupported(1,3,0)) &&
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
    case VV_GLSL_SHD:
      return vvShaderFactory::isSupported(VV_GLSL_MANAGER);
    default: return false;
  }
}

//----------------------------------------------------------------------------
/** Return true if a feature is supported.
 */
bool vvTexRend::isSupported(const FeatureType feature) const
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
vvTexRend::GeometryType vvTexRend::getGeomType() const
{
  vvDebugMsg::msg(3, "vvTexRend::getGeomType()");
  return geomType;
}

//----------------------------------------------------------------------------
/** Return the currently used voxel type.
  This is expecially useful if VV_AUTO was passed in the constructor.
*/
vvTexRend::VoxelType vvTexRend::getVoxelType() const
{
  vvDebugMsg::msg(3, "vvTexRend::getVoxelType()");
  return voxelType;
}

//----------------------------------------------------------------------------
/** Return the currently used pixel shader [0..numShaders-1].
 */
int vvTexRend::getCurrentShader() const
{
  vvDebugMsg::msg(3, "vvTexRend::getCurrentShader()");
  return _currentShader;
}

//----------------------------------------------------------------------------
/** Set the currently used pixel shader [0..numShaders-1].
 */
void vvTexRend::setCurrentShader(const int shader)
{
  vvDebugMsg::msg(3, "vvTexRend::setCurrentShader()");
  if(shader >= NUM_PIXEL_SHADERS || shader < 0)
  {
    _currentShader = 0;
  }
  else
  {
    _currentShader = shader;
  }
}

//----------------------------------------------------------------------------
/// inherited from vvRenderer, only valid for planar textures
void vvTexRend::renderQualityDisplay()
{
  const int numSlices = int(_renderState._quality * 100.0f);
  vvPrintGL* printGL = new vvPrintGL();
  printGL->print(-0.9f, 0.9f, "Textures: %d", numSlices);
  delete printGL;
}

//----------------------------------------------------------------------------
void vvTexRend::enableTexture(const GLenum target) const
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
void vvTexRend::disableTexture(const GLenum target) const
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
void vvTexRend::enableNVShaders() const
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
void vvTexRend::disableNVShaders() const
{
  glDisable(GL_TEXTURE_SHADER_NV);
  glActiveTextureARB(GL_TEXTURE1_ARB);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexEnvi(GL_TEXTURE_SHADER_NV, GL_SHADER_OPERATION_NV, GL_TEXTURE_2D);
  glActiveTextureARB(GL_TEXTURE0_ARB);
}

//----------------------------------------------------------------------------
void vvTexRend::enableFragProg(GLuint& lutName, GLuint progName[VV_FRAG_PROG_MAX]) const
{
  glActiveTextureARB(GL_TEXTURE1_ARB);
  glBindTexture(GL_TEXTURE_2D, lutName);
  glActiveTextureARB(GL_TEXTURE0_ARB);

  glEnable(GL_FRAGMENT_PROGRAM_ARB);
  switch(geomType)
  {
    case VV_CUBIC2D:
    case VV_SLICES:
      glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, progName[VV_FRAG_PROG_2D]);
      break;
    case VV_VIEWPORT:
    case VV_SPHERICAL:
    case VV_BRICKS:
      if(usePreIntegration)
      {
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, progName[VV_FRAG_PROG_PREINT]);
      }
      else
      {
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, progName[VV_FRAG_PROG_3D]);
      }
      break;
    default:
      vvDebugMsg::msg(1, "vvTexRend::enableFragProg(): unknown method used\n");
      break;
  }
}

//----------------------------------------------------------------------------
void vvTexRend::disableFragProg() const
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
      values[parameterCount] = reinterpret_cast<void*>(lutName);
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
void vvTexRend::disablePixelShaders(vvShaderManager* pixelShader) const
{
  if (voxelType == VV_PIX_SHD)
  {
    if ((_currentShader != 4) && (pixelShader->parametersInitialized(_currentShader)))
    {
      pixelShader->disableTexture(_currentShader, "pixLUT");
    }
    pixelShader->disableShader(_currentShader);
  }
}

//----------------------------------------------------------------------------
void vvTexRend::enableIntersectionShader(vvShaderManager* isectShader, vvShaderManager* pixelShader) const
{
  if (_proxyGeometryOnGpu)
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
void vvTexRend::disableIntersectionShader(vvShaderManager* isectShader, vvShaderManager* pixelShader) const
{
  if (_proxyGeometryOnGpu)
  {
    isectShader->disableShader(0);
    if (pixelShader != NULL)
    {
      // Deactivate the passthrough shader.
      pixelShader->disableShader(0);
    }
  }
}

void vvTexRend::initPostClassificationStage(vvShaderManager* pixelShader, GLuint progName[VV_FRAG_PROG_MAX])
{
  if (voxelType == VV_FRG_PRG)
  {
    initArbFragmentProgram(progName);
  }
  else if (voxelType==VV_PIX_SHD)
  {
    if (!initPixelShaders(pixelShader))
    {
      voxelType = VV_RGBA;
    }
  }
}

void vvTexRend::initArbFragmentProgram(GLuint progName[VV_FRAG_PROG_MAX]) const
{
  glGenProgramsARB(VV_FRAG_PROG_MAX, progName);

  const char fragProgString2D[] = "!!ARBfp1.0\n"
    "TEMP temp;\n"
    "TEX  temp, fragment.texcoord[0], texture[0], 2D;\n"
    "TEX  result.color, temp, texture[1], 2D;\n"
    "END\n";
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, progName[VV_FRAG_PROG_2D]);
  glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB,
    GL_PROGRAM_FORMAT_ASCII_ARB,
    strlen(fragProgString2D),
    fragProgString2D);

  const char fragProgString3D[] = "!!ARBfp1.0\n"
    "TEMP temp;\n"
    "TEX  temp, fragment.texcoord[0], texture[0], 3D;\n"
    "TEX  result.color, temp, texture[1], 2D;\n"
    "END\n";
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, progName[VV_FRAG_PROG_3D]);
  glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB,
    GL_PROGRAM_FORMAT_ASCII_ARB,
    strlen(fragProgString3D),
    fragProgString3D);

  const char fragProgStringPreint[] = "!!ARBfp1.0\n"
    "TEMP temp;\n"
    "TEX  temp.x, fragment.texcoord[0], texture[0], 3D;\n"
    "TEX  temp.y, fragment.texcoord[1], texture[0], 3D;\n"
    "TEX  result.color, temp, texture[1], 2D;\n"
    "END\n";
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, progName[VV_FRAG_PROG_PREINT]);
  glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB,
    GL_PROGRAM_FORMAT_ASCII_ARB,
    strlen(fragProgStringPreint),
    fragProgStringPreint);
}

//----------------------------------------------------------------------------
/** @return true if initialization successful
 */
bool vvTexRend::initPixelShaders(vvShaderManager* pixelShader) const
{
  const char* shaderFileName = "vv_shader";
  const char* shaderExt = ".cg";
  const char* unixShaderDir = NULL;
  char* shaderFile = NULL;
  char* shaderPath = NULL;

  cerr << "enable PIX called"<< endl;

  pixelShader->printCompatibilityInfo();

  // Specify shader path:
  cerr << "Searching for shader files..." << endl;

  unixShaderDir = pixelShader->getShaderDir();
  cerr << "Using shader path: " << unixShaderDir << endl;

  bool ok = true;
  // Load shader files:
  for (int i=0; i<NUM_PIXEL_SHADERS; ++i)
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

    ok &= pixelShader->loadShader(shaderPath, vvShaderManager::VV_FRAG_SHD);

    delete[] shaderFile;
    delete[] shaderPath;

    if (!ok)
      break;
  }

  cerr << "Fragment programs ready." << endl;

  return ok;
}

//----------------------------------------------------------------------------
/** @return true if initialization successful
 */
bool vvTexRend::initIntersectionShader(vvShaderManager* isectShader, vvShaderManager* pixelShader) const
{
#ifdef ISECT_GLSL_GEO
  const char* shaderFileName = "vv_intersection_geo";
#else
  const char* shaderFileName = "vv_intersection";
#endif

#ifdef ISECT_GLSL_INST
  const char* instStr = "_inst";
#else
  const char* instStr = "";
#endif
  const char* shaderExt = ".glsl";
  const char* unixShaderDir = NULL;
  char* shaderFile = NULL;
  char* shaderPath = NULL;

  shaderFile = new char[strlen(shaderFileName) + 2 + strlen(instStr) + strlen(shaderExt) + 1];
  sprintf(shaderFile, "%s%s%s", shaderFileName, instStr, shaderExt);

  unixShaderDir = isectShader->getShaderDir();

  shaderPath = new char[strlen(unixShaderDir) + 1 + strlen(shaderFile) + 1];
#ifdef _WIN32
  sprintf(shaderPath, "%s\\%s", unixShaderDir, shaderFile);
#else
  sprintf(shaderPath, "%s/%s", unixShaderDir, shaderFile);
#endif

#ifdef ISECT_GLSL_GEO
  const char* vertexFileName = "vv_intersection_ver";
  char* vertexFile = new char[strlen(vertexFileName) + strlen(instStr) + strlen(shaderExt) + 1];
  sprintf(vertexFile, "%s%s%s", vertexFileName, instStr, shaderExt);
  char* vertexPath = new char[strlen(unixShaderDir) + 2 + strlen(vertexFileName) + 1];
#ifdef _WIN32
  sprintf(vertexPath, "%s\\%s", unixShaderDir, vertexFile);
#else
  sprintf(vertexPath, "%s/%s", unixShaderDir, vertexFile);
#endif
  bool ok = isectShader->loadGeomShader(vertexPath, shaderPath);
#else
  bool ok = isectShader->loadShader(shaderPath, vvShaderManager::VV_VERT_SHD);
#endif

  if (ok)
  {
    cerr << "Using intersection shader from: " << shaderPath << endl;
  }
  else
  {
    cerr << "An error occurred when trying to load: " << shaderPath << endl;
  }

  if (ok && pixelShader != NULL)
  {
    const char* passthroughFile = "vv_shader00.cg";
    char* passthroughShader = new char[strlen(unixShaderDir) + 1 + strlen(passthroughFile) + 1];
#ifdef _WIN32
    sprintf(passthroughShader, "%s\\%s", unixShaderDir, passthroughFile);
#else
    sprintf(passthroughShader, "%s/%s", unixShaderDir, passthroughFile);
#endif
    ok &= pixelShader->loadShader(passthroughShader, vvShaderManager::VV_FRAG_SHD);
    delete[] passthroughShader;
  }

  delete[] shaderFile;
  delete[] shaderPath;

  if (ok)
    setupIntersectionParameters(isectShader);

  return ok;
}

void vvTexRend::setupIntersectionParameters(vvShaderManager* isectShader) const
{
  const int parameterCount = 15;

  const char** parameterNames = new const char*[parameterCount];
  vvShaderParameterType* parameterTypes = new vvShaderParameterType[parameterCount];
  parameterNames[ISECT_SHADER_SEQUENCE] = "sequence";            parameterTypes[ISECT_SHADER_SEQUENCE] = VV_SHD_ARRAY;
  parameterNames[ISECT_SHADER_V1] = "v1";                        parameterTypes[ISECT_SHADER_V1] = VV_SHD_ARRAY;
  parameterNames[ISECT_SHADER_V2] = "v2";                        parameterTypes[ISECT_SHADER_V2] = VV_SHD_ARRAY;

  parameterNames[ISECT_SHADER_BRICKMIN] = "brickMin";            parameterTypes[ISECT_SHADER_BRICKMIN] = VV_SHD_VEC4;
  parameterNames[ISECT_SHADER_BRICKDIMINV] = "brickDimInv";      parameterTypes[ISECT_SHADER_BRICKDIMINV] = VV_SHD_VEC3;
  parameterNames[ISECT_SHADER_TEXRANGE] = "texRange";            parameterTypes[ISECT_SHADER_TEXRANGE] = VV_SHD_VEC3;
  parameterNames[ISECT_SHADER_TEXMIN] = "texMin";                parameterTypes[ISECT_SHADER_TEXMIN] = VV_SHD_VEC3;
  parameterNames[ISECT_SHADER_MODELVIEWPROJ] = "modelViewProj";  parameterTypes[ISECT_SHADER_MODELVIEWPROJ] = VV_SHD_ARRAY;
  parameterNames[ISECT_SHADER_DELTA] = "delta";                  parameterTypes[ISECT_SHADER_DELTA] = VV_SHD_SCALAR;
  parameterNames[ISECT_SHADER_PLANENORMAL] = "planeNormal";      parameterTypes[ISECT_SHADER_PLANENORMAL] = VV_SHD_VEC3;
  parameterNames[ISECT_SHADER_FRONTINDEX] = "frontIndex";        parameterTypes[ISECT_SHADER_FRONTINDEX] = VV_SHD_SCALAR;
  parameterNames[ISECT_SHADER_VERTICES] = "vertices";            parameterTypes[ISECT_SHADER_VERTICES] = VV_SHD_ARRAY;
  parameterNames[ISECT_SHADER_V1_MAYBE] = "v1Maybe";             parameterTypes[ISECT_SHADER_V1_MAYBE] = VV_SHD_ARRAY;
  parameterNames[ISECT_SHADER_V2_MAYBE] = "v2Maybe";             parameterTypes[ISECT_SHADER_V2_MAYBE] = VV_SHD_ARRAY;
  parameterNames[ISECT_SHADER_FIRSTPLANE] = "firstPlane";        parameterTypes[ISECT_SHADER_FIRSTPLANE] = VV_SHD_SCALAR;

  isectShader->enableShader(0);
  isectShader->initParameters(0, parameterNames, parameterTypes, parameterCount);

  delete[] parameterNames;

  // Global scope, values will never be changed.

#ifdef ISECT_GLSL_GEO
  int v1[9] = { 0, 1, 2,
                0, 5, 4,
                0, 3, 6 };

  int v2[9] = { 1, 2, 7,
                5, 4, 7,
                3, 6, 7 };

  isectShader->setArray1i(0, ISECT_SHADER_V1, v1, 9);
  isectShader->setArray1i(0, ISECT_SHADER_V2, v2, 9);
#else
  int v1[24] = { 0, 1, 2, 7,
                 0, 1, 4, 7,
                 0, 5, 4, 7,
                 0, 5, 6, 7,
                 0, 3, 6, 7,
                 0, 3, 2, 7 };
  isectShader->setArray1i(0, ISECT_SHADER_V1, v1, 24);
#endif

  isectShader->disableShader(0);
}

//----------------------------------------------------------------------------
void vvTexRend::printLUT() const
{
  int lutEntries[3];

  const int total = getLUTSize(lutEntries);
  for (int i=0; i<total; ++i)
  {
    cerr << "#" << i << ": ";
    for (int c=0; c<4; ++c)
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

float vvTexRend::getManhattenDist(float p1[3], float p2[3]) const
{
  float dist = 0;

  for (int i=0; i<3; ++i)
  {
    dist += float(fabs(p1[i] - p2[i])) / float(vd->getSize()[i] * vd->vox[i]);
  }

  std::cerr << "Manhattan Distance: " << dist << endl;

  return dist;
}

void vvTexRend::prepareDistributedRendering(const int numSlaveNodes)
{
  _numSlaveNodes = numSlaveNodes;
  // No load balancing for now, distribute equally among nodes.
  float* part = new float[_numSlaveNodes];
  for (unsigned int i = 0; i < _numSlaveNodes; ++i)
  {
    part[i] = 1.0f / static_cast<float>(_numSlaveNodes);
  }

  delete _bspTree;
  _bspTree = new vvBspTree(part, _numSlaveNodes, _brickList);

  int i = 0;
  for (std::vector<vvHalfSpace*>::const_iterator it = _bspTree->getLeafs()->begin();
       it != _bspTree->getLeafs()->end(); ++it)
  {
    vvHalfSpace* hs = (*it);
    hs->setId(i);
    ++i;
  }

  delete[] part;
}

std::vector<BrickList>** vvTexRend::getBrickListsToDistribute()
{
  std::vector<vvHalfSpace*>* _bspTreeLeafs = _bspTree->getLeafs();
  std::vector<BrickList>** result = new std::vector<BrickList>*[_bspTreeLeafs->size()];

  int i = 0;
  for (std::vector<vvHalfSpace*>::const_iterator it = _bspTreeLeafs->begin();
       it != _bspTreeLeafs->end(); ++it)
  {
    result[i] = new std::vector<BrickList>[vd->frames];

    for (int f=0; f<vd->frames; ++f)
    {
      result[i]->push_back(((*it)->getBrickList()[f]));
    }
    ++i;
  }

  return result;
}

int vvTexRend::getNumBrickListsToDistribute() const
{
  return _bspTree->getLeafs()->size();
}

void vvTexRend::calcProjectedScreenRects()
{
  vvVector3 probePosObj;
  vvVector3 probeSizeObj;
  vvVector3 probeMin, probeMax;

  calcProbeDims(probePosObj, probeSizeObj, probeMin, probeMax);

  for (std::vector<vvHalfSpace*>::const_iterator it = _bspTree->getLeafs()->begin();
       it != _bspTree->getLeafs()->end(); ++it)
  {
    (*it)->getProjectedScreenRect(&probeMin, &probeMax, true);
  }
}

//----------------------------------------------------------------------------
/** Get the quality to adjust numSlices. If the render target is an offscreen
    buffer, that one gets scaled and the quality return value is adjusted since
    part of the quality reduction is accomodated through image scaling.
    @return The quality.
*/
float vvTexRend::calcQualityAndScaleImage()
{
  float quality = _renderState._quality;
  if (quality < 1.0f)
  {
    vvOffscreenBuffer* offscreenBuffer = dynamic_cast<vvOffscreenBuffer*>(_renderTarget);
    if (offscreenBuffer != NULL)
    {
      quality = powf(quality, 1.0f/3.0f);
      offscreenBuffer->setScale(quality);
    }
  }
  return quality;
}

int vvTexRend::get2DTextureShader()
{
  // this is trivial, but a better idea than x times writing
  // shader = 9 in your code and later changing that value... .
  return 9;
}

int vvTexRend::getLocalIlluminationShader()
{
  return 12;
}

void vvTexRend::initVertArray(const int numSlices)
{
  if(static_cast<int>(_elemCounts.size()) >= numSlices)
    return;

  _elemCounts.resize(numSlices);
  _vertIndices.resize(numSlices);
#ifdef ISECT_GLSL_GEO
  _vertIndicesAll.resize(numSlices*3);
  _vertArray.resize(numSlices*6);
#else
  _vertIndicesAll.resize(numSlices*6);
  _vertArray.resize(numSlices*12);
#endif

  int idxIterator = 0;
  int vertIterator = 0;

  // Spare some instructions in shader:
#ifdef ISECT_GLSL_GEO
  int mul = 3; // ==> x-values: 0, 3, 6 instead of 0, 1, 2
#else
  int mul = 4; // ==> x-values: 0, 4, 8, 12, 16, 20 instead of 0, 1, 2, 3, 4, 5
#endif
  for (int i = 0; i < numSlices; ++i)
  {
#ifdef ISECT_GLSL_GEO
    _elemCounts[i] = 3;
#else
      _elemCounts[i] = 6;
#endif
     _vertIndices[i] = &_vertIndicesAll[i*_elemCounts[i]];
    for (int j = 0; j < _elemCounts[i]; ++j)
    {
      _vertIndices[i][j] = idxIterator;
      ++idxIterator;

      _vertArray[vertIterator] = j * mul;
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
    _renderState._emptySpaceLeaping &= (geomType == VV_BRICKS);
    _renderState._emptySpaceLeaping &= (voxelType != VV_PIX_SHD) || (_currentShader == 0) || (_currentShader == 12);
    _renderState._emptySpaceLeaping &= (voxelType != VV_RGBA);
    // TODO: only implemented for 8-bit 1-channel volumes. Support higher volume resolutions.
    _renderState._emptySpaceLeaping &= ((vd->chan == 1) && (vd->bpc == 1));
  }
}

void vvTexRend::evaluateLocalIllumination(vvShaderManager*& pixelShader, const vvVector3& normal)
{
  // Local illumination based on blinn-phong shading.
  if (voxelType == VV_PIX_SHD && _currentShader == 12)
  {
    // Light direction.
    const vvVector3 L(-normal);

    // Viewing direction.
    const vvVector3 V(-normal);

    // Half way vector.
    vvVector3 H(L + V);
    H.normalize();
    pixelShader->setParameter3f(_currentShader, "L", L[0], L[1], L[2]);
    pixelShader->setParameter3f(_currentShader, "H", H[0], H[1], H[2]);
  }
}

//============================================================================
// End of File
//============================================================================
