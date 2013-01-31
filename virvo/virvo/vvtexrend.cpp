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

#include <GL/glew.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits.h>
#include <math.h>
#include <cstring>

#include "vvopengl.h"
#include "vvdynlib.h"

#ifdef HAVE_X11
#include <GL/glx.h>
#include <X11/Xlib.h>
#endif

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvbrick.h"
#include "vvvecmath.h"
#include "vvdebugmsg.h"
#include "vvtoolshed.h"
#include "vvgltools.h"
#include "vvsphere.h"
#include "vvtexrend.h"
#include "vvclock.h"
#include "vvoffscreenbuffer.h"
#include "vvprintgl.h"
#include "vvshaderfactory.h"
#include "vvshaderprogram.h"
#include "vvvoldesc.h"
#include "vvpthread.h"

using namespace std;

//----------------------------------------------------------------------------
const int vvTexRend::NUM_PIXEL_SHADERS = 13;

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
vvTexRend::vvTexRend(vvVolDesc* vd, vvRenderState renderState, GeometryType geom, VoxelType vox)
  : vvRenderer(vd, renderState)
  , _renderTarget(NULL)
{
  vvDebugMsg::msg(1, "vvTexRend::vvTexRend()");

  glewInit();

  if (vvDebugMsg::isActive(2))
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
#ifdef GL_VERSION_4_2
    cerr << ", 4.2";
#endif
    cerr << endl;
  }

  _shaderFactory = new vvShaderFactory();

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

  if (_useOffscreenBuffer)
  {
    _renderTarget = new vvOffscreenBuffer(_imageScale, virvo::Byte);
    if (_opaqueGeometryPresent)
    {
      _renderTarget->setPreserveDepthBuffer(true);
    }
  }

  _currentShader = vd->chan - 1;
  _previousShader = _currentShader;

  _useOnlyOneBrick = false;
  _areEmptyBricksCreated = false;
  _areBricksCreated = false;
  _lastFrame = -1;
  lutDistance = -1.0;
  _isROIChanged = true;

  // Find out which OpenGL extensions are supported:
  extTex3d  = vvGLTools::isGLextensionSupported("GL_EXT_texture3D") || vvGLTools::isGLVersionSupported(1,2,1);
  arbMltTex = vvGLTools::isGLextensionSupported("GL_ARB_multitexture") || vvGLTools::isGLVersionSupported(1,3,0);

  _shader = NULL;

  extMinMax = vvGLTools::isGLextensionSupported("GL_EXT_blend_minmax") || vvGLTools::isGLVersionSupported(1,4,0);
  extBlendEquation = vvGLTools::isGLextensionSupported("GL_EXT_blend_equation") || vvGLTools::isGLVersionSupported(1,1,0);
  extColLUT  = isSupported(VV_SGI_LUT);
  extPalTex  = isSupported(VV_PAL_TEX);
  extTexShd  = isSupported(VV_TEX_SHD);
  extPixShd  = isSupported(VV_PIX_SHD);
  arbFrgPrg  = isSupported(VV_FRG_PRG);

  extNonPower2 = vvGLTools::isGLextensionSupported("GL_ARB_texture_non_power_of_two") || vvGLTools::isGLVersionSupported(2,0,0);

  // Determine best rendering algorithm for current hardware:
  setVoxelType(findBestVoxelType(vox));
  geomType  = findBestGeometry(geom, voxelType);

  _proxyGeometryOnGpuSupported = vvGLTools::isGLVersionSupported(2,0,0);
  if (!_proxyGeometryOnGpuSupported || geomType != VV_BRICKS)
  {
    _isectType = CPU;
  }

  if(geomType==VV_SLICES || geomType==VV_CUBIC2D)
  {
    _currentShader = 9;
  }

  if (geomType == VV_BRICKS)
  {
    validateEmptySpaceLeaping();
  }

  pixLUTName = 0;
  _shader = initShader();
  if(_shader) _proxyGeometryOnGpuSupported = true;
  if(voxelType == VV_PIX_SHD && !_shader)
    setVoxelType(VV_RGBA);
  initClassificationStage(&pixLUTName, fragProgName);

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
    cerr << ", proxy geometry generation: ";
    switch (_isectType)
    {
    case VERT_SHADER_ONLY:
      cerr << "on the GPU, vertex shader";
      break;
    case GEOM_SHADER_ONLY:
      cerr << "on the GPU, geometry shader";
      break;
    case VERT_GEOM_COMBINED:
      cerr << "on the GPU, vertex shader and geometry shader";
      break;
    case CPU:
      // fall-through
    default:
      cerr << "on the CPU";
      break;
    }
  }
  cerr << endl;

  textures = 0;

  if ((geomType == VV_BRICKS) && _computeBrickSize)
  {
    _areBricksCreated = false;
    computeBrickSize();
  }

  if (voxelType != VV_RGBA)
  {
    makeTextures();      // we only have to do this once for non-RGBA textures
  }
  updateTransferFunction();
}

//----------------------------------------------------------------------------
/// Destructor
vvTexRend::~vvTexRend()
{
  vvDebugMsg::msg(1, "vvTexRend::~vvTexRend()");

  delete _renderTarget;

  freeClassificationStage(pixLUTName, fragProgName);
  removeTextures();

  delete[] rgbaTF;
  delete[] rgbaLUT;
  delete _shader;
  _shader = NULL;

  delete[] preintTable;

  for(std::vector<BrickList>::iterator frame = _brickList.begin();
      frame != _brickList.end();
      ++frame)
  {
    for(BrickList::iterator brick = frame->begin(); brick != frame->end(); ++brick)
      delete *brick;
  }
}

//------------------------------------------------
/** Initialize texture parameters for a voxel type
  @param vt voxeltype
*/
void vvTexRend::setVoxelType(vvTexRend::VoxelType vt)
{
  voxelType = vt;
  switch(voxelType)
  {
    case VV_PAL_TEX:
      texelsize=1;
      internalTexFormat = GL_COLOR_INDEX8_EXT;
      texFormat = GL_COLOR_INDEX;
      break;
    case VV_FRG_PRG:
      texelsize=1;
      internalTexFormat = GL_LUMINANCE;
      texFormat = GL_LUMINANCE;
      break;
    case VV_PIX_SHD:
      if(vd->chan == 1)
      {
        texelsize=1;
        internalTexFormat = GL_LUMINANCE;
        texFormat = GL_LUMINANCE;
      }
      else
      {
        texelsize=4;
        internalTexFormat = GL_RGBA;
        texFormat = GL_RGBA;
      }
      break;
    case VV_TEX_SHD:
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
    if (extTex3d) return VV_VIEWPORT;
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
void vvTexRend::removeTextures()
{
  vvDebugMsg::msg(1, "vvTexRend::removeTextures()");

  if (textures > 0)
  {
    glDeleteTextures(textures, texNames);
    delete[] texNames;
    texNames = NULL;
    textures = 0;
  }
}

//----------------------------------------------------------------------------
/// Generate textures for all rendering modes.
vvTexRend::ErrorType vvTexRend::makeTextures()
{
  ErrorType err = OK;

  vvDebugMsg::msg(2, "vvTexRend::makeTextures()");

  vvVector3i vox = _paddingRegion.getMax() - _paddingRegion.getMin();
  for (int i = 0; i < 3; ++i)
  {
    vox[i] = std::min(vox[i], vd->vox[i]);
  }

  if (vox[0] == 0 || vox[1] == 0 || vox[2] == 0)
    return err;

  // Compute texture dimensions (must be power of 2):
  if (geomType != VV_BRICKS)
  {
    texels[0] = vvToolshed::getTextureSize(vox[0]);
    texels[1] = vvToolshed::getTextureSize(vox[1]);
    texels[2] = vvToolshed::getTextureSize(vox[2]);
  }
  else
  {
    texels[0] = vvToolshed::getTextureSize(_brickSize[0]);
    texels[1] = vvToolshed::getTextureSize(_brickSize[1]);
    texels[2] = vvToolshed::getTextureSize(_brickSize[2]);
  }

  if (geomType == VV_BRICKS)
  {
    // compute number of bricks
    if ((_useOnlyOneBrick) ||
      ((texels[0] == vd->vox[0]) && (texels[1] == vd->vox[1]) && (texels[2] == vd->vox[2])))
      _numBricks[0] = _numBricks[1] = _numBricks[2] = 1;

    else
    {
      _numBricks[0] = (int) ceil((float) (vd->vox[0]) / (float) (_brickSize[0]));
      _numBricks[1] = (int) ceil((float) (vd->vox[1]) / (float) (_brickSize[1]));
      _numBricks[2] = (int) ceil((float) (vd->vox[2]) / (float) (_brickSize[2]));
    }
  }

  switch (geomType)
  {
    case VV_SLICES:  err=makeTextures2D(1); updateTextures2D(1, 0, 10, 20, 15, 10, 5); break;
    case VV_CUBIC2D: err=makeTextures2D(3); updateTextures2D(3, 0, 10, 20, 15, 10, 5); break;
    case VV_BRICKS:
      if (!_areEmptyBricksCreated)
      {
        err = makeEmptyBricks();
      }

      if (err == OK)
      {
        err = makeTextureBricks(_brickList, _areBricksCreated);
      }
      break;
    default: updateTextures3D(0, 0, 0, texels[0], texels[1], texels[2], true); break;
  }
  vvGLTools::printGLError("vvTexRend::makeTextures");

  if (voxelType==VV_PIX_SHD || voxelType==VV_FRG_PRG || voxelType==VV_TEX_SHD)
  {
    makeLUTTexture();
  }
  return err;
}

//----------------------------------------------------------------------------
/// Generate texture for look-up table.
void vvTexRend::makeLUTTexture() const
{
  vvVector3i size;

  vvGLTools::printGLError("enter makeLUTTexture");
  if(voxelType!=VV_PIX_SHD)
     glActiveTextureARB(GL_TEXTURE1_ARB);
  getLUTSize(size);
  glBindTexture(GL_TEXTURE_2D, pixLUTName);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size[0], size[1], 0,
    GL_RGBA, GL_UNSIGNED_BYTE, rgbaLUT);
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
  vvVector3i texSize;                             // size of a 2D texture in bytes for each principal axis
  uchar* raw;                                     // raw volume data
  int texIndex=0;                                 // index of current texture
  int texSliceIndex;                              // index of current voxel in texture
  vvVector3i tw, th;                              // current texture width and height for each principal axis
  vvVector3i rw, rh, rs;                          // raw data width, height, slices for each principal axis
  vvVector3i rawStart;                            // starting offset into raw data, for each principal axis
  vvVector3i rawStepW;                            // raw data step size for texture row, for each principal axis
  vvVector3i rawStepH;                            // raw data step size for texture column, for each principal axis
  vvVector3i rawStepS;                            // raw data step size for texture slices, for each principal axis
  uchar* rawVoxel;                                // current raw data voxel
  bool accommodated = true;                       // false if a texture cannot be accommodated in TRAM
  ErrorType err = OK;

  vvDebugMsg::msg(1, "vvTexRend::makeTextures2D()");

  assert(axes==1 || axes==3);

  removeTextures();                                // first remove previously generated textures from TRAM

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
                case VV_PIX_SHD:
                  for (int c=0; c<ts_min(4, vd->chan); ++c)
                  {
                    rgbaSlice[i][texSliceIndex + c]   = (uchar)rawVal[0];
                  }
                  break;
                case VV_TEX_SHD:
                  for (int c=0; c<4; ++c)
                  {
                    rgbaSlice[i][texSliceIndex + c]   = (uchar)rawVal[0];
                  }
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
  vvVector3i tmpTexels;                           // number of texels in each dimension for current brick

  vvDebugMsg::msg(2, "vvTexRend::makeEmptyBricks()");

  if (!extTex3d) return NO3DTEX;

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

  const vvVector3 halfBrick(float(texels[0]-_brickTexelOverlap) * 0.5f,
                            float(texels[1]-_brickTexelOverlap) * 0.5f,
                            float(texels[2]-_brickTexelOverlap) * 0.5f);

  const vvVector3 halfVolume(float(vd->vox[0] - 1) * 0.5f,
                             float(vd->vox[1] - 1) * 0.5f,
                             float(vd->vox[2] - 1) * 0.5f);

  _brickList.resize(vd->frames);
  for (int f = 0; f < vd->frames; f++)
  {
    for (int bx = 0; bx < _numBricks[0]; bx++)
      for (int by = 0; by < _numBricks[1]; by++)
        for (int bz = 0; bz < _numBricks[2]; bz++)
        {
          // offset to first voxel of current brick
          const int startOffset[3] = { bx * _brickSize[0], by * _brickSize[1], bz * _brickSize[2] };

          // Guarantee that startOffset[i] + brickSize[i] won't exceed actual size of the volume.
          if ((startOffset[0] + _brickSize[0]) >= vd->vox[0])
            tmpTexels[0] = vvToolshed::getTextureSize(vd->vox[0] - startOffset[0]);
          else
            tmpTexels[0] = texels[0];
          if ((startOffset[1] + _brickSize[1]) >= vd->vox[1])
            tmpTexels[1] = vvToolshed::getTextureSize(vd->vox[1] - startOffset[1]);
          else
            tmpTexels[1] = texels[1];
          if ((startOffset[2] + _brickSize[2]) >= vd->vox[2])
            tmpTexels[2] = vvToolshed::getTextureSize(vd->vox[2] - startOffset[2]);
          else
            tmpTexels[2] = texels[2];

          vvBrick* currBrick = new vvBrick();
          int bs[3];
          bs[0] = _brickSize[0];
          bs[1] = _brickSize[1];
          bs[2] = _brickSize[2];
          if (_useOnlyOneBrick)
          {
            bs[0] += _brickTexelOverlap;
            bs[1] += _brickTexelOverlap;
            bs[2] += _brickTexelOverlap;
          }

          int brickTexelOverlap[3];
          for (int d = 0; d < 3; ++d)
          {
            brickTexelOverlap[d] = _brickTexelOverlap;
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
            currBrick->texMin[d] = (1.0f / (2.0f * (float)(_brickTexelOverlap) * (float)tmpTexels[d]));
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

vvTexRend::ErrorType vvTexRend::makeTextureBricks(std::vector<BrickList>& brickList, bool& areBricksCreated)
{
  ErrorType err = OK;
  vvVector4i rawVal;                              // raw values for R,G,B,A
  bool accommodated = true;                       // false if a texture cannot be accommodated in TRAM

  removeTextures();

  const int frames = vd->frames;

  const int texSize = texels[0] * texels[1] * texels[2] * texelsize;
  uchar* texData = new uchar[texSize];

  // number of textures needed
  textures = frames * _numBricks[0] * _numBricks[1] * _numBricks[2];

  texNames = new GLuint[textures];
  glGenTextures(textures, texNames);

  // generate textures contents:
  vvDebugMsg::msg(2, "Transferring textures to TRAM. Total size [KB]: ",
    textures * texSize / 1024);

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
                case VV_PIX_SHD:
                  texData[texelsize * texOffset] = (uchar) rawVal[0];
                  break;
                case VV_TEX_SHD:
                  for (int c = 0; c < 4; c++)
                  {
                    texData[4 * texOffset + c] = (uchar) rawVal[0];
                  }
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
                    texData[4 * texOffset + 3] = rgbaLUT[rawVal[3] * 4 + 3];
                  else
                    texData[4 * texOffset + 3] = (uchar) rawVal[3];
                }
                else if(vd->chan > 0) // compute alpha from color components
                {
                  int alpha = 0;
                  for (int c = 0; c < vd->chan; c++)
                  {
                    // alpha: mean of sum of RGB conversion table results:
                    alpha += (int) rgbaLUT[rawVal[c] * 4 + c];
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

      accommodated = currBrick->upload3DTexture(texNames[currBrick->index], texData,
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

void vvTexRend::setComputeBrickSize(const bool flag)
{
  _computeBrickSize = flag;
  if (_computeBrickSize)
  {
    computeBrickSize();
    if(!_areBricksCreated)
    {
      makeTextures();
    }
  }
}

void vvTexRend::setBrickSize(const int newSize)
{
  vvDebugMsg::msg(3, "vvRenderer::setBricksize()");
  _brickSize[0] = _brickSize[1] = _brickSize[2] = newSize-1;
  _useOnlyOneBrick = false;

  makeTextures();
}

int vvTexRend::getBrickSize() const
{
  vvDebugMsg::msg(3, "vvRenderer::getBricksize()");
  return _brickSize[0]+1;
}

void vvTexRend::setTexMemorySize(const int newSize)
{
  if (_texMemorySize == newSize)
    return;

  _texMemorySize = newSize;
  if (_computeBrickSize)
  {
    computeBrickSize();

    if(!_areBricksCreated)
    {
      makeTextures();
    }
  }
}

int vvTexRend::getTexMemorySize() const
{
  return _texMemorySize;
}

void vvTexRend::computeBrickSize()
{
  vvVector3 probeSize;
  int newBrickSize[3];

  int texMemorySize = _texMemorySize;
  if (texMemorySize == 0)
  {
     vvDebugMsg::msg(1, "vvTexRend::computeBrickSize(): unknown texture memory size, assuming 32 M");
     texMemorySize = 32;
  }

  _useOnlyOneBrick = true;
  for(int i=0; i<3; ++i)
  {
    newBrickSize[i] = vvToolshed::getTextureSize(vd->vox[i]);
    if(newBrickSize[i] > _maxBrickSize[i])
    {
      newBrickSize[i] = _maxBrickSize[i];
      _useOnlyOneBrick = false;
    }
  }

  if(_useOnlyOneBrick)
  {
    setROIEnable(false);
  }
  else
  {
    probeSize[0] = 2 * (newBrickSize[0] - _brickTexelOverlap) / (float) vd->vox[0];
    probeSize[1] = 2 * (newBrickSize[1] - _brickTexelOverlap) / (float) vd->vox[1];
    probeSize[2] = 2 * (newBrickSize[2] - _brickTexelOverlap) / (float) vd->vox[2];

    setProbeSize(&probeSize);
    //setROIEnable(true);
  }
  if (newBrickSize[0]-_brickTexelOverlap != _brickSize[0]
      || newBrickSize[1]-_brickTexelOverlap != _brickSize[1]
      || newBrickSize[2]-_brickTexelOverlap != _brickSize[2]
      || !_areBricksCreated)
  {
    _brickSize[0] = newBrickSize[0]-_brickTexelOverlap;
    _brickSize[1] = newBrickSize[1]-_brickTexelOverlap;
    _brickSize[2] = newBrickSize[2]-_brickTexelOverlap;
    _areBricksCreated = false;
  }
}

//----------------------------------------------------------------------------
/// Update transfer function from volume description.
void vvTexRend::updateTransferFunction()
{
  vvVector3i size;

  vvDebugMsg::msg(1, "vvTexRend::updateTransferFunction()");
  if (preIntegration &&
      arbMltTex && 
      geomType==VV_VIEWPORT && 
      !(_clipMode && (_clipSingleSlice || _clipOpaque)) &&
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
    updateLUT(1.0f);                                // generate color/alpha lookup table
  else
    lutDistance = -1.;                              // invalidate LUT

  fillNonemptyList(_nonemptyList, _brickList);

  _isROIChanged = true; // have to update list of visible bricks
}

//----------------------------------------------------------------------------
// see parent in vvRenderer
void vvTexRend::updateVolumeData()
{
  vvRenderer::updateVolumeData();
  if (_computeBrickSize)
  {
    _areEmptyBricksCreated = false;
    _areBricksCreated = false;
    computeBrickSize();
  }

  makeTextures();
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
  if (_emptySpaceLeaping)
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
        if (_mipMode > 0)
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
  ErrorType err = OK;
  int srcIndex;
  int texOffset=0;
  vvVector4i rawVal;
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

    removeTextures();
    textures  = vd->frames;
    delete[] texNames;
    texNames = new GLuint[textures];
    glGenTextures(vd->frames, texNames);
  }

  vvDebugMsg::msg(2, "Transferring textures to TRAM. Total size [KB]: ",
    vd->frames * texSize / 1024);

  vvVector3i offsets = vvVector3i(offsetX, offsetY, offsetZ);
  offsets += _paddingRegion.getMin();

  // Generate sub texture contents:
  for (int f = 0; f < vd->frames; f++)
  {
    raw = vd->getRaw(f);
    for (int s = offsets[2]; s < (offsets[2] + sizeZ); s++)
    {
      const int rawSliceOffset = (vd->vox[2] - min(s,vd->vox[2]-1) - 1) * sliceSize;
      for (int y = offsets[1]; y < (offsets[1] + sizeY); y++)
      {
        const int heightOffset = (vd->vox[1] - min(y,vd->vox[1]-1) - 1) * vd->vox[0] * vd->bpc * vd->chan;
        const int texLineOffset = (y - offsets[1] - offsetY) * sizeX + (s - offsets[2] - offsetZ) * sizeX * sizeY;
        if (vd->chan == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
        {
          for (int x = offsets[0]; x < (offsets[0] + sizeX); x++)
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
            texOffset = (x - offsets[0] - offsetX) + texLineOffset;
            switch(voxelType)
            {
              case VV_SGI_LUT:
                texData[2 * texOffset] = texData[2 * texOffset + 1] = (uchar) rawVal[0];
                break;
              case VV_PAL_TEX:
              case VV_FRG_PRG:
              case VV_PIX_SHD:
                texData[texelsize * texOffset] = (uchar) rawVal[0];
                break;
              case VV_TEX_SHD:
                for (int c = 0; c < 4; c++)
                {
                  texData[4 * texOffset + c] = (uchar) rawVal[0];
                }
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
  return err;
}

vvTexRend::ErrorType vvTexRend::updateTextures2D(const int axes,
                                                 const int offsetX, const int offsetY, const int offsetZ,
                                                 const int sizeX, const int sizeY, const int sizeZ)
{
  vvVector4i rawVal;
  int rawSliceSize;
  int rawLineSize;
  vvVector3i texSize;
  int texIndex = 0;
  int texSliceIndex;
  vvVector3i texW, texH;
  vvVector3i tw, th;
  vvVector3i rw, rh, rs;
  vvVector3i sw, sh, ss;
  vvVector3i rawStart;
  vvVector3i rawStepW;
  vvVector3i rawStepH;
  vvVector3i rawStepS;
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
  vvVector4i rawVal;
  int alpha;
  vvVector3i startOffset, endOffset;
  vvVector3i start, end, size;
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

      endOffset[0] = startOffset[0] + _brickSize[0];
      endOffset[1] = startOffset[1] + _brickSize[1];
      endOffset[2] = startOffset[2] + _brickSize[2];

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
                  texData[texelsize * texOffset] = (uchar) rawVal[0];
                  break;
                case VV_TEX_SHD:
                  for (c = 0; c < 4; c++)
                    texData[4 * texOffset + c] = (uchar) rawVal[0];
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

  if (glBlendFuncSeparate)
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
  switch (_mipMode)
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
void vvTexRend::enableLUTMode(GLuint& lutName, GLuint progName[VV_FRAG_PROG_MAX])
{
  switch(voxelType)
  {
    case VV_FRG_PRG:
      enableFragProg(lutName, progName);
      break;
    case VV_TEX_SHD:
      enableNVShaders();
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
void vvTexRend::disableLUTMode()
{
  switch(voxelType)
  {
    case VV_FRG_PRG:
      disableFragProg();
      break;
    case VV_TEX_SHD:
      disableNVShaders();
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
void vvTexRend::renderTex3DPlanar(const vvMatrix& mv)
{
  vvMatrix invMV;                                 // inverse of model-view matrix
  vvMatrix pm;                                    // OpenGL projection matrix
  vvVector3 vissize, vissize2;                    // full and half object visible sizes
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
  int       numSlices;

  vvDebugMsg::msg(3, "vvTexRend::renderTex3DPlanar()");

  if (!extTex3d) return;                          // needs 3D texturing extension

  // determine visible size and half object size as shortcut
  vvVector3i minVox = _visibleRegion.getMin();
  vvVector3i maxVox = _visibleRegion.getMax();
  for (int i = 0; i < 3; ++i)
  {
    minVox[i] = std::max(minVox[i], 0);
    maxVox[i] = std::min(maxVox[i], vd->vox[i]);
  }
  const vvVector3 minCorner = vd->objectCoords(minVox);
  const vvVector3 maxCorner = vd->objectCoords(maxVox);
  vissize = maxCorner - minCorner;
  const vvVector3 center = vvAABB(minCorner, maxCorner).getCenter();

  for (int i=0; i<3; ++i)
  {
    texSize[i] = vissize[i] * (float)texels[i] / (float)vd->vox[i];
    vissize2[i]   = 0.5f * vissize[i];
  }
  pos = center;

  // Calculate inverted modelview matrix:
  invMV = vvMatrix(mv);
  invMV.invert();

  // Find eye position:
  getEyePosition(&eye);
  eye.multiply(invMV);

  if (_isROIUsed)
  {
    const vvVector3 size = vd->getSize();
    const vvVector3 size2 = size * 0.5f;
    // Convert probe midpoint coordinates to object space w/o position:
    probePosObj = _roiPos;
    probePosObj.sub(pos);                        // eliminate object position from probe position

    // Compute probe min/max coordinates in object space:
    for (int i=0; i<3; ++i)
    {
      probeMin[i] = probePosObj[i] - (_roiSize[i] * size[i]) * 0.5f;
      probeMax[i] = probePosObj[i] + (_roiSize[i] * size[i]) * 0.5f;
    }

    // Constrain probe boundaries to volume data area:
    for (int i=0; i<3; ++i)
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
    for (int i=0; i<3; ++i)
      probeSizeObj[i] = probeMax[i] - probeMin[i];
  }
  else                                            // probe mode off
  {
    const vvVector3 size = vd->getSize();
    probeSizeObj = size;
    probeMin = minCorner;
    probeMax = maxCorner;
    probePosObj = center;
  }

  // Initialize texture counters
  if (_isROIUsed)
  {
    probeTexels.zero();
    for (int i=0; i<3; ++i)
    {
      probeTexels[i] = texels[i] * probeSizeObj[i] / texSize[i];
    }
  }
  else                                            // probe mode off
  {
    probeTexels.set((float)vd->vox[0], (float)vd->vox[1], (float)vd->vox[2]);
  }

  // Get projection matrix:
  vvGLTools::getProjectionMatrix(&pm);
  bool isOrtho = pm.isProjOrtho();

  getObjNormal(normal, origin, eye, invMV, isOrtho);
  evaluateLocalIllumination(_shader, normal);

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
  if(_isROIUsed && _quality < 2.0)
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
      updateLUT(thickness);
    }
  }

  delta = normal;
  delta.scale(sliceDistance);

  // Compute farthest point to draw texture at:
  farthest = delta;
  farthest.scale((float)(numSlices - 1) * -0.5f);
  farthest.add(vd->pos);

  if (_clipMode)                     // clipping plane present?
  {
    // Adjust numSlices and set farthest point so that textures are only
    // drawn up to the clipPoint. (Delta may not be changed
    // due to the automatic opacity correction.)
    // First find point on clipping plane which is on a line perpendicular
    // to clipping plane and which traverses the origin:
    temp = delta;
    temp.scale(-0.5f);
    farthest.add(temp);                          // add a half delta to farthest
    clipPosObj = _clipPoint;
    clipPosObj.sub(pos);
    temp = probePosObj;
    temp.add(normal);
    normClipPoint.isectPlaneLine(normal, clipPosObj, probePosObj, temp);
    maxDist = farthest.distance(normClipPoint);
    numSlices = (int)(maxDist / delta.length()) + 1;
    temp = delta;
    temp.scale((float)(1 - numSlices));
    farthest = normClipPoint;
    farthest.add(temp);
    if (_clipSingleSlice)
    {
      // Compute slice position:
      temp = delta;
      temp.scale((float)(numSlices-1));
      farthest.add(temp);
      numSlices = 1;

      // Make slice opaque if possible:
      if (instantClassification())
      {
        updateLUT(0.0f);
      }
    }
  }

  vvVector3 texPoint;                             // arbitrary point on current texture
  int isectCnt;                                   // intersection counter
  int j,k;                                        // counters
  int drawn = 0;                                  // counter for drawn textures
  vvVector3 deltahalf;
  deltahalf = delta;
  deltahalf.scale(0.5f);

  // Relative viewing position
  vvVector3 releye;
  releye = eye;
  releye.sub(pos);

  // Volume render a 3D texture:
  if(voxelType == VV_PIX_SHD && _shader)
  {
    _shader->setParameterTex3D("pix3dtex", texNames[vd->getCurrentFrame()]);
  }
  else
  {
    enableTexture(GL_TEXTURE_3D_EXT);
    glBindTexture(GL_TEXTURE_3D_EXT, texNames[vd->getCurrentFrame()]);
  }
  texPoint = farthest;
  for (int i=0; i<numSlices; ++i)                     // loop thru all drawn textures
  {
    // Search for intersections between texture plane (defined by texPoint and
    // normal) and texture object (0..1):
    isectCnt = isect->isectPlaneCuboid(normal, texPoint, probeMin, probeMax);

    texPoint.add(delta);

    if (isectCnt<3) continue;                     // at least 3 intersections needed for drawing

    // Check volume section mode:
    if (minSlice != -1 && i < minSlice) continue;
    if (maxSlice != -1 && i > maxSlice) continue;

    // Put the intersecting 3 to 6 vertices in cyclic order to draw adjacent
    // and non-overlapping triangles:
    isect->cyclicSort(isectCnt, normal);

    // Generate vertices in texture coordinates:
    if(usePreIntegration)
    {
      for (j=0; j<isectCnt; ++j)
      {
        vvVector3 front, back;

        if(isOrtho)
        {
          back = isect[j];
          back.sub(deltahalf);
        }
        else
        {
          vvVector3 v = isect[j];
          v.sub(deltahalf);
          back.isectPlaneLine(normal, v, releye, isect[j]);
        }

        if(isOrtho)
        {
          front = isect[j];
          front.add(deltahalf);
        }
        else
        {
          vvVector3 v;
          v = isect[j];
          v.add(deltahalf);
          front.isectPlaneLine(normal, v, releye, isect[j]);
        }

        for (k=0; k<3; ++k)
        {
          texcoord[j][k] = (back[k] - minCorner[k]) / vissize[k];
          texcoord[j][k] = texcoord[j][k] * (texMax[k] - texMin[k]) + texMin[k];

          texcoord[j+6][k] = (front[k] - minCorner[k]) / vissize[k];
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
          texcoord[j][k] = (isect[j][k] - minCorner[k]) / vissize[k];
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
}

void vvTexRend::renderTexBricks(const vvMatrix& mv)
{
  vvGLTools::printGLError("Enter vvTexRend::renderTexBricks(const vvMatrix* mv)");
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

  vvGLTools::printGLError("enter vvTexRend::renderTexBricks()");

  // needs 3D texturing extension
  if (!extTex3d) return;

  if (_brickList.empty()) return;

  // Calculate inverted modelview matrix:
  vvMatrix invMV(mv);
  invMV.invert();

  // Find eye position:
  getEyePosition(&eye);
  eye.multiply(invMV);

  calcProbeDims(probePosObj, probeSizeObj, probeMin, probeMax);

  vvVector3 clippedProbeSizeObj = probeSizeObj;
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
  vvGLTools::getProjectionMatrix(&pm);
  const bool isOrtho = pm.isProjOrtho();

  getObjNormal(normal, origin, eye, invMV, isOrtho);
  evaluateLocalIllumination(_shader, normal);

  // Use alpha correction in indexed mode: adapt alpha values to number of textures:
  if (instantClassification())
  {
    const float thickness = diagonalVoxels / float(numSlices);
    if(lutDistance/thickness < 0.88 || thickness/lutDistance < 0.88)
    {
      updateLUT(thickness);
    }
  }

  delta = normal;
  delta.scale(diagonal / ((float)numSlices));

  // Compute farthest point to draw texture at:
  farthest = delta;
  farthest.scale((float)(numSlices - 1) * -0.5f);

  if (_clipMode)                     // clipping plane present?
  {
    // Adjust numSlices and set farthest point so that textures are only
    // drawn up to the clipPoint. (Delta may not be changed
    // due to the automatic opacity correction.)
    // First find point on clipping plane which is on a line perpendicular
    // to clipping plane and which traverses the origin:
    vvVector3 temp(delta);
    temp.scale(-0.5f);
    farthest.add(temp);                          // add a half delta to farthest
    vvVector3 clipPosObj(_clipPoint);
    clipPosObj.sub(vd->pos);
    temp = probePosObj;
    temp.add(normal);
    vvVector3 normClipPoint;
    normClipPoint.isectPlaneLine(normal, clipPosObj, probePosObj, temp);
    const float maxDist = farthest.distance(normClipPoint);
    numSlices = (int)(maxDist / delta.length()) + 1;
    temp = delta;
    temp.scale((float)(1 - numSlices));
    farthest = normClipPoint;
    farthest.add(temp);
    if (_clipSingleSlice)
    {
      // Compute slice position:
      delta.scale((float)(numSlices-1));
      farthest.add(delta);
      numSlices = 1;

      // Make slice opaque if possible:
      if (instantClassification())
      {
        updateLUT(1.0f);
      }
    }
  }

  initVertArray(numSlices);

  getBricksInProbe(_nonemptyList, _insideList, _sortedList, probePosObj, probeSizeObj, _isROIChanged);

  markBricksInFrustum(probeMin, probeMax);

  // Volume render a 3D texture:
  enableTexture(GL_TEXTURE_3D_EXT);
  sortBrickList(_sortedList, eye, normal, isOrtho);

  if (_showBricks)
  {
    // Debugging mode: render the brick outlines and deactivate shaders,
    // lighting and texturing.
    glDisable(GL_TEXTURE_3D_EXT);
    glDisable(GL_LIGHTING);
    disableShader(_shader);
    disableFragProg();
    for(BrickList::iterator it = _sortedList.begin(); it != _sortedList.end(); ++it)
    {
      (*it)->renderOutlines(probeMin, probeMax);
    }
  }
  else
  {
    if (_isectType != CPU)
    {
      _shader->setParameter1f("delta", delta.length());
      _shader->setParameter3f("planeNormal", normal[0], normal[1], normal[2]);

      glEnableClientState(GL_VERTEX_ARRAY);
    }
    for(BrickList::iterator it = _sortedList.begin(); it != _sortedList.end(); ++it)
    {
      (*it)->render(this, normal, farthest, delta, probeMin, probeMax,
                   texNames,
                   _shader);
    }
    if (_isectType != CPU)
    {
      glDisableClientState(GL_VERTEX_ARRAY);
    }
  }

  vvDebugMsg::msg(3, "Bricks discarded: ",
                  static_cast<int>(_brickList[vd->getCurrentFrame()].size() - _sortedList.size()));

  disableTexture(GL_TEXTURE_3D_EXT);
}
void vvTexRend::updateFrustum()
{
  float pm[16];
  float mvm[16];
  vvMatrix proj, modelview, clip;

  // Get the current projection matrix from OpenGL
  glGetFloatv(GL_PROJECTION_MATRIX, pm);
  proj.setGL(pm);

  // Get the current modelview matrix from OpenGL
  glGetFloatv(GL_MODELVIEW_MATRIX, mvm);
  modelview.setGL(mvm);

  clip = proj;
  clip.multiplyRight(modelview);

  // extract the planes of the viewing frustum

  // left plane
  _frustum[0].set(clip(3, 0)+clip(0, 0), clip(3, 1)+clip(0, 1),
    clip(3, 2)+clip(0, 2), clip(3, 3)+clip(0, 3));
  // right plane
  _frustum[1].set(clip(3, 0)-clip(0, 0), clip(3, 1)-clip(0, 1),
    clip(3, 2)-clip(0, 2), clip(3, 3)-clip(0, 3));
  // top plane
  _frustum[2].set(clip(3, 0)-clip(1, 0), clip(3, 1)-clip(1, 1),
    clip(3, 2)-clip(1, 2), clip(3, 3)-clip(1, 3));
  // bottom plane
  _frustum[3].set(clip(3, 0)+clip(1, 0), clip(3, 1)+clip(1, 1),
    clip(3, 2)+clip(1, 2), clip(3, 3)+clip(1, 3));
  // near plane
  _frustum[4].set(clip(3, 0)+clip(2, 0), clip(3, 1)+clip(2, 1),
    clip(3, 2)+clip(2, 2), clip(3, 3)+clip(2, 3));
  // far plane
  _frustum[5].set(clip(3, 0)-clip(2, 0), clip(3, 1)-clip(2, 1),
    clip(3, 2)-clip(2, 2), clip(3, 3)-clip(2, 3));
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

      if ((pv.dot(normal) + _frustum[i][3]) < 0)
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

    if ((pv.dot(normal) + _frustum[i][3]) < 0)
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
    const float x = brick->min[0] + xStep * i;
    for(int j = 0; j < numSteps; j++)
    {
      const float y = brick->min[1] + yStep * j;
      for(int k = 0; k < numSteps; k++)
      {
        const float z = brick->min[2] + zStep * k;
        vvVector3 clipPnt(x, y, z);
        clipPnt.multiply(mvpMat);

        //test if this point falls within screen space
        if(clipPnt[0] >= -1.0 && clipPnt[0] <= 1.0 &&
          clipPnt[1] >= -1.0 && clipPnt[1] <= 1.0)
        {
          return true;
        }
      }
    }
  }

  return false;
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
                                 const vvVector3 pos, const vvVector3 size, bool& roiChanged)
{
  // Single gpu mode.
  if(!roiChanged && vd->getCurrentFrame() == _lastFrame)
    return;
  _lastFrame = vd->getCurrentFrame();
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
    if ((tmp->min[0] <= max[0]) && (tmp->max[0] >= min[0]) &&
      (tmp->min[1] <= max[1]) && (tmp->max[1] >= min[1]) &&
      (tmp->min[2] <= max[2]) && (tmp->max[2] >= min[2]))
    {
      insideList.push_back(tmp);
      if ((tmp->min[0] >= min[0]) && (tmp->max[0] <= max[0]) &&
        (tmp->min[1] >= min[1]) && (tmp->max[1] <= max[1]) &&
        (tmp->min[2] >= min[2]) && (tmp->max[2] <= max[2]))
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
      (*it)->dist = -(*it)->pos.dot(normal);
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

//----------------------------------------------------------------------------
/** Render the volume using a 3D texture (needs 3D texturing extension).
  Spherical slices are surrounding the observer.
  @param mv       model-view matrix
*/
void vvTexRend::renderTex3DSpherical(const vvMatrix& mv)
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
  const int numShells = max(1, static_cast<int>(_quality * 100.0f));

  // Determine texture object dimensions:
  const vvVector3 size(vd->getSize());
  for (int i=0; i<3; ++i)
  {
    texSize[i]  = size[i] * (float)texels[i] / (float)vd->vox[i];
    texSize2[i] = 0.5f * texSize[i];
  }

  invView = vvMatrix(mv);
  invView.invert();

  // generates the vertices of the cube (volume) in world coordinates
  int vertexIdx = 0;
  for (int ix=0; ix<2; ++ix)
    for (int iy=0; iy<2; ++iy)
      for (int iz=0; iz<2; ++iz)
      {
        volumeVertices[vertexIdx][0] = (float)ix;
        volumeVertices[vertexIdx][1] = (float)iy;
        volumeVertices[vertexIdx][2] = (float)iz;
        // transfers vertices to world coordinates:
        for (int k=0; k<3; ++k)
          volumeVertices[vertexIdx][k] =
            (volumeVertices[vertexIdx][k] * 2.0f - 1.0f) * texSize2[k];
        volumeVertices[vertexIdx].multiply(mv);
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
  eye.multiply(invView);
  bool inside = true;
  for (int k=0; k<3; ++k)
  {
    if (eye[k] < -texSize2[k] || eye[k] > texSize2[k])
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
    updateLUT(diagonalVoxels / numShells);
  }

  vvSphere shell;
  shell.subdivide();
  shell.subdivide();
  shell.setVolumeDim(texSize);
  shell.setViewMatrix(mv);
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
  if(voxelType == VV_PIX_SHD && _shader)
  {
    _shader->setParameterTex3D("pix3dtex", texNames[vd->getCurrentFrame()]);
  }
  else
  {
    enableTexture(GL_TEXTURE_3D_EXT);
    glBindTexture(GL_TEXTURE_3D_EXT, texNames[vd->getCurrentFrame()]);
  }

  // Enable clipping plane if appropriate:
  if (_clipMode) activateClippingPlane();

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

  // Enable clipping plane if appropriate:
  if (_clipMode) activateClippingPlane();

  // Generate half object size as shortcut:
  size = vd->getSize();
  size2[0] = 0.5f * size[0];
  size2[1] = 0.5f * size[1];
  size2[2] = 0.5f * size[2];

  numTextures = int(_quality * 100.0f);
  if (numTextures < 1) numTextures = 1;

  normal.set(0.0f, 0.0f, 1.0f);
  zPos = -size2[2];
  if (numTextures>1)                              // prevent division by zero
  {
    texSpacing = size[2] / (float)(numTextures - 1);
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
    normal[2]   = -normal[2];
  }

  if (instantClassification())
  {
    float diagVoxels = sqrtf(float(vd->vox[0]*vd->vox[0]
      + vd->vox[1]*vd->vox[1]
      + vd->vox[2]*vd->vox[2]));
    updateLUT(diagVoxels/numTextures);
  }

  // Volume rendering with multiple 2D textures:
  enableTexture(GL_TEXTURE_2D);

  for (int i=0; i<numTextures; ++i)
  {
    if(voxelType == VV_PIX_SHD && _shader)
    {
      _shader->setParameterTex2D("pix2dtex", texNames[vvToolshed::round(texIndex)]);
    }
    else
    {
      glBindTexture(GL_TEXTURE_2D, texNames[vvToolshed::round(texIndex)]);
    }


    if(voxelType==VV_PAL_TEX)
    {
      vvVector3i size;
      getLUTSize(size);
      glColorTableEXT(GL_TEXTURE_2D, GL_RGBA,
        size[0], GL_RGBA, GL_UNSIGNED_BYTE, rgbaLUT);
    }

    glBegin(GL_QUADS);
      glColor4f(1.0, 1.0, 1.0, 1.0);
      glNormal3f(normal[0], normal[1], normal[2]);
      glTexCoord2f(texMin[0], texMax[1]); glVertex3f(-size2[0],  size2[1], zPos);
      glTexCoord2f(texMin[0], texMin[1]); glVertex3f(-size2[0], -size2[1], zPos);
      glTexCoord2f(texMax[0], texMin[1]); glVertex3f( size2[0], -size2[1], zPos);
      glTexCoord2f(texMax[0], texMax[1]); glVertex3f( size2[0],  size2[1], zPos);
    glEnd();

    zPos += texSpacing;
    texIndex += texStep;
  }

  disableTexture(GL_TEXTURE_2D);
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
void vvTexRend::renderTex2DCubic(vvVecmath::AxisType principal, float zx, float zy, float zz)
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

  // Enable clipping plane if appropriate:
  if (_clipMode) activateClippingPlane();

  // Initialize texture parameters:
  numTextures = int(_quality * 100.0f);
  frameTextures = vd->vox[0] + vd->vox[1] + vd->vox[2];
  if (numTextures < 2)  numTextures = 2;          // make sure that at least one slice is drawn to prevent division by zero

  // Generate half object size as a shortcut:
  size = vd->getSize();
  size2 = size;
  size2.scale(0.5f);

  // Initialize parameters upon principal viewing direction:
  switch (principal)
  {
    case vvVecmath::X_AXIS:                                  // zx>0 -> draw left to right
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

      texSpacing.set(size[0] / float(numTextures - 1), 0.0f, 0.0f);
      texStep = -1.0f * float(vd->vox[0] - 1) / float(numTextures - 1);
      normal.set(1.0f, 0.0f, 0.0f);
      texIndex = float(vd->getCurrentFrame() * frameTextures);
      if (zx<0)                                   // reverse order? draw right to left
      {
        normal[0]       = -normal[0];
        objTL[0]        = objTR[0] = objBL[0] = objBR[0] = size2[0];
        texSpacing[0]   = -texSpacing[0];
        texStep         = -texStep;
      }
      else
      {
        texIndex += float(vd->vox[0] - 1);
      }
      break;

    case vvVecmath::Y_AXIS:                                  // zy>0 -> draw bottom to top
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

      texSpacing.set(0.0f, size[1] / float(numTextures - 1), 0.0f);
      texStep = -1.0f * float(vd->vox[1] - 1) / float(numTextures - 1);
      normal.set(0.0f, 1.0f, 0.0f);
      texIndex = float(vd->getCurrentFrame() * frameTextures + vd->vox[0]);
      if (zy<0)                                   // reverse order? draw top to bottom
      {
        normal[1]       = -normal[1];
        objTL[1]        = objTR[1] = objBL[1] = objBR[1] = size2[1];
        texSpacing[1]   = -texSpacing[1];
        texStep         = -texStep;
      }
      else
      {
        texIndex += float(vd->vox[1] - 1);
      }
      break;

    case vvVecmath::Z_AXIS:                                  // zz>0 -> draw back to front
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

      texSpacing.set(0.0f, 0.0f, size[2] / float(numTextures - 1));
      normal.set(0.0f, 0.0f, 1.0f);
      texStep = -1.0f * float(vd->vox[2] - 1) / float(numTextures - 1);
      texIndex = float(vd->getCurrentFrame() * frameTextures + vd->vox[0] + vd->vox[1]);
      if (zz<0)                                   // reverse order? draw front to back
      {
        normal[2]       = -normal[2];
        objTL[2]        = objTR[2] = objBL[2] = objBR[2] = size2[2];
        texSpacing[2]   = -texSpacing[2];
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
    updateLUT(diagVoxels/numTextures);
  }

  // Volume render a 2D texture:
  enableTexture(GL_TEXTURE_2D);
  for (i=0; i<numTextures; ++i)
  {
    if(voxelType == VV_PIX_SHD && _shader)
    {
      _shader->setParameterTex2D("pix2dtex", texNames[vvToolshed::round(texIndex)]);
    }
    else
    {
      glBindTexture(GL_TEXTURE_2D, texNames[vvToolshed::round(texIndex)]);
    }

    if(voxelType==VV_PAL_TEX)
    {
      vvVector3i size;
      getLUTSize(size);
      glColorTableEXT(GL_TEXTURE_2D, GL_RGBA,
        size[0], GL_RGBA, GL_UNSIGNED_BYTE, rgbaLUT);
    }

    glBegin(GL_QUADS);
    glColor4f(1.0, 1.0, 1.0, 1.0);
    glNormal3f(normal[0], normal[1], normal[2]);
    glTexCoord2f(texTL[0], texTL[1]); glVertex3f(objTL[0], objTL[1], objTL[2]);
    glTexCoord2f(texBL[0], texBL[1]); glVertex3f(objBL[0], objBL[1], objBL[2]);
    glTexCoord2f(texBR[0], texBR[1]); glVertex3f(objBR[0], objBR[1], objBR[2]);
    glTexCoord2f(texTR[0], texTR[1]); glVertex3f(objTR[0], objTR[1], objTR[2]);
    glEnd();
    objTL.add(texSpacing);
    objBL.add(texSpacing);
    objBR.add(texSpacing);
    objTR.add(texSpacing);

    texIndex += texStep;
  }
  disableTexture(GL_TEXTURE_2D);
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
  float zx, zy, zz;                               // base vector z coordinates

  vvDebugMsg::msg(3, "vvTexRend::renderVolumeGL()");

  vvGLTools::printGLError("enter vvTexRend::renderVolumeGL()");

  vvVector3i vox = _paddingRegion.getMax() - _paddingRegion.getMin();
  for (int i = 0; i < 3; ++i)
  {
    vox[i] = std::min(vox[i], vd->vox[i]);
  }

  if (vox[0] * vox[1] * vox[2] == 0)
    return;

  if (_measureRenderTime)
  {
    sw.start();
  }

  const vvVector3 size(vd->getSize());            // volume size [world coordinates]

  // Draw boundary lines (must be done before setGLenvironment()):
  if (_boundaries)
  {
    drawBoundingBox(size, vd->pos, _boundColor);
  }
  if (_isROIUsed)
  {
    const vvVector3 probeSizeObj(size[0] * _roiSize[0], size[1] * _roiSize[1], size[2] * _roiSize[2]);
    drawBoundingBox(probeSizeObj, _roiPos, _probeColor);
  }
  if (_clipMode && _clipPerimeter)
  {
    drawPlanePerimeter(size, vd->pos, _clipPoint, _clipNormal, _clipColor);
  }

  if (_renderTarget != NULL)
  {
    _renderTarget->bind();
  }

  setGLenvironment();

  if (_renderTarget != NULL)
  {
    _renderTarget->clear();
  }

  // Determine texture object extensions:
  for (int i = 0; i < 3; ++i)
  {
    // padded borders for (trilinear) interpolation
    const int paddingLeft = abs(_visibleRegion.getMin()[i] - _paddingRegion.getMin()[i]);
    const int paddingRight = abs(_visibleRegion.getMax()[i] - _paddingRegion.getMax()[i]);
    // a voxels size
    const float vsize = 1.0f / (float)texels[i];
    // half a voxels size
    const float vsize2 = 0.5f / (float)texels[i];
    if (paddingLeft == 0)
    {
      texMin[i] = vsize2;
    }
    else
    {
      texMin[i] = vsize * (float)paddingLeft;
    }

    texMax[i] = (float)vox[i] / (float)texels[i];
    if (paddingRight == 0)
    {
      texMax[i] -= vsize2;
    }
    else
    {
      texMax[i] -= vsize * (float)paddingRight;
    }
  }

  // Get OpenGL modelview matrix:
  vvGLTools::getModelviewMatrix(&mv);

  if (geomType != VV_BRICKS || !_showBricks)
  {
	  enableShader(_shader, pixLUTName);
    enableLUTMode(pixLUTName, fragProgName);
  }

  switch (geomType)
  {
    default:
    case VV_SLICES:
      getPrincipalViewingAxis(mv, zx, zy, zz);renderTex2DSlices(zz);
      break;
    case VV_CUBIC2D:
      {
        const vvVecmath::AxisType at = getPrincipalViewingAxis(mv, zx, zy, zz);
        renderTex2DCubic(at, zx, zy, zz);
      }
      break;
    case VV_SPHERICAL: renderTex3DSpherical(mv); break;
    case VV_VIEWPORT:  renderTex3DPlanar(mv); break;
    case VV_BRICKS:
        renderTexBricks(mv);
      break;
  }

  disableLUTMode();
  unsetGLenvironment();
  disableShader(_shader);
  vvRenderer::renderVolumeGL();

  if (_renderTarget != NULL)
  {
    _renderTarget->unbind();
  }

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
  planeEq[0] = -_clipNormal[0];
  planeEq[1] = -_clipNormal[1];
  planeEq[2] = -_clipNormal[2];
  planeEq[3] = _clipNormal.dot(_clipPoint);
  glClipPlane(GL_CLIP_PLANE0, planeEq);
  glEnable(GL_CLIP_PLANE0);

  // Generate second clipping plane in single slice mode:
  if (_clipSingleSlice)
  {
    thickness = vd->_scale * vd->dist[0] * (vd->vox[0] * 0.01f);
    clipNormal2 = _clipNormal;
    clipNormal2.negate();
    planeEq[0] = -clipNormal2[0];
    planeEq[1] = -clipNormal2[1];
    planeEq[2] = -clipNormal2[2];
    planeEq[3] = clipNormal2.dot(_clipPoint) + thickness;
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
  if (_clipSingleSlice) glDisable(GL_CLIP_PLANE1);
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
int vvTexRend::getLUTSize(vvVector3i& size) const
{
  int x, y, z;

  vvDebugMsg::msg(3, "vvTexRend::getLUTSize()");
  if (vd->bpc==2 && voxelType==VV_SGI_LUT)
  {
    x = 4096;
    y = z = 1;
  }
  else if (_currentShader==8 && voxelType==VV_PIX_SHD)
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

  size[0] = x;
  size[1] = y;
  size[2] = z;

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
void vvTexRend::updateLUT(const float dist)
{
  vvDebugMsg::msg(3, "Generating texture LUT. Slice distance = ", dist);

  vvVector4f corr;                                // gamma/alpha corrected RGBA values [0..1]
  vvVector3i lutSize;                             // number of entries in the RGBA lookup table
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
      if (_gammaCorrection)
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
      if (dist<=0.0 || (_clipMode && _clipOpaque)) corr[3] = 1.0f;
      else if (opacityCorrection) corr[3] = 1.0f - powf(1.0f - corr[3], dist);

      // Convert float to uchar and copy to rgbaLUT array:
      for (int c=0; c<4; ++c)
      {
        rgbaLUT[i * 4 + c] = uchar(corr[c] * 255.0f);
      }
    }
  }

  // Copy LUT to graphics card:
  vvGLTools::printGLError("enter updateLUT()");
  switch (voxelType)
  {
    case VV_RGBA:
      makeTextures();// this mode doesn't use a hardware LUT, so every voxel has to be updated
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
      glBindTexture(GL_TEXTURE_2D, pixLUTName);
      if(usePreIntegration)
      {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, getPreintTableSize(), getPreintTableSize(), 0,
            GL_RGBA, GL_UNSIGNED_BYTE, preintTable);
      }
      else
      {
         glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, lutSize[0], lutSize[1], 0,
               GL_RGBA, GL_UNSIGNED_BYTE, rgbaLUT);
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
void vvTexRend::setViewingDirection(const vvVector3& vd)
{
  vvDebugMsg::msg(3, "vvTexRend::setViewingDirection()");
  viewDir = vd;
}

//----------------------------------------------------------------------------
/** Set the direction from the viewer to the object.
  This information is needed to correctly orientate the texture slices
  in 3D texturing mode if the viewer is outside of the volume.
  @param vd  object direction in object coordinates
*/
void vvTexRend::setObjectDirection(const vvVector3& od)
{
  vvDebugMsg::msg(3, "vvTexRend::setObjectDirection()");
  objDir = od;
}

//----------------------------------------------------------------------------
// see parent
void vvTexRend::setParameter(ParameterType param, const vvParam& newValue)
{
  bool newInterpol;

  vvDebugMsg::msg(3, "vvTexRend::setParameter()");
  switch (param)
  {
    case vvRenderer::VV_GAMMA:
      // fall-through
    case vvRenderer::VV_GAMMA_CORRECTION:
      vvRenderer::setParameter(param, newValue);
      updateTransferFunction();
      break;
    case vvRenderer::VV_SLICEINT:
      newInterpol = newValue;
      if (interpolation!=newInterpol)
      {
        interpolation = newInterpol;
        makeTextures();
        updateTransferFunction();
      }
      break;
    case vvRenderer::VV_MIN_SLICE:
      minSlice = newValue;
      break;
    case vvRenderer::VV_MAX_SLICE:
      maxSlice = newValue;
      break;
    case vvRenderer::VV_OPCORR:
      opacityCorrection = newValue;
      break;
    case vvRenderer::VV_SLICEORIENT:
      _sliceOrientation = (SliceOrientation)newValue.asInt();
      break;
    case vvRenderer::VV_PREINT:
      preIntegration = newValue;
      updateTransferFunction();
      disableShader(_shader);
      delete _shader;
      _shader = initShader();
      break;
    case vvRenderer::VV_BINNING:
      vd->_binning = (vvVolDesc::BinningType)newValue.asInt();
      break;
    case vvRenderer::VV_ISECT_TYPE:
      _isectType = (IsectType)newValue.asInt();
      if ((_isectType != CPU) && (!_proxyGeometryOnGpuSupported || geomType != VV_BRICKS))
      {
        cerr << "Cannot generate proxy geometry on GPU" << std::endl;
        _isectType = CPU;
      }

      // need to set these up anew
      _elemCounts.clear();
      _vertIndices.clear();
      _vertIndicesAll.clear();
      _vertArray.clear();

      disableShader(_shader);
      delete _shader;
      _shader = initShader();
      break;
    case vvRenderer::VV_LEAPEMPTY:
      _emptySpaceLeaping = newValue;
      // Maybe a tf type was chosen which is incompatible with empty space leaping.
      validateEmptySpaceLeaping();
      updateTransferFunction();
      break;
    case vvRenderer::VV_OFFSCREENBUFFER:
      _useOffscreenBuffer = newValue;
      if (_useOffscreenBuffer)
      {
        if (_renderTarget == NULL)
        {
          delete _renderTarget;
          _renderTarget = new vvOffscreenBuffer(_imageScale, _imagePrecision);
          if (_opaqueGeometryPresent)
          {
            _renderTarget->setPreserveDepthBuffer(true);
          }
        }
      }
      else
      {
        delete _renderTarget;
        _renderTarget = NULL;
      }
      break;
    case vvRenderer::VV_IMG_SCALE:
      _imageScale = newValue;
      if (_useOffscreenBuffer)
      {
        if (_renderTarget != NULL)
        {
          _renderTarget->setScale(_imageScale);
        }
        else
        {
          delete _renderTarget;
          _renderTarget = new vvOffscreenBuffer(_imageScale, _imagePrecision);
        }
      }
      break;
    case vvRenderer::VV_IMG_PRECISION:
      if (_useOffscreenBuffer)
      {
        if (int(newValue) <= 8)
        {
          if (_renderTarget != NULL)
          {
            _renderTarget->setPrecision(virvo::Byte);
          }
          else
          {
            delete _renderTarget;
            _renderTarget = new vvOffscreenBuffer(_imageScale, virvo::Byte);
          }
          break;
        }
        else if ((int(newValue) > 8) && (int(newValue) < 32))
        {
          if (_renderTarget != NULL)
          {
            _renderTarget->setPrecision(virvo::Short);
          }
          else
          {
            delete _renderTarget;
            _renderTarget = new vvOffscreenBuffer(_imageScale, virvo::Short);
          }
          break;
        }
        else if (int(newValue) >= 32)
        {
          if (_renderTarget != NULL)
          {
            _renderTarget->setPrecision(virvo::Float);
          }
          else
          {
            delete _renderTarget;
            _renderTarget = new vvOffscreenBuffer(_imageScale, virvo::Float);
          }
          break;
        }
      }
      break;
    case vvRenderer::VV_LIGHTING:
      if(geomType != VV_SLICES && geomType != VV_CUBIC2D)
      {
        if (newValue.asBool())
        {
          _previousShader = _currentShader;
          _currentShader = 12;
        }
        else
        {
          _currentShader = _previousShader;
        }
        disableShader(_shader);
        delete _shader;
        _shader = initShader();
      }
      break;
    case vvRenderer::VV_MEASURETIME:
      _measureRenderTime = newValue;
      break;
    case vvRenderer::VV_PIX_SHADER:
      setCurrentShader(newValue);
      break;
    case vvRenderer::VV_PADDING_REGION:
      vvRenderer::setParameter(param, newValue);
      makeTextures();
      break;
    default:
      vvRenderer::setParameter(param, newValue);
      break;
  }
}

//----------------------------------------------------------------------------
// see parent for comments
vvParam vvTexRend::getParameter(ParameterType param) const
{
  vvDebugMsg::msg(3, "vvTexRend::getParameter()");

  switch (param)
  {
    case vvRenderer::VV_SLICEINT:
      return interpolation;
    case vvRenderer::VV_MIN_SLICE:
      return minSlice;
    case vvRenderer::VV_MAX_SLICE:
      return maxSlice;
    case vvRenderer::VV_SLICEORIENT:
      return (int)_sliceOrientation;
    case vvRenderer::VV_PREINT:
      return preIntegration;
    case vvRenderer::VV_BINNING:
      return (int)vd->_binning;
    case vvRenderer::VV_PIX_SHADER:
      return getCurrentShader();
    default:
      return vvRenderer::getParameter(param);
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
      {
        return (vvShaderFactory::isSupported("cg")
          || vvShaderFactory::isSupported("glsl"));
      }
    case VV_FRG_PRG:
      return vvGLTools::isGLextensionSupported("GL_ARB_fragment_program");
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
    _currentShader = 0;
  else
    _currentShader = shader;

  disableShader(_shader);
  delete _shader;
  _shader = initShader();
}

//----------------------------------------------------------------------------
/// inherited from vvRenderer, only valid for planar textures
void vvTexRend::renderQualityDisplay()
{
  const int numSlices = int(_quality * 100.0f);
  vvPrintGL* printGL = new vvPrintGL();
  vvVector4 clearColor = vvGLTools::queryClearColor();
  vvVector4 fontColor = vvVector4(1.0f - clearColor[0], 1.0f - clearColor[1], 1.0f - clearColor[2], 1.0f);
  printGL->setFontColor(fontColor);
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
void vvTexRend::enableShader(vvShaderProgram* shader, GLuint lutName)
{
  vvGLTools::printGLError("Enter vvTexRend::enablePixelShaders()");

  if(!shader)
    return;

  shader->enable();

  if(VV_PIX_SHD == voxelType)
  {
    shader->setParameterTex2D("pixLUT", lutName);

    if (_channel4Color != NULL)
    {
      shader->setParameter3f("chan4color", _channel4Color[0], _channel4Color[1], _channel4Color[2]);
    }
    if (_opacityWeights != NULL)
    {
      shader->setParameter4f("opWeights", _opacityWeights[0], _opacityWeights[1], _opacityWeights[2], _opacityWeights[3]);
    }
  }

  vvGLTools::printGLError("Leaving vvTexRend::enablePixelShaders()");
}

//----------------------------------------------------------------------------
void vvTexRend::disableShader(vvShaderProgram* shader) const
{
  if (shader)
  {
    shader->disable();
  }
}

void vvTexRend::initClassificationStage(GLuint *pixLUTName, GLuint progName[VV_FRAG_PROG_MAX]) const
{
  if(voxelType==VV_TEX_SHD || voxelType==VV_PIX_SHD || voxelType==VV_FRG_PRG)
  {
    glGenTextures(1, pixLUTName);
  }

  if (voxelType == VV_FRG_PRG)
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
      (GLsizei)strlen(fragProgString2D),
      fragProgString2D);

    const char fragProgString3D[] = "!!ARBfp1.0\n"
      "TEMP temp;\n"
      "TEX  temp, fragment.texcoord[0], texture[0], 3D;\n"
      "TEX  result.color, temp, texture[1], 2D;\n"
      "END\n";
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, progName[VV_FRAG_PROG_3D]);
    glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB,
      GL_PROGRAM_FORMAT_ASCII_ARB,
      (GLsizei)strlen(fragProgString3D),
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
      (GLsizei)strlen(fragProgStringPreint),
      fragProgStringPreint);
  }
}

void vvTexRend::freeClassificationStage(GLuint pixLUTName, GLuint progname[VV_FRAG_PROG_MAX]) const
{
  if (voxelType==VV_FRG_PRG)
  {
    glDeleteProgramsARB(VV_FRAG_PROG_MAX, progname);
  }
  if (voxelType==VV_FRG_PRG || voxelType==VV_TEX_SHD || voxelType==VV_PIX_SHD)
  {
    glDeleteTextures(1, &pixLUTName);
  }
}


//----------------------------------------------------------------------------
/** @return Pointer of initialized ShaderProgram or NULL
 */
vvShaderProgram* vvTexRend::initShader()
{
  vvGLTools::printGLError("Enter vvTexRend::initShader()");

  std::ostringstream fragName;
  if(voxelType == VV_PIX_SHD)
  {
    fragName << "shader" << std::setw(2) << std::setfill('0') << (_currentShader+1);
  }

  vvShaderProgram* shader = NULL;
  std::string vertName;
  std::string geoName;
  if (_isectType == VERT_SHADER_ONLY)
  {
    vertName = "isect_vert_only";
    geoName = "";
    shader = _shaderFactory->createProgram(vertName.c_str(), geoName.c_str(), fragName.str());
  }
  else if (_isectType == GEOM_SHADER_ONLY)
  {
    vertName = "isect_geom_only";
    geoName = "isect_geom_only";
    vvShaderProgram::GeoShaderArgs args;
    args.inputType = vvShaderProgram::VV_POINTS;
    args.outputType = vvShaderProgram::VV_TRIANGLE_STRIP;
    args.numOutputVertices = 6;
    shader = _shaderFactory->createProgram(vertName.c_str(), geoName.c_str(), fragName.str(), args);
  }
  else if(_isectType == VERT_GEOM_COMBINED)
  {
    vertName = "isect_vert_geom_combined";
    geoName = "isect_vert_geom_combined";
    vvShaderProgram::GeoShaderArgs args;
    args.inputType = vvShaderProgram::VV_TRIANGLES;
    args.outputType = vvShaderProgram::VV_TRIANGLE_STRIP;
    args.numOutputVertices = 6;
    shader = _shaderFactory->createProgram(vertName.c_str(), geoName.c_str(), fragName.str(), args);
  }

  if (!shader && _isectType != CPU)
  {
    vvDebugMsg::msg(0, "Cannot load shader, falling back to CPU proxy geometry");
    _isectType = CPU;
  }

  if (_isectType != CPU)
  {
    setupIntersectionParameters(shader);
  }
  
  if(!shader)
  {
    // intersection on CPU, try to create fragment program
    shader = _shaderFactory->createProgram("", "", fragName.str());
  }

  vvGLTools::printGLError("Leave vvTexRend::initShader()");

  return shader;
}

void vvTexRend::setupIntersectionParameters(vvShaderProgram* shader)
{
  vvGLTools::printGLError("Enter vvTexRend::setupIntersectionParameters()");

  if(shader)
  {
    shader->enable();
  }
  else
  {
    cerr << "invalid isectShader!!" << endl;
    return;
  }

  // Global scope, values will never be changed.

  if (_isectType == VERT_SHADER_ONLY)
  {
    int v1[24] = { 0, 1, 2, 7,
                   0, 1, 4, 7,
                   0, 5, 4, 7,
                   0, 5, 6, 7,
                   0, 3, 6, 7,
                   0, 3, 2, 7 };
    shader->setParameterArray1i("v1", v1, 24);
  }
  else if (_isectType == GEOM_SHADER_ONLY)
  {
    int v1[24] = { 0, 1, 4, 7,
                   0, 1, 2, 7,
                   0, 5, 4, 7,
                   0, 3, 2, 7,
                   0, 5, 6, 7,
                   0, 3, 6, 7 };
    shader->setParameterArray1i("v1", v1, 24);
  }
  else if (_isectType == VERT_GEOM_COMBINED)
  {
    int v1[9] = { 0, 1, 2,
                  0, 5, 4,
                  0, 3, 6 };

    int v2[9] = { 1, 2, 7,
                  5, 4, 7,
                  3, 6, 7 };

    shader->setParameterArray1i("v1", v1, 9);
    shader->setParameterArray1i("v2", v2, 9);
  }
  shader->disable();

  vvGLTools::printGLError("Leaving vvTexRend::setupIntersectionParameters()");
}

//----------------------------------------------------------------------------
void vvTexRend::printLUT() const
{
  vvVector3i lutEntries;

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

  size = vd->getSize();
  for (i = 0; i < 3; ++i)
    size2[i]   = 0.5f * size[i];

  for (j = 0; j < 4; j++)
    for (k = 0; k < 3; k++)
  {
    texcoord[j][k] = (points[j][k] + size2[k]) / size[k];
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
        case VV_TEX_SHD:
        case VV_PIX_SHD:
          result[index] = data[texelsize*index];
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

//----------------------------------------------------------------------------
/** Get the quality to adjust numSlices. If the render target is an offscreen
    buffer, that one gets scaled and the quality return value is adjusted since
    part of the quality reduction is accomodated through image scaling.
    @return The quality.
*/
float vvTexRend::calcQualityAndScaleImage()
{
  float quality = _quality;
  if (quality < 1.0f)
  {
    if (_renderTarget != NULL)
    {
      quality = powf(quality, 1.0f/3.0f);
      _renderTarget->setScale(quality);
    }
  }
  return quality;
}

void vvTexRend::initVertArray(const int numSlices)
{
  if (_isectType == CPU)
  {
    return;
  }

  if(static_cast<int>(_elemCounts.size()) >= numSlices)
    return;

  _elemCounts.resize(numSlices);
  _vertIndices.resize(numSlices);

  if (_isectType == VERT_SHADER_ONLY)
  {
    _vertIndicesAll.resize(numSlices*6);
    _vertArray.resize(numSlices*12);
  }
  else if (_isectType == GEOM_SHADER_ONLY)
  {
    _vertIndicesAll.resize(numSlices);
    _vertArray.resize(numSlices*2);
  }
  else if (_isectType == VERT_GEOM_COMBINED)
  {
    _vertIndicesAll.resize(numSlices*3);
    _vertArray.resize(numSlices*6);
  }

  int idxIterator = 0;
  int vertIterator = 0;

  // Spare some instructions in shader:
  int mul = 4; // ==> x-values: 0, 4, 8, 12, 16, 20 instead of 0, 1, 2, 3, 4, 5
  if (_isectType == GEOM_SHADER_ONLY)
  {
    mul = 1;
  }
  else if (_isectType == VERT_GEOM_COMBINED)
  {
    mul = 3; // ==> x-values: 0, 3, 6 instead of 0, 1, 2
  }

  for (int i = 0; i < numSlices; ++i)
  {
    if (_isectType == VERT_SHADER_ONLY)
    {
      _elemCounts[i] = 6;
    }
    else if (_isectType == GEOM_SHADER_ONLY)
    {
      _elemCounts[i] = 1;
    }
    else if (_isectType == VERT_GEOM_COMBINED)
    {
      _elemCounts[i] = 3;
    }
    
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
  if (_emptySpaceLeaping == true)
  {
    _emptySpaceLeaping &= (geomType == VV_BRICKS);
    _emptySpaceLeaping &= (voxelType != VV_PIX_SHD) || (_currentShader == 0) || (_currentShader == 12);
    _emptySpaceLeaping &= (voxelType != VV_RGBA);
    // TODO: only implemented for 8-bit 1-channel volumes. Support higher volume resolutions.
    _emptySpaceLeaping &= ((vd->chan == 1) && (vd->bpc == 1));
  }
}

void vvTexRend::evaluateLocalIllumination(vvShaderProgram* shader, const vvVector3& normal)
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
    shader->setParameter3f("L", L[0], L[1], L[2]);
    shader->setParameter3f("H", H[0], H[1], H[2]);
  }
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
