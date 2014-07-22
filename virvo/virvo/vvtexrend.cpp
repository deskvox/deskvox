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

#include "vvvecmath.h"
#include "vvdebugmsg.h"
#include "vvtoolshed.h"
#include "vvsphere.h"
#include "vvtexrend.h"
#include "vvprintgl.h"
#include "vvshaderfactory.h"
#include "vvshaderprogram.h"
#include "vvvoldesc.h"
#include "vvpthread.h"

#include "gl/util.h"

#include "private/vvgltools.h"
#include "private/vvlog.h"

using namespace std;

namespace gl = virvo::gl;

using virvo::mat4;
using virvo::vec3f;
using virvo::vec3;
using virvo::vec4f;
using virvo::vec4i;
using virvo::vec4;

using virvo::PixelFormat;


//----------------------------------------------------------------------------
const int vvTexRend::NUM_PIXEL_SHADERS = 13;

static virvo::BufferPrecision mapBitsToBufferPrecision(int bits)
{
    switch (bits)
    {
    case 8:
        return virvo::Byte;
    case 16:
        return virvo::Short;
    case 32:
        return virvo::Float;
    default:
        assert(!"unknown bit size");
        return virvo::Byte;
    }
}

static PixelFormat mapBufferPrecisionToFormat(virvo::BufferPrecision bp)
{
    switch (bp)
    {
    case virvo::Byte:
        return virvo::PF_RGBA8;
    case virvo::Short:
        return virvo::PF_RGBA16F;
    case virvo::Float:
        return virvo::PF_RGBA32F;
    default:
        assert(!"unknown format");
        return virvo::PF_UNSPECIFIED;
    }
}

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
{
  vvDebugMsg::msg(1, "vvTexRend::vvTexRend()");

  glewInit();

  if (this->_useOffscreenBuffer)
    setRenderTarget( virvo::FramebufferObjectRT::create( mapBufferPrecisionToFormat(this->_imagePrecision), virvo::PF_DEPTH24_STENCIL8) );

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
  minSlice = maxSlice = -1;
  rgbaTF  = new float[256 * 256 * 4];
  rgbaLUT = new uint8_t[256 * 256 * 4];
  preintTable = new uint8_t[getPreintTableSize()*getPreintTableSize()*4];
  usePreIntegration = false;
  textures = 0;

  _currentShader = vd->chan - 1;
  _previousShader = _currentShader;

  _lastFrame = std::numeric_limits<size_t>::max();
  lutDistance = -1.0;
  _isROIChanged = true;

  // Find out which OpenGL extensions are supported:
  extTex3d  = vvGLTools::isGLextensionSupported("GL_EXT_texture3D") || vvGLTools::isGLVersionSupported(1,2,1);
  arbMltTex = vvGLTools::isGLextensionSupported("GL_ARB_multitexture") || vvGLTools::isGLVersionSupported(1,3,0);

  _shader = NULL;

  extMinMax = vvGLTools::isGLextensionSupported("GL_EXT_blend_minmax") || vvGLTools::isGLVersionSupported(1,4,0);
  extBlendEquation = vvGLTools::isGLextensionSupported("GL_EXT_blend_equation") || vvGLTools::isGLVersionSupported(1,1,0);
  extPalTex  = isSupported(VV_PAL_TEX);
  extTexShd  = isSupported(VV_TEX_SHD);
  extPixShd  = isSupported(VV_PIX_SHD);
  arbFrgPrg  = isSupported(VV_FRG_PRG);

  extNonPower2 = vvGLTools::isGLextensionSupported("GL_ARB_texture_non_power_of_two") || vvGLTools::isGLVersionSupported(2,0,0);

  // Determine best rendering algorithm for current hardware:
  setVoxelType(findBestVoxelType(vox));
  geomType  = findBestGeometry(geom, voxelType);

  _proxyGeometryOnGpuSupported = vvGLTools::isGLVersionSupported(2,0,0);
  if(geomType==VV_SLICES || geomType==VV_CUBIC2D)
  {
    _currentShader = 9;
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
    case VV_SPHERICAL: cerr << "VV_SPHERICAL"; break;
    default: assert(0); break;
  }
  cerr << ", ";
  switch(voxelType)
  {
    case VV_RGBA:    cerr << "VV_RGBA";    break;
    case VV_PAL_TEX: cerr << "VV_PAL_TEX"; break;
    case VV_TEX_SHD: cerr << "VV_TEX_SHD"; break;
    case VV_PIX_SHD: cerr << "VV_PIX_SHD"; break;
    case VV_FRG_PRG: cerr << "VV_FRG_PRG"; break;
    default: assert(0); break;
  }
  cerr << endl;

  textures = 0;

  if (voxelType != VV_RGBA)
  {
    makeTextures(true);      // we only have to do this once for non-RGBA textures
  }
  updateTransferFunction();
}

//----------------------------------------------------------------------------
/// Destructor
vvTexRend::~vvTexRend()
{
  vvDebugMsg::msg(1, "vvTexRend::~vvTexRend()");

  freeClassificationStage(pixLUTName, fragProgName);
  removeTextures();

  delete[] rgbaTF;
  delete[] rgbaLUT;
  delete _shader;
  _shader = NULL;

  delete[] preintTable;
}


void vvTexRend::setVolDesc(vvVolDesc* vd)
{
  vvRenderer::setVolDesc(vd);
  makeTextures(true);
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
      else if (vd->chan == 2)
      {
        texelsize=2;
        internalTexFormat = GL_LUMINANCE_ALPHA;
        texFormat = GL_LUMINANCE_ALPHA;
      }
      else if (vd->chan == 3)
      {
        texelsize=3;
        internalTexFormat = GL_RGB;
        texFormat = GL_RGB;
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
    if (!extTex3d && (geom==VV_VIEWPORT || geom==VV_SPHERICAL))
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
vvTexRend::ErrorType vvTexRend::makeTextures(bool newTex)
{
  ErrorType err = OK;

  vvDebugMsg::msg(2, "vvTexRend::makeTextures()");

  vvssize3 vox = _paddingRegion.getMax() - _paddingRegion.getMin();
  for (size_t i = 0; i < 3; ++i)
  {
    vox[i] = std::min(vox[i], vd->vox[i]);
  }

  if (vox[0] == 0 || vox[1] == 0 || vox[2] == 0)
    return err;

  // Compute texture dimensions (perhaps must be power of 2):
  texels[0] = getTextureSize(vox[0]);
  texels[1] = getTextureSize(vox[1]);
  texels[2] = getTextureSize(vox[2]);

  switch (geomType)
  {
    case VV_SLICES:  err=makeTextures2D(1); updateTextures2D(1, 0, 10, 20, 15, 10, 5); break;
    case VV_CUBIC2D: err=makeTextures2D(3); updateTextures2D(3, 0, 10, 20, 15, 10, 5); break;
    default: updateTextures3D(0, 0, 0, texels[0], texels[1], texels[2], newTex); break;
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
  vvsize3 size;

  vvGLTools::printGLError("enter makeLUTTexture");
  if(voxelType!=VV_PIX_SHD)
     glActiveTextureARB(GL_TEXTURE1_ARB);
  getLUTSize(size);
  glBindTexture(GL_TEXTURE_2D, pixLUTName);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size[0], size[1], 0,
    GL_RGBA, GL_UNSIGNED_BYTE, rgbaLUT);
  if(voxelType!=VV_PIX_SHD)
     glActiveTextureARB(GL_TEXTURE0_ARB);
  vvGLTools::printGLError("leave makeLUTTexture");
}

//----------------------------------------------------------------------------
/// Generate 2D textures for cubic 2D mode.
vvTexRend::ErrorType vvTexRend::makeTextures2D(size_t axes)
{
  GLint glWidth;                                  // return value from OpenGL call
  uint8_t* rgbaSlice[3];                          // RGBA slice data for texture memory for each principal axis
  vec4i rawVal;                                   // raw values for R,G,B,A
  size_t rawSliceSize;                            // number of bytes in a slice of the raw data array
  size_t rawLineSize;                             // number of bytes in a row of the raw data array
  vvsize3 texSize;                                // size of a 2D texture in bytes for each principal axis
  uint8_t* raw;                                   // raw volume data
  size_t texIndex=0;                              // index of current texture
  size_t texSliceIndex;                           // index of current voxel in texture
  vvsize3 tw, th;                                 // current texture width and height for each principal axis
  vvsize3 rw, rh, rs;                             // raw data width, height, slices for each principal axis
  vvsize3 rawStart;                               // starting offset into raw data, for each principal axis
  vvVector3i rawStepW;                            // raw data step size for texture row, for each principal axis
  vvVector3i rawStepH;                            // raw data step size for texture column, for each principal axis
  vvVector3i rawStepS;                            // raw data step size for texture slices, for each principal axis
  uint8_t* rawVoxel;                              // current raw data voxel
  bool accommodated = true;                       // false if a texture cannot be accommodated in TRAM
  ErrorType err = OK;

  vvDebugMsg::msg(1, "vvTexRend::makeTextures2D()");

  assert(axes==1 || axes==3);

  removeTextures();                                // first remove previously generated textures from TRAM

  size_t frames = vd->frames;
  
  // Determine total number of textures:
  if (axes==1)
  {
    textures = vd->vox[2] * frames;
  }
  else
  {
    textures = (vd->vox[0] + vd->vox[1] + vd->vox[2]) * frames;
  }

  VV_LOG(1) << "Total number of 2D textures:    " << textures << std::endl;
  VV_LOG(1) << "Total size of 2D textures [KB]: " << frames * axes * texels[0] * texels[1] * texels[2] * texelsize / 1024;

  // Generate texture names:
  assert(texNames==NULL);
  texNames = new GLuint[textures];
  glGenTextures(textures, texNames);

  // Initialize texture sizes:
  th[1] = tw[2] = texels[0];
  tw[0] = th[2] = texels[1];
  tw[1] = th[0] = texels[2];
  for (size_t i=3-axes; i<3; ++i)
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
  for (size_t i=3-axes; i<3; ++i)
  {
    rgbaSlice[i] = new uint8_t[texSize[i]];
  }

  // Generate texture data:
  for (size_t f=0; f<frames; ++f)
  {
    raw = vd->getRaw(f);                          // points to beginning of frame in raw data
    for (size_t i=3-axes; i<3; ++i)                      // generate textures for each principal axis
    {
      memset(rgbaSlice[i], 0, texSize[i]);        // initialize with 0's for invisible empty regions

      // Generate texture contents:
      for (size_t s=0; s<rs[i]; ++s)                 // loop thru texture and raw data slices
      {
        for (size_t h=0; h<rh[i]; ++h)               // loop thru raw data rows
        {
          // Set voxel to starting position in raw data array:
          rawVoxel = raw + rawStart[i] + s * rawStepS[i] + h * rawStepH[i];

          for (size_t w=0; w<rw[i]; ++w)             // loop thru raw data columns
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
                rawVal[0] = *((uint16_t*)(rawVoxel));
                rawVal[0] >>= 8;
              }
              else                                // float voxels
              {
                const float fval = *((float*)(rawVoxel));
                rawVal[0] = vd->mapFloat2Int(fval);
              }
              switch(voxelType)
              {
                case VV_PAL_TEX:
                case VV_FRG_PRG:
                  rgbaSlice[i][texSliceIndex] = (uint8_t)rawVal[0];
                  break;
                case VV_PIX_SHD:
                  for (size_t c=0; c<ts_min(size_t(4), vd->chan); ++c)
                  {
                    rgbaSlice[i][texSliceIndex + c]   = (uint8_t)rawVal[0];
                  }
                  break;
                case VV_TEX_SHD:
                  for (size_t c=0; c<4; ++c)
                  {
                    rgbaSlice[i][texSliceIndex + c]   = (uint8_t)rawVal[0];
                  }
                  break;
                case VV_RGBA:
                  for (size_t c=0; c<4; ++c)
                  {
                    rgbaSlice[i][texSliceIndex + c] = rgbaLUT[size_t(rawVal[0]) * 4 + c];
                  }
                  break;
                default: assert(0); break;
              }
            }
            else if (vd->bpc==1)                  // for up to 4 channels
            {
              //XXX: das in Ordnung bringen fuer 2D-Texturen mit LUT
              // Fetch component values from memory:
              for (size_t c=0; c<ts_min(vd->chan, size_t(4)); ++c)
              {
                rawVal[c] = *(rawVoxel + c);
              }

              // Copy color components:
              for (size_t c=0; c<ts_min(vd->chan, size_t(3)); ++c)
              {
                rgbaSlice[i][texSliceIndex + c] = (uint8_t)rawVal[c];
              }

              // Alpha channel:
              if (vd->chan>=4)                    // RGBA?
              {
                rgbaSlice[i][texSliceIndex + 3] = rgbaLUT[rawVal[3] * 4 + 3];
              }
              else                                // compute alpha from color components
              {
                size_t alpha = 0;
                for (size_t c=0; c<vd->chan; ++c)
                {
                  // Alpha: mean of sum of RGB conversion table results:
                  alpha += (size_t)rgbaLUT[size_t(rawVal[c]) * 4 + c];
                }
                rgbaSlice[i][texSliceIndex + 3] = (uint8_t)(alpha / vd->chan);
              }
            }
            rawVoxel += rawStepW[i];
          }
        }
        glBindTexture(GL_TEXTURE_2D, texNames[texIndex]);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
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

void vvTexRend::setTexMemorySize(size_t newSize)
{
  if (_texMemorySize == newSize)
    return;

  _texMemorySize = newSize;
}

size_t vvTexRend::getTexMemorySize() const
{
  return _texMemorySize;
}

//----------------------------------------------------------------------------
/// Update transfer function from volume description.
void vvTexRend::updateTransferFunction()
{
  vvsize3 size;

  vvDebugMsg::msg(1, "vvTexRend::updateTransferFunction()");
  if (_preIntegration &&
      arbMltTex && 
      geomType==VV_VIEWPORT && 
      !(_clipMode == 1 && (_clipSingleSlice || _clipOpaque)) &&
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
}

//----------------------------------------------------------------------------
// see parent in vvRenderer
void vvTexRend::updateVolumeData()
{
  vvRenderer::updateVolumeData();

  makeTextures(true);
}

//----------------------------------------------------------------------------
void vvTexRend::updateVolumeData(size_t offsetX, size_t offsetY, size_t offsetZ,
                                 size_t sizeX, size_t sizeY, size_t sizeZ)
{
  switch (geomType)
  {
    case VV_VIEWPORT:
      updateTextures3D(offsetX, offsetY, offsetZ, sizeX, sizeY, sizeZ, false);
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
vvTexRend::ErrorType vvTexRend::updateTextures3D(ssize_t offsetX, ssize_t offsetY, ssize_t offsetZ,
                                                 ssize_t sizeX, ssize_t sizeY, ssize_t sizeZ, bool newTex)
{
  ErrorType err = OK;
  size_t srcIndex;
  size_t texOffset=0;
  vec4i rawVal;
  uint8_t* texData = NULL;
  bool accommodated = true;
  GLint glWidth;

  vvDebugMsg::msg(1, "vvTexRend::updateTextures3D()");

  if (!extTex3d) return NO3DTEX;

  size_t texSize = sizeX * sizeY * sizeZ * texelsize;
  VV_LOG(1) << "3D Texture width     = " << sizeX << std::endl;
  VV_LOG(1) << "3D Texture height    = " << sizeY << std::endl;
  VV_LOG(1) << "3D Texture depth     = " << sizeZ << std::endl;
  VV_LOG(1) << "3D Texture size (KB) = " << texSize / 1024 << std::endl;

  size_t sliceSize = vd->getSliceBytes();

  if (vd->frames != textures)
    newTex = true;

  if (newTex)
  {
    VV_LOG(2) << "Creating texture names. # of names: " << vd->frames << std::endl;

    removeTextures();
    textures  = vd->frames;
    delete[] texNames;
    texNames = new GLuint[textures];
    glGenTextures(vd->frames, texNames);
  }

  VV_LOG(2) << "Transferring textures to TRAM. Total size [KB]: " << vd->frames * texSize / 1024 << std::endl;

  vvssize3 offsets(offsetX, offsetY, offsetZ);
  offsets += _paddingRegion.getMin();

  bool useRaw = geomType!=VV_SPHERICAL && vd->bpc==1 && vd->chan<=4 && vd->chan==texelsize;
  if (sizeX != vd->vox[0])
    useRaw = false;
  if (sizeY != vd->vox[1])
    useRaw = false;
  if (sizeZ != vd->vox[2])
    useRaw = false;
  for (int i=0; i<3; ++i) {
    if (offsets[i] != 0)
      useRaw = false;
  }
  if (!useRaw)
  {
    texData = new uint8_t[texSize];
    memset(texData, 0, texSize);
  }

  // Generate sub texture contents:
  for (size_t f = 0; f < vd->frames; f++)
  {
    uint8_t *raw = vd->getRaw(f);
    if (useRaw) {
      texData = raw;
    }
    else
    {
      for (ssize_t s = offsets[2]; s < (offsets[2] + sizeZ); s++)
      {
        size_t rawSliceOffset = (ts_min(ts_max(s,ssize_t(0)),vd->vox[2]-1)) * sliceSize;
        for (ssize_t y = offsets[1]; y < (offsets[1] + sizeY); y++)
        {
          size_t heightOffset = (ts_min(ts_max(y,ssize_t(0)),vd->vox[1]-1)) * vd->vox[0] * vd->bpc * vd->chan;
          size_t texLineOffset = (y - offsets[1] - offsetY) * sizeX + (s - offsets[2] - offsetZ) * sizeX * sizeY;
          
          if (vd->chan == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
          {
            if (vd->bpc == 1 && texelsize == 1)
            {
              // one byte, one color channel ==> can use memcpy for consecutive memory chunks
              ssize_t x1 = offsets[0];
              ssize_t x2 = offsets[0] + sizeX;
              size_t srcMin = vd->bpc * min(x1, vd->vox[0] - 1) + rawSliceOffset + heightOffset;
              size_t srcMax = vd->bpc * min(x2, vd->vox[0] - 1) + rawSliceOffset + heightOffset;
              texOffset = texLineOffset - offsetX;
              memcpy(&texData[texelsize * texOffset], &raw[srcMin], srcMax - srcMin);
            }
            else
            {
              for (ssize_t x = offsets[0]; x < (offsets[0] + sizeX); x++)
              {
                srcIndex = vd->bpc * min(x,vd->vox[0]-1) + rawSliceOffset + heightOffset;
                if (vd->bpc == 1) rawVal[0] = int(raw[srcIndex]);
                else if (vd->bpc == 2)
                {
                  rawVal[0] = *(uint16_t*)(raw+srcIndex);
                  rawVal[0] >>= 8;
                }
                else // vd->bpc==4: convert floating point to 8bit value
                {
                  const float fval = *((float*)(raw + srcIndex));      // fetch floating point data value
                  rawVal[0] = vd->mapFloat2Int(fval);
                }
                texOffset = (x - offsets[0] - offsetX) + texLineOffset;
                switch(voxelType)
                {
                case VV_PAL_TEX:
                case VV_FRG_PRG:
                case VV_PIX_SHD:
                  texData[texelsize * texOffset] = (uint8_t) rawVal[0];
                  break;
                case VV_TEX_SHD:
                  for (size_t c = 0; c < 4; c++)
                  {
                    texData[4 * texOffset + c] = (uint8_t) rawVal[0];
                  }
                  break;
                case VV_RGBA:
                  for (size_t c = 0; c < 4; c++)
                  {
                    texData[4 * texOffset + c] = rgbaLUT[size_t(rawVal[0]) * 4 + c];
                  }
                  break;
                default:
                  assert(0);
                  break;
                }
              }
            }
          }
          else if (vd->bpc==1 || vd->bpc==2 || vd->bpc==4)
          {
            if (voxelType == VV_RGBA || voxelType == VV_PIX_SHD)
            {
              for (ssize_t x = offsets[0]; x < (offsets[0] + sizeX); x++)
              {
                texOffset = (x - offsets[0] - offsetX) + texLineOffset;
                for (size_t c = 0; c < ts_min(vd->chan, size_t(4)); c++)
                {
                  srcIndex = vd->bpc * (min(x,vd->vox[0]-1)*vd->chan+c) + rawSliceOffset + heightOffset;
                  if (vd->bpc == 1)
                    rawVal[c] = (int) raw[srcIndex];
                  else if (vd->bpc == 2)
                  {
                    rawVal[c] = *((uint16_t *)(raw + srcIndex));
                    rawVal[c] >>= 8;
                  }
                  else  // vd->bpc == 4
                  {
                    const float fval = *((float*)(raw + srcIndex));      // fetch floating point data value
                    rawVal[c] = vd->mapFloat2Int(fval);
                  }
                }

                // Copy color components:
                for (size_t c = 0; c < ts_min(vd->chan, size_t(3)); c++)
                {
                  texData[4 * texOffset + c] = (uint8_t) rawVal[c];
                }
              }

              // Alpha channel:
              if (vd->chan >= 4)
              {
                texData[4 * texOffset + 3] = (uint8_t)rawVal[3];
              }
              else
              {
                size_t alpha = 0;
                for (size_t c = 0; c < vd->chan; c++)
                {
                  // Alpha: mean of sum of RGB conversion table results:
                  alpha += (size_t) rgbaLUT[size_t(rawVal[c]) * 4 + c];
                }
                texData[4 * texOffset + 3] = (uint8_t) (alpha / vd->chan);
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
        for (ssize_t s = offsetZ; s < (offsetZ + sizeZ); ++s)
        {
          for (ssize_t y = offsetY; y < (offsetY + sizeY); ++y)
          {
            for (ssize_t x = offsetX; x < (offsetX + sizeX); ++x)
            {
              if ((s == 0) || (s>=vd->vox[2]-1) ||
                  (y == 0) || (y>=vd->vox[1]-1) ||
                  (x == 0) || (x>=vd->vox[0]-1))
              {
                texOffset = x + y * texels[0] + s * texels[0] * texels[1];
                for(size_t i=0; i<texelsize; i++)
                  texData[texelsize*texOffset+i] = 0;
              }
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
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP);

      glTexImage3D(GL_PROXY_TEXTURE_3D_EXT, 0, internalTexFormat,
        texels[0], texels[1], texels[2], 0, texFormat, GL_UNSIGNED_BYTE, NULL);
      glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D_EXT, 0, GL_TEXTURE_WIDTH, &glWidth);

      if (glWidth==texels[0])
      {
        glTexImage3D(GL_TEXTURE_3D_EXT, 0, internalTexFormat, texels[0], texels[1], texels[2], 0,
          texFormat, GL_UNSIGNED_BYTE, texData);
      }
      else
      {
        accommodated = false;
        vvGLTools::printGLError("Tried to accomodate 3D textures");

        cerr << "Insufficient texture memory for 3D texture(s)." << endl;
        err = TRAM_ERROR;
      }
    }
    else
    {
      glBindTexture(GL_TEXTURE_3D_EXT, texNames[f]);
      glTexSubImage3D(GL_TEXTURE_3D_EXT, 0, offsetX, offsetY, offsetZ,
        sizeX, sizeY, sizeZ, texFormat, GL_UNSIGNED_BYTE, texData);
    }
  }

  if (!useRaw)
  {
    delete[] texData;
  }
  return err;
}

vvTexRend::ErrorType vvTexRend::updateTextures2D(size_t axes, ssize_t offsetX, ssize_t offsetY, ssize_t offsetZ,
  ssize_t sizeX, ssize_t sizeY, ssize_t sizeZ)
{
  vec4i rawVal;
  size_t rawSliceSize;
  size_t rawLineSize;
  vvsize3 texSize;
  size_t texIndex = 0;
  size_t texSliceIndex;
  vvsize3 texW, texH;
  vvsize3 tw, th;
  vvsize3 rw, rh, rs;
  vvsize3 sw, sh, ss;
  vvsize3 rawStart;
  vvVector3i rawStepW;
  vvVector3i rawStepH;
  vvVector3i rawStepS;
  uint8_t* rgbaSlice[3];
  uint8_t* raw;
  uint8_t* rawVoxel;

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

  rs[0] = ts_clamp(vd->vox[0] - offsetX, ssize_t(0), vd->vox[0]);
  rw[0] = ts_clamp(offsetY + sizeY, ssize_t(0), vd->vox[1]);
  rh[0] = ts_clamp(offsetZ + sizeZ, ssize_t(0), vd->vox[2]);

  rs[1] = ts_clamp(vd->vox[1] - offsetY, ssize_t(0), vd->vox[1]);
  rh[1] = ts_clamp(offsetZ + sizeZ, ssize_t(0), vd->vox[2]);
  rw[1] = ts_clamp(offsetX + sizeX, ssize_t(0), vd->vox[0]);

  rs[2] = ts_clamp(vd->vox[2] - offsetZ, ssize_t(0), vd->vox[2]);
  rw[2] = ts_clamp(offsetX + sizeX, ssize_t(0), vd->vox[0]);
  rh[2] = ts_clamp(offsetY + sizeY, ssize_t(0), vd->vox[1]);

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

  for (size_t i = 3-axes; i < 3; i++)
  {
    texSize[i] = tw[i] * th[i] * texelsize;
  }

  // generate texture data arrays
  for (size_t i=3-axes; i<3; ++i)
  {
    rgbaSlice[i] = new uint8_t[texSize[i]];
  }

  // generate texture data
  for (size_t f = 0; f < vd->frames; f++)
  {
    raw = vd->getRaw(f);

    for (size_t i = 3-axes; i < 3; i++)
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
      for (size_t s = ss[i]; s < rs[i]; s++)
      {
        for (size_t h = sh[i]; h < rh[i]; h++)
        {
          // set voxel to starting position in raw data array
          rawVoxel = raw + rawStart[i] + s * rawStepS[i] + h * rawStepH[i] + sw[i];

          for (size_t w = sw[i]; w < rw[i]; w++)
          {
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

              for (size_t c = 0; c < texelsize; c++)
                rgbaSlice[i][texSliceIndex + c] = rgbaLUT[size_t(rawVal[0]) * 4 + c];
            }
            else if (vd->bpc == 1)
            {
              // fetch component values from memory
              for (size_t c = 0; c < ts_min(vd->chan, size_t(4)); c++)
                rawVal[c] = *(rawVoxel + c);

              // copy color components
              for (size_t c = 0; c < ts_min(vd->chan, size_t(3)); c++)
                rgbaSlice[i][texSliceIndex + c] = (uint8_t) rawVal[c];

              // alpha channel
              if (vd->chan >= 4)
                rgbaSlice[i][texSliceIndex + 3] = rgbaLUT[rawVal[3] * 4 + 3];
              else
              {
                size_t alpha = 0;
                for (size_t c = 0; c < vd->chan; c++)
                  // alpha: mean of sum of RGB conversion table results
                  alpha += (size_t)rgbaLUT[size_t(rawVal[c]) * 4 + c];

                rgbaSlice[i][texSliceIndex + 3] = (uint8_t)(alpha / vd->chan);
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
    case VV_PAL_TEX:
      if (glsSharedTexPal==(uint8_t)true) glEnable(GL_SHARED_TEXTURE_PALETTE_EXT);
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
void vvTexRend::renderTex3DPlanar(mat4 const& mv)
{
  vec3f vissize, vissize2;                        // full and half object visible sizes
  vvVector3 isect[6];                             // intersection points, maximum of 6 allowed when intersecting a plane and a volume [object space]
  vec3f texcoord[12];                             // intersection points in texture coordinate space [0..1]
  vec3f farthest;                                 // volume vertex farthest from the viewer
  vec3f delta;                                    // distance vector between textures [object space]
  vec3f normal;                                   // normal vector of textures
  vec3f origin;                                   // origin (0|0|0) transformed to object space
  vvVector3 normClipPoint;                        // normalized point on clipping plane
  vec3f clipPosObj;                               // clipping plane position in object space w/o position
  vec3f probePosObj;                              // probe midpoint [object space]
  vec3f probeSizeObj;                             // probe size [object space]
  vec3f probeTexels;                              // number of texels in each probe dimension
  vec3f probeMin, probeMax;                       // probe min and max coordinates [object space]
  vec3f texSize;                                  // size of 3D texture [object space]
  float     maxDist;                              // maximum length of texture drawing path
  size_t    numSlices;

  vvDebugMsg::msg(3, "vvTexRend::renderTex3DPlanar()");

  if (!extTex3d) return;                          // needs 3D texturing extension

  // determine visible size and half object size as shortcut
  vvssize3 minVox = _visibleRegion.getMin();
  vvssize3 maxVox = _visibleRegion.getMax();
  for (size_t i = 0; i < 3; ++i)
  {
    minVox[i] = std::max(minVox[i], ssize_t(0));
    maxVox[i] = std::min(maxVox[i], vd->vox[i]);
  }
  const vvVector3 minCorner = vd->objectCoords(minVox);
  const vvVector3 maxCorner = vd->objectCoords(maxVox);
  vissize = maxCorner - minCorner;
  vec3f center = vvAABB(minCorner, maxCorner).getCenter();

  for (size_t i=0; i<3; ++i)
  {
    texSize[i] = vissize[i] * (float)texels[i] / (float)vd->vox[i];
    vissize2[i]   = 0.5f * vissize[i];
  }
  vec3f pos = vd->pos + center;

  if (_isROIUsed)
  {
    vec3f size = vd->getSize();
    vec3f size2 = size * 0.5f;
    // Convert probe midpoint coordinates to object space w/o position:
    probePosObj = roi_pos_;
    probePosObj -= pos;                        // eliminate object position from probe position

    // Compute probe min/max coordinates in object space:
    probeMin = probePosObj - (roi_size_ * size) * 0.5f;
    probeMax = probePosObj + (roi_size_ * size) * 0.5f;

    // Constrain probe boundaries to volume data area:
    for (size_t i=0; i<3; ++i)
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
    for (size_t i=0; i<3; ++i)
      probeSizeObj[i] = probeMax[i] - probeMin[i];
  }
  else                                            // probe mode off
  {
    probeSizeObj = vd->getSize();
    probeMin = minCorner;
    probeMax = maxCorner;
    probePosObj = center;
  }

  // Initialize texture counters
  if (_isROIUsed)
  {
    probeTexels = vec3f(0.0f, 0.0f, 0.0f);
    for (size_t i=0; i<3; ++i)
    {
      probeTexels[i] = texels[i] * probeSizeObj[i] / texSize[i];
    }
  }
  else                                            // probe mode off
  {
    probeTexels = vec3f( (float)vd->vox[0], (float)vd->vox[1], (float)vd->vox[2] );
  }

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glTranslatef(vd->pos[0], vd->pos[1], vd->pos[2]);

  // Calculate inverted modelview matrix:
  mat4 invMV = inverse(mv);

  // Find eye position (object space):
  vec3f eye = getEyePosition();

  // Get projection matrix:
  vvMatrix pm = gl::getProjectionMatrix();
  bool isOrtho = pm.isProjOrtho();

  getObjNormal(normal, origin, eye, invMV, isOrtho);
  evaluateLocalIllumination(_shader, normal);

  // compute number of slices to draw
  float depth = fabs(normal[0]*probeSizeObj[0]) + fabs(normal[1]*probeSizeObj[1]) + fabs(normal[2]*probeSizeObj[2]);
  size_t minDistanceInd = 0;
  if(probeSizeObj[1]/probeTexels[1] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
    minDistanceInd=1;
  if(probeSizeObj[2]/probeTexels[2] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
    minDistanceInd=2;
  float voxelDistance = probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd];

  float sliceDistance = voxelDistance / _quality;
  if(_isROIUsed && _quality < 2.0)
  {
    // draw at least twice as many slices as there are samples in the probe depth.
    sliceDistance = voxelDistance * 0.5f;
  }
  numSlices = 2*(size_t)ceilf(depth/sliceDistance*.5f);

  if (numSlices < 1)                              // make sure that at least one slice is drawn
    numSlices = 1;
  // don't render an insane amount of slices
  {
    vvssize3 sz = maxVox - minVox;
    ssize_t maxV = ts_max(sz[0], sz[1]);
    maxV = ts_max(maxV, sz[2]);
    ssize_t lim = maxV * 10. * ts_max(_quality, 1.f);
    if (numSlices > lim)
    {
      numSlices = lim;
      VV_LOG(1) << "Limiting number of slices to " << numSlices << std::endl;
    }
  }

  VV_LOG(3) << "Number of textures to render: " << numSlices << std::endl;

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
  delta *= vec3f(sliceDistance);

  // Compute farthest point to draw texture at:
  farthest = delta;
  farthest *= vec3f((float)(numSlices - 1) * -0.5f);
  farthest += probePosObj; // will be vd->pos if no probe present

  if (_clipMode == 1)                     // clipping plane present?
  {
    // Adjust numSlices and set farthest point so that textures are only
    // drawn up to the clipPoint. (Delta may not be changed
    // due to the automatic opacity correction.)
    // First find point on clipping plane which is on a line perpendicular
    // to clipping plane and which traverses the origin:
    vec3f temp = delta * vec3f(-0.5f);
    farthest += temp;                          // add a half delta to farthest
    clipPosObj = clip_plane_point_;
    clipPosObj -= pos;
    temp = probePosObj;
    temp += normal;
    normClipPoint.isectPlaneLine(normal, clipPosObj, probePosObj, temp);
    maxDist = length( farthest - vec3f(normClipPoint) );
    numSlices = (size_t)( maxDist / length(delta) ) + 1;
    temp = delta;
    temp *= vec3f( ((float)(1 - static_cast<ptrdiff_t>(numSlices))) );
    farthest = normClipPoint;
    farthest += temp;
    if (_clipSingleSlice)
    {
      // Compute slice position:
      temp = delta;
      temp *= vec3f( ((float)(numSlices-1)) );
      farthest += temp;
      numSlices = 1;

      // Make slice opaque if possible:
      if (instantClassification())
      {
        updateLUT(0.0f);
      }
    }
  }

  vec3 texPoint;                                  // arbitrary point on current texture
  int drawn = 0;                                  // counter for drawn textures
  vec3 deltahalf = delta * 0.5f;

  // Relative viewing position
  vec3 releye = eye - pos;

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
  for (size_t i=0; i<numSlices; ++i)                     // loop thru all drawn textures
  {
    // Search for intersections between texture plane (defined by texPoint and
    // normal) and texture object (0..1):
    size_t isectCnt = isect->isectPlaneCuboid(normal, texPoint, probeMin, probeMax);

    texPoint += delta;

    if (isectCnt<3) continue;                     // at least 3 intersections needed for drawing

    // Check volume section mode:
    if (minSlice != -1 && static_cast<ptrdiff_t>(i) < minSlice) continue;
    if (maxSlice != -1 && static_cast<ptrdiff_t>(i) > maxSlice) continue;

    // Put the intersecting 3 to 6 vertices in cyclic order to draw adjacent
    // and non-overlapping triangles:
    isect->cyclicSort(isectCnt, normal);

    // Generate vertices in texture coordinates:
    if(usePreIntegration)
    {
      for (size_t j=0; j<isectCnt; ++j)
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

        for (size_t k=0; k<3; ++k)
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
      for (size_t j=0; j<isectCnt; ++j)
      {
        for (size_t k=0; k<3; ++k)
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
    for (size_t j=0; j<isectCnt; ++j)
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

//----------------------------------------------------------------------------
/** Render the volume using a 3D texture (needs 3D texturing extension).
  Spherical slices are surrounding the observer.
  @param mv       model-view matrix
*/
void vvTexRend::renderTex3DSpherical(mat4 const& mv)
{
  float  maxDist = 0.0;
  float  minDist = 0.0;
  vec3 texSize;                                   // size of 3D texture [object space]
  vec3 texSize2;                                  // half size of 3D texture [object space]
  vec3 volumeVertices[8];

  vvDebugMsg::msg(3, "vvTexRend::renderTex3DSpherical()");

  if (!extTex3d) return;

  // make sure that at least one shell is drawn
  const int numShells = max(1, static_cast<int>(_quality * 100.0f));

  // Determine texture object dimensions:
  const vvVector3 size(vd->getSize());
  for (size_t i=0; i<3; ++i)
  {
    texSize[i]  = size[i] * (float)texels[i] / (float)vd->vox[i];
    texSize2[i] = 0.5f * texSize[i];
  }

  mat4 invView = inverse(mv);

  // generates the vertices of the cube (volume) in world coordinates
  size_t vertexIdx = 0;
  for (size_t ix=0; ix<2; ++ix)
    for (size_t iy=0; iy<2; ++iy)
      for (size_t iz=0; iz<2; ++iz)
      {
        volumeVertices[vertexIdx][0] = (float)ix;
        volumeVertices[vertexIdx][1] = (float)iy;
        volumeVertices[vertexIdx][2] = (float)iz;
        // transfers vertices to world coordinates:
        for (size_t k=0; k<3; ++k)
          volumeVertices[vertexIdx][k] =
            (volumeVertices[vertexIdx][k] * 2.0f - 1.0f) * texSize2[k];
        vec4 tmp( volumeVertices[vertexIdx], 1.0f );
        tmp = mv * tmp;
        volumeVertices[vertexIdx] = tmp.xyz() / tmp.w;
        vertexIdx++;
  }

  // Determine maximal and minimal distance of the volume from the eyepoint:
  maxDist = minDist = length( volumeVertices[0] );
  for (size_t i = 1; i<7; i++)
  {
    float dist = length( volumeVertices[i] );
    if (dist > maxDist)  maxDist = dist;
    if (dist < minDist)  minDist = dist;
  }

  maxDist *= 1.4f;
  minDist *= 0.5f;

  // transfer the eyepoint to the object coordinates of the volume
  // to check whether the camera is inside the volume:
  vec4 eye4(0.0, 0.0, 0.0, 1.0);
  eye4 = invView * eye4;
  vec3 eye = eye4.xyz() / eye4.w;
  bool inside = true;
  for (size_t k=0; k<3; ++k)
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
  if (_clipMode == 1) activateClippingPlane();

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
  float     texSpacing;                           // spacing for texture coordinates
  float     zPos;                                 // texture z position
  float     texStep;                              // step between texture indices
  float     texIndex;                             // current texture index
  size_t    numTextures;                          // number of textures drawn

  vvDebugMsg::msg(3, "vvTexRend::renderTex2DSlices()");

  // Enable clipping plane if appropriate:
  if (_clipMode == 1) activateClippingPlane();

  // Generate half object size as shortcut:
  vec3 size = vd->getSize();
  vec3 size2 = size * 0.5f;

  numTextures = size_t(_quality * 100.0f);
  if (numTextures < 1) numTextures = 1;

  vec3 normal(0.0f, 0.0f, 1.0f);
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
    normal.z    = -normal.z;
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

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glTranslatef(vd->pos[0], vd->pos[1], vd->pos[2]);
  for (size_t i=0; i<numTextures; ++i)
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
      vvsize3 size;
      getLUTSize(size);
      glColorTableEXT(GL_TEXTURE_2D, GL_RGBA,
        size[0], GL_RGBA, GL_UNSIGNED_BYTE, rgbaLUT);
    }

    glBegin(GL_QUADS);
      glColor4f(1.0, 1.0, 1.0, 1.0);
      glNormal3f(normal.x, normal.y, normal.z);
      glTexCoord2f(texMin[0], texMax[1]); glVertex3f(-size2.x,  size2.y, zPos);
      glTexCoord2f(texMin[0], texMin[1]); glVertex3f(-size2.x, -size2.y, zPos);
      glTexCoord2f(texMax[0], texMin[1]); glVertex3f( size2.x, -size2.y, zPos);
      glTexCoord2f(texMax[0], texMax[1]); glVertex3f( size2.x,  size2.y, zPos);
    glEnd();

    zPos += texSpacing;
    texIndex += texStep;
  }

  disableTexture(GL_TEXTURE_2D);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  deactivateClippingPlane();
  VV_LOG(3) << "Number of textures stored: " << vd->vox[2] << std::endl;
  VV_LOG(3) << "Number of textures drawn: " << numTextures << std::endl;
}

//----------------------------------------------------------------------------
/** Render the volume using 2D textures, switching to the optimum
    texture set to prevent holes.
  @param principal  principal viewing axis
  @param zx,zy,zz   z coordinates of transformed base vectors
*/
void vvTexRend::renderTex2DCubic(vvVecmath::AxisType principal, float zx, float zy, float zz)
{
  vec3 normal;                                    // normal vector for slices
  vec3 texTL, texTR, texBL, texBR;                // texture coordinates (T=top etc.)
  vec3 objTL, objTR, objBL, objBR;                // object coordinates in world space
  vec3 texSpacing;                                // distance between textures
  float  texStep;                                 // step size for texture names
  float  texIndex;                                // textures index
  size_t numTextures;                             // number of textures drawn
  size_t frameTextures;                           // number of textures per frame

  vvDebugMsg::msg(3, "vvTexRend::renderTex2DCubic()");

  // Enable clipping plane if appropriate:
  if (_clipMode == 1) activateClippingPlane();

  // Initialize texture parameters:
  numTextures = size_t(_quality * 100.0f);
  frameTextures = vd->vox[0] + vd->vox[1] + vd->vox[2];
  if (numTextures < 2)  numTextures = 2;          // make sure that at least one slice is drawn to prevent division by zero

  // Generate half object size as a shortcut:
  vec3 size = vd->getSize();
  vec3 size2 = size * 0.5f;

  // Initialize parameters upon principal viewing direction:
  switch (principal)
  {
    case vvVecmath::X_AXIS:                                  // zx>0 -> draw left to right
      // Coordinate system:
      //     z
      //     |__y
      //   x/
      objTL = vec3(-size2.x,-size2.y, size2.z);
      objTR = vec3(-size2.x, size2.y, size2.z);
      objBL = vec3(-size2.x,-size2.y,-size2.z);
      objBR = vec3(-size2.x, size2.y,-size2.z);

      texTL = vec3(texMin[1], texMax[2], 0.0f);
      texTR = vec3(texMax[1], texMax[2], 0.0f);
      texBL = vec3(texMin[1], texMin[2], 0.0f);
      texBR = vec3(texMax[1], texMin[2], 0.0f);

      texSpacing = vec3(size[0] / float(numTextures - 1), 0.0f, 0.0f);
      texStep = -1.0f * float(vd->vox[0] - 1) / float(numTextures - 1);
      normal = vec3(1.0f, 0.0f, 0.0f);
      texIndex = float(vd->getCurrentFrame() * frameTextures);
      if (zx<0)                                   // reverse order? draw right to left
      {
        normal.x        = -normal.x;
        objTL.x         = objTR.x = objBL.x = objBR.x = size2.x;
        texSpacing.x    = -texSpacing.x;
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
      objTL = vec3( size2.x,-size2.y,-size2.z);
      objTR = vec3( size2.x,-size2.y, size2.z);
      objBL = vec3(-size2.x,-size2.y,-size2.z);
      objBR = vec3(-size2.x,-size2.y, size2.z);

      texTL = vec3(texMin[2], texMax[0], 0.0f);
      texTR = vec3(texMax[2], texMax[0], 0.0f);
      texBL = vec3(texMin[2], texMin[0], 0.0f);
      texBR = vec3(texMax[2], texMin[0], 0.0f);

      texSpacing = vec3(0.0f, size.y / float(numTextures - 1), 0.0f);
      texStep = -1.0f * float(vd->vox[1] - 1) / float(numTextures - 1);
      normal = vec3(0.0f, 1.0f, 0.0f);
      texIndex = float(vd->getCurrentFrame() * frameTextures + vd->vox[0]);
      if (zy<0)                                   // reverse order? draw top to bottom
      {
        normal.y        = -normal.y;
        objTL.y         = objTR.y = objBL.y = objBR.y = size2.y;
        texSpacing.y    = -texSpacing.y;
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
      objTL = vec3(-size2.x, size2.y,-size2.z);
      objTR = vec3( size2.x, size2.y,-size2.z);
      objBL = vec3(-size2.x,-size2.y,-size2.z);
      objBR = vec3( size2.x,-size2.y,-size2.z);

      texTL = vec3(texMin[0], texMax[1], 0.0f);
      texTR = vec3(texMax[0], texMax[1], 0.0f);
      texBL = vec3(texMin[0], texMin[1], 0.0f);
      texBR = vec3(texMax[0], texMin[1], 0.0f);

      texSpacing = vec3(0.0f, 0.0f, size.z / float(numTextures - 1));
      normal = vec3(0.0f, 0.0f, 1.0f);
      texStep = -1.0f * float(vd->vox[2] - 1) / float(numTextures - 1);
      texIndex = float(vd->getCurrentFrame() * frameTextures + vd->vox[0] + vd->vox[1]);
      if (zz<0)                                   // reverse order? draw front to back
      {
        normal.z        = -normal.z;
        objTL.z         = objTR.z = objBL.z = objBR.z = size2.z;
        texSpacing.z    = -texSpacing.z;
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
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glTranslatef(vd->pos[0], vd->pos[1], vd->pos[2]);
  for (size_t i=0; i<numTextures; ++i)
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
      vvsize3 size;
      getLUTSize(size);
      glColorTableEXT(GL_TEXTURE_2D, GL_RGBA,
        size[0], GL_RGBA, GL_UNSIGNED_BYTE, rgbaLUT);
    }

    glBegin(GL_QUADS);
    glColor4f(1.0, 1.0, 1.0, 1.0);
    glNormal3f(normal.x, normal.y, normal.z);
    glTexCoord2f(texTL.x, texTL.y); glVertex3f(objTL.x, objTL.y, objTL.z);
    glTexCoord2f(texBL.x, texBL.y); glVertex3f(objBL.x, objBL.y, objBL.z);
    glTexCoord2f(texBR.x, texBR.y); glVertex3f(objBR.x, objBR.y, objBR.z);
    glTexCoord2f(texTR.x, texTR.y); glVertex3f(objTR.x, objTR.y, objTR.z);
    glEnd();
    objTL += texSpacing;
    objBL += texSpacing;
    objBR += texSpacing;
    objTR += texSpacing;

    texIndex += texStep;
  }
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  disableTexture(GL_TEXTURE_2D);
  deactivateClippingPlane();
}

//----------------------------------------------------------------------------
/** Render the volume onto currently selected drawBuffer.
 Viewport size in world coordinates is -1.0 .. +1.0 in both x and y direction
*/
void vvTexRend::renderVolumeGL()
{
  float zx, zy, zz;                               // base vector z coordinates

  vvDebugMsg::msg(3, "vvTexRend::renderVolumeGL()");

  vvGLTools::printGLError("enter vvTexRend::renderVolumeGL()");

  vvssize3 vox = _paddingRegion.getMax() - _paddingRegion.getMin();
  for (size_t i = 0; i < 3; ++i)
  {
    vox[i] = std::min(vox[i], vd->vox[i]);
  }

  if (vox[0] * vox[1] * vox[2] == 0)
    return;

  vec3f size = vd->getSize();                  // volume size [world coordinates]

  // Draw boundary lines (must be done before setGLenvironment()):
  if (_isROIUsed)
  {
    vec3f probeSizeObj = size * roi_size_;
    drawBoundingBox(probeSizeObj, roi_pos_, _probeColor);
  }
  if (_clipMode == 1 && _clipPlanePerimeter)
  {
    drawPlanePerimeter(size, vd->pos, clip_plane_point_, clip_plane_normal_, _clipPlaneColor);
  }

  setGLenvironment();

  // Determine texture object extensions:
  for (size_t i = 0; i < 3; ++i)
  {
    // padded borders for (trilinear) interpolation
    size_t paddingLeft = size_t(abs(ptrdiff_t(_visibleRegion.getMin()[i] - _paddingRegion.getMin()[i])));
    size_t paddingRight = size_t(abs(ptrdiff_t(_visibleRegion.getMax()[i] - _paddingRegion.getMax()[i])));
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

  if (geomType==VV_SPHERICAL || geomType==VV_VIEWPORT) {
    // allow for using raw volume data from vvVolDesc for textures without re-shuffeling
    std::swap(texMin[2], texMax[2]);
    std::swap(texMin[1], texMax[1]);
  }

  // Get OpenGL modelview matrix:
  mat4 mv = gl::getModelviewMatrix();

  enableShader(_shader, pixLUTName);
  enableLUTMode(pixLUTName, fragProgName);

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
  }

  disableLUTMode();
  unsetGLenvironment();
  disableShader(_shader);

  if (_fpsDisplay)
  {
    // Make sure rendering is done to measure correct time.
    // Since this operation is costly, only do it if necessary.
    glFinish();
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
  planeEq[0] = -clip_plane_normal_[0];
  planeEq[1] = -clip_plane_normal_[1];
  planeEq[2] = -clip_plane_normal_[2];
  planeEq[3] = dot( clip_plane_normal_, clip_plane_point_ );
  glClipPlane(GL_CLIP_PLANE0, planeEq);
  glEnable(GL_CLIP_PLANE0);

  // Generate second clipping plane in single slice mode:
  if (_clipSingleSlice)
  {
    thickness = vd->_scale * vd->dist[0] * (vd->vox[0] * 0.01f);
    clipNormal2 = -clip_plane_normal_;
    planeEq[0] = -clipNormal2[0];
    planeEq[1] = -clipNormal2[1];
    planeEq[2] = -clipNormal2[2];
    planeEq[3] = clipNormal2.dot(clip_plane_point_) + thickness;
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
size_t vvTexRend::getLUTSize(vvsize3& size) const
{
  size_t x, y, z;

  vvDebugMsg::msg(3, "vvTexRend::getLUTSize()");
  if (_currentShader==11 && voxelType==VV_PIX_SHD)
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
size_t vvTexRend::getPreintTableSize() const
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
  vvsize3 lutSize;                                // number of entries in the RGBA lookup table
  lutDistance = dist;
  size_t total = 0;

  if(usePreIntegration)
  {
    vd->tf.makePreintLUTCorrect(getPreintTableSize(), preintTable, dist);
  }
  else
  {
    total = getLUTSize(lutSize);
    for (size_t i=0; i<total; ++i)
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
      if (dist<=0.0 || (_clipMode == 1 && _clipOpaque)) corr[3] = 1.0f;
      else if (_opacityCorrection) corr[3] = 1.0f - powf(1.0f - corr[3], dist);

      // Convert float to uint8_t and copy to rgbaLUT array:
      for (size_t c=0; c<4; ++c)
      {
        rgbaLUT[i * 4 + c] = uint8_t(corr[c] * 255.0f);
      }
    }
  }

  // Copy LUT to graphics card:
  vvGLTools::printGLError("enter updateLUT()");
  switch (voxelType)
  {
    case VV_RGBA:
      makeTextures(false);// this mode doesn't use a hardware LUT, so every voxel has to be updated
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
void vvTexRend::setViewingDirection(vec3f const& vd)
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
void vvTexRend::setObjectDirection(vec3f const& od)
{
  vvDebugMsg::msg(3, "vvTexRend::setObjectDirection()");
  objDir = od;
}


bool vvTexRend::checkParameter(ParameterType param, vvParam const& value) const
{
  switch (param)
  {
  case VV_SLICEINT:

    {
      virvo::tex_filter_mode mode = static_cast< virvo::tex_filter_mode >(value.asInt());

      if (mode == virvo::Nearest || mode == virvo::Linear)
      {
        return true;
      }
    }

    return false;;

  default:

    return vvRenderer::checkParameter(param, value);

  }
}


//----------------------------------------------------------------------------
// see parent
void vvTexRend::setParameter(ParameterType param, const vvParam& newValue)
{
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
      if (_interpolation != static_cast< virvo::tex_filter_mode >(newValue.asInt()))
      {
        _interpolation = static_cast< virvo::tex_filter_mode >(newValue.asInt());
        for (size_t f = 0; f < vd->frames; ++f)
        {
          glBindTexture(GL_TEXTURE_3D_EXT, texNames[f]);
          glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
          glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
        }
        updateTransferFunction();
      }
      break;
    case vvRenderer::VV_MIN_SLICE:
      minSlice = newValue;
      break;
    case vvRenderer::VV_MAX_SLICE:
      maxSlice = newValue;
      break;
    case vvRenderer::VV_SLICEORIENT:
      _sliceOrientation = (SliceOrientation)newValue.asInt();
      break;
    case vvRenderer::VV_PREINT:
      _preIntegration = newValue;
      updateTransferFunction();
      disableShader(_shader);
      delete _shader;
      _shader = initShader();
      break;
    case vvRenderer::VV_BINNING:
      vd->_binning = (vvVolDesc::BinningType)newValue.asInt();
      break;
    case vvRenderer::VV_OFFSCREENBUFFER:
    case vvRenderer::VV_USE_OFFSCREEN_BUFFER:
      {
        bool fbo = static_cast<bool>(newValue);

        this->_useOffscreenBuffer = fbo;

        if (fbo)
          setRenderTarget( virvo::FramebufferObjectRT::create() );
        else
          setRenderTarget( virvo::NullRT::create() );
      }
      break;
    case vvRenderer::VV_IMG_SCALE:
      //_imageScale = newValue;
      break;
    case vvRenderer::VV_IMG_PRECISION:
    case vvRenderer::VV_IMAGE_PRECISION:
      {
        virvo::BufferPrecision bp = mapBitsToBufferPrecision(static_cast<int>(newValue));

        this->_imagePrecision = bp;

//      setRenderTarget( virvo::FramebufferObjectRT::create(mapBufferPrecisionToFormat(bp), virvo::PF_DEPTH32F_STENCIL8) );
        setRenderTarget( virvo::FramebufferObjectRT::create(mapBufferPrecisionToFormat(bp), virvo::PF_DEPTH24_STENCIL8) );
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
    case vvRenderer::VV_PIX_SHADER:
      setCurrentShader(newValue);
      break;
    case vvRenderer::VV_PADDING_REGION:
      vvRenderer::setParameter(param, newValue);
      makeTextures(true);
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
    case vvRenderer::VV_MIN_SLICE:
      return minSlice;
    case vvRenderer::VV_MAX_SLICE:
      return maxSlice;
    case vvRenderer::VV_SLICEORIENT:
      return (int)_sliceOrientation;
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
void vvTexRend::renderQualityDisplay() const
{
  const int numSlices = int(_quality * 100.0f);
  vvPrintGL printGL;
  vec4f clearColor = vvGLTools::queryClearColor();
  vec4f fontColor( 1.0f - clearColor[0], 1.0f - clearColor[1], 1.0f - clearColor[2], 1.0f );
  printGL.setFontColor(fontColor);
  printGL.print(-0.9f, 0.9f, "Textures: %d", numSlices);
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
  vvGLTools::printGLError("Enter vvTexRend::enableShader()");

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
  vvGLTools::printGLError("Enter vvTexRend::disableShader()");

  if (shader)
  {
    shader->disable();
  }

  vvGLTools::printGLError("Leaving vvTexRend::disableShader()");
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

  // intersection on CPU, try to create fragment program
  vvShaderProgram* shader = _shaderFactory->createProgram("", "", fragName.str());

  vvGLTools::printGLError("Leave vvTexRend::initShader()");

  return shader;
}

//----------------------------------------------------------------------------
void vvTexRend::printLUT() const
{
  vvsize3 lutEntries;

  size_t total = getLUTSize(lutEntries);
  for (size_t i=0; i<total; ++i)
  {
    cerr << "#" << i << ": ";
    for (size_t c=0; c<4; ++c)
    {
      cerr << int(rgbaLUT[i * 4 + c]);
      if (c<3) cerr << ", ";
    }
    cerr << endl;
  }
}

uint8_t* vvTexRend::getHeightFieldData(float points[4][3], size_t& width, size_t& height)
{
  GLint viewport[4];
  uint8_t *pixels, *data, *result=NULL;
  size_t numPixels;
  size_t index;
  float sizeX, sizeY;
  vvVector3 size, size2;
  vvVector3 texcoord[4];

  std::cerr << "getHeightFieldData" << endl;

  glGetIntegerv(GL_VIEWPORT, viewport);

  width = size_t(ceil(getManhattenDist(points[0], points[1])));
  height = size_t(ceil(getManhattenDist(points[0], points[3])));

  numPixels = width * height;
  pixels = new uint8_t[4*numPixels];

  glReadPixels(viewport[0], viewport[1], width, height,
    GL_RGBA, GL_UNSIGNED_BYTE, pixels);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();

  size = vd->getSize();
  for (size_t i = 0; i < 3; ++i)
    size2[i]   = 0.5f * size[i];

  for (size_t j = 0; j < 4; j++)
    for (size_t k = 0; k < 3; k++)
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

  data = new uint8_t[texelsize * numPixels];
  memset(data, 0, texelsize * numPixels);
  glReadPixels(viewport[0], viewport[1], width, height,
    GL_RGB, GL_UNSIGNED_BYTE, data);

  std::cerr << "data read" << endl;

  if (vd->chan == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
  {
    result = new uint8_t[numPixels];
    for (size_t y = 0; y < height; y++)
      for (size_t x = 0; x < width; x++)
    {
      index = y * width + x;
      switch (voxelType)
      {
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
      result = new uint8_t[vd->chan * numPixels];

      for (size_t y = 0; y < height; y++)
        for (size_t x = 0; x < width; x++)
      {
        index = (y * width + x) * vd->chan;
        for (size_t c = 0; c < vd->chan; c++)
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

  for (size_t i=0; i<3; ++i)
  {
    dist += float(fabs(p1[i] - p2[i])) / float(vd->getSize()[i] * vd->vox[i]);
  }

  std::cerr << "Manhattan Distance: " << dist << endl;

  return dist;
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

size_t vvTexRend::getTextureSize(size_t sz) const
{
  if (extNonPower2)
    return sz;

  return vvToolshed::getTextureSize(sz);
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
