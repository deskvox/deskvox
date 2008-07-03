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
#include <iomanip>

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#if defined(__linux) || defined(LINUX)
#define GL_GLEXT_PROTOTYPES 1
#define GL_GLEXT_LEGACY 1
#include <string.h>
#endif

#include "vvopengl.h"
#include "vvglext.h"

#include "vvdynlib.h"
#if !defined(_WIN32) && !defined(__APPLE__)
#include <dlfcn.h>
#include <GL/glx.h>
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
#include "vvprintgl.h"

#ifdef HAVE_CG
  static void checkCgError(CGcontext ctx, CGerror err, void *appdata);
#endif

using namespace std;

//----------------------------------------------------------------------------
const int vvTexRend::NUM_PIXEL_SHADERS = 12;

//----------------------------------------------------------------------------
/** Constructor.
  @param vd  volume description
  @param m   render geometry (default: automatic)
*/
vvTexRend::vvTexRend(vvVolDesc* vd, vvRenderState renderState, GeometryType geom, VoxelType vox) : vvRenderer(vd, renderState)
{
  vvDebugMsg::msg(1, "vvTexRend::vvTexRend()");

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
#ifdef HAVE_CG
  _currentShader = vd->chan - 1;
#else
  _currentShader = 0;
#endif
  _useOnlyOneBrick = false;
  _areBricksCreated = false;
  lutDistance = -1.0;

  // Find out which OpenGL extensions are supported:
#if defined(GL_VERSION_1_2) && defined(__APPLE__)
  extTex3d  = true;
  arbMltTex = true;
#else
  extTex3d  = vvGLTools::isGLextensionSupported("GL_EXT_texture3D");
  arbMltTex = vvGLTools::isGLextensionSupported("GL_ARB_multitexture");
#endif

  extMinMax = vvGLTools::isGLextensionSupported("GL_EXT_blend_minmax");
  extBlendEquation = vvGLTools::isGLextensionSupported("GL_EXT_blend_equation");
  extColLUT = isSupported(VV_SGI_LUT);
  extPalTex = isSupported(VV_PAL_TEX);
  extTexShd = isSupported(VV_TEX_SHD);
#ifdef HAVE_CG
  extPixShd = isSupported(VV_PIX_SHD);
#else
  extPixShd = false;
#endif
  arbFrgPrg = isSupported(VV_FRG_PRG);

  extNonPower2 = vvGLTools::isGLextensionSupported("GL_ARB_texture_non_power_of_two");

#if defined(_WIN32)
#ifdef GL_VERSION_1_2
  glTexImage3DEXT = (PFNGLTEXIMAGE3DEXTPROC)vvDynLib::glSym("glTexImage3D");
  if (glTexImage3DEXT==NULL) extTex3d = false;
#else
  glTexImage3DEXT = (PFNGLTEXIMAGE3DEXTPROC)vvDynLib::glSym("glTexImage3DEXT");
  if (glTexImage3DEXT==NULL) extTex3d = false;
#endif
  glTexSubImage3DEXT = (PFNGLTEXSUBIMAGE3DEXTPROC)vvDynLib::glSym("glTexSubImage3D");
  if (glTexSubImage3DEXT==NULL) extTex3d = false;
  glColorTableSGI = (PFNGLCOLORTABLESGIPROC)vvDynLib::glSym("glColorTableSGI");
  if (glColorTableSGI==NULL) extColLUT = false;
  glColorTableEXT = (PFNGLCOLORTABLEEXTPROC)vvDynLib::glSym("glColorTableEXT");
  if (glColorTableEXT==NULL) extPalTex = false;
  if (extBlendEquation) glBlendEquationVV = (PFNGLBLENDEQUATIONEXTPROC)vvDynLib::glSym("glBlendEquationEXT");
  else glBlendEquationVV = (PFNGLBLENDEQUATIONPROC)vvDynLib::glSym("glBlendEquation");
  glActiveTextureARB = (PFNGLACTIVETEXTUREARBPROC)vvDynLib::glSym("glActiveTextureARB");
  if (glActiveTextureARB==NULL)
  {
     extTexShd = false;
     arbMltTex = false;
  }
  glMultiTexCoord3fARB = (PFNGLMULTITEXCOORD3FARBPROC)vvDynLib::glSym("glMultiTexCoord3fARB");
  if (glMultiTexCoord3fARB==NULL) arbMltTex = false;
  glGenProgramsARB = (PFNGLGENPROGRAMSARBPROC)vvDynLib::glSym("glGenProgramsARB");
  if(glGenProgramsARB==NULL) arbFrgPrg = false;
  glBindProgramARB = (PFNGLBINDPROGRAMARBPROC)vvDynLib::glSym("glBindProgramARB");
  if(glBindProgramARB==NULL) arbFrgPrg = false;
  glDeleteProgramsARB = (PFNGLDELETEPROGRAMSARBPROC)vvDynLib::glSym("glDeleteProgramsARB");
  if(glDeleteProgramsARB==NULL) arbFrgPrg = false;
  glProgramStringARB = (PFNGLPROGRAMSTRINGARBPROC)vvDynLib::glSym("glProgramStringARB");
  if(glProgramStringARB==NULL) arbFrgPrg = false;
#else                                           // not WIN32
  if (extBlendEquation) glBlendEquationVV = (glBlendEquationEXT_type*)vvDynLib::glSym("glBlendEquationEXT");
  else glBlendEquationVV = (glBlendEquationEXT_type*)vvDynLib::glSym("glBlendEquation");
#ifdef GL_VERSION_1_2
  glTexImage3DEXT = (glTexImage3DEXT_type*)vvDynLib::glSym("glTexImage3D");
  if (glTexImage3DEXT==NULL) extTex3d = false;
  glTexSubImage3DEXT = (glTexSubImage3DEXT_type*)vvDynLib::glSym("glTexSubImage3D");
  if (glTexSubImage3DEXT==NULL) extTex3d = false;
#else
  glTexImage3DEXT = (glTexImage3DEXT_type*)vvDynLib::glSym("glTexImage3DEXT");
  if (glTexImage3DEXT==NULL) extTex3d = false;
  glTexSubImage3DEXT = (glTexSubImage3DEXT_type*)vvDynLib::glSym("glTexSubImage3DEXT");
  if (glTexSubImage3DEXT==NULL) extSubTex3d = false;
#endif
  glColorTableSGI = (glColorTableSGI_type*)vvDynLib::glSym("glColorTableSGI");
  if (glColorTableSGI==NULL) extColLUT = false;
  glColorTableEXT = (glColorTableEXT_type*)vvDynLib::glSym("glColorTableEXT");
  if (glColorTableEXT==NULL) extPalTex = false;
  glMultiTexCoord3fARB = (glMultiTexCoord3fARB_type*)vvDynLib::glSym("glMultiTexCoord3fARB");
  if (glMultiTexCoord3fARB==NULL) arbMltTex = false;
  glActiveTextureARB = (glActiveTextureARB_type*)vvDynLib::glSym("glActiveTextureARB");
  if (glActiveTextureARB==NULL)
  {
     extTexShd = false;
     arbMltTex = false;
  }
  glGenProgramsARB = (glGenProgramsARB_type*)vvDynLib::glSym("glGenProgramsARB");
  if(glGenProgramsARB==NULL) arbFrgPrg = false;
  glDeleteProgramsARB = (glDeleteProgramsARB_type*)vvDynLib::glSym("glDeleteProgramsARB");
  if(glDeleteProgramsARB==NULL) arbFrgPrg = false;
  glBindProgramARB = (glBindProgramARB_type*)vvDynLib::glSym("glBindProgramARB");
  if(glBindProgramARB==NULL) arbFrgPrg = false;
  glProgramStringARB = (glProgramStringARB_type*)vvDynLib::glSym("glProgramStringARB");
  if(glProgramStringARB==NULL) arbFrgPrg = false;
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

#ifdef HAVE_CG
  if(geomType==VV_SLICES || geomType==VV_CUBIC2D)
  {
    _currentShader = 8;
  }
#endif

  if(voxelType==VV_TEX_SHD || voxelType==VV_PIX_SHD || voxelType==VV_FRG_PRG)
  {
    glGenTextures(1, &pixLUTName);
  }

  // Init fragment program:
  if(voxelType==VV_FRG_PRG)
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

#ifdef HAVE_CG
  _cgProgram     = new CGprogram[NUM_PIXEL_SHADERS];
  _cgFragProfile = new CGprofile[NUM_PIXEL_SHADERS];
#endif

  if (voxelType==VV_PIX_SHD)
  {
    if (!initPixelShaders())
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
  updateTransferFunction();
  if (voxelType != VV_RGBA)
  {
    makeTextures();                             // we only have to do this once for non-RGBA textures
  }
}

//----------------------------------------------------------------------------
/// Destructor
vvTexRend::~vvTexRend()
{
  vvDebugMsg::msg(1, "vvTexRend::~vvTexRend()");

  if (voxelType==VV_PIX_SHD)
  {
#ifdef HAVE_CG
    cgDestroyContext(_cgContext);                 // destroy our Cg context and all programs contained within it
#endif
  }

  if (voxelType==VV_FRG_PRG)
  {
    glDeleteProgramsARB(3, fragProgName);
  }
  if (voxelType==VV_FRG_PRG || voxelType==VV_TEX_SHD || voxelType==VV_PIX_SHD)
  {
    glDeleteTextures(1, &pixLUTName);
  }
  removeTextures();

  delete[] rgbaTF;
  delete[] rgbaLUT;

#ifdef HAVE_CG
  delete[] _cgProgram;
  delete[] _cgFragProfile;
#endif
  delete[] preintTable;

  _brickList.first();
  for (int f = 0; f < _brickList.count(); f++)
  {
    _brickList.getData()->removeAll();
    _brickList.next();
  }
  _brickList.removeAll();
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
      if (extPixShd) return VV_PIX_SHD;
      else if (arbFrgPrg) return VV_FRG_PRG;
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
vvTexRend::ErrorType vvTexRend::makeTextures()
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
    case VV_BRICKS:  err=makeTextureBricks(); break;
    default: updateTextures3D(0, 0, 0, texels[0], texels[1], texels[2], true); break;
  }
  vvGLTools::printGLError("vvTexRend::makeTextures");

  if (voxelType==VV_PIX_SHD || voxelType==VV_FRG_PRG || voxelType==VV_TEX_SHD)
  {
    //if (first)
    {
      makeLUTTexture();                           // FIXME: works only once, then generates OpenGL error
      first = false;
    }
  }
  return err;
}

//----------------------------------------------------------------------------
/// Generate texture for look-up table.
void vvTexRend::makeLUTTexture()
{
  int size[3];

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

vvTexRend::ErrorType vvTexRend::makeTextureBricks()
{
  ErrorType err = OK;
  GLint glWidth;                                  // return value from OpenGL call
  uchar* texData;                                 // data for texture memory
  uchar* raw;                                     // raw volume data
  vvSLList<Brick*>* tmp;                          // tmp variable for creating brick lists
  vvVector3 voxSize;                              // size of a voxel
  vvVector3 halfBrick;                            // middle of the current brick in texels
  vvVector3 halfVolume;                           // middle of the volume in voxel
  float fval;                                     // floating point voxel value
  int rawVal[4];                                  // raw values for R,G,B,A
  int frames;                                     // number of time steps in sequence
  int numBricks[3];                               // number of bricks used for each dimension of the volume
  int tmpTexels[3];                               // number of texels in each dimension for current brick
  int texSize;                                    // texture size in bytes
  int sliceSize;                                  // number of voxels in a raw slice
  int startOffset[3];                             // offset to first voxel of current brick
  int texOffset;                                  // currently processed texel [texels]
  int texLineOffset;                              // index into currently processed line of texture data array [texels]
  int rawSliceOffset;                             // slice offset in raw data
  int heightOffset, srcIndex;
  int bx, by, bz, x, y, s, f, c, alpha, i, texIndex;
  bool accommodated = true;                       // false if a texture cannot be accommodated in TRAM
  Brick* currBrick;

  if (!extTex3d) return NO3DTEX;

  if (_renderState._brickSize == 0)
  {
    vvDebugMsg::msg(1, "3D Texture brick size unknown");
    return err;
  }

  removeTextures();

  frames = vd->frames;

  _brickList.first();
  for (i = 0; i < _brickList.count(); i++)
  {
    _brickList.getData()->removeAll();
    _brickList.next();
  }
  _brickList.removeAll();

  for (f = 0; f < frames; f++)
  {
    tmp = new vvSLList<Brick*>;
    _brickList.append(tmp, vvSLNode<vvSLList<Brick*> *>::NORMAL_DELETE);
  }

  // compute number of texels / per brick (should be of power 2)
  texels[0] = vvToolshed::getTextureSize(_renderState._brickSize[0]);
  texels[1] = vvToolshed::getTextureSize(_renderState._brickSize[1]);
  texels[2] = vvToolshed::getTextureSize(_renderState._brickSize[2]);

  // compute number of bricks
  if ((_useOnlyOneBrick) ||
    ((texels[0] == vd->vox[0]) && (texels[1] == vd->vox[1]) && (texels[2] == vd->vox[2])))
    numBricks[0] = numBricks[1] = numBricks[2] = 1;
  else
  {
    numBricks[0] = (int) ceil((float) (vd->vox[0]) / (float) (_renderState._brickSize[0]));
    numBricks[1] = (int) ceil((float) (vd->vox[1]) / (float) (_renderState._brickSize[1]));
    numBricks[2] = (int) ceil((float) (vd->vox[2]) / (float) (_renderState._brickSize[2]));
  }

  // number of textures needed
  textures = frames * numBricks[0] * numBricks[1] * numBricks[2];

  texSize = texels[0] * texels[1] * texels[2] * texelsize;

  vvDebugMsg::msg(1, "3D Texture (bricking) width     = ", texels[0]);
  vvDebugMsg::msg(1, "3D Texture (bricking) height    = ", texels[1]);
  vvDebugMsg::msg(1, "3D Texture (bricking) depth     = ", texels[2]);
  vvDebugMsg::msg(1, "3D Texture (bricking) size (KB) = ", texSize / 1024);

  texData = new uchar[texSize];

  sliceSize = vd->getSliceBytes();

  vvDebugMsg::msg(2, "Creating texture names. # of names: ", textures);

  texNames = new GLuint[textures];
  glGenTextures(textures, texNames);

  // generate textures contents:
  vvDebugMsg::msg(2, "Transferring textures to TRAM. Total size [KB]: ",
    textures * texSize / 1024);

  // helper variables
  voxSize = vd->getSize();
  voxSize[0] /= (vd->vox[0]-1);
  voxSize[1] /= (vd->vox[1]-1);
  voxSize[2] /= (vd->vox[2]-1);

  halfBrick.set(float(texels[0]-1), float(texels[1]-1), float(texels[2]-1));
  halfBrick.scale(0.5);

  halfVolume.set(float(vd->vox[0]), float(vd->vox[1]), float(vd->vox[2]));
  halfVolume.sub(1.0);
  halfVolume.scale(0.5);

  _brickList.first();

  for (f = 0; f < frames; f++)
  {
    raw = vd->getRaw(f);

    for (bx = 0; bx < numBricks[0]; bx++)
      for (by = 0; by < numBricks[1]; by++)
        for (bz = 0; bz < numBricks[2]; bz++)
        {
          startOffset[0] = bx * _renderState._brickSize[0];
          startOffset[1] = by * _renderState._brickSize[1];
          startOffset[2] = bz * _renderState._brickSize[2];

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

      for (s = startOffset[2]; (s < (startOffset[2] + tmpTexels[2])) && (s < vd->vox[2]); s++)
      {
        if (s < 0) continue;
        rawSliceOffset = (vd->vox[2] - s - 1) * sliceSize;
        for (y = startOffset[1]; (y < (startOffset[1] + tmpTexels[1])) && (y < vd->vox[1]); y++)
        {
          if (y < 0) continue;
          heightOffset = (vd->vox[1] - y - 1) * vd->vox[0] * vd->bpc * vd->chan;
          texLineOffset = (y - startOffset[1]) * tmpTexels[0] + (s - startOffset[2]) * tmpTexels[0] * tmpTexels[1];
          if (vd->chan == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
          {
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

                // copy color components:
                for (c = 0; c < ts_min(vd->chan, 3); c++)
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
                  alpha = 0;
                  for (c = 0; c < vd->chan; c++)
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
        }
      }

      texIndex = (f * numBricks[0] * numBricks[1] * numBricks[2]) + (bx * numBricks[2] * numBricks[1])
        + (by * numBricks[2]) + bz;

      currBrick = new Brick();
      currBrick->index = texIndex;
      currBrick->pos.set(vd->pos[0] + voxSize[0] * (startOffset[0] + halfBrick[0] - halfVolume[0]),
        vd->pos[1] + voxSize[1] * (startOffset[1] + halfBrick[1] - halfVolume[1]),
        vd->pos[2] + voxSize[2] * (startOffset[2] + halfBrick[2] - halfVolume[2]));
      currBrick->min.set(vd->pos[0] + voxSize[0] * (startOffset[0] - halfVolume[0]),
        vd->pos[1] + voxSize[1] * (startOffset[1] - halfVolume[1]),
        vd->pos[2] + voxSize[2] * (startOffset[2] - halfVolume[2]));
      currBrick->max.set(vd->pos[0] + voxSize[0] * (startOffset[0] + (tmpTexels[0] - 1) - halfVolume[0]),
        vd->pos[1] + voxSize[1] * (startOffset[1] + (tmpTexels[1] - 1) - halfVolume[1]),
        vd->pos[2] + voxSize[2] * (startOffset[2] + (tmpTexels[2] - 1) - halfVolume[2]));
      currBrick->texels[0] = tmpTexels[0];
      currBrick->texels[1] = tmpTexels[1];
      currBrick->texels[2] = tmpTexels[2];
      currBrick->startOffset[0] = startOffset[0];
      currBrick->startOffset[1] = startOffset[1];
      currBrick->startOffset[2] = startOffset[2];

      _brickList.getData()->append(currBrick, vvSLNode<Brick*>::NORMAL_DELETE);

      glBindTexture(GL_TEXTURE_3D_EXT, texNames[texIndex]);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP);

      glTexImage3DEXT(GL_PROXY_TEXTURE_3D_EXT, 0, internalTexFormat,
        tmpTexels[0], tmpTexels[1], tmpTexels[2], 0, texFormat, GL_UNSIGNED_BYTE, NULL);

      glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D_EXT, 0, GL_TEXTURE_WIDTH, &glWidth);
      if (glWidth != 0)
      {
        glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, internalTexFormat, tmpTexels[0], tmpTexels[1], tmpTexels[2], 0,
          texFormat, GL_UNSIGNED_BYTE, texData);
      }
      else accommodated = false;
    }
    _brickList.next();
  }

  _brickList.first();

  if (!accommodated)
  {
    cerr << "Insufficient texture memory for 3D textures." << endl;
    err = TRAM_ERROR;
  }

  delete[] texData;
  _areBricksCreated = true;
  return err;
}

void vvTexRend::updateBrickGeom()
{
  int c, f;
  Brick* tmp;
  vvVector3 voxSize;
  vvVector3 halfBrick;
  vvVector3 halfVolume;

  _brickList.first();

  for (f = 0; f < _brickList.count(); f++)
  {
    _brickList.getData()->first();

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

    for (c = 0; c < _brickList.getData()->count(); c++)
    {
      tmp = _brickList.getData()->getData();
      tmp->pos.set(vd->pos[0] + voxSize[0] * (tmp->startOffset[0] + halfBrick[0] - halfVolume[0]),
        vd->pos[1] + voxSize[1] * (tmp->startOffset[1] + halfBrick[1] - halfVolume[1]),
        vd->pos[2] + voxSize[2] * (tmp->startOffset[2] + halfBrick[2] - halfVolume[2]));
      tmp->min.set(vd->pos[0] + voxSize[0] * (tmp->startOffset[0] - halfVolume[0]),
        vd->pos[1] + voxSize[1] * (tmp->startOffset[1] - halfVolume[1]),
        vd->pos[2] + voxSize[2] * (tmp->startOffset[2] - halfVolume[2]));
      tmp->max.set(vd->pos[0] + voxSize[0] * (tmp->startOffset[0] + (tmp->texels[0] - 1) - halfVolume[0]),
        vd->pos[1] + voxSize[1] * (tmp->startOffset[1] + (tmp->texels[1] - 1) - halfVolume[1]),
        vd->pos[2] + voxSize[2] * (tmp->startOffset[2] + (tmp->texels[2] - 1) - halfVolume[2]));
      _brickList.getData()->next();
    }
    _brickList.next();
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

void vvTexRend::setComputeBrickSize(bool flag)
{
  _renderState._computeBrickSize = flag;
  if (_renderState._computeBrickSize)
  {
    computeBrickSize();
    if(!_areBricksCreated)
       makeTextures();
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
  makeTextures();
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
       makeTextures();
  }
}

int vvTexRend::getTexMemorySize()
{
  return _renderState._texMemorySize;
}

void vvTexRend::computeBrickSize()
{
  int powerOf2[3];
  int max;
  int neededMemory;
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

  powerOf2[0] = vvToolshed::getTextureSize(vd->vox[0]);
  powerOf2[1] = vvToolshed::getTextureSize(vd->vox[1]);
  powerOf2[2] = vvToolshed::getTextureSize(vd->vox[2]);

  neededMemory = (powerOf2[0] * powerOf2[1] * powerOf2[2]) / (1024*1024) * texelsize;

  if (neededMemory < texMemorySize)
  {
    // use only one brick
    _useOnlyOneBrick = true;
    newBrickSize[0] = powerOf2[0];
    newBrickSize[1] = powerOf2[1];
    newBrickSize[2] = powerOf2[2];
    setROIEnable(false);
  }
  else
  {
    _useOnlyOneBrick = false;

    max = ts_max(vd->vox[0], vd->vox[1], vd->vox[2]);

    int tmp[3] = { vd->vox[0], vd->vox[1], vd->vox[2] };
    bool done = false;
    while (!done)
    {
      int i = 0;
      int maxSize = -1;
      for(int j=0; j<3; ++j)
      {
        newBrickSize[j] = vvToolshed::getTextureSize(tmp[j]);
        if(maxSize < newBrickSize[j])
        {
          i = j;
          maxSize = newBrickSize[j];
        }
      }

      // compute needed memory for 27 bricks
      neededMemory = newBrickSize[0] * newBrickSize[1] * newBrickSize[2] / (1024 * 1024) * texelsize * 27;
      if (neededMemory < texMemorySize)
      {
        done = true;
        break;
      }

      tmp[i] = newBrickSize[i] / 2;
      if(tmp[i] < 1)
         tmp[i] = 1;
    }

    probeSize[0] = 2 * (newBrickSize[0]-1) / (float) vd->vox[0];
    probeSize[1] = 2 * (newBrickSize[1]-1) / (float) vd->vox[1];
    probeSize[2] = 2 * (newBrickSize[2]-1) / (float) vd->vox[2];

    setProbeSize(&probeSize);
    //setROIEnable(true);
  }
  if (newBrickSize[0]-1 != _renderState._brickSize[0]
      || newBrickSize[1]-1 != _renderState._brickSize[1]
      || newBrickSize[2]-1 != _renderState._brickSize[2]
      || !_areBricksCreated)
  {
    _renderState._brickSize[0] = newBrickSize[0]-1;
    _renderState._brickSize[1] = newBrickSize[1]-1;
    _renderState._brickSize[2] = newBrickSize[2]-1;
    _areBricksCreated = false;
  }
}

//----------------------------------------------------------------------------
/// Update transfer function from volume description.
void vvTexRend::updateTransferFunction()
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

  updateLUT(1.0f);                                // generate color/alpha lookup table

  //printLUT();
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

      glTexImage3DEXT(GL_PROXY_TEXTURE_3D_EXT, 0, internalTexFormat,
        texels[0], texels[1], texels[2], 0, texFormat, GL_UNSIGNED_BYTE, NULL);
      glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D_EXT, 0, GL_TEXTURE_WIDTH, &glWidth);
      if (glWidth!=0)
      {
        glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, internalTexFormat, texels[0], texels[1], texels[2], 0,
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
  _brickList.first();

  for (f = 0; f < frames; f++)
  {
    raw = vd->getRaw(f);

    while (true)
    {
      startOffset[0] = _brickList.getData()->getData()->startOffset[0];
      startOffset[1] = _brickList.getData()->getData()->startOffset[1];
      startOffset[2] = _brickList.getData()->getData()->startOffset[2];

      endOffset[0] = startOffset[0] + _renderState._brickSize[0];
      endOffset[1] = startOffset[1] + _renderState._brickSize[1];
      endOffset[2] = startOffset[2] + _renderState._brickSize[2];

      endOffset[0] = ts_clamp(endOffset[0], 0, vd->vox[0] - 1);
      endOffset[1] = ts_clamp(endOffset[1], 0, vd->vox[1] - 1);
      endOffset[2] = ts_clamp(endOffset[2], 0, vd->vox[2] - 1);

      if ((offsetX > endOffset[0]) || ((offsetX + sizeX - 1) < startOffset[0]))
      {
        if (_brickList.getData()->next()) continue;
        else break;
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
        if (_brickList.getData()->next()) continue;
        else break;
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
        if (_brickList.getData()->next()) continue;
        else break;
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

      glBindTexture(GL_TEXTURE_3D_EXT, texNames[_brickList.getData()->getData()->index]);

      glTexSubImage3DEXT(GL_TEXTURE_3D_EXT, 0, start[0] - startOffset[0], start[1] - startOffset[1], start[2] - startOffset[2],
        size[0], size[1], size[2], texFormat, GL_UNSIGNED_BYTE, texData);

      if (!_brickList.getData()->next()) break;
    }
    _brickList.next();
  }

  _brickList.first();

  delete[] texData;
  return OK;
}

//----------------------------------------------------------------------------
/// Set GL environment for texture rendering.
void vvTexRend::setGLenvironment()
{
  vvDebugMsg::msg(3, "vvTexRend::setGLenvironment()");

  // Save current GL state:
  glGetBooleanv(GL_CULL_FACE, &glsCulling);
  glGetBooleanv(GL_BLEND, &glsBlend);
  glGetBooleanv(GL_COLOR_MATERIAL, &glsColorMaterial);
  glGetIntegerv(GL_BLEND_SRC, &glsBlendSrc);
  glGetIntegerv(GL_BLEND_DST, &glsBlendDst);
  glGetBooleanv(GL_LIGHTING, &glsLighting);
  glGetBooleanv(GL_DEPTH_TEST, &glsDepthTest);
  glGetIntegerv(GL_MATRIX_MODE, &glsMatrixMode);
  glGetIntegerv(GL_DEPTH_FUNC, &glsDepthFunc);

  if (extMinMax) glGetIntegerv(GL_BLEND_EQUATION_EXT, &glsBlendEquation);
  glGetBooleanv(GL_DEPTH_WRITEMASK, &glsDepthMask);

  switch (voxelType)
  {
    case VV_SGI_LUT:
      glGetBooleanv(GL_TEXTURE_COLOR_TABLE_SGI, &glsTexColTable);
      break;
    case VV_PAL_TEX:
      glGetBooleanv(GL_SHARED_TEXTURE_PALETTE_EXT, &glsSharedTexPal);
      break;
    default: break;
  }

  // Set new GL state:
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);                           // default depth function
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
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

  if (glBlendEquationVV)
  {
    switch (_renderState._mipMode)
    {
                                                  // alpha compositing
      case 0: glBlendEquationVV(GL_FUNC_ADD); break;
      case 1: glBlendEquationVV(GL_MAX); break;   // maximum intensity projection
      case 2: glBlendEquationVV(GL_MIN); break;   // minimum intensity projection
      default: break;
    }
  }

  vvDebugMsg::msg(3, "vvTexRend::setGLenvironment() done");
}

//----------------------------------------------------------------------------
/// Unset GL environment for texture rendering.
void vvTexRend::unsetGLenvironment()
{
  vvDebugMsg::msg(3, "vvTexRend::unsetGLenvironment()");

  if (glsCulling==(GLboolean)true) glEnable(GL_CULL_FACE);
  else glDisable(GL_CULL_FACE);

  if (glsBlend==(GLboolean)true) glEnable(GL_BLEND);
  else glDisable(GL_BLEND);

  if (glsColorMaterial==(GLboolean)true) glEnable(GL_COLOR_MATERIAL);
  else glDisable(GL_COLOR_MATERIAL);

  if (glsDepthTest==(GLboolean)true) glEnable(GL_DEPTH_TEST);
  else glDisable(GL_DEPTH_TEST);

  if (glsLighting==(GLboolean)true) glEnable(GL_LIGHTING);
  else glDisable(GL_LIGHTING);

  glDepthMask(glsDepthMask);
  glDepthFunc(glsDepthFunc);
  glBlendFunc(glsBlendSrc, glsBlendDst);
  if (glBlendEquationVV) glBlendEquationVV(glsBlendEquation);
  glMatrixMode(glsMatrixMode);
  vvDebugMsg::msg(3, "vvTexRend::unsetGLenvironment() done");
}

//----------------------------------------------------------------------------
void vvTexRend::enableLUTMode()
{
  switch(voxelType)
  {
    case VV_FRG_PRG:
      enableFragProg();
      break;
    case VV_TEX_SHD:
      enableNVShaders();
      break;
    case VV_PIX_SHD:
      enablePixelShaders();
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
    case VV_PIX_SHD:
      disablePixelShaders();
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
      probeMin[i] = probePosObj[i] - (_renderState._roiSize[i] * size[i]) / 2.0f;
      probeMax[i] = probePosObj[i] + (_renderState._roiSize[i] * size[i]) / 2.0f;
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
      probePosObj[i] = (probeMax[i] + probeMin[i]) / 2.0f;
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
      updateLUT(thickness);
    }
  }

  delta.copy(&normal);
  delta.scale(sliceDistance);

  // Compute farthest point to draw texture at:
  farthest.copy(&delta);
  farthest.scale((float)(numSlices - 1) / -2.0f);
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
        updateLUT(0.0f);
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
  vvVector3 size, size2;                          // full and half object sizes
  vvVector3 isect[6];                             // intersection points, maximum of 6 allowed when intersecting a plane and a volume [object space]
  vvVector3 texcoord[6];                          // intersection points in texture coordinate space [0..1]
  vvVector3 farthest;                             // volume vertex farthest from the viewer
  vvVector3 texPoint;                             // arbitrary point on current texture
  vvVector3 delta;                                // distance vector between textures [object space]
  vvVector3 normal;                               // normal vector of textures
  vvVector3 origin;                               // origin (0|0|0) transformed to object space
  vvVector3 eye;                                  // user's eye position [object space]
  vvVector3 probePosObj;                          // probe midpoint [object space]
  vvVector3 probeSizeObj;                         // probe size [object space]
  vvVector3 probeMin, probeMax;                   // probe min and max coordinates [object space]
  vvVector3 pos;                                  // volume location
  vvVector3 min, max;                             // min and max pos of current brick (cut with probe)
  vvVector3 texRange;                             // range of texture coordinates
  vvVector3 texMin;                               // minimum texture coordinate
  vvVector3 dist;                                 // dimensions of current brick
  Brick* tmp;                                     // current brick
  float     diagonal;                             // probe diagonal [object space]
  vvVector3 maxSize;                              // probe edge length [object space]
  int       isectCnt;                             // intersection counter
  int       numSlices;                            // number of texture slices along diagonal
  int       drawn;                                // counter for drawn textures
  int       i, j, k;                              // general counters

  vvDebugMsg::msg(3, "vvTexRend::renderTexBricks()");

  // needs 3D texturing extension
  if (!extTex3d) return;

  if (_brickList.isEmpty()) return;

  // Determine texture object dimensions and half object size as a shortcut:
  size.copy(vd->getSize());
  size2 = size;
  size2.scale(0.5);

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

  // Compute length of probe diagonal [object space]:
  diagonal = (float)sqrt(
    probeSizeObj[0] * probeSizeObj[0] +
    probeSizeObj[1] * probeSizeObj[1] +
    probeSizeObj[2] * probeSizeObj[2]);

  numSlices = int(_renderState._quality * 100.0f);
  if (numSlices < 1) numSlices = 1;               // make sure that at least one slice is drawn

  vvDebugMsg::msg(3, "Number of textures rendered per brick: ", numSlices);

  // Use alpha correction in indexed mode: adapt alpha values to number of textures:
  if (instantClassification())
  {
    float diagonalVoxels;
    diagonalVoxels = sqrtf(float(vd->vox[0] * vd->vox[0] +
      vd->vox[1] * vd->vox[1] +
      vd->vox[2] * vd->vox[2]));
    updateLUT(diagonalVoxels / float(numSlices));
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
  farthest.scale((float)(numSlices - 1) / -2.0f);
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
      if (instantClassification())
      {
        updateLUT(1.0f);
      }
    }
  }

  getBricksInProbe(probePosObj, probeSizeObj);

  sortBrickList(eye, normal, isOrtho);

  // Translate object by its position:
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  //glTranslatef(pos.e[0], pos.e[1], pos.e[2]);

  _sortedList.first();

  // Volume render a 3D texture:
  enableTexture(GL_TEXTURE_3D_EXT);

  while ((tmp = _sortedList.getData()) != 0)
  {
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

    dist = tmp->max;
    dist.sub(&tmp->min);

    for (i = 0; i < 3; i++)
    {
      texRange[i] = 1.0f - 1.0f / (float) tmp->texels[i];
      texMin[i] = 1.0f / (2.0f * (float) tmp->texels[i]);
    }

    glBindTexture(GL_TEXTURE_3D_EXT, texNames[tmp->index]);

    texPoint.copy(&farthest);

    for (i = 0; i < numSlices; ++i)               // loop thru all drawn textures
    {
      // Search for intersections between texture plane (defined by texPoint and
      // normal) and texture object (0..1):
      isectCnt = isect->isectPlaneCuboid(&normal, &texPoint, &min, &max);

      texPoint.add(&delta);

      if (isectCnt < 3) continue;                 // at least 3 intersections needed for drawing

      // Check volume section mode:
      if (minSlice != -1 && i < minSlice) continue;
      if (maxSlice != -1 && i > maxSlice) continue;

      // Put the intersecting 3 to 6 vertices in cyclic order to draw adjacent
      // and non-overlapping triangles:
      isect->cyclicSort(isectCnt, &normal);

      // Generate vertices in texture coordinates:
      for (j = 0; j < isectCnt; ++j)
      {
        for (k = 0; k < 3; ++k)
        {
          texcoord[j][k] = (isect[j][k] - tmp->min.e[k]) / dist[k];
          texcoord[j][k] = texcoord[j][k] * texRange[k] + texMin[k];
        }
      }

      glBegin(GL_TRIANGLE_FAN);
      glColor4f(1.0, 1.0, 1.0, 1.0);
      glNormal3f(normal[0], normal[1], normal[2]);
      ++drawn;

      for (j = 0; j < isectCnt; ++j)
      {
        // The following lines are the bottleneck of this method:
        glTexCoord3f(texcoord[j][0], texcoord[j][1], texcoord[j][2]);
        glVertex3f(isect[j][0], isect[j][1], isect[j][2]);
      }
      glEnd();
    }

    vvDebugMsg::msg(3, "Number of textures drawn: ", drawn);

    if (!_sortedList.next()) break;
  }

  disableTexture(GL_TEXTURE_3D_EXT);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

/** Renders the outline of the bricks, but not the volume data set.
 */
void vvTexRend::renderBricks(vvMatrix* mv)
{
  vvMatrix invMV;                                 // inverse of model-view matrix
  vvVector3 size, size2;                          // full and half object sizes
  vvVector3 eye;                                  // user's eye position [object space]
  vvVector3 probePosObj;                          // probe midpoint [object space]
  vvVector3 probeSizeObj;                         // probe size [object space]
  vvVector3 probeMin, probeMax;                   // probe min and max coordinates [object space]
  vvVector3 pos;                                  // volume location
  vvVector3 maxSize;                              // probe edge length [object space]
  int       i;                                    // general counters

  disableLUTMode();

  // Determine texture object dimensions and half object size as a shortcut:
  size.copy(vd->getSize());
  size2 = size;
  size2.scale(0.5);

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

  getBricksInProbe(probePosObj, probeSizeObj);

  // Translate object by its position:
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  //glTranslatef(pos.e[0], pos.e[1], pos.e[2]);

  Brick* tmp;
  vvVector3 min, max;

  _insideList.first();

  while ((tmp = _insideList.getData()) != 0)
  {
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

    glBegin(GL_LINES);
    glColor4f(1.0, 1.0, 1.0, 1.0);
    glVertex3f(min.e[0], min.e[1], min.e[2]);
    glVertex3f(max.e[0], min.e[1], min.e[2]);

    glVertex3f(min.e[0], min.e[1], min.e[2]);
    glVertex3f(min.e[0], max.e[1], min.e[2]);

    glVertex3f(min.e[0], min.e[1], min.e[2]);
    glVertex3f(min.e[0], min.e[1], max.e[2]);

    glVertex3f(max.e[0], max.e[1], max.e[2]);
    glVertex3f(min.e[0], max.e[1], max.e[2]);

    glVertex3f(max.e[0], max.e[1], max.e[2]);
    glVertex3f(max.e[0], min.e[1], max.e[2]);

    glVertex3f(max.e[0], max.e[1], max.e[2]);
    glVertex3f(max.e[0], max.e[1], min.e[2]);

    glVertex3f(max.e[0], min.e[1], min.e[2]);
    glVertex3f(max.e[0], max.e[1], min.e[2]);

    glVertex3f(max.e[0], min.e[1], min.e[2]);
    glVertex3f(max.e[0], min.e[1], max.e[2]);

    glVertex3f(min.e[0], max.e[1], min.e[2]);
    glVertex3f(max.e[0], max.e[1], min.e[2]);

    glVertex3f(min.e[0], max.e[1], min.e[2]);
    glVertex3f(min.e[0], max.e[1], max.e[2]);

    glVertex3f(min.e[0], min.e[1], max.e[2]);
    glVertex3f(max.e[0], min.e[1], max.e[2]);

    glVertex3f(min.e[0], min.e[1], max.e[2]);
    glVertex3f(min.e[0], max.e[1], max.e[2]);
    glEnd();

    if (!_insideList.next()) break;
  }

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  enableLUTMode();
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

bool vvTexRend::testBrickVisibility(Brick* brick)
{
  vvVector3 pv, normal;

  // get p-vertex (that's the farthest vertex in the direction of the normal plane
  for (int i = 0; i < 6; i++)
  {
    if (_frustum[i][0] > 0.0)
      pv[0] = brick->max[0];
    else
      pv[0] = brick->min[0];
    if (_frustum[i][1] > 0.0)
      pv[1] = brick->max[1];
    else
      pv[1] = brick->min[1];
    if (_frustum[i][2] > 0.0)
      pv[2] = brick->max[2];
    else
      pv[2] = brick->min[2];

    normal.set(_frustum[i][0], _frustum[i][1], _frustum[i][2]);

    if ((pv.dot(&normal) + _frustum[i][3]) < 0)
      return false;
  }

  return true;
}

bool vvTexRend::testBrickVisibility(Brick* brick, const vvMatrix& mvpMat)
{
  //sample the brick at many point and test them all for visibility
  float numSteps = 3;
  float xStep = (brick->max[0] - brick->min[0]) / (numSteps - 1.0f);
  float yStep = (brick->max[1] - brick->min[1]) / (numSteps - 1.0f);
  float zStep = (brick->max[2] - brick->min[2]) / (numSteps - 1.0f);
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

void vvTexRend::getBricksInProbe(vvVector3 pos, vvVector3 size)
{
  _brickList.first();
  for (int f = 1; f < vd->getCurrentFrame(); f++)
    _brickList.next();

  _brickList.getData()->first();
  _insideList.removeAll();

  vvVector3 tmpVec = size;
  tmpVec.scale(0.5);

  vvVector3 min = pos - tmpVec;
  vvVector3 max = pos + tmpVec;

  //   vvMatrix mvMat;
  //   vvMatrix pMat;
  //   vvMatrix mvpMat;
  //   getModelviewMatrix(&mvMat);
  //   getProjectionMatrix(&pMat);
  //   mvpMat = mvMat * pMat;

  updateFrustum();

  int countVisible = 0, countInvisible = 0;

  while (Brick *tmp = _brickList.getData()->getData())
  {
    if ((tmp->min.e[0] <= max.e[0]) && (tmp->max.e[0] >= min.e[0]) &&
      (tmp->min.e[1] <= max.e[1]) && (tmp->max.e[1] >= min.e[1]) &&
      (tmp->min.e[2] <= max.e[2]) && (tmp->max.e[2] >= min.e[2]))
    {
      //check if the brick is visible
      if (testBrickVisibility(tmp))
      {
        countVisible++;
        _insideList.append(tmp, vvSLNode<Brick*>::NO_DELETE);
      }
      else
        countInvisible++;
    }

    if (!_brickList.getData()->next())
    {
      //        cerr << "Bricks visible: " << countVisible << " Bricks invisible: " << countInvisible << endl;
      return;
    }
  }
}

void vvTexRend::sortBrickList(vvVector3 pos, vvVector3 normal, bool isOrtho)
{
  Brick* tmp;
  Brick* farthest = NULL;
  float max;

  _insideList.first();
  _sortedList.removeAll();

  while ((tmp = _insideList.getData()) != 0)
  {
    if (isOrtho)
      tmp->dist = -tmp->pos.dot(&normal);
    else
      tmp->dist = (tmp->pos - pos).length();

    if (!_insideList.next()) break;
  }

  while (_insideList.count() != _sortedList.count())
  {
    _insideList.first();
    max = -FLT_MAX;
    farthest = NULL;
    while (true)
    {
      tmp = _insideList.getData();
      if (tmp->dist > max)
      {
        farthest = tmp;
        max = farthest->dist;
      }

      if (!_insideList.next()) break;
    }

    if(!farthest)
      break;

    _sortedList.append(farthest, vvSLNode<Brick*>::NO_DELETE);
    farthest->dist = -FLT_MAX;
  }
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
    updateLUT(diagonalVoxels / numShells);
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
    updateLUT(diagVoxels/numTextures);
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
  //glTranslatef(pos.e[0], pos.e[1], pos.e[2]);

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
    updateLUT(diagVoxels/numTextures);
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
  AxisType principal;                             // principal viewing direction
  vvMatrix mv;                                    // current modelview matrix
  vvVector3 origin(0.0f, 0.0f, 0.0f);             // zero vector
  vvVector3 xAxis(1.0f, 0.0f, 0.0f);              // vector in x axis direction
  vvVector3 yAxis(0.0f, 1.0f, 0.0f);              // vector in y axis direction
  vvVector3 zAxis(0.0f, 0.0f, 1.0f);              // vector in z axis direction
  vvVector3 probeSizeObj;                         // probe size [object space]
  vvVector3 size;                                 // volume size [world coordinates]
  float zx, zy, zz;                               // base vector z coordinates
  int i;

  vvDebugMsg::msg(3, "vvTexRend::renderVolumeGL()");

  sw.start();

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

  // Determine texture object extensions:
  for (i=0; i<3; ++i)
  {
    texMin[i] = 0.5f / (float)texels[i];
    texMax[i] = (float)vd->vox[i] / (float)texels[i] - texMin[i];
  }

  // Find principle viewing direction:

  // Get OpenGL modelview matrix:
  getModelviewMatrix(&mv);

  // Transform 4 point vectors with the modelview matrix:
  origin.multiply(&mv);
  xAxis.multiply(&mv);
  yAxis.multiply(&mv);
  zAxis.multiply(&mv);

  // Generate coordinate system base vectors from those vectors:
  xAxis.sub(&origin);
  yAxis.sub(&origin);
  zAxis.sub(&origin);

  xAxis.normalize();
  yAxis.normalize();
  zAxis.normalize();

  // Only z component of base vectors is needed:
  zx = xAxis.e[2];
  zy = yAxis.e[2];
  zz = zAxis.e[2];

  if (fabs(zx) > fabs(zy))
  {
    if (fabs(zx) > fabs(zz)) principal = X_AXIS;
    else principal = Z_AXIS;
  }
  else
  {
    if (fabs(zy) > fabs(zz)) principal = Y_AXIS;
    else principal = Z_AXIS;
  }

  enableLUTMode();
  switch (geomType)
  {
    default:
    case VV_SLICES:    renderTex2DSlices(zz); break;
    case VV_CUBIC2D:   renderTex2DCubic(principal, zx, zy, zz); break;
    case VV_SPHERICAL: renderTex3DSpherical(&mv); break;
    case VV_VIEWPORT:  renderTex3DPlanar(&mv); break;
    case VV_BRICKS:
      if (_renderState._showBricks)
        renderBricks(&mv);
      else
        renderTexBricks(&mv);
      break;
  }
  disableLUTMode();

  vvRenderer::renderVolumeGL();

  unsetGLenvironment();

  glFinish();                                     // make sure rendering is done to measure correct time
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
#ifdef HAVE_CG
  else if (_currentShader==8 && voxelType==VV_PIX_SHD)
  {
    x = y = 256;
    z = 1;
  }
#endif
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
void vvTexRend::updateLUT(float dist)
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
        rgbaLUT[i * 4 + c] = uchar(corr[c] * 255.0f);
      }
    }
  }

  // Copy LUT to graphics card:
  vvGLTools::printGLError("enter updateLUT()");
  switch (voxelType)
  {
    case VV_RGBA:
      makeTextures();                           // this mode doesn't use a hardware LUT, so every voxel has to be updated
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
          makeTextures();
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
        updateTransferFunction();
        break;
      case vvRenderer::VV_BINNING:
        if (newValue==0.0f) vd->_binning = vvVolDesc::LINEAR;
        else if (newValue==1.0f) vd->_binning = vvVolDesc::ISO_DATA;
        else vd->_binning = vvVolDesc::OPACITY;
        break;      
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
#ifdef HAVE_CG
        return vvGLTools::isGLextensionSupported("GL_ARB_fragment_program");
#else
        return false;
#endif
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
      case VV_MIP: if (glBlendEquationVV) return true; else return false;
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
    makeTextures();
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
  void vvTexRend::enableFragProg()
  {
    glActiveTextureARB(GL_TEXTURE1_ARB);
    glBindTexture(GL_TEXTURE_2D, pixLUTName);
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

#ifdef HAVE_CG
  //----------------------------------------------------------------------------
  /// Automatically called when a Cg error occurs.
  void checkCgError(CGcontext ctx, CGerror cgerr, void *)
  {
    if(cgerr != CG_NO_ERROR) 
      cerr << cgGetErrorString(cgerr) << "(" << static_cast<int>(cgerr) << ")" << endl;
    for(GLint glerr = glGetError(); glerr != GL_NO_ERROR; glerr = glGetError())
    {
      cerr << "GL error: " << gluErrorString(glerr) << endl;
    }
    if(ctx && cgerr==CG_COMPILER_ERROR)
    {
       if(const char *listing = cgGetLastListing(ctx))
       {
          cerr << "last listing:" << endl;
          cerr << listing << endl;
       }
    }
  }
#endif

  //----------------------------------------------------------------------------
  void vvTexRend::enablePixelShaders()
  {
#ifdef HAVE_CG
    if(VV_PIX_SHD == voxelType)
    {
      // Load, enable, and bind fragment shader:
      cgGLLoadProgram(_cgProgram[_currentShader]);
      cgGLEnableProfile(_cgFragProfile[_currentShader]);
      cgGLBindProgram(_cgProgram[_currentShader]);

      // Set fragment program parameters:
      if (_currentShader != 4)                    // pixLUT, doesn't work with grayscale shader
      {
        glBindTexture(GL_TEXTURE_2D, pixLUTName);
        _cgPixLUT = cgGetNamedParameter(_cgProgram[_currentShader], "pixLUT");
        cgGLSetTextureParameter(_cgPixLUT, pixLUTName);
        cgGLEnableTextureParameter(_cgPixLUT);
      }
                                                  // chan4color
      if (_currentShader == 3 || _currentShader == 7)
      {
        _cgChannel4Color = cgGetNamedParameter(_cgProgram[_currentShader], "chan4color");
        cgGLSetParameter3fv(_cgChannel4Color, _channel4Color);
      }
                                                  // opWeights
      if (_currentShader > 4 && _currentShader < 8)
      {
        _cgOpacityWeights = cgGetNamedParameter(_cgProgram[_currentShader], "opWeights");
        cgGLSetParameter4fv(_cgOpacityWeights, _opacityWeights);
      }
    }
#endif
  }

  //----------------------------------------------------------------------------
  void vvTexRend::disablePixelShaders()
  {
#ifdef HAVE_CG
    if(VV_PIX_SHD == voxelType)
    {
      cgGLDisableTextureParameter(_cgPixLUT);
      cgGLDisableProfile(_cgFragProfile[_currentShader]);
    }
#endif
  }

  //----------------------------------------------------------------------------
  /** @return true if initialization successful
   */
  bool vvTexRend::initPixelShaders()
  {
#ifdef HAVE_CG
#ifdef _WIN32
    const char* primaryWin32ShaderDir = "..\\..\\..\\virvo\\shader";
#endif
    const char* shaderFileName = "vv_shader";
    const char* shaderEnv = "VV_SHADER_PATH";
    const char* shaderExt = ".cg";
    const char* unixShaderDir = NULL;
    char* shaderFile = NULL;
    char* shaderPath = NULL;
    char shaderDir[256];
    int i;

    cerr << "enable PIX called"<< endl;

    _cgContext = cgCreateContext();               // Create context for fragment programs
    if (!_cgContext) cerr << "Could not create Cg context." << endl;
    cgSetErrorHandler(checkCgError, NULL);

    // Check if correct version of pixel shaders is available:
    if(cgGLIsProfileSupported(CG_PROFILE_ARBFP1)) // test for GL_ARB_fragment_program
    {
                                                  // FIXME: why isn't this a show stopper?
      cerr << "Hardware may not support extension CG_PROFILE_ARBFP1" << endl;
      //    return false;
    }
                                                  // test for GL_NV_fragment_program
    if(cgGLIsProfileSupported(CG_PROFILE_FP20)==CG_TRUE)
    {
      cerr << "Hardware may not support extension CG_PROFILE_VP20" << endl;
      //    return false;
    }
                                                  // test for GL_NV_fragment_program
    if(cgGLIsProfileSupported(CG_PROFILE_FP30)==CG_TRUE)
    {
      cerr << "Hardware may not support extension CG_PROFILE_VP30" << endl;
      //    return false;
    }

    // Specify shader path:
    cerr << "Searching for shader files..." << endl;
    if (getenv(shaderEnv))
    {
      cerr << "Environment variable " << shaderEnv << " found: " << getenv(shaderEnv) << endl;
      unixShaderDir = getenv(shaderEnv);
    }
    else
    {
      cerr << "Warning: you should set the environment variable " << shaderEnv << " to point to your shader directory" << endl;
#ifdef _WIN32
      vvToolshed::getProgramDirectory(shaderDir, 256);
      strcat(shaderDir, primaryWin32ShaderDir);
      cerr << "Trying shader path: " << shaderDir << endl;
      if (!vvToolshed::isDirectory(shaderDir))
      {
	 vvToolshed::getProgramDirectory(shaderDir, 256);
      }
      cerr << "Using shader path: " << shaderDir << endl;
      unixShaderDir = shaderDir;
#else
      const char* deskVoxShaderPath = "../";
#ifdef SHADERDIR
      unixShaderDir = SHADERDIR;
#else
      vvToolshed::getProgramDirectory(shaderDir, 256);
      strcat(shaderDir, deskVoxShaderPath);
      unixShaderDir = shaderDir;
#endif
#endif
    }
    cerr << "Using shader path: " << unixShaderDir << endl;

    // Load shader files:
    for (i=0; i<NUM_PIXEL_SHADERS; ++i)
    {
      shaderFile = new char[strlen(shaderFileName) + 2 + strlen(shaderExt) + 1];
      sprintf(shaderFile, "%s%02d%s", shaderFileName, i+1, shaderExt);

      _cgFragProfile[i] = CG_PROFILE_ARBFP1;      // The GL Fragment Profile
      cgGLSetOptimalOptions(_cgFragProfile[i]);

      // Load Vertex Shader From File:
      // FIXME: why don't relative paths work under Linux?
      shaderPath = new char[strlen(unixShaderDir) + 1 + strlen(shaderFile) + 1];
#ifdef _WIN32
      sprintf(shaderPath, "%s\\%s", unixShaderDir, shaderFile);
#else
      sprintf(shaderPath, "%s/%s", unixShaderDir, shaderFile);
#endif

      cerr << "Loading shader file: " << shaderPath << endl;
      _cgProgram[i] = cgCreateProgramFromFile(_cgContext, CG_SOURCE, shaderPath, _cgFragProfile[i], "main", 0);

      delete[] shaderFile;
      delete[] shaderPath;

      // Validate success:
      if (_cgProgram[i] == NULL)
      {
        cerr << "Error: failed to compile fragment program " << i+1 << endl;
        return false;
      }
      else cerr << "Fragment program " << i+1 << " compiled" << endl;
    }

    cerr << "Fragment programs ready." << endl;
#endif
    return true;
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

  //============================================================================
  // End of File
  //============================================================================
