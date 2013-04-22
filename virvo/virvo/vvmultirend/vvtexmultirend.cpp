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
#include <GL/glew.h>
#define GLEW_INCLUDED
#endif

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

#include "../vvopengl.h"

#include "../vvdynlib.h"
#if !defined(_WIN32) && !defined(__APPLE__)
#include <dlfcn.h>
#include <GL/glx.h>
#endif

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "../vvvecmath.h"
#include "../vvdebugmsg.h"
#include "../vvtoolshed.h"
#include "../private/vvgltools.h"
#include "../private/vvlog.h"
#include "../vvsphere.h"
#include "../vvclock.h"
#include "../vvprintgl.h"
#include "vvtexmultirend.h"

using namespace std;

//----------------------------------------------------------------------------
const int vvTexMultiRend::NUM_PIXEL_SHADERS = 10;

//----------------------------------------------------------------------------
/** Constructor.
  @param vd  volume description
  @param m   render geometry (default: automatic)
*/
vvTexMultiRend::vvTexMultiRend(vvVolDesc* vd, vvRenderState renderState, GeometryType geom, VoxelType vox) : vvRenderer(vd, renderState)
{
  vvDebugMsg::msg(1, "vvTexMultiRend::vvTexMultiRend()");

  rendererType = TEXREND;
  texNames = NULL;
  _sliceOrientation = VV_VARIABLE;
  viewDir.zero();
  objDir.zero();
  minSlice = maxSlice = -1;
  rgbaTF  = new float[256 * 256 * 4];
  rgbaLUT = new uchar[256 * 256 * 4];
  preintTable = new uchar[getPreintTableSize()*getPreintTableSize()*4];

  _ntextures = 0;
  interpolation = true;
  quality = 1.0f;
  rotation.identity();
  tfmode = GAMMATF;
  lutDistance = 1.0;


  // Find out which OpenGL extensions are supported:
#if defined(GL_VERSION_1_2)
  extTex3d  = true;
#else
  extTex3d  = vvGLTools::isGLextensionSupported("GL_EXT_texture3D");
#endif
  extMinMax = vvGLTools::isGLextensionSupported("GL_EXT_blend_minmax");
  extBlendEquation = vvGLTools::isGLextensionSupported("GL_EXT_blend_equation");

  // Init glew.
  glewInit();

  // Determine best rendering algorithm for current hardware:
#ifdef NDEBUG
  (void)vox;
  (void)geom;
#else
  assert(vox == VV_GLSL);
  assert(geom == VV_VIEWPORT);
#endif

  voxelType = VV_GLSL;
  geomType = VV_VIEWPORT;

  cerr << "Rendering algorithm: VV_GLSL, VV_VIEWPORT";
  init();
}

//----------------------------------------------------------------------------
/// Destructor
vvTexMultiRend::~vvTexMultiRend()
{
  vvDebugMsg::msg(1, "vvTexMultiRend::~vvTexMultiRend()");
  removeTextures();

  glDeleteTextures(1, &tfTexName);
  glDeleteTextures(vd->chan+1, pixLUTName);

  delete[] rgbaTF;
  delete[] rgbaLUT;
  delete[] preintTable;

  delete[] chanWeight;
  delete[] color;
  delete[] tfGamma;
  delete[] tfHPOrder;
  delete[] tfHPCutoff;
  delete[] tfOffset;
  delete[] histCDF;
}

//----------------------------------------------------------------------------
/// Remove all textures from texture memory.
void vvTexMultiRend::removeTextures()
{
  vvDebugMsg::msg(1, "vvTexMultiRend::removeTextures()");

  if (_ntextures>0)
  {
    glDeleteTextures(_ntextures, texNames);
    delete[] texNames;
    texNames = NULL;
    _ntextures = 0;
  }
}

//----------------------------------------------------------------------------
/// Generate textures for all rendering modes.
vvTexMultiRend::ErrorType vvTexMultiRend::makeTextures()
{
  ErrorType err = OK;
  vvDebugMsg::msg(2, "vvTexMultiRend::makeTextures()");

  // Compute texture dimensions (must be power of 2):
  texels[0] = vvToolshed::getTextureSize(vd->vox[0]);
  texels[1] = vvToolshed::getTextureSize(vd->vox[1]);
  texels[2] = vvToolshed::getTextureSize(vd->vox[2]);

  updateTextures3D(0, 0, 0, texels[0], texels[1], texels[2], true); 

  makeLUTTexture(); // FIXME: works only once, then generates OpenGL error

  vvGLTools::printGLError("vvTexMultiRend::makeTextures");
  return err;
}

//----------------------------------------------------------------------------
/// Generate texture for look-up table.
void vvTexMultiRend::makeLUTTexture()
{
  int size[3];

  vvGLTools::printGLError("enter makeLUTTexture");
  getLUTSize(size);
  if (vd->chan > 1)
  {
	  for (size_t i = 0; i < vd->chan+1; i++)
	  {
		vvDebugMsg::msg(1, "makeLUTTexture(): ", (int)pixLUTName[i]);
		  glBindTexture(GL_TEXTURE_1D, pixLUTName[i]);
		  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
		  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		  glTexImage1D(GL_TEXTURE_1D, 0, GL_LUMINANCE, size[0], 0,
			  GL_LUMINANCE, GL_UNSIGNED_BYTE, &rgbaLUT[i*size[0]]);
	  }
  }
  else
  {
	  glBindTexture(GL_TEXTURE_2D, pixLUTName[0]);
	  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size[0], size[1], 0,
		  GL_RGBA, GL_UNSIGNED_BYTE, rgbaLUT);
  }
  vvGLTools::printGLError("leave makeLUTTexture");
}

//----------------------------------------------------------------------------
/// Update transfer function from volume description.
void vvTexMultiRend::updateTransferFunction()
{
  int size[3];

  // Generate arrays from pins:
  getLUTSize(size);
 
  // 2D transfer functions for GLSL
  if (vd->chan > 1)
  {
	switch (tfmode)
	{
	  case GAMMATF:
		vvDebugMsg::msg(1, "size, chan, tfGamma, tfOffset: ", size[0], vd->chan);
		vd->tf.computeTFTextureGamma(size[0], rgbaTF, vd->real[0], vd->real[1], vd->chan, tfGamma, tfOffset );
		break;
	  case HIGHPASSTF:
		vd->tf.computeTFTextureHighPass(size[0], rgbaTF, vd->real[0], vd->real[1], vd->chan, tfHPOrder, tfHPCutoff, tfOffset );
		break;
	  case HISTCDFTF:
		vd->tf.computeTFTextureHistCDF(size[0], rgbaTF, vd->real[0], vd->real[1], vd->chan, vd->getCurrentFrame(), histCDF, tfGamma, tfOffset);
		break;
	}
  } else {
	vd->computeTFTexture(size[0], size[1], size[2], rgbaTF);
  }

  updateLUT(1.0f/quality);               // generate color/alpha lookup table

  //printLUT();
}

//----------------------------------------------------------------------------
// see parent in vvRenderer
void vvTexMultiRend::updateVolumeData()
{
  vvRenderer::updateVolumeData();
  makeTextures();
}

//----------------------------------------------------------------------------
void vvTexMultiRend::updateVolumeData(int offsetX, int offsetY, int offsetZ,
  int sizeX, int sizeY, int sizeZ)
{
  updateTextures3D(offsetX, offsetY, offsetZ, sizeX, sizeY, sizeZ, false);
}

//----------------------------------------------------------------------------
/**
   Method to create a new 3D texture or update parts of an existing 3D texture.
   @param offsetX, offsetY, offsetZ: lower left corner of texture
   @param sizeX, sizeY, sizeZ: size of texture
   @param newTex: true: create a new texture
                  false: update an existing texture
*/
vvTexMultiRend::ErrorType vvTexMultiRend::updateTextures3D(int offsetX, int offsetY, int offsetZ,
int sizeX, int sizeY, int sizeZ, bool newTex)
{
  ErrorType err = OK;
  size_t frames;
  size_t texSize;
  size_t sliceSize;
  size_t rawSliceOffset;
  size_t heightOffset;
  size_t texLineOffset;
  size_t srcIndex;
  size_t texOffset = 0;
  int rawVal[10];
  //int alpha;
  float fval;
  uint8_t* raw;
  uint8_t* texData;
  bool accommodated = true;
  GLint glWidth;

  vvDebugMsg::msg(1, "vvTexMultiRend::updateTextures3D()");

  if (!extTex3d) return NO3DTEX;

  frames = vd->frames;

  if (vd->chan > 1) texelsize = vd->chan;
  texSize = sizeX * sizeY * sizeZ * texelsize;

  vvDebugMsg::msg(1, "3D Texture width     = ", sizeX);
  vvDebugMsg::msg(1, "3D Texture height    = ", sizeY);
  vvDebugMsg::msg(1, "3D Texture depth     = ", sizeZ);
  VV_LOG(1) << "3D Texture size (KB) = " << texSize / 1024 << std::endl;

  texData = new uchar[texSize];
  memset(texData, 0, texSize);

  assert(vd->chan <= 10);

  sliceSize = vd->getSliceBytes();

  if (newTex)
  {
    VV_LOG(1) << "Creating texture names. # of names: " << frames << std::endl;

    removeTextures();
	
	if (vd->chan > 1)
		_ntextures  = frames * vd->chan;
	else
		_ntextures  = frames;
    delete[] texNames;
    texNames = new GLuint[_ntextures];
	glGenTextures(_ntextures, texNames);
  }

  VV_LOG(1) << "Transferring textures to TRAM. Total size [KB]: " << frames * texSize / 1024 << std::endl;

  // Generate sub texture contents:
  for (size_t f = 0; f < frames; f++)
  {
    raw = vd->getRaw(f);
    for (size_t s = offsetZ; s < (offsetZ + sizeZ); s++)
    {
      rawSliceOffset = (vd->vox[2] - min(s,vd->vox[2]-1) - 1) * sliceSize;
      for (size_t y = offsetY; y < (offsetY + sizeY); y++)
      {
        heightOffset = (vd->vox[1] - min(y,vd->vox[1]-1) - 1) * vd->vox[0] * vd->bpc * vd->chan;
        texLineOffset = (y - offsetY) * sizeX + (s - offsetZ) * sizeX * sizeY;
        if (vd->chan == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
        {
          for (size_t x = offsetX; x < (offsetX + sizeX); x++)
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
			texData[4 * texOffset] = (uchar) rawVal[0];
          }
        }
        else if (vd->bpc==1 || vd->bpc==2 || vd->bpc==4)
        {
		  size_t size = sizeX * sizeY * sizeZ;
		  for (size_t x = offsetX; x < (offsetX + sizeX); x++)
		  {
			texOffset = (x - offsetX) + texLineOffset;
			for (size_t c = 0; c < vd->chan; c++)
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
			for (size_t c = 0; c < vd->chan; c++)					
			  texData[c * size + texOffset] = (uchar) rawVal[c];
		  }
		}
		else cerr << "Cannot create texture: unsupported voxel format (3)." << endl;
	  }
	}

	if (newTex)
	{
	  if (vd->chan > 1)   
	  {
		size_t size = sizeX * sizeY * sizeZ;
		for (size_t c = 0; c < vd->chan; c++)
		{
		  // interleave channels within each frame for texName
		  glBindTexture(GL_TEXTURE_3D_EXT, texNames[f*vd->chan + c]);
		  glPixelStorei(GL_UNPACK_ALIGNMENT,1);
		  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
		  glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
		  glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
		  glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP);
		  glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP);
		  glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP);

		  // consecutive data in each channel for texData
		  // GL_LUMINANCE will make R,G,B equal to this value and alpha 1 in the shader
		  glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_LUMINANCE, texels[0], texels[1], texels[2], 0,
			  GL_LUMINANCE, GL_UNSIGNED_BYTE, &texData[c * size]);
		  glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D_EXT, 0, GL_TEXTURE_WIDTH, &glWidth);
		  if (glWidth!=0)	accommodated = false;

		  // FIXME: WARNING! DOESN'T WORK FOR >1 FRAME
		  updateChannelHistCDF(c, f, &texData[c * size]);
		}
	  }
	  else
	  {
		glBindTexture(GL_TEXTURE_3D_EXT, texNames[f]);
		glPixelStorei(GL_UNPACK_ALIGNMENT,1);
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
		glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
		glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
		glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP);

		glTexImage3DEXT(GL_PROXY_TEXTURE_3D_EXT, 0, internalTexFormat, texels[0], texels[1], texels[2], 
						0, texFormat, GL_UNSIGNED_BYTE, NULL);
		glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D_EXT, 0, GL_TEXTURE_WIDTH, &glWidth);
		if (glWidth!=0)
		  glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, internalTexFormat, texels[0], texels[1], texels[2], 
			  			0, texFormat, GL_UNSIGNED_BYTE, texData);
		else
		  accommodated = false;
	  }
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

//----------------------------------------------------------------------------
/** Render a volume entirely if probeSize=0 or a cubic sub-volume of size probeSize.
  @param mv        model-view matrix
*/
void vvTexMultiRend::renderTex3DPlanar(vvMatrix* mv)
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
  //int       numSlices;

  vvDebugMsg::msg(3, "vvTexMultiRend::renderTex3DPlanar()");

  if (!extTex3d) return;                          // needs 3D texturing extension

  // Determine texture object dimensions and half object size as a shortcut:
  size = vvVector3(vd->getSize());
  for (i=0; i<3; ++i)
  {
    texSize[i] = size[i] * (float)texels[i] / (float)vd->vox[i];
    size2[i]   = 0.5f * size[i];
  }
  pos = vvVector3(vd->pos);

  // Calculate inverted modelview matrix:
  invMV = vvMatrix(*mv);
  invMV.invert();

  // Find eye position:
  getEyePosition(&eye);

  if (_isROIUsed)
  {
    // Convert probe midpoint coordinates to object space w/o position:
    probePosObj = vvVector3(_roiPos);
    probePosObj.sub(pos);                        // eliminate object position from probe position

    // Compute probe min/max coordinates in object space:
    for (i=0; i<3; ++i)
    {
      probeMin[i] = probePosObj[i] - (_roiSize[i] * size[i]) / 2.0f;
      probeMax[i] = probePosObj[i] + (_roiSize[i] * size[i]) / 2.0f;
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
    probeSizeObj = vvVector3(size);
    probeMin.set(-size2[0], -size2[1], -size2[2]);
    probeMax = vvVector3(size2);
    probePosObj.zero();
  }

  // Initialize texture counters
  if (_roiSize[0])
  {
    probeTexels.zero();
    for (i=0; i<3; ++i)
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

  // Compute normal vector of textures using the following strategy:
  // For orthographic projections or if viewDir is (0|0|0) use
  // (0|0|1) as the normal vector.
  // Otherwise use objDir as the normal.
  // Exception: if user's eye is inside object and probe mode is off,
  // then use viewDir as the normal.
  if (_sliceOrientation==VV_CLIPPLANE ||
     (_sliceOrientation==VV_VARIABLE && _clipMode == 1))
  {
    normal = vvVector3(_clipPlaneNormal);
  }
  else if(_sliceOrientation==VV_VIEWPLANE)
  {
    normal.set(0.0f, 0.0f, 1.0f);                 // (0|0|1) is normal on projection plane
    vvMatrix invPM;
    invPM = vvMatrix(pm);
    invPM.invert();
    normal.multiply(invPM);
    normal.multiply(invMV);
    normal.negate();
  }
  else if (_sliceOrientation==VV_ORTHO ||
          (_sliceOrientation==VV_VARIABLE &&
          (isOrtho || (viewDir[0]==0.0f && viewDir[1]==0.0f && viewDir[2]==0.0f))))
  {
    // Draw slices parallel to projection plane:
    normal.set(0.0f, 0.0f, 1.0f);                 // (0|0|1) is normal on projection plane
    normal.multiply(invMV);
    origin.zero();
    origin.multiply(invMV);
    normal.sub(origin);
  }
  else if (_sliceOrientation==VV_VIEWDIR || 
          (_sliceOrientation==VV_VARIABLE && (!_isROIUsed && isInVolume(&eye))))
  {
    // Draw slices perpendicular to viewing direction:
    normal = vvVector3(viewDir);
    normal.negate();                              // viewDir points away from user, the normal should point towards them
  }
  else
  {
    // Draw slices perpendicular to line eye->object:
    normal = vvVector3(objDir);
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

  float sliceDistance = voxelDistance / _quality;
  if(_isROIUsed && _quality < 2.0)
  {
    // draw at least twice as many slices as there are samples in the probe depth.
    sliceDistance = voxelDistance / 2.0f;
  }
  numSlices = 2*(int)ceilf(depth/sliceDistance*.5f);

  if (numSlices < 1)                              // make sure that at least one slice is drawn
    numSlices = 1;

  vvDebugMsg::msg(3, "Number of textures rendered: ", numSlices);

  // Use alpha correction in indexed mode: adapt alpha values to number of textures:
  float thickness = sliceDistance/voxelDistance;

  // just tolerate slice distance differences imposed on us
  // by trying to keep the number of slices constant
  if(lutDistance/thickness < 0.88 || thickness/lutDistance < 0.88)
  {
	updateLUT(thickness);
  }

  //vvDebugMsg::msg(1, "voxelDistance, sliceDistance, thickness: ", voxelDistance, sliceDistance, thickness, 1.0f/thickness);

  delta = vvVector3(normal);
  delta.scale(sliceDistance);

  // Compute farthest point to draw texture at:
  farthest = vvVector3(delta);
  farthest.scale((float)(numSlices - 1) / -2.0f);
  farthest.add(probePosObj);

  if (_clipMode == 1)                     // clipping plane present?
  {
    // Adjust numSlices and set farthest point so that textures are only
    // drawn up to the clipPoint. (Delta may not be changed
    // due to the automatic opacity correction.)
    // First find point on clipping plane which is on a line perpendicular
    // to clipping plane and which traverses the origin:
    temp = vvVector3(delta);
    temp.scale(-0.5f);
    farthest.add(temp);                          // add a half delta to farthest
    clipPosObj = vvVector3(_clipPlanePoint);
    clipPosObj.sub(pos);
    temp = vvVector3(probePosObj);
    temp.add(normal);
    normClipPoint.isectPlaneLine(normal, clipPosObj, probePosObj, temp);
    maxDist = farthest.distance(normClipPoint);
    numSlices = (int)(maxDist / delta.length()) + 1;
    temp = vvVector3(delta);
    temp.scale((float)(1 - numSlices));
    farthest = vvVector3(normClipPoint);
    farthest.add(temp);
    if (_clipSingleSlice)
    {
      // Compute slice position:
      temp = vvVector3(delta);
      temp.scale((float)(numSlices-1));
      farthest.add(temp);
      numSlices = 1;

      // Make slice opaque if possible:
	  updateLUT(0.0f);
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
  deltahalf = vvVector3(delta);
  deltahalf.scale(0.5f);

  // Relative viewing position
  vvVector3 releye;
  releye = vvVector3(eye);
  releye.sub(pos);

  // Volume render a 3D texture:
  glEnable(GL_TEXTURE_3D_EXT);
  glBindTexture(GL_TEXTURE_3D_EXT, texNames[vd->getCurrentFrame()]);
  texPoint = vvVector3(farthest);
  for (i=0; i<numSlices; ++i)                     // loop thru all drawn textures
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
	for (j=0; j<isectCnt; ++j)
	{
	  for (k=0; k<3; ++k)
	  {
            texcoord[j][k] = (isect[j][k] + size2[k]) / size[k];
	    texcoord[j][k] = texcoord[j][k] * (texMax[k] - texMin[k]) + texMin[k];
	  }
	}

    glBegin(GL_TRIANGLE_FAN);
    glColor4f(1.0, 1.0, 1.0, 1.0);
    glNormal3f(normal[0], normal[1], normal[2]);
    ++drawn;
    for (j=0; j<isectCnt; ++j)
    {
	  glTexCoord3f(texcoord[j][0], texcoord[j][1], texcoord[j][2]);
      glVertex3f(isect[j][0], isect[j][1], isect[j][2]);
    }
    glEnd();
  }
  vvDebugMsg::msg(3, "Number of textures drawn: ", drawn);
  glDisable(GL_TEXTURE_3D_EXT);

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

//----------------------------------------------------------------------------
/** Render the volume onto currently selected drawBuffer.
 Viewport size in world coordinates is -1.0 .. +1.0 in both x and y direction
*/
void vvTexMultiRend::renderVolumeGL()
{
  static vvStopwatch sw;                          // stop watch for performance measurements
  vvMatrix mv;                                    // current modelview matrix
  vvVector3 origin(0.0f, 0.0f, 0.0f);             // zero vector
  vvVector3 xAxis(1.0f, 0.0f, 0.0f);              // vector in x axis direction
  vvVector3 yAxis(0.0f, 1.0f, 0.0f);              // vector in y axis direction
  vvVector3 zAxis(0.0f, 0.0f, 1.0f);              // vector in z axis direction
  vvVector3 probeSizeObj;                         // probe size [object space]
  vvVector3 size;                                 // volume size [world coordinates]
  int i;

  vvDebugMsg::msg(3, "vvTexMultiRend::renderVolumeGL()");

  sw.start();

  size = vvVector3(vd->getSize());

  // Draw boundary lines (must be done before setGLenvironment()):
  if (_boundaries)
  {
    drawBoundingBox(size, vd->pos, _boundColor);
  }
  if (_isROIUsed)
  {
    probeSizeObj.set(size[0] * _roiSize[0], size[1] * _roiSize[1], size[2] * _roiSize[2]);
    drawBoundingBox(probeSizeObj, _roiPos, _probeColor);
  }
  if (_clipMode == 1 && _clipPlanePerimeter)
  {
    drawPlanePerimeter(size, vd->pos, _clipPlanePoint, _clipPlaneNormal, _clipPlaneColor);
  }

  //setGLenvironment();

  // Determine texture object extensions:
  for (i=0; i<3; ++i)
  {
    texMin[i] = 0.5f / (float)texels[i];
    texMax[i] = (float)vd->vox[i] / (float)texels[i] - texMin[i];
  }

  // Find principle viewing direction:

  // Get OpenGL modelview matrix:
  vvGLTools::getModelviewMatrix(&mv);

  // Transform 4 point vectors with the modelview matrix:
  origin.multiply(mv);
  xAxis.multiply(mv);
  yAxis.multiply(mv);
  zAxis.multiply(mv);

  // Generate coordinate system base vectors from those vectors:
  xAxis.sub(origin);
  yAxis.sub(origin);
  zAxis.sub(origin);

  xAxis.normalize();
  yAxis.normalize();
  zAxis.normalize();

  renderTex3DPlanar(&mv); 
  vvRenderer::renderVolumeGL();

  glFinish();                                     // make sure rendering is done to measure correct time
  _lastRenderTime = sw.getTime();

  vvDebugMsg::msg(3, "vvTexMultiRend::renderVolumeGL() done");
}


//----------------------------------------------------------------------------
/// Returns the number of entries in the RGBA lookup table.
int vvTexMultiRend::getLUTSize(int* size)
{
  int x, y, z;

  vvDebugMsg::msg(3, "vvTexMultiRend::getLUTSize()");
  x = 256;
  y = z = 1;

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
int vvTexMultiRend::getPreintTableSize()
{
  vvDebugMsg::msg(1, "vvTexMultiRend::getPreintTableSize()");
  return 256;
}

//----------------------------------------------------------------------------
/** Update the color/alpha look-up table.
 Note: glColorTableSGI can have a maximum width of 1024 RGBA entries on IR2 graphics!
 @param dist  slice distance relative to 3D texture sample point distance
              (1.0 for original distance, 0.0 for all opaque).
*/
void vvTexMultiRend::updateLUT(float dist)
{
  vvDebugMsg::msg(3, "Generating texture LUT. Slice distance = ", dist);

  float corr[4];                                  // gamma/alpha corrected RGBA values [0..1]
  int lutSize[3];                                 // number of entries in the RGBA lookup table
  size_t i,c;
  size_t total;

  lutDistance = dist;
	
  if (vd->chan > 1)  // multi channel
  {
	//vvDebugMsg::msg(1, "updateLUT with VV_GLSL");
	total = getLUTSize(lutSize);
	  
	for (i=0; i<total*vd->chan; ++i)
	  // Convert float to uchar and copy to vd->rgbaLUT array:
	  rgbaLUT[i] = uchar(rgbaTF[i] * 255.0f);

	// additional texture for opacity tf
	for (i=total*vd->chan; i<total*(vd->chan+1); ++i)
	{
	  float alpha = ts_clamp(1.0f - powf(1.0f - rgbaTF[i], dist), 0.0f, 1.0f);
	  rgbaLUT[i] = uchar(alpha * 255.0f);
	}	  

	// Copy LUT to graphics card:
	vvGLTools::printGLError("enter updateLUT()");

	for (size_t i = 0; i < vd->chan+1; i++)
	{
	  glBindTexture(GL_TEXTURE_1D, pixLUTName[i]);
	  glTexImage1D(GL_TEXTURE_1D, 0, GL_LUMINANCE, lutSize[0], 0, GL_LUMINANCE, 
					GL_UNSIGNED_BYTE, rgbaLUT + i * lutSize[0]);
	}

	vvGLTools::printGLError("leave updateLUT()");
  }
  else    // single channel
  {
    total = getLUTSize(lutSize);
    for (i=0; i<total; ++i)
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
      
	  corr[3] = 1.0f - powf(1.0f - corr[3], dist);

      // Convert float to uchar and copy to vd->rgbaLUT array:
      for (c=0; c<4; ++c)
      {
        rgbaLUT[i * 4 + c] = uchar(corr[c] * 255.0f);
      }
    }

	// Copy LUT to graphics card:
	vvGLTools::printGLError("enter updateLUT()");

	glBindTexture(GL_TEXTURE_2D, pixLUTName[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, lutSize[0], lutSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, rgbaLUT);
	  
	vvGLTools::printGLError("leave updateLUT()");
  }
}

void vvTexMultiRend::enableLUTMode(vvShaderProgram* glslShader)
{
  glslShader->enable();

  // initialize transfer function
  if(vd->chan == 1)
    glslShader->setParameterTex2D("tfTex0", pixLUTName[0]);
  else
  {
    glslShader->setParameterTex1D("tfTex[0]", pixLUTName[vd->chan]);

	for(size_t c = 0; c < vd->chan; c++)
	{
	  char varName[20];
#ifdef WIN32
	  _snprintf(varName, sizeof(varName), "tfTex[%" VV_PRIiSIZE "]", c+1);
#else
	  snprintf(varName, sizeof(varName), "tfTex[%" VV_PRIiSIZE "]", c+1);
#endif
    glslShader->setParameterTex1D(varName, pixLUTName[c]);
  }
  }

  // initialize 3D channel texture
  for(size_t c = 0; c < vd->chan; c++)
  {
	char varName[20];
#ifdef WIN32
	_snprintf(varName, sizeof(varName), "gl3dTex%" VV_PRIiSIZE, c);
#else
	snprintf(varName, sizeof(varName), "gl3dTex%" VV_PRIiSIZE, c);
#endif
  glslShader->setParameterTex3D(varName, texNames[c]);
  }

  // copy parameters
  float weight[10];
  assert ( vd->chan  <= 10 );

  float maxWeight = 0.0f, sumWeight = 0.0f;
  // channel weights
  for (size_t c = 0; c < vd->chan; c++)
  {
	weight[c] =  chanWeight[c] * volWeight;
	maxWeight = max(weight[c], maxWeight);
	sumWeight += weight[c];
  }

  glslShader->setParameter1f("weight", weight[vd->chan]);

  for(size_t c = 0; c < vd->chan; c++)
  {
	char varName[20];
#ifdef WIN32
   _snprintf(varName, sizeof(varName), "color[%" VV_PRIdSIZE "]", c);
#else
	snprintf(varName, sizeof(varName), "color[%" VV_PRIdSIZE "]", c);
#endif
        float e[3] = { color[c][0], color[c][1], color[c][2] };
        glslShader->setParameter3f(varName, e);
  }

  //glslShader->setValue(program, "numChan", 1, &(vd->chan));
  //_alphaMode == 0 ? &maxWeight : &sumWeight;
  glslShader->setParameter1i("normAlpha", _alphaMode);
  glslShader->setParameter1i("alphaMode", _alphaMode);
}


//----------------------------------------------------------------------------
/** Set user's viewing direction.
  This information is needed to correctly orientate the texture slices
  in 3D texturing mode if the user is inside the volume.
  @param vd  viewing direction in object coordinates
*/
void vvTexMultiRend::setViewingDirection(const vvVector3& vd)
{
  vvDebugMsg::msg(3, "vvTexMultiRend::setViewingDirection()");
  viewDir = vd;
}

//----------------------------------------------------------------------------
/** Set the direction from the viewer to the object.
  This information is needed to correctly orientate the texture slices
  in 3D texturing mode if the viewer is outside of the volume.
  @param od  object direction in object coordinates
*/
void vvTexMultiRend::setObjectDirection(const vvVector3& od)
{
  vvDebugMsg::msg(3, "vvTexMultiRend::setObjectDirection()");
  objDir = od;
}

//----------------------------------------------------------------------------
// see parent
void vvTexMultiRend::setParameter(ParameterType param, const vvParam& newValue)
{
  bool newInterpol;

  vvDebugMsg::msg(3, "vvTexMultiRend::setParameter()");
  switch (param)
  {
	case vvRenderer::VV_SLICEINT:
	  newInterpol = newValue;
	  if (interpolation!=newInterpol)
	  {
	    interpolation = newInterpol;
	    makeTextures();
	  }
	  break;
	case vvRenderer::VV_MIN_SLICE:
	  minSlice = newValue;
	  break;
	case vvRenderer::VV_MAX_SLICE:
	  maxSlice = newValue;
	  break;
	case vvRenderer::VV_OPCORR:
	  //opacityCorrection = newValue;;
	  break;
	case vvRenderer::VV_SLICEORIENT:
	  _sliceOrientation = SliceOrientation(newValue.asInt());
	  break;
	case vvRenderer::VV_BINNING:
    vd->_binning = (vvVolDesc::BinningType)newValue.asInt();
	  break;      
	default:
    vvRenderer::setParameter(param, newValue);
	  break;
  }
}

//----------------------------------------------------------------------------
// see parent for comments
vvParam vvTexMultiRend::getParameter(ParameterType param) const
{
  vvDebugMsg::msg(3, "vvTexMultiRend::getParameter()");

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
	case vvRenderer::VV_BINNING:
	  return (int)vd->_binning;
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
bool vvTexMultiRend::isSupported(GeometryType geom)
{
  vvDebugMsg::msg(3, "vvTexMultiRend::isSupported(0)");

  switch (geom)
  {
	case VV_VIEWPORT:
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
/** Return the currently used rendering geometry.
  This is expecially useful if VV_AUTO was passed in the constructor.
*/
vvTexMultiRend::GeometryType vvTexMultiRend::getGeomType()
{
  vvDebugMsg::msg(3, "vvTexMultiRend::getGeomType()");
  return geomType;
}

//----------------------------------------------------------------------------
/** Return the currently used voxel type.
  This is expecially useful if VV_AUTO was passed in the constructor.
*/
vvTexMultiRend::VoxelType vvTexMultiRend::getVoxelType()
{
  vvDebugMsg::msg(3, "vvTexMultiRend::getVoxelType()");
  return voxelType;
}

//----------------------------------------------------------------------------
/// inherited from vvRenderer, only valid for planar textures
void vvTexMultiRend::renderQualityDisplay()
{
  //int numSlices = int(_renderState._quality * 100.0f);
  numSlices = int(_quality * 100.0f);
  vvPrintGL* printGL = new vvPrintGL();
  printGL->print(-0.9f, 0.9f, "Textures: %d", numSlices);
  delete printGL;
}

//----------------------------------------------------------------------------
void vvTexMultiRend::printLUT()
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

void vvTexMultiRend::init()
{
  vvRenderer::init();
 
  lutDistance = 1.0;
  rgbaTF  = new float[256 * 256 * 4];
  rgbaLUT = new uchar[256 * 256 * 4];
  preintTable = new uchar[getPreintTableSize()*getPreintTableSize()*4];
  _ntextures = 0;
  chanWeight = new float[vd->chan];

  pixLUTName = new GLuint[vd->chan+1];
  glGenTextures(vd->chan+1, pixLUTName);

  tfGamma = new float[vd->chan+1];
  tfHPOrder = new float[vd->chan+1];
  tfHPCutoff = new float[vd->chan+1];
  tfOffset = new float[vd->chan+1];

  for (size_t c = 0; c < vd->chan+1; c++)
  {
	tfGamma[c] = 1.0f;
	tfHPOrder[c] = 1.0f;
	tfHPCutoff[c] = 0.5f;
	tfOffset[c] = 0.0f;
  }
  tfGamma[vd->chan] = 3.0f;

  for (size_t c = 0; c < vd->chan; c++)
	chanWeight[c] = 1.0f;
  volWeight = 1.0f;
  color = new vvVector3[vd->chan];

  // Set first 3 channels to be RGB
  color[0].set(1.0f, 0.0f, 0.0f);
	
  histCDF = new uint[256 * vd->chan * vd->frames];

  if (vd->chan > 1)
  {
	color[1].set(0.0f, 1.0f, 0.0f);
	if (vd->chan > 2)
		color[2].set(0.0f, 0.0f, 1.0f);
  }

  texelsize=4;
  internalTexFormat = GL_RGBA;
  texFormat = GL_RGBA;

  updateTransferFunction();
    
  makeTextures();                             // we only have to do this once for non-RGBA textures
}

void vvTexMultiRend::preRendering()
{
  vvMatrix pm;
  vvMatrix invMV;                                 // inverse of model-view matrix
  vvVector3 eye;                                  // user's eye position [object space]
  vvVector3 probePosObj;                          // probe midpoint [object space]
  vvVector3 probeSizeObj;                         // probe size [object space]
  vvVector3 probeTexels;                          // number of texels in each probe dimension
  vvVector3 texSize;                              // size of 3D texture [object space]
  vvVector3 pos;                                  // volume location
  vvVector3 size(vd->getSize());                  // volume size [world coordinates]
  

  // Get projection matrix:
  vvGLTools::getProjectionMatrix(&pm);

  // Compute normal vector of view planes
  vvVector3 planeNormal(0.0f, 0.0f, -1.0f);                 // (0|0|1) is normal on projection plane
  vvMatrix invPM(pm);
  invPM.invert();
  planeNormal.multiply(invPM);

  // Get OpenGL modelview matrix:
  vvGLTools::getModelviewMatrix(&tr.mv);
  tr.mv.translate(translation);
  tr.mv.multiplyRight(rotation);
  tr.mv.getGL(tr.glMV);

  // drawing bounding boxes
  if(_boundaries)
  {
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadMatrixf(tr.glMV);
        drawBoundingBox(size, vd->pos, _boundColor);
	glPopMatrix();
  }

  // Determine texture object extensions:
  for (int i=0; i<3; ++i)
  {
	texMin[i] = 0.5f / (float)texels[i];
	texMax[i] = (float)vd->vox[i] / (float)texels[i] - texMin[i];
  }

  // Determine texture object dimensions and half object size as a shortcut:
  for (int i=0; i<3; ++i)
  {
        texSize[i] = size[i] * (float)texels[i] / (float)vd->vox[i];
        tr.size2[i]   = 0.5f * size[i];
  }
  pos = vd->pos;

  // Calculate inverted modelview matrix:
  invMV = tr.mv;
  invMV.invert();

  // Find eye position:
  getEyePosition(&eye);

  probeSizeObj = size;
  tr.probeMin.set(-tr.size2[0], -tr.size2[1], -tr.size2[2]);
  tr.probeMax = tr.size2;
  probePosObj.zero();

  // Initialize texture counters
  if (_roiSize[0])
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
  
  //if (_sliceOrientation != VV_VIEWPLANE)
	  //cerr << "sliceOrientation != VV_VIEWPLANE\n";

  // Compute distance vector between textures:
  tr.normal = planeNormal;
  tr.normal.multiply(invMV);
  tr.normal.normalize();

  // compute number of slices to draw
  float depth = fabs(tr.normal[0]*probeSizeObj[0]) + fabs(tr.normal[1]*probeSizeObj[1]) + fabs(tr.normal[2]*probeSizeObj[2]);

  int minDistanceInd = 0;
  if(probeSizeObj[1]/probeTexels[1] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
	  minDistanceInd=1;
  if(probeSizeObj[2]/probeTexels[2] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
	  minDistanceInd=2;
  float voxelDistance = probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd];

  float sliceDistance = voxelDistance / quality;
  /*
  if(_renderState._roiSize[0] && _renderState._quality < 2.0)
  {
	  // draw at least twice as many slices as there are samples in the probe depth.
	  sliceDistance = voxelDistance / 2.0f;
  }
  */
  numSlices = 2*(int)ceilf(depth/sliceDistance*.5f);

  /*set num slices to constant*/
  //aRenderer->numSlices = (int) ( min(min(vd->texels[0], vd->texels[1]),vd->texels[2]) * vd->quality );
  //sliceDistance = depth/aRenderer->numSlices; 

  if (numSlices < 1)                              // make sure that at least one slice is drawn
	  numSlices = 1;

  // Use alpha correction in indexed mode: adapt alpha values to number of textures:
  float thickness = sliceDistance/voxelDistance;

  // just tolerate slice distance differences imposed on us
  // by trying to keep the number of slices constant
  if(lutDistance/thickness < 0.88 || thickness/lutDistance < 0.88)
  {
	  updateLUT(thickness);
  }

  //vvDebugMsg::msg(1, "voxelDistance, sliceDistance, thickness: ", voxelDistance, sliceDistance, thickness, 1.0f/thickness);

  tr.delta = vvVector3(tr.normal);
  tr.delta.scale(sliceDistance);

  // Compute farthest point to draw texture at:
  tr.farthest = vvVector3(tr.delta);
  tr.farthest.scale((float)(numSlices - 1) / -2.0f);
  tr.farthest.add(probePosObj);
  tr.farWS = vvVector3(tr.farthest);
  tr.farWS.multiply(tr.mv);
}


void vvTexMultiRend::updateChannelHistCDF(int channel, int frame, uchar* data)
{
	int size = texels[0] * texels[1] * texels[2];
	uint *hist = histCDF + (channel + frame*vd->chan)*256;

	memset(hist, 0, 256*sizeof(uint));

	for (int i = 0; i < size; i++)
		hist[data[i]]++;

	for (int i = 1; i < 256; i++)
		hist[i] += hist[i-1];
}

//----------------------------------------------------------------------------
/** Render the volume onto currently selected drawBuffer.
 Viewport size in world coordinates is -1.0 .. +1.0 in both x and y direction
*/

void vvTexMultiRend::renderMultipleVolume()
{	
  vvVector3 isect[6];             // intersection points, maximum of 6 allowed when intersecting a plane and a volume [object space]
  vvVector3 texcoord[12];         // intersection points in texture coordinate space [0..1]

  // Search for intersections between texture plane (defined by texPoint and
  // normal) and texture object (0..1):
  int isectCnt = isect->isectPlaneCuboid(tr.normal, tr.farthest, tr.probeMin, tr.probeMax);

  tr.farthest.add(tr.delta);

  tr.farWS = vvVector3(tr.farthest);
  tr.farWS.multiply(tr.mv);

  if (isectCnt<3) return;                     // at least 3 intersections needed for drawing

  // Put the intersecting 3 to 6 vertices in cyclic order to draw adjacent
  // and non-overlapping triangles:
  isect->cyclicSort(isectCnt, tr.normal);

  // Generate vertices in texture coordinates:

  for (int j=0; j<isectCnt; ++j)
  {
	  for (int k=0; k<3; ++k)
	  {
                  texcoord[j][k] = (isect[j][k] + tr.size2[k]) / vd->getSize()[k];
		  texcoord[j][k] = texcoord[j][k] * (texMax[k] - texMin[k]) + texMin[k];
	  }
  }

  //swGL.start();
  glPushMatrix();
  glLoadMatrixf(tr.glMV);

  glBegin(GL_TRIANGLE_FAN);
  glColor4f(1.0, 1.0, 1.0, 1.0);
  glNormal3f(tr.normal[0], tr.normal[1], tr.normal[2]);

  for (int j=0; j<isectCnt; ++j)
  {
	  glTexCoord3f(texcoord[j][0], texcoord[j][1], texcoord[j][2]);
	  glVertex3f(isect[j][0], isect[j][1], isect[j][2]);
  }
  glEnd();

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

}


//============================================================================
// End of File
//============================================================================


// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
