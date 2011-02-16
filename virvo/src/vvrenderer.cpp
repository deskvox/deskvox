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

#include "vvglew.h"

#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "vvgltools.h"
#include "vvopengl.h"
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvrenderer.h"
#include "vvdebugmsg.h"
#include "vvprintgl.h"

/*ARGSUSED*/

//----------------------------------------------------------------------------
vvRenderState::vvRenderState()
{
  int i;

  _mipMode = 0;
  _alphaMode = 0;
  _emptySpaceLeaping = true;
  _clipPerimeter = true;
  _boundaries = false;
  _orientation = false;
  _palette = false;
  _qualityDisplay = false;
  _fpsDisplay = false;
  _quality = 1.0f;
  _clipSingleSlice = false;
  _clipOpaque = false;
  _clipMode = false;
  _roiPos.zero();
  _roiSize.set(0.5f, 0.5f, 0.5f);
  _isROIUsed = false;
  _sphericalROI = false;
  _brickSize[0] = _brickSize[1] = _brickSize[2] = 0;
  _maxBrickSize[0] = 64;
  _maxBrickSize[1] = 64;
  _maxBrickSize[2] = 64;
  _brickTexelOverlap = 1;
  _showBricks = false;
  _computeBrickSize = true;
  _texMemorySize = 0;
  _clipPoint.set(0.0f, 0.0f, 0.0f);
  _clipNormal.set(0.0f, 0.0f, 1.0f);
  _gammaCorrection = false;
  _opacityWeights = false;

  for (i=0; i<4; ++i)
  {
    _gamma[i] = 1.0f;
  }

  for (i=0; i<3; ++i)
  {
    _clipColor[i]  = 1.0f;
    _probeColor[i] = 1.0f;
    _boundColor[i] = 1.0f;
  }

  _useOffscreenBuffer = false;
  _imageScale = 1.0f;
  _imagePrecision = VV_BYTE;
  _showTexture = true;	// added by Han, Feb 2008
  _opaqueGeometryPresent = false;
}

void vvRenderState::setParameter(const ParameterType param, const float newValue)
{
  vvDebugMsg::msg(3, "vvRenderState::setParameter()");

  switch (param)
  {
  case VV_QUALITY:
    _quality = newValue;
    break;
  case VV_MIP_MODE:
    _mipMode = (int)newValue;
    break;
  case VV_ALPHA_MODE:
    _alphaMode = (int)newValue;
    break;
  case VV_LEAPEMPTY:
    _emptySpaceLeaping = (bool)newValue;
    break;
  case VV_CLIP_PERIMETER:
    _clipPerimeter = (bool)newValue;
    break;
  case VV_BOUNDARIES:
    _boundaries = (bool)newValue;
    break;
  case VV_ORIENTATION:
    _orientation = (bool)newValue;
    break;
  case VV_PALETTE:
    _palette = (bool)newValue;
    break;
  case VV_QUALITY_DISPLAY:
    _qualityDisplay = (bool)newValue;
    break;
  case VV_CLIP_MODE:
    _clipMode = (bool)newValue;
    break;
  case VV_CLIP_SINGLE_SLICE:
    _clipSingleSlice = (bool)newValue;
    break;
  case VV_CLIP_OPAQUE:
    _clipOpaque = (bool)newValue;
    break;
  case VV_IS_ROI_USED:
    _isROIUsed = (bool)newValue;
    break;
  case VV_IS_ROI_CHANGED:
    _isROIChanged = (bool)newValue;
    break;
  case VV_SPHERICAL_ROI:
    _sphericalROI = (bool)newValue;
    break;
  case VV_BRICK_TEXEL_OVERLAP:
    _brickTexelOverlap = (int)newValue;
    break;
  case VV_SHOW_BRICKS:
    _showBricks = (bool)newValue;
    break;
  case VV_COMPUTE_BRICK_SIZE:
    _computeBrickSize = (bool)newValue;
    break;
  case VV_TEX_MEMORY_SIZE:
    _texMemorySize = (int)newValue;
    break;
  case VV_FPS_DISPLAY:
    _fpsDisplay = (bool)newValue;
    break;
  case VV_GAMMA_CORRECTION:
    _gammaCorrection = (bool)newValue;
    break;
  case VV_OPACITY_WEIGHTS:
    _opacityWeights = (bool)newValue;
    break;
  case VV_USE_OFFSCREEN_BUFFER:
    _useOffscreenBuffer = (bool)newValue;
    break;
  case VV_IMAGE_SCALE:
    _imageScale = newValue;
    break;
  case VV_IMAGE_PRECISION:
    _imagePrecision = (BufferPrecision)newValue;
    break;
  case VV_SHOW_TEXTURE:
    _showTexture = (bool)newValue;
    break;
  case VV_OPAQUE_GEOMETRY_PRESENT:
    _opaqueGeometryPresent = (bool)newValue;
    break;
  default:
    break;
  }
}

void vvRenderState::setParameterV3(const ParameterType param, const vvVector3& newValue)
{
  vvDebugMsg::msg(3, "vvRenderState::setParameterV3()");

  switch (param)
  {
  case VV_CLIP_POINT:
    _clipPoint = newValue;
    break;
  case VV_CLIP_NORMAL:
    _clipNormal = newValue;
    _clipNormal.normalize();
    break;
  case VV_CLIP_COLOR:
    _clipColor = vvColor(newValue[0], newValue[1], newValue[2]);
    break;
  case VV_ROI_POS:
    _roiPos = newValue;
    break;
  case VV_ROI_SIZE:
    _roiSize = newValue;
    break;
  case VV_BRICK_SIZE:
    {
      for (int i=0; i<3; ++i)
      {
        _brickSize[i] = newValue[i];
      }
    }
    break;
  case VV_MAX_BRICK_SIZE:
    {
      for (int i=0; i<3; ++i)
      {
        _maxBrickSize[i] = newValue[i];
      }
    }
    break;
  case VV_BOUND_COLOR:
    _boundColor = vvColor(newValue[0], newValue[1], newValue[2]);
    break;
  case VV_PROBE_COLOR:
    _probeColor = vvColor(newValue[0], newValue[1], newValue[2]);
    break;
  default:
    break;
  }
}

void vvRenderState::setParameterV4(const ParameterType param, const vvVector4& newValue)
{
  vvDebugMsg::msg(3, "vvRenderState::setParameterV4()");

  switch (param)
  {
  case VV_GAMMA:
    _gamma = newValue;
    break;
  default:
    break;
  }
}

float vvRenderState::getParameter(const ParameterType param) const
{
  vvDebugMsg::msg(3, "vvRenderState::getParameter()");

  switch (param)
  {
  case VV_QUALITY:
    return _quality;
  case VV_MIP_MODE:
    return _mipMode;
  case VV_ALPHA_MODE:
    return _alphaMode;
  case VV_CLIP_PERIMETER:
    return _clipPerimeter;
  case VV_BOUNDARIES:
    return _boundaries;
  case VV_ORIENTATION:
    return _orientation;
  case VV_PALETTE:
    return _palette;
  case VV_QUALITY_DISPLAY:
    return _qualityDisplay;
  case VV_CLIP_MODE:
    return _clipMode;
  case VV_CLIP_SINGLE_SLICE:
    return _clipSingleSlice;
  case VV_CLIP_OPAQUE:
    return _clipOpaque;
  case VV_IS_ROI_CHANGED:
    return _isROIChanged;
  case VV_IS_ROI_USED:
    return _isROIUsed;
  case VV_SPHERICAL_ROI:
    return _sphericalROI;
  case VV_BRICK_TEXEL_OVERLAP:
    return _brickTexelOverlap;
  case VV_TEX_MEMORY_SIZE:
    return _texMemorySize;
  case VV_FPS_DISPLAY:
    return _fpsDisplay;
  case VV_GAMMA_CORRECTION:
    return _gammaCorrection;
  case VV_OPACITY_WEIGHTS:
    return _opacityWeights;
  case VV_USE_OFFSCREEN_BUFFER:
    return _useOffscreenBuffer;
  case VV_IMAGE_SCALE:
    return _imageScale;
  case VV_IMAGE_PRECISION:
    return _imagePrecision;
  case VV_SHOW_TEXTURE:
    return _showTexture;
  case VV_OPAQUE_GEOMETRY_PRESENT:
    return _opaqueGeometryPresent;
  default:
    return -VV_FLT_MAX;
    break;
  }
  return -VV_FLT_MAX;
}

vvVector3 vvRenderState::getParameterV3(const ParameterType param) const
{
  vvDebugMsg::msg(3, "vvRenderState::getParameterV3()");

  switch (param)
  {
  case VV_CLIP_POINT:
    return _clipPoint;
  case VV_CLIP_NORMAL:
    return _clipNormal;
  case VV_CLIP_COLOR:
    return _clipColor;
  case VV_ROI_SIZE:
    return _roiSize;
  case VV_ROI_POS:
    return _roiPos;
  case VV_BRICK_SIZE:
    return vvVector3(_brickSize[0], _brickSize[1], _brickSize[2]);
  case VV_MAX_BRICK_SIZE:
    return vvVector3(_maxBrickSize[0], _maxBrickSize[1], _maxBrickSize[2]);
  case VV_BOUND_COLOR:
    return _boundColor;
  case VV_PROBE_COLOR:
    return _probeColor;
  default:
    break;
  }
  return vvVector3(-VV_FLT_MAX, -VV_FLT_MAX, -VV_FLT_MAX);
}

vvVector4 vvRenderState::getParameterV4(const ParameterType param) const
{
  vvDebugMsg::msg(3, "vvRenderState::getParameterV4()");

  switch (param)
  {
  case VV_GAMMA:
    return _gamma;
  default:
    break;
  }
  return vvVector4(-VV_FLT_MAX, -VV_FLT_MAX, -VV_FLT_MAX, -VV_FLT_MAX);
}

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
/** Constructor, called when program starts and when the rendering method changes
  @param voldesc volume descriptor, should be deleted when not needed anymore
  @param renderState contains variables for the renderer
*/
vvRenderer::vvRenderer(vvVolDesc* voldesc, vvRenderState renderState)
  : vvRenderState(renderState)
{
  vvDebugMsg::msg(1, "vvRenderer::vvRenderer()");
  assert(voldesc!=NULL);
  vd = voldesc;
  init();
}

//----------------------------------------------------------------------------
// Added by Han, March 2008
void vvRenderer::setVolDesc(vvVolDesc* voldesc)
{
  vvDebugMsg::msg(2, "vvRenderer::setVolDesc()");
  assert(voldesc != NULL);
  vd = voldesc;
}

vvVolDesc* vvRenderer::getVolDesc()
{
  return vd;
}

vvRenderer::AxisType vvRenderer::getPrincipalViewingAxis(const vvMatrix& mv,
                                                         float& zx, float& zy, float& zz) const
{
  vvMatrix invMV;
  invMV.copy(&mv);
  invMV.invert();

  vvVector3 eye;
  getEyePosition(&eye);
  eye.multiply(&invMV);

  vvVector3 normal;
  vvVector3 origin;
  getObjNormal(normal, origin, eye, invMV);

  zx = normal[0];
  zy = normal[1];
  zz = normal[2];

  if (fabs(zx) > fabs(zy))
  {
    if (fabs(zx) > fabs(zz)) return X_AXIS;
    else return Z_AXIS;
  }
  else
  {
    if (fabs(zy) > fabs(zz)) return Y_AXIS;
    else return Z_AXIS;
  }
}

//----------------------------------------------------------------------------
/// Initialization routine for class variables.
void vvRenderer::init()
{
  vvDebugMsg::msg(1, "vvRenderer::init()");

  rendererType = UNKNOWN;
  _lastRenderTime = 0.0f;
  _lastComputeTime = 0.0f;
  _lastPlaneSortingTime = 0.0f;
  for(int i=0; i<3; ++i)
  {
    _channel4Color[i] = 1.0f;
  }
  for(int i=0; i<4; ++i)
  {
    _opacityWeights[i] = 1.0f;
  }
}

void vvRenderer::getObjNormal(vvVector3& normal, vvVector3& origin,
                              const vvVector3& eye, const vvMatrix& invMV,
                              const bool isOrtho) const
{
  // Compute normal vector of textures using the following strategy:
  // For orthographic projections or if viewDir is (0|0|0) use
  // (0|0|1) as the normal vector.
  // Otherwise use objDir as the normal.
  // Exception: if user's eye is inside object and probe mode is off,
  // then use viewDir as the normal.
  if (_clipMode)
  {
    normal.copy(&_clipNormal);
  }
  else if (isOrtho || (viewDir[0] == 0.0f && viewDir[1] == 0.0f && viewDir[2] == 0.0f))
  {
    // Draw slices parallel to projection plane:
    normal.set(0.0f, 0.0f, 1.0f);                 // (0|0|1) is normal on projection plane
    normal.multiply(&invMV);
    origin.zero();
    origin.multiply(&invMV);
    normal.sub(&origin);
  }
  else if (!_isROIUsed && isInVolume(&eye))
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
}

void vvRenderer::getShadingNormal(vvVector3& normal, vvVector3& origin,
                                  const vvVector3& eye, const vvMatrix& invMV,
                                  const bool isOrtho) const
{
  // See calcutions in getObjNormal(). Only difference: if clip plane
  // is active, this is ignored. This normal isn't used to align
  // slices or such with the clipping plane, but for shading calculations.
  if (isOrtho || (viewDir[0] == 0.0f && viewDir[1] == 0.0f && viewDir[2] == 0.0f))
  {
    // Draw slices parallel to projection plane:
    normal.set(0.0f, 0.0f, 1.0f);                 // (0|0|1) is normal on projection plane
    normal.multiply(&invMV);
    origin.zero();
    origin.multiply(&invMV);
    normal.sub(&origin);
  }
  else if (!_isROIUsed && isInVolume(&eye))
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
}

void vvRenderer::calcProbeDims(vvVector3& probePosObj, vvVector3& probeSizeObj, vvVector3& probeMin, vvVector3& probeMax) const
{
  // Determine texture object dimensions and half object size as a shortcut:
  const vvVector3 size(vd->getSize());
  const vvVector3 size2 = size * 0.5f;

  if (_isROIUsed)
  {
    // Convert probe midpoint coordinates to object space w/o position:
    probePosObj = _roiPos;

    // Compute probe min/max coordinates in object space:
    const vvVector3 maxSize = _roiSize * size2;

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

//----------------------------------------------------------------------------
/** Destructor, called when program ends or when rendering method changes.
   Clear up all dynamically allocated memory space here.
*/
vvRenderer::~vvRenderer()
{
  vvDebugMsg::msg(1, "vvRenderer::~vvRenderer()");
}

//----------------------------------------------------------------------------
/** Adapt image quality to frame rate.
  The method assumes a reciprocal dependence of quality and frame rate.
  The threshold region is always entered from below.
  @param quality    current image quality (>0, the bigger the better)
  @param curFPS     current frame rate [fps]
  @param desFPS     desired frame rate [fps]
  @param threshold  threshold to prevent flickering [fraction of frame rate difference]
  @return the adapted quality
*/
float vvRenderer::adaptQuality(float quality, float curFPS, float desFPS, float threshold)
{
  const float MIN_QUALITY = 0.001f;
  const float MAX_QUALITY = 10.0f;

  vvDebugMsg::msg(1, "vvRenderer::adaptQuality()");
                                                  // plausibility check
  if (curFPS>0.0f && curFPS<1000.0f && desFPS>0.0f && threshold>=0.0f)
  {
    if (curFPS>desFPS || ((desFPS - curFPS) / desFPS) > threshold)
    {
      quality *= curFPS / desFPS;
      if (quality < MIN_QUALITY) quality = MIN_QUALITY;
      else if (quality > MAX_QUALITY) quality = MAX_QUALITY;
    }
  }
  return quality;
}

//----------------------------------------------------------------------------
/** Returns the type of the renderer used.
 */
vvRenderer::RendererType vvRenderer::getRendererType() const
{
  return rendererType;
}

//----------------------------------------------------------------------------
/** Core display rendering routine.
  Should render the volume to the currently selected draw buffer. This parent
  method renders the coordinate axes and the palette, if the respective
  modes are set.
*/
void vvRenderer::renderMultipleVolume()
{
  vvDebugMsg::msg(3, "vvRenderer::renderMultipleVolume()");
}

void vvRenderer::renderVolumeGL()
{
  vvDebugMsg::msg(3, "vvRenderer::renderVolumeGL()");

  // Draw legend if requested:
  if (_orientation) renderCoordinates();
  if (_palette) renderPalette();
  if (_qualityDisplay) renderQualityDisplay();
  if (_fpsDisplay) renderFPSDisplay();
}

//----------------------------------------------------------------------------
/** Copy the currently displayed renderer image to a memory buffer and
  resize the image if necessary.
  Important: OpenGL canvas size should be larger than rendererd image size
  for optimum quality!
  @param w,h image size in pixels
  @param data _allocated_ memory space providing w*h*3 bytes of memory space
  @return memory space to which volume was rendered. This need not be the same
          as data, if internal space is used.
*/
void vvRenderer::renderVolumeRGB(int w, int h, uchar* data)
{
  uchar* screenshot;
  GLint viewPort[4];                              // x, y, width, height of viewport
  int x, y;
  int srcIndex, dstIndex, srcX, srcY;
  int offX, offY;                                 // offsets in source image to maintain aspect ratio
  int srcWidth, srcHeight;                        // actually used area of source image

  vvDebugMsg::msg(3, "vvRenderer::renderVolumeRGB(), size: ", w, h);

  // Save GL state:
  glPushAttrib(GL_ALL_ATTRIB_BITS);

  // Prepare reading:
  glGetIntegerv(GL_VIEWPORT, viewPort);
  screenshot = new uchar[viewPort[2] * viewPort[3] * 3];
  glDrawBuffer(GL_FRONT);                         // set draw buffer to front in order to read image data
  glPixelStorei(GL_PACK_ALIGNMENT, 1);            // Important command: default value is 4, so allocated memory wouldn't suffice

  // Read image data:
  glReadPixels(0, 0, viewPort[2], viewPort[3], GL_RGB, GL_UNSIGNED_BYTE, screenshot);

  // Restore GL state:
  glPopAttrib();

  // Maintain aspect ratio:
  if (viewPort[2]==w && viewPort[3]==h)
  {                                               // movie image same aspect ratio as OpenGL window?
    srcWidth  = viewPort[2];
    srcHeight = viewPort[3];
    offX = 0;
    offY = 0;
  }
  else if ((float)viewPort[2] / (float)viewPort[3] > (float)w / (float)h)
  {                                               // movie image more narrow than OpenGL window?
    srcHeight = viewPort[3];
    srcWidth = srcHeight * w / h;
    offX = (viewPort[2] - srcWidth) / 2;
    offY = 0;
  }
  else                                            // movie image wider than OpenGL window
  {
    srcWidth = viewPort[2];
    srcHeight = h * srcWidth / w;
    offX = 0;
    offY = (viewPort[3] - srcHeight) / 2;
  }

  // Now resample image data:
  for (y=0; y<h; ++y)
  {
    for (x=0; x<w; ++x)
    {
      dstIndex = 3 * (x + (h - y - 1) * w);
      srcX = offX + srcWidth  * x / w;
      srcY = offY + srcHeight * y / h;
      srcIndex = 3 * (srcX + srcY * viewPort[2]);
      memcpy(data + dstIndex, screenshot + srcIndex, 3);
    }
  }
  delete[] screenshot;
}

//----------------------------------------------------------------------------
/// Update transfer function in renderer
void vvRenderer::updateTransferFunction()
{
  vvDebugMsg::msg(1, "vvRenderer::updateTransferFunction()");
}

//----------------------------------------------------------------------------
/** Update volume data in renderer.
  This function is called when the volume data in vvVolDesc were modified.
*/
void vvRenderer::updateVolumeData()
{
  vvDebugMsg::msg(1, "vvRenderer::updateVolumeData()");
}

//----------------------------------------------------------------------------
/** Returns the number of animation frames.
 */
int vvRenderer::getNumFrames()
{
  vvDebugMsg::msg(3, "vvRenderer::getNumFrames()");
  return vd->frames;
}

//----------------------------------------------------------------------------
/** Returns index of current animation frame.
  (first frame = 0, <0 if undefined)
*/
int vvRenderer::getCurrentFrame()
{
  vvDebugMsg::msg(3, "vvRenderer::getCurrentFrame()");
  return vd->getCurrentFrame();
}

//----------------------------------------------------------------------------
/** Set new frame index.
  @param index  new frame index (0 for first frame)
*/
void vvRenderer::setCurrentFrame(int index)
{
  vvDebugMsg::msg(3, "vvRenderer::setCurrentFrame()");
  if (index == vd->getCurrentFrame()) return;
  if (index < 0) index = 0;
  if (index >= vd->frames) index = vd->frames - 1;
  vvDebugMsg::msg(3, "New frame index: ", index);
  vd->setCurrentFrame(index);
}

//----------------------------------------------------------------------------
/** Get last render time.
  @return time to render last image
*/
float vvRenderer::getLastRenderTime()
{
  vvDebugMsg::msg(3, "vvRenderer::getLastRenderTime()");
  return _lastRenderTime;
}

//----------------------------------------------------------------------------
/** Render axis coordinates in bottom right corner.
  Arrows are of length 1.0<BR>
  Colors: x-axis=red, y-axis=green, z-axis=blue
*/
void vvRenderer::renderCoordinates()
{
  vvMatrix mv;                                    // current modelview matrix
  vvVector3 column;                               // column vector
  GLboolean glsLighting;                          // stores GL_LIGHTING
  GLint viewPort[4];                              // x, y, width, height of viewport
  float aspect;                                   // viewport aspect ratio
  float half[2];                                  // half viewport dimensions (x,y)
  int i;

  // Save lighting mode:
  glGetBooleanv(GL_LIGHTING, &glsLighting);
  glDisable(GL_LIGHTING);

  // Get viewport parameters:
  glGetIntegerv(GL_VIEWPORT, viewPort);
  aspect = (float)viewPort[2] / (float)viewPort[3];

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();

  half[0] = (aspect < 1.0f) ? 1.0f : aspect;
  half[1] = (aspect > 1.0f) ? 1.0f : (1.0f / aspect);
  glOrtho(-half[0], half[0], -half[1], half[1], 10.0f, -10.0f);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();

  // Compute modelview matrix:
  getModelviewMatrix(&mv);
  mv.killTrans();
  for (i=0; i<3; ++i)                             // normalize base vectors to remove scaling
  {
    mv.getColumn(i, &column);
    column.normalize();
    mv.setColumn(i, &column);
  }
  mv.translate(0.8f * half[0], -0.8f * half[1], 0.0f);
  mv.scale(0.2f, 0.2f, 0.2f);
  setModelviewMatrix(&mv);

  // Draw axis cross:
  glBegin(GL_LINES);
  glColor3f(1.0f, 0.0f, 0.0f);                    // red
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(1.0f, 0.0f, 0.0f);

  glColor3f(0.0f, 1.0f, 0.0f);                    // green
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(0.0f, 1.0f, 0.0f);

  glColor3f(0.0f, 0.0f, 1.0f);                    // blue
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(0.0f, 0.0f, 1.0f);
  glEnd();

  // Draw arrows:
  glBegin(GL_TRIANGLES);
  glColor3f(1.0f, 0.0f, 0.0f);                    // red
  glVertex3f(1.0f, 0.0f, 0.0f);
  glVertex3f(0.8f, 0.0f,-0.2f);
  glVertex3f(0.8f, 0.0f, 0.2f);

  glColor3f(0.0f, 1.0f, 0.0f);                    // green
  glVertex3f(0.0f, 1.0f, 0.0f);
  glVertex3f(-0.2f, 0.8f, 0.0f);
  glVertex3f(0.2f, 0.8f, 0.0f);

  glColor3f(0.0f, 0.0f, 1.0f);                    // blue
  glVertex3f(0.0f, 0.0f, 1.0f);
  glVertex3f(-0.2f,-0.0f, 0.8f);
  glVertex3f(0.2f, 0.0f, 0.8f);
  glEnd();

  // Restore matrix states:
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);

  // Restore lighting mode:
  if (glsLighting==(uchar)true) glEnable(GL_LIGHTING);
  else glDisable(GL_LIGHTING);
}

//----------------------------------------------------------------------------
/// Render transfer function palette at left border.
void vvRenderer::renderPalette()
{
  const int WIDTH = 10;                           // palette width [pixels]
  GLfloat viewport[4];                            // OpenGL viewport information (position and size)
  GLfloat glsRasterPos[4];                        // current raster position (glRasterPos)
  float* colors;                                  // palette colors
  uchar* image;                                   // palette image
  int w, h;                                       // width and height of palette
  int x, y, c;

  vvDebugMsg::msg(3, "vvRenderer::renderPalette()");

  if (vd->chan > 1) return;                       // palette only makes sense with scalar data

  // Get viewport size:
  glGetFloatv(GL_VIEWPORT, viewport);
  if (viewport[2]<=0 || viewport[3]<=0) return;   // safety first

  // Save matrix states:
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // Store raster position:
  glGetFloatv(GL_CURRENT_RASTER_POSITION, glsRasterPos);

  // Compute palette image:
  w = WIDTH;
  h = (int)viewport[3];
  colors = new float[h * 4];
  vd->tf.computeTFTexture(h, 1, 1, colors, vd->real[0], vd->real[1]);
  image = new uchar[w * h * 3];
  for (x=0; x<w; ++x)
    for (y=0; y<h; ++y)
      for (c=0; c<3; ++c)
        image[c + 3 * (x + w * y)] = (uchar)(colors[4 * y + c] * 255.99f);

  // Draw palette:
  glRasterPos2f(-1.0f,-1.0f);                     // pixmap origin is bottom left corner of output window
  glDrawPixels(w, h, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)image);
  delete[] image;
  delete[] colors;

  // Display min and max values:
  vvPrintGL* printGL;
  printGL = new vvPrintGL();
  printGL->print(-0.90f,  0.9f,  "%-9.2f", vd->real[1]);
  printGL->print(-0.90f, -0.95f, "%-9.2f", vd->real[0]);
  delete printGL;

  // Restore raster position:
  glRasterPos4fv(glsRasterPos);

  // Restore matrix states:
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

//----------------------------------------------------------------------------
/// Display rendering quality.
void vvRenderer::renderQualityDisplay()
{
  vvPrintGL* printGL = new vvPrintGL();
  vvVector4 clearColor = vvGLTools::queryClearColor();
  vvVector4 fontColor = vvVector4(1.0f - clearColor[0], 1.0f - clearColor[1], 1.0f - clearColor[2], 1.0f);
  printGL->setFontColor(fontColor);
  printGL->print(-0.9f, 0.9f, "Quality: %-9.2f", _quality);
  delete printGL;
}

//----------------------------------------------------------------------------
/// Display frame rate.
void vvRenderer::renderFPSDisplay()
{
  float fps = getLastRenderTime();
  if (fps > 0.0f) fps = 1.0f / fps;
  else fps = -1.0f;
  vvPrintGL* printGL = new vvPrintGL();
  vvVector4 clearColor = vvGLTools::queryClearColor();
  vvVector4 fontColor = vvVector4(1.0f - clearColor[0], 1.0f - clearColor[1], 1.0f - clearColor[2], 1.0f);
  printGL->setFontColor(fontColor);
  printGL->print(0.3f, 0.9f, "fps: %-9.1f", fps);
  delete printGL;
}

//----------------------------------------------------------------------------
/** Render an axis aligned box.
  Volume vertex names:<PRE>
      4____ 7        y
     /___ /|         |
   0|   3| |         |___x
    | 5  | /6       /
    |/___|/        z
    1    2
  </PRE>
  @param oSize  boundary box size [object coordinates]
  @param oPos   position of boundary box center [object coordinates]
@param color  bounding box color (R,G,B) [0..1], array of 3 floats expected
*/
void vvRenderer::drawBoundingBox(const vvVector3* oSize, const vvVector3* oPos, const vvColor* color) const
{
  vvVector3 vertvec[8];                           // vertex vectors in object space
  GLboolean glsLighting;                          // stores GL_LIGHTING
  GLfloat   glsColor[4];                          // stores GL_CURRENT_COLOR
  GLfloat   glsLineWidth;                         // stores GL_LINE_WIDTH
  float vertices[8][3] =                          // volume vertices
  {
    {-0.5, 0.5, 0.5},
    {-0.5,-0.5, 0.5},
    { 0.5,-0.5, 0.5},
    { 0.5, 0.5, 0.5},
    {-0.5, 0.5,-0.5},
    {-0.5,-0.5,-0.5},
    { 0.5,-0.5,-0.5},
    { 0.5, 0.5,-0.5}
  };
  int faces[6][4] =                               // volume faces. first 3 values are used for normal compuation
  {
    {7, 3, 2, 6},
    {0, 3, 7, 4},
    {2, 3, 0, 1},
    {4, 5, 1, 0},
    {1, 5, 6, 2},
    {6, 5, 4, 7}
  };
  int i,j;

  vvDebugMsg::msg(3, "vvRenderer::drawBoundingBox()");

  // Save lighting state:
  glGetBooleanv(GL_LIGHTING, &glsLighting);

  // Save color:
  glGetFloatv(GL_CURRENT_COLOR, glsColor);

  // Save line width:
  glGetFloatv(GL_LINE_WIDTH, &glsLineWidth);

  // Translate boundaries by volume position:
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();                                 // save modelview matrix
  glTranslatef((*oPos)[0], (*oPos)[1], (*oPos)[2]);

  // Set box color:
  glColor4f((*color)[0], (*color)[1], (*color)[2], 1.0);

  // Disable lighting:
  glDisable(GL_LIGHTING);

  // Create vertex vectors:
  for (i=0; i<8; ++i)
  {
    vertvec[i].set(vertices[i][0], vertices[i][1], vertices[i][2]);
    vertvec[i].scale(oSize);
  }

  glLineWidth(3.0f);

  // Draw faces:
  for (i=0; i<6; ++i)
  {
    glBegin(GL_LINE_STRIP);
    for (j=0; j<4; ++j)
    {
      glVertex3f(vertvec[faces[i][j]][0], vertvec[faces[i][j]][1], vertvec[faces[i][j]][2]);
    }
    glVertex3f(vertvec[faces[i][0]][0], vertvec[faces[i][0]][1], vertvec[faces[i][0]][2]);
    glEnd();
  }

  glLineWidth(glsLineWidth);                      // restore line width

  glPopMatrix();                                  // restore modelview matrix

  // Restore lighting state:
  if (glsLighting==(uchar)true) glEnable(GL_LIGHTING);
  else glDisable(GL_LIGHTING);

  // Restore draw color:
  glColor4fv(glsColor);
}

//----------------------------------------------------------------------------
/** Render the intersection of a plane and an axis aligned box.
  @param oSize   box size [object coordinates]
  @param oPos    box center [object coordinates]
  @param oPlane  a point on the plane [object coordinates]
  @param oNorm   normal of plane [object coordinates]
  @param color   box color (R,G,B) [0..1], array of 3 floats expected
*/
void vvRenderer::drawPlanePerimeter(const vvVector3* oSize, const vvVector3* oPos,
                                    const vvVector3* oPlane, const vvVector3* oNorm, const vvColor* color) const
{
  GLboolean glsLighting;                          // stores GL_LIGHTING
  GLfloat   glsColor[4];                          // stores GL_CURRENT_COLOR
  GLfloat   glsLineWidth;                         // stores GL_LINE_WIDTH
  vvVector3 isect[6];                             // intersection points, maximum of 6 when intersecting a plane and a volume [object space]
  vvVector3 boxMin,boxMax;                        // minimum and maximum box coordinates
  int isectCnt;
  int j;

  vvDebugMsg::msg(3, "vvRenderer::drawPlanePerimeter()");

  // Save lighting state:
  glGetBooleanv(GL_LIGHTING, &glsLighting);

  // Save color:
  glGetFloatv(GL_CURRENT_COLOR, glsColor);

  // Save line width:
  glGetFloatv(GL_LINE_WIDTH, &glsLineWidth);

  // Translate by volume position:
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();                                 // save modelview matrix
  glTranslatef((*oPos)[0], (*oPos)[1], (*oPos)[2]);

  // Set color:
  glColor4f((*color)[0], (*color)[1], (*color)[2], 1.0);

  // Disable lighting:
  glDisable(GL_LIGHTING);

  glLineWidth(3.0f);

  boxMin.set((*oPos)[0] - (*oSize)[0] * 0.5f, (*oPos)[1] - (*oSize)[1] * 0.5f, (*oPos)[2] - (*oSize)[2] * 0.5f);
  boxMax.set((*oPos)[0] + (*oSize)[0] * 0.5f, (*oPos)[1] + (*oSize)[1] * 0.5f, (*oPos)[2] + (*oSize)[2] * 0.5f);

  isectCnt = isect->isectPlaneCuboid(oNorm, oPlane, &boxMin, &boxMax);

  if (isectCnt>1)
  {
    // Put the intersecting vertices in cyclic order:
    isect->cyclicSort(isectCnt, oNorm);

    // Draw line strip:
    glBegin(GL_LINE_STRIP);
    for (j=0; j<isectCnt; ++j)
    {
      glVertex3f(isect[j][0], isect[j][1], isect[j][2]);
    }
    glVertex3f(isect[0][0], isect[0][1], isect[0][2]);
    glEnd();
  }

  glLineWidth(glsLineWidth);                      // restore line width

  glPopMatrix();                                  // restore modelview matrix

  // Restore lighting state:
  if (glsLighting==(uchar)true) glEnable(GL_LIGHTING);
  else glDisable(GL_LIGHTING);

  // Restore draw color:
  glColor4fv(glsColor);
}

//----------------------------------------------------------------------------
/** Find out if classification can be done in real time.
  @return true if updateTransferFunction() can be processed immediately
          (eg. for color indexed textures), otherwise false is returned.
*/
bool vvRenderer::instantClassification() const
{
  vvDebugMsg::msg(1, "vvRenderer::instantClassification()");
  return false;
}

//----------------------------------------------------------------------------
/// Set volume position.
void vvRenderer::setPosition(const vvVector3* p)
{
  vvDebugMsg::msg(3, "vvRenderer::setPosition()");
  vd->pos.copy(p);
}

//----------------------------------------------------------------------------
/// Get volume position.
void vvRenderer::getPosition(vvVector3* p)
{
  vvDebugMsg::msg(3, "vvRenderer::getPosition()");
  p->copy(&vd->pos);
}

//----------------------------------------------------------------------------
/** Set the direction in which the user is currently viewing.
  The vector originates in the user's eye and points along the
  viewing direction.
*/
void vvRenderer::setViewingDirection(const vvVector3*)
{
  vvDebugMsg::msg(3, "vvRenderer::setViewingDirection()");
}

//----------------------------------------------------------------------------
/// Set the direction from the user to the object.
void vvRenderer::setObjectDirection(const vvVector3*)
{
  vvDebugMsg::msg(3, "vvRenderer::setObjectDirection()");
}

void vvRenderer::setROIEnable(const bool flag)
{
  vvDebugMsg::msg(1, "vvRenderer::setROIEnable()");
  _isROIChanged = true;
  _isROIUsed = flag;
}

void vvRenderer::setSphericalROI(const bool sphericalROI)
{
  vvDebugMsg::msg(1, "vvRenderer::setSphericalROI()");
  _sphericalROI = sphericalROI;
}

bool vvRenderer::isROIEnabled() const
{
  return _isROIUsed;
}

//----------------------------------------------------------------------------
/** Set the probe position.
  @param pos  position [object space]
*/
void vvRenderer::setProbePosition(const vvVector3* pos)
{
  vvDebugMsg::msg(3, "vvRenderer::setProbePosition()");
  _isROIChanged = true;
  _roiPos.copy(pos);
}

//----------------------------------------------------------------------------
/** Get the probe position.
  @param pos  returned position [object space]
*/
void vvRenderer::getProbePosition(vvVector3* pos) const
{
  vvDebugMsg::msg(3, "vvRenderer::getProbePosition()");
  pos->copy(&_roiPos);
}

//----------------------------------------------------------------------------
/** Set the probe size.
  @param newSize  probe size. 0.0 turns off probe draw mode
*/
void vvRenderer::setProbeSize(const vvVector3* newSize)
{
  vvDebugMsg::msg(3, "vvRenderer::setProbeSize()");
  _isROIChanged = true;
  _roiSize.copy(newSize);
}

//----------------------------------------------------------------------------
/** Get the probe size.
  @return probe size (0.0 = probe mode off)
*/
void vvRenderer::getProbeSize(vvVector3* size) const
{
  vvDebugMsg::msg(3, "vvRenderer::getProbeSize()");
  size->copy(&_roiSize);
}

//----------------------------------------------------------------------------
/** Get the current modelview matrix.
  @param a matrix which will be set to the current modelview matrix
*/
void vvRenderer::getModelviewMatrix(vvMatrix* mv) const
{
  GLfloat glmatrix[16];                           // OpenGL compatible matrix

  vvDebugMsg::msg(3, "vvRenderer::getModelviewMatrix()");
  glGetFloatv(GL_MODELVIEW_MATRIX, glmatrix);
  mv->getGL((float*)glmatrix);
}

//----------------------------------------------------------------------------
/** Get the current projection matrix.
  @param a matrix which will be set to the current projection matrix
*/
void vvRenderer::getProjectionMatrix(vvMatrix* pm) const
{
  vvDebugMsg::msg(3, "vvRenderer::getProjectionMatrix()");

  GLfloat glmatrix[16];                           // OpenGL compatible matrix
  glGetFloatv(GL_PROJECTION_MATRIX, glmatrix);
  pm->getGL((float*)glmatrix);
}

//----------------------------------------------------------------------------
/** Set the OpenGL modelview matrix.
  @param new OpenGL modelview matrix
*/
void vvRenderer::setModelviewMatrix(const vvMatrix* mv)
{
  vvDebugMsg::msg(3, "vvRenderer::setModelviewMatrix()");

  GLfloat glmatrix[16];                           // OpenGL compatible matrix
  mv->makeGL((float*)glmatrix);
  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixf(glmatrix);
}

//----------------------------------------------------------------------------
/** Set the OpenGL projection matrix.
  @param new OpenGL projection matrix
*/
void vvRenderer::setProjectionMatrix(const vvMatrix* pm)
{
  vvDebugMsg::msg(3, "vvRenderer::setProjectionMatrix()");

  GLfloat glmatrix[16];                           // OpenGL compatible matrix
  pm->makeGL((float*)glmatrix);
  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf(glmatrix);
  glMatrixMode(GL_MODELVIEW);
}

//----------------------------------------------------------------------------
/** Compute user's eye position.
  @param eye  vector to receive eye position [world space]
*/
void vvRenderer::getEyePosition(vvVector3* eye) const
{
  vvMatrix invPM;                                 // inverted projection matrix
  vvVector4 projEye;                              // eye x PM

  vvDebugMsg::msg(3, "vvRenderer::getEyePosition()");

  getProjectionMatrix(&invPM);
  invPM.invert();
  projEye.set(0.0f, 0.0f, -1.0f, 0.0f);
  projEye.multiply(&invPM);
  eye->copy(&projEye);
}

//----------------------------------------------------------------------------
/** Find out if user is inside of volume.
  @param point  point to test [object coordinates]
  @return true if the given point is inside or on the volume boundaries.
*/
bool vvRenderer::isInVolume(const vvVector3* point) const
{
  vvVector3 size;                                 // object size
  vvVector3 size2;                                // half object size
  vvVector3 pos;                                  // object location
  int i;

  vvDebugMsg::msg(3, "vvRenderer::isInVolume()");

  pos.copy(&vd->pos);
  size = vd->getSize();
  size2.copy(&size);
  size2.scale(0.5f);
  for (i=0; i<3; ++i)
  {
    if ((*point)[i] < (pos[i] - size2[i])) return false;
    if ((*point)[i] > (pos[i] + size2[i])) return false;
  }
  return true;
}

//----------------------------------------------------------------------------
/** Gets the alpha value nearest to the point specified in x, y and z
  coordinates with consideration of the alpha transfer function.
  @param x,y,z  point to test [object coordinates]
  @return normalized alpha value, -1.0 if point is outside of volume, or -2
    if wrong data type
*/
float vvRenderer::getAlphaValue(float x, float y, float z)
{
  vvVector3 size, size2;                          // full and half object sizes
  vvVector3 point(x,y,z);                         // point to get alpha value at
  vvVector3 pos;
  float index;                                    // floating point index value into alpha TF [0..1]
  int vp[3];                                      // position of nearest voxel to x/y/z [voxel space]
  int i;
  uchar* ptr;

  vvDebugMsg::msg(3, "vvRenderer::getAlphaValue()");

  size = vd->getSize();
  size2.copy(&size);
  size2.scale(0.5f);
  pos.copy(&vd->pos);

  for (i=0; i<3; ++i)
  {
    if (point[i] < (pos[i] - size2[i])) return -1.0f;
    if (point[i] > (pos[i] + size2[i])) return -1.0f;

    vp[i] = int(float(vd->vox[i]) * (point[i] - pos[i] + size2[i]) / size[i]);
    vp[i] = ts_clamp(vp[i], 0, vd->vox[i]-1);
  }

  vp[1] = vd->vox[1] - vp[1] - 1;
  vp[2] = vd->vox[2] - vp[2] - 1;
  ptr = vd->getRaw(getCurrentFrame()) + vd->bpc * vd->chan * (vp[0] + vp[1] * vd->vox[0] + vp[2] * vd->vox[0] * vd->vox[1]);

  // Determine index into alpha LUT:
  if (vd->bpc==1) index = float(*ptr) / 255.0f;
  else if (vd->bpc==2) index = (float(*(ptr+1)) + float(int(*ptr) << 8)) / 65535.0f;
  else if (vd->bpc==1 && vd->chan==3) index = (float(*ptr) + float(*(ptr+1)) + float(*(ptr+2))) / (3.0f * 255.0f);
  else if (vd->bpc==1 && vd->chan==4) index = float(*(ptr+3)) / 255.0f;
  else return -2.0f;

  // Determine alpha value:
  return vd->tf.computeOpacity(index);
}

//----------------------------------------------------------------------------
/** Set a new value for a parameter. This function can be interpreted
    differently by every actual renderer implementation.
  @param param      parameter to change
  @param newValue   new value
  @param objName    name of the object to change (default: NULL)
*/
void vvRenderer::setParameter(const ParameterType param, const float newValue)
{
  vvDebugMsg::msg(3, "vvRenderer::setParameter()");
  vvRenderState::setParameter(param, newValue);
}

//----------------------------------------------------------------------------
/** Get a parameter value.
  @param param    parameter to get value of
  @param objName  name of the object to get value of (default: NULL)
*/
float vvRenderer::getParameter(const ParameterType param) const
{
  vvDebugMsg::msg(3, "vvRenderer::getParameter()");
  return vvRenderState::getParameter(param);
}

//----------------------------------------------------------------------------
/// Start benchmarking.
void vvRenderer::profileStart()
{
  vvDebugMsg::msg(1, "vvRenderer::profileStart()");
}

//----------------------------------------------------------------------------
/// Stop benchmarking.
void vvRenderer::profileStop()
{
  vvDebugMsg::msg(1, "vvRenderer::profileStop()");
}

//----------------------------------------------------------------------------
/** Set gamma correction value for one basic color.
  @param color color
  @param val  new gamma value for that color
*/
void  vvRenderer::setGamma(BasicColorType color, float val)
{
  vvDebugMsg::msg(3, "vvRenderer::setGamma()");
  switch(color)
  {
    case VV_RED:   _gamma[0] = val; break;
    case VV_GREEN: _gamma[1] = val; break;
    case VV_BLUE:  _gamma[2] = val; break;
    case VV_ALPHA: _gamma[3] = val; break;
    default: assert(0); break;
  }
}

//----------------------------------------------------------------------------
/** @return gamma correction value for one basic color.
  @param color color
  @return current gamma for that color, or -1 on error
*/
float vvRenderer::getGamma(BasicColorType color)
{
  vvDebugMsg::msg(3, "vvRenderer::getGamma()");
  switch(color)
  {
    case VV_RED:   return _gamma[0];
    case VV_GREEN: return _gamma[1];
    case VV_BLUE:  return _gamma[2];
    case VV_ALPHA: return _gamma[3];
    default: assert(0); return -1.0f;
  }
}

//----------------------------------------------------------------------------
/** Performs gamma correction.
  gamma_corrected_image = image ^ (1/crt_gamma)
  image is brightness between [0..1]
  For most CRTs, the crt_gamma is somewhere between 1.0 and 3.0.
  @param val value to gamma correct, expected to be in [0..1]
  @param color color
  @return gamma corrected value
*/
float vvRenderer::gammaCorrect(float val, BasicColorType color)
{
  vvDebugMsg::msg(3, "vvRenderer::gammaCorrect()");
  float g = getGamma(color);
  if (g==0.0f) return 0.0f;
  else return ts_clamp(powf(val, 1.0f / g), 0.0f, 1.0f);
}

//----------------------------------------------------------------------------
/** Set color of 4th data channel.
  @param color color to set (VV_RED, VV_GREEN, or VV_BLUE)
  @param val weight in [0..1]
*/
void vvRenderer::setChannel4Color(BasicColorType color, float val)
{
  vvDebugMsg::msg(3, "vvRenderer::setChannel4Color()");

  val = ts_clamp(val, 0.0f, 1.0f);
  switch(color)
  {
    case VV_RED:   _channel4Color[0] = val; break;
    case VV_GREEN: _channel4Color[1] = val; break;
    case VV_BLUE:  _channel4Color[2] = val; break;
    default: assert(0); break;
  }
}

//----------------------------------------------------------------------------
/** @return color components for rendering the 4th channel
  @param color basic color component
  @return current weight for that component, or -1 on error
*/
float vvRenderer::getChannel4Color(BasicColorType color)
{
  vvDebugMsg::msg(3, "vvRenderer::getChannel4Color()");
  switch(color)
  {
    case VV_RED:   return _channel4Color[0];
    case VV_GREEN: return _channel4Color[1];
    case VV_BLUE:  return _channel4Color[2];
    default: assert(0); return -1.0f;
  }
}

//----------------------------------------------------------------------------
/** Set opacity weights for multi-channel data sets.
  @param color color to set (VV_RED, VV_GREEN, VV_BLUE, or VV_ALPHA)
  @param val weight [0..1]
*/
void  vvRenderer::setOpacityWeight(BasicColorType color, float val)
{
  vvDebugMsg::msg(3, "vvRenderer::setOpacityWeights()");

  val = ts_clamp(val, 0.0f, 1.0f);
  switch(color)
  {
    case VV_RED:   _opacityWeights[0] = val; break;
    case VV_GREEN: _opacityWeights[1] = val; break;
    case VV_BLUE:  _opacityWeights[2] = val; break;
    case VV_ALPHA: _opacityWeights[3] = val; break;
    default: assert(0); break;
  }
}

//----------------------------------------------------------------------------
/** @return weights for opacity of multi-channel data sets
  @param color basic color component (VV_RED, VV_GREEN, VV_BLUE, or VV_ALPHA)
  @return current weight for that component, or -1 on error
*/
float vvRenderer::getOpacityWeight(BasicColorType color)
{
  vvDebugMsg::msg(3, "vvRenderer::getOpacityWeights()");
  switch(color)
  {
    case VV_RED:   return _opacityWeights[0];
    case VV_GREEN: return _opacityWeights[1];
    case VV_BLUE:  return _opacityWeights[2];
    case VV_ALPHA: return _opacityWeights[3];
    default: assert(0); return -1.0f;
  }
}

//============================================================================
// End of File
//============================================================================
