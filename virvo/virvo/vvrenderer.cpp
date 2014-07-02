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

#include <GL/glew.h>

#include <stdlib.h>
#include "vvclock.h"
#include "vvplatform.h"
#include "vvopengl.h"
#include "vvvoldesc.h"
#include "vvtoolshed.h"
#include <string.h>
#include <math.h>
#include <assert.h>
#include <limits>

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvrenderer.h"
#include "vvdebugmsg.h"
#include "vvprintgl.h"

#include "gl/util.h"

#include "private/vvgltools.h"
#include "private/vvlog.h"

namespace gl = virvo::gl;
namespace math = virvo::math;


//----------------------------------------------------------------------------
vvRenderState::vvRenderState()
  : _quality(1.0f)
  , _mipMode(0)
  , _alphaMode(0)
  , _emptySpaceLeaping(true)
  , _boundaries(false)
  , _orientation(false)
  , _palette(false)
  , _qualityDisplay(false)
  , _clipMode(0)
  , clip_plane_point_(math::vec3f(0.0f, 0.0f, 0.0f))
  , clip_plane_normal_(math::vec3f(0.0f, 0.0f, 1.0f))
  , _clipPlanePerimeter(true)
  , _clipPlaneColor(vvColor(1.0f, 1.0f, 1.0f))
  , clip_sphere_center_(math::vec3f(0.0f, 0.0f, 0.0f))
  , _clipSphereRadius(100.0f)
  , _clipSingleSlice(false)
  , _clipOpaque(false)
  , _isROIUsed(false)
  , roi_pos_(math::vec3f(0.0f, 0.0f, 0.0f))
  , roi_size_(math::vec3f(0.5f, 0.5f, 0.5f))
  , _sphericalROI(false)
  , _brickSize(vvsize3(0, 0, 0))
  , _maxBrickSize(vvsize3(64, 64, 64))
  , _brickTexelOverlap(1)
  , _showBricks(false)
  , _computeBrickSize(true)
  , _texMemorySize(0)
  , _fpsDisplay(false)
  , _gammaCorrection(false)
  , gamma_(math::vec4f(1.0f, 1.0f, 1.0f, 1.0f))
  , _opacityWeights(false)
  , _boundColor(vvColor(1.0f, 1.0f, 1.0f))
  , _probeColor(vvColor(1.0f, 1.0f, 1.0f))
  , _useOffscreenBuffer(false)
  , _imageScale(1.0f)
  , _imagePrecision(virvo::Byte)
  , _lighting(false)
  , _showTexture(true)
  , _useIbr(false)
  , _ibrMode(VV_REL_THRESHOLD)
  , _visibleRegion(vvAABBss(vvssize3(0), vvssize3(std::numeric_limits<ssize_t>::max())))
  , _paddingRegion(vvAABBss(vvssize3(0), vvssize3(std::numeric_limits<ssize_t>::max())))
  , _opacityCorrection(true)
  , _interpolation(virvo::Linear)
  , _earlyRayTermination(true)
  , _preIntegration(false)
  , _depthPrecision(8)
  , depth_range_(0.0f, 0.0f)
{
  
}


bool vvRenderState::checkParameter(ParameterType param, vvParam const& value) const
{
  VV_UNUSED(param);
  VV_UNUSED(value);

  return false;
}


void vvRenderState::setParameter(ParameterType param, const vvParam& value)
{
  vvDebugMsg::msg(3, "vvRenderState::setParameter()");

  switch (param)
  {
  case VV_QUALITY:
    _quality = value;
    break;
  case VV_MIP_MODE:
    _mipMode = value;
    break;
  case VV_ALPHA_MODE:
    _alphaMode = value;
    break;
  case VV_LEAPEMPTY:
    _emptySpaceLeaping = value;
    break;
  case VV_CLIP_PERIMETER:
    _clipPlanePerimeter = value;
    break;
  case VV_BOUNDARIES:
    _boundaries = value;
    break;
  case VV_ORIENTATION:
    _orientation = value;
    break;
  case VV_PALETTE:
    _palette = value;
    break;
  case VV_QUALITY_DISPLAY:
    _qualityDisplay = value;
    break;
  case VV_CLIP_MODE:
    _clipMode = value;
    break;
  case VV_CLIP_SINGLE_SLICE:
    _clipSingleSlice = value;
    break;
  case VV_CLIP_OPAQUE:
    _clipOpaque = value;
    break;
  case VV_IS_ROI_USED:
    _isROIUsed = value;
    _isROIChanged = true;
    break;
  case VV_SPHERICAL_ROI:
    _sphericalROI = value;
    _isROIChanged = true;
    break;
  case VV_BRICK_TEXEL_OVERLAP:
    _brickTexelOverlap = value;
    break;
  case VV_SHOW_BRICKS:
    _showBricks = value;
    break;
  case VV_COMPUTE_BRICK_SIZE:
    _computeBrickSize = value;
    break;
  case VV_TEX_MEMORY_SIZE:
    _texMemorySize = value;
    break;
  case VV_FPS_DISPLAY:
    _fpsDisplay = value;
    break;
  case VV_GAMMA_CORRECTION:
    _gammaCorrection = value;
    break;
  case VV_OPACITY_WEIGHTS:
    _opacityWeights = value;
    break;
  case VV_USE_OFFSCREEN_BUFFER:
    _useOffscreenBuffer = value;
    break;
  case VV_IMAGE_SCALE:
    _imageScale = value;
    break;
  case VV_IMAGE_PRECISION:
    _imagePrecision = (virvo::BufferPrecision)value.asInt();
    break;
  case VV_SHOW_TEXTURE:
    _showTexture = value;
    break;
  case VV_USE_IBR:
    _useIbr = value;
    break;
  case VV_IBR_MODE:
    _ibrMode = (IbrMode)value.asInt();
    break;
  case VV_VISIBLE_REGION:
    _visibleRegion = value;
    break;
  case VV_PADDING_REGION:
    _paddingRegion = value;
    break;
  case VV_CLIP_PLANE_POINT:
    clip_plane_point_ = value;
    break;
  case VV_CLIP_PLANE_NORMAL:
    clip_plane_normal_ = value;
    if ( math::length(clip_plane_normal_) < 1e-20 )
    {
        clip_plane_normal_ = math::vec3f(0.0f, 0.0f, 1.0f);
    }
    clip_plane_normal_ = math::normalize( clip_plane_normal_ );
    if ( math::length(clip_plane_normal_) < 0.5f )
    {
        clip_plane_normal_ = math::vec3f(0.0f, 0.0f, 1.0f);
    }
    break;
  case VV_CLIP_COLOR:
    _clipPlaneColor = value;
    break;
  case VV_ROI_POS:
    roi_pos_ = value;
    _isROIChanged = true;
    break;
  case VV_ROI_SIZE:
    roi_size_ = value;
    _isROIChanged = true;
    break;
  case VV_BRICK_SIZE:
    _brickSize = value;
    break;
  case VV_MAX_BRICK_SIZE:
    _maxBrickSize = value;
    break;
  case VV_BOUND_COLOR:
    _boundColor = value;
    break;
  case VV_PROBE_COLOR:
    _probeColor = value;
    break;
  case VV_GAMMA:
    gamma_ = value;
    break;
  case VV_LIGHTING:
    _lighting = value;
    break;
  case VV_OPCORR:
    _opacityCorrection = value;
    break;
  case VV_SLICEINT:
    _interpolation = static_cast< virvo::tex_filter_mode >(value.asInt());
    break;
  case VV_PREINT:
    _preIntegration = value;
    break;
  case VV_TERMINATEEARLY:
    _earlyRayTermination = value;
    break;
  case VV_IBR_DEPTH_PREC:
    _depthPrecision = value;
    break;
  case VV_IBR_DEPTH_RANGE:
    depth_range_ = value;
    break;
  default:
    break;
  }
}

vvParam vvRenderState::getParameter(ParameterType param) const
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
  case VV_LEAPEMPTY:
    return _emptySpaceLeaping;
  case VV_CLIP_PERIMETER:
    return _clipPlanePerimeter;
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
    return (int)_imagePrecision;
  case VV_SHOW_TEXTURE:
    return _showTexture;
  case VV_USE_IBR:
    return _useIbr;
  case VV_IBR_MODE:
    return (int)_ibrMode;
  case VV_VISIBLE_REGION:
    return _visibleRegion;
  case VV_PADDING_REGION:
    return _paddingRegion;
  case VV_CLIP_PLANE_POINT:
    return clip_plane_point_;
  case VV_CLIP_PLANE_NORMAL:
    return clip_plane_normal_;
  case VV_CLIP_COLOR:
    return _clipPlaneColor;
  case VV_ROI_SIZE:
    return roi_size_;
  case VV_ROI_POS:
    return roi_pos_;
  case VV_BRICK_SIZE:
    return _brickSize;
  case VV_MAX_BRICK_SIZE:
    return _maxBrickSize;
  case VV_BOUND_COLOR:
    return _boundColor;
  case VV_PROBE_COLOR:
    return _probeColor;
  case VV_GAMMA:
    return gamma_;
  case VV_SHOW_BRICKS:
    return _showBricks;
  case VV_COMPUTE_BRICK_SIZE:
    return _computeBrickSize;
  case VV_LIGHTING:
    return _lighting;
  case VV_OPCORR:
    return _opacityCorrection;
  case VV_SLICEINT:
    return _interpolation;
  case VV_PREINT:
    return _preIntegration;
  case VV_TERMINATEEARLY:
    return _earlyRayTermination;
  case VV_IBR_DEPTH_PREC:
    return _depthPrecision;
  case VV_IBR_DEPTH_RANGE:
    return depth_range_;
  default:
    return vvParam();
  }
}

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
/** Constructor, called when program starts and when the rendering method changes
  @param voldesc volume descriptor, should be deleted when not needed anymore
  @param renderState contains variables for the renderer
*/
vvRenderer::vvRenderer(vvVolDesc* voldesc, vvRenderState renderState)
  : vvRenderState(renderState)
  , viewDir(math::vec3f(0.0f, 0.0f, 0.0f))
  , objDir(math::vec3f(0.0f, 0.0f, 0.0f))
  , renderTarget_(virvo::NullRT::create())
  , stopwatch_(new vvStopwatch)
{
  vvDebugMsg::msg(1, "vvRenderer::vvRenderer()");
  assert(voldesc!=NULL);
  rendererType = GENERIC;
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

vvVecmath::AxisType vvRenderer::getPrincipalViewingAxis(math::mat4 const& mv,
                                                        float& zx, float& zy, float& zz) const
{
  math::mat4 invMV = inverse(mv);

  math::vec3 eye = getEyePosition();

  math::vec3 normal;
  math::vec3 origin;
  getObjNormal(normal, origin, eye, invMV);

  zx = normal.x;
  zy = normal.y;
  zz = normal.z;

  if (fabs(zx) > fabs(zy))
  {
    if (fabs(zx) > fabs(zz)) return vvVecmath::X_AXIS;
    else return vvVecmath::Z_AXIS;
  }
  else
  {
    if (fabs(zy) > fabs(zz)) return vvVecmath::Y_AXIS;
    else return vvVecmath::Z_AXIS;
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

void vvRenderer::getObjNormal(math::vec3f& normal, math::vec3f& origin,
                              math::vec3f const& eye, const vvMatrix& invMV,
                              bool isOrtho) const
{
  // Compute normal vector of textures using the following strategy:
  // For orthographic projections or if viewDir is (0|0|0) use
  // (0|0|1) as the normal vector.
  // Otherwise use objDir as the normal.
  // Exception: if user's eye is inside object and probe mode is off,
  // then use viewDir as the normal.
  if (_clipMode == 1)
  {
    normal = clip_plane_normal_;
  }
  else if ( isOrtho || viewDir == math::vec3f(0.0f) )
  {
    // Draw slices parallel to projection plane:
    normal = math::vec3f(0.0f, 0.0f, 1.0f);       // (0|0|1) is normal on projection plane
    vvVector3 tmp( normal );                      // TODO
    tmp.multiply(invMV);
    normal = tmp;                                 // TODO
    origin = math::vec3f(0.0f, 0.0f, 0.0f);
    tmp = vvVector3( origin );                    // TODO
    tmp.multiply(invMV);
    origin = tmp;                                 // TODO
    normal -= origin;
  }
  else if (!_isROIUsed && isInVolume(eye))
  {
    // Draw slices perpendicular to viewing direction:
    normal = -viewDir;                            // viewDir points away from user, the normal should point towards them
  }
  else
  {
    // Draw slices perpendicular to line eye->object:
    normal = -objDir;
  }

  if (normal == math::vec3f(0.f))
  {
    normal = -viewDir;
  }
  if (normal == math::vec3f(0.f))
  {
    normal = -objDir;
  }
  if (normal == math::vec3f(0.f))
  {
    normal = math::vec3f(0.f, 0.f, 1.f);
  }

  normal = normalize(normal);
}

void vvRenderer::getShadingNormal(math::vec3f& normal, math::vec3f& origin,
                                  math::vec3f const& eye, const vvMatrix& invMV,
                                  bool isOrtho) const
{
  // See calcutions in getObjNormal(). Only difference: if clip plane
  // is active, this is ignored. This normal isn't used to align
  // slices or such with the clipping plane, but for shading calculations.
  if ( isOrtho || viewDir == math::vec3f(0.0f) )
  {
    // Draw slices parallel to projection plane:
    normal = math::vec3f(0.0f, 0.0f, 1.0f);       // (0|0|1) is normal on projection plane
    vvVector3 tmp( normal );                      // TODO
    tmp.multiply(invMV);
    normal = tmp;                                 // TODO
    origin = math::vec3f(0.0f, 0.0f, 0.0f);
    tmp = vvVector3( origin );                    // TODO
    tmp.multiply(invMV);
    origin = tmp;
    normal -= origin;
  }
  else if (!_isROIUsed && isInVolume(eye))
  {
    // Draw slices perpendicular to viewing direction:
    normal = -viewDir;                            // viewDir points away from user, the normal should point towards them
  }
  else
  {
    // Draw slices perpendicular to line eye->object:
    normal = -objDir;
  }

  normal = normalize(normal);
}

void vvRenderer::calcProbeDims(math::vec3f& probePosObj, math::vec3f& probeSizeObj, math::vec3f& probeMin, math::vec3f& probeMax) const
{
  // Determine texture object dimensions and half object size as a shortcut:
  math::vec3f size(vd->getSize());
  math::vec3f size2 = size * 0.5f;

  if (_isROIUsed)
  {
    // Convert probe midpoint coordinates to object space w/o position:
    probePosObj = roi_pos_;

    // Compute probe min/max coordinates in object space:
    math::vec3f maxSize = roi_size_ * size2;

    probeMin = probePosObj - maxSize;
    probeMax = probePosObj + maxSize;

    // Constrain probe boundaries to volume data area:
    for (size_t i = 0; i < 3; ++i)
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
    probeSizeObj = size;
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

  vvGLTools::printGLError("enter vvRenderer::renderVolumeGL");
}

void vvRenderer::renderOpaqueGeometry() const
{
  renderBoundingBox();
}

void vvRenderer::renderHUD() const
{
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
  glDrawBuffer(GL_FRONT);                         // set draw buffer to front in order to read image data
  glPixelStorei(GL_PACK_ALIGNMENT, 1);            // Important command: default value is 4, so allocated memory wouldn't suffice

  // Read image data:
  std::vector< uchar > screenshot(viewPort[2] * viewPort[3] * 3);
  glReadPixels(0, 0, viewPort[2], viewPort[3], GL_RGB, GL_UNSIGNED_BYTE, &screenshot[0]);

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
      memcpy(data + dstIndex, &screenshot[0] + srcIndex, 3);
    }
  }
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
size_t vvRenderer::getNumFrames()
{
  vvDebugMsg::msg(3, "vvRenderer::getNumFrames()");
  return vd->frames;
}

//----------------------------------------------------------------------------
/** Returns index of current animation frame.
  (first frame = 0, <0 if undefined)
*/
size_t vvRenderer::getCurrentFrame()
{
  vvDebugMsg::msg(3, "vvRenderer::getCurrentFrame()");
  return vd->getCurrentFrame();
}

//----------------------------------------------------------------------------
/** Set new frame index.
  @param index  new frame index (0 for first frame)
*/
void vvRenderer::setCurrentFrame(size_t index)
{
  vvDebugMsg::msg(3, "vvRenderer::setCurrentFrame()");
  if (index == vd->getCurrentFrame()) return;
  if (index >= vd->frames) index = vd->frames - 1;
  VV_LOG(3) << "New frame index: " << index << std::endl;
  vd->setCurrentFrame(index);
}

//----------------------------------------------------------------------------
/** Get last render time.
  @return time to render last image
*/
float vvRenderer::getLastRenderTime() const
{
  vvDebugMsg::msg(3, "vvRenderer::getLastRenderTime()");
  return _lastRenderTime;
}

//----------------------------------------------------------------------------
/** Render axis coordinates in bottom right corner.
  Arrows are of length 1.0<BR>
  Colors: x-axis=red, y-axis=green, z-axis=blue
*/
void vvRenderer::renderCoordinates() const
{
  glPushAttrib(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_CURRENT_BIT | GL_TRANSFORM_BIT);

  vvMatrix mv;                                    // current modelview matrix
  vvVector3 column;                               // column vector
  GLint viewPort[4];                              // x, y, width, height of viewport
  float aspect;                                   // viewport aspect ratio
  float half[2];                                  // half viewport dimensions (x,y)
  int i;

  glDepthFunc(GL_ALWAYS);
  glDepthMask(GL_FALSE);

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
  vvGLTools::getModelviewMatrix(&mv);
  mv.killTrans();
  for (i=0; i<3; ++i)                             // normalize base vectors to remove scaling
  {
    mv.getColumn(i, column);
    column.normalize();
    mv.setColumn(i, column);
  }
  mv.translate(0.8f * half[0], -0.8f * half[1], 0.0f);
  mv.scaleLocal(0.2f, 0.2f, 0.2f);
  vvGLTools::setModelviewMatrix(mv);

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
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  glPopAttrib();
}

//----------------------------------------------------------------------------
/// Render transfer function palette at left border.
void vvRenderer::renderPalette() const
{
  const int WIDTH = 10;                           // palette width [pixels]
  GLfloat viewport[4];                            // OpenGL viewport information (position and size)
  GLfloat glsRasterPos[4];                        // current raster position (glRasterPos)
  int w, h;                                       // width and height of palette
  int x, y, c;

  vvDebugMsg::msg(3, "vvRenderer::renderPalette()");

  if (vd->chan > 1) return;                       // palette only makes sense with scalar data

  // Get viewport size:
  glGetFloatv(GL_VIEWPORT, viewport);
  if (viewport[2]<=0 || viewport[3]<=0) return;   // safety first

  glPushAttrib(GL_ALL_ATTRIB_BITS);

  glDepthFunc(GL_ALWAYS);
  glDepthMask(GL_FALSE);

  // Save matrix states:
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Store raster position:
  glGetFloatv(GL_CURRENT_RASTER_POSITION, glsRasterPos);

  // Compute palette image:
  w = WIDTH;
  h = (int)viewport[3];
  std::vector< float > colors(h * 4);
  vd->tf.computeTFTexture(h, 1, 1, &colors[0], vd->real[0], vd->real[1]);
  std::vector< uint8_t > image(w * h * 3);
  for (x=0; x<w; ++x)
    for (y=0; y<h; ++y)
      for (c=0; c<3; ++c)
        image[c + 3 * (x + w * y)] = (uint8_t)(colors[4 * y + c] * 255.99f);

  // Draw palette:
  glRasterPos2f(-1.0f,-1.0f);                     // pixmap origin is bottom left corner of output window
  glDrawPixels(w, h, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)&image[0]);

  // Display min and max values:
  vvPrintGL printGL;
  printGL.print(-0.90f,  0.9f,  "%-9.2f", vd->real[1]);
  printGL.print(-0.90f, -0.95f, "%-9.2f", vd->real[0]);

  // Restore state:
  glPopAttrib();
}

//----------------------------------------------------------------------------
/// Display rendering quality.
void vvRenderer::renderQualityDisplay() const
{
  vvPrintGL printGL;
  math::vec4f clearColor = vvGLTools::queryClearColor();
  math::vec4f fontColor( 1.0f - clearColor[0], 1.0f - clearColor[1], 1.0f - clearColor[2], 1.0f );
  printGL.setFontColor(fontColor);
  printGL.print(-0.9f, 0.9f, "Quality: %-9.2f", _quality);
}

//----------------------------------------------------------------------------
/// Display frame rate.
void vvRenderer::renderFPSDisplay() const
{
  float fps = getLastRenderTime();
  if (fps > 0.0f) fps = 1.0f / fps;
  else fps = -1.0f;
  vvPrintGL printGL;
  math::vec4f clearColor = vvGLTools::queryClearColor();
  math::vec4f fontColor( 1.0f - clearColor[0], 1.0f - clearColor[1], 1.0f - clearColor[2], 1.0f );
  printGL.setFontColor(fontColor);
  printGL.print(0.3f, 0.9f, "fps: %-9.1f", fps);
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
void vvRenderer::drawBoundingBox(math::vec3f const& oSize, math::vec3f const& oPos, const vvColor& color) const
{
  math::vec3f vertvec[8];                         // vertex vectors in object space
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
  glTranslatef(oPos[0], oPos[1], oPos[2]);

  // Set box color:
  glColor4f(color[0], color[1], color[2], 1.0);

  // Disable lighting:
  glDisable(GL_LIGHTING);

  // Create vertex vectors:
  for (i=0; i<8; ++i)
  {
    vertvec[i] = math::vec3f(vertices[i][0], vertices[i][1], vertices[i][2]);
    vertvec[i] *= oSize;
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
void vvRenderer::drawPlanePerimeter(math::vec3f const& oSize, math::vec3f const& oPos,
                                    math::vec3f const& oPlane, math::vec3f const& oNorm, const vvColor& color) const
{
  GLboolean glsLighting;                          // stores GL_LIGHTING
  GLfloat   glsColor[4];                          // stores GL_CURRENT_COLOR
  GLfloat   glsLineWidth;                         // stores GL_LINE_WIDTH
  vvVector3 isect[6];                             // intersection points, maximum of 6 when intersecting a plane and a volume [object space]
  math::vec3f boxMin,boxMax;                      // minimum and maximum box coordinates
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
  glTranslatef(oPos[0], oPos[1], oPos[2]);

  // Set color:
  glColor4f(color[0], color[1], color[2], 1.0);

  // Disable lighting:
  glDisable(GL_LIGHTING);

  glLineWidth(3.0f);

  boxMin = math::vec3f(oPos[0] - oSize[0] * 0.5f, oPos[1] - oSize[1] * 0.5f, oPos[2] - oSize[2] * 0.5f);
  boxMax = math::vec3f(oPos[0] + oSize[0] * 0.5f, oPos[1] + oSize[1] * 0.5f, oPos[2] + oSize[2] * 0.5f);

  isectCnt = isect->isectPlaneCuboid(oNorm, oPlane, boxMin, boxMax);

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
void vvRenderer::setPosition(math::vec3f const& p)
{
  vvDebugMsg::msg(3, "vvRenderer::setPosition()");
  vd->pos = p;
}

//----------------------------------------------------------------------------
/// Get volume position.
math::vec3f vvRenderer::getPosition() const
{
    return vd->pos;
}

//----------------------------------------------------------------------------
/** Set the direction in which the user is currently viewing.
  The vector originates in the user's eye and points along the
  viewing direction.
*/
void vvRenderer::setViewingDirection(math::vec3f const&)
{
  vvDebugMsg::msg(3, "vvRenderer::setViewingDirection()");
}

//----------------------------------------------------------------------------
/// Set the direction from the user to the object.
void vvRenderer::setObjectDirection(math::vec3f const&)
{
  vvDebugMsg::msg(3, "vvRenderer::setObjectDirection()");
}

void vvRenderer::setROIEnable(const bool flag)
{
  vvDebugMsg::msg(1, "vvRenderer::setROIEnable()");
  setParameter(VV_IS_ROI_USED, flag);
}

void vvRenderer::setSphericalROI(const bool sphericalROI)
{
  vvDebugMsg::msg(1, "vvRenderer::setSphericalROI()");
  _sphericalROI = sphericalROI;
}

bool vvRenderer::isROIEnabled() const
{
  return getParameter(VV_IS_ROI_USED);
}

//----------------------------------------------------------------------------
/** Set the probe position.
  @param pos  position [object space]
*/
void vvRenderer::setProbePosition(math::vec3f const& pos)
{
  vvDebugMsg::msg(3, "vvRenderer::setProbePosition()");
  setParameter(VV_ROI_POS, pos);
}

//----------------------------------------------------------------------------
/** Get the probe position.
  @param pos  returned position [object space]
*/
math::vec3f vvRenderer::getProbePosition() const
{
   return getParameter(VV_ROI_POS); 
}

//----------------------------------------------------------------------------
/** Set the probe size.
  @param newSize  probe size. 0.0 turns off probe draw mode
*/
void vvRenderer::setProbeSize(math::vec3f const& newSize)
{
  vvDebugMsg::msg(3, "vvRenderer::setProbeSize()");
  setParameter(VV_ROI_SIZE, newSize);
}

//----------------------------------------------------------------------------
/** Get the probe size.
  @return probe size (0.0 = probe mode off)
*/
math::vec3f vvRenderer::getProbeSize() const
{
  vvDebugMsg::msg(3, "vvRenderer::getProbeSize()");
  return getParameter(VV_ROI_SIZE);
}

//----------------------------------------------------------------------------
/** Compute user's eye position.
  @param eye  vector to receive eye position [world space]
*/
math::vec3f vvRenderer::getEyePosition() const
{
    math::mat4 invPM = inverse( gl::getProjectionMatrix() );
    math::mat4 invMV = inverse( gl::getModelviewMatrix() );

    math::vec4 projEye = invMV * ( invPM * math::vec4(0.0f, 0.0f, -1.0f, 0.0f) );
    return projEye.xyz() / projEye.w;
}

//----------------------------------------------------------------------------
/** Find out if user is inside of volume.
  @param point  point to test [object coordinates]
  @return true if the given point is inside or on the volume boundaries.
*/
bool vvRenderer::isInVolume(math::vec3f const& point) const
{
  math::vec3f pos = vd->pos;
  math::vec3f size = vd->getSize();
  math::vec3f size2 = size * 0.5f;
  for (size_t i=0; i<3; ++i)
  {
    if (point[i] < (pos[i] - size2[i])) return false;
    if (point[i] > (pos[i] + size2[i])) return false;
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
  math::vec3f point(x,y,z);                       // point to get alpha value at
  float index;                                    // floating point index value into alpha TF [0..1]
  ssize_t vp[3];                                  // position of nearest voxel to x/y/z [voxel space]
  uchar* ptr;

  vvDebugMsg::msg(3, "vvRenderer::getAlphaValue()");

  math::vec3f size = vd->getSize();
  math::vec3f size2 = size * 0.5f;
  math::vec3f pos = vd->pos;

  for (size_t i=0; i<3; ++i)
  {
    if (point[i] < (pos[i] - size2[i])) return -1.0f;
    if (point[i] > (pos[i] + size2[i])) return -1.0f;

    vp[i] = ssize_t(float(vd->vox[i]) * (point[i] - pos[i] + size2[i]) / size[i]);
    vp[i] = ts_clamp(vp[i], ssize_t(0), vd->vox[i]-1);
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
void vvRenderer::setParameter(ParameterType param, const vvParam& value)
{
  vvDebugMsg::msg(3, "vvRenderer::setParameter()");
  vvRenderState::setParameter(param, value);
}

//----------------------------------------------------------------------------
/** Get a parameter value.
  @param param    parameter to get value of
  @param objName  name of the object to get value of (default: NULL)
*/
vvParam vvRenderer::getParameter(ParameterType param) const
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
    case VV_RED:   gamma_[0] = val; break;
    case VV_GREEN: gamma_[1] = val; break;
    case VV_BLUE:  gamma_[2] = val; break;
    case VV_ALPHA: gamma_[3] = val; break;
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
    case VV_RED:   return gamma_[0];
    case VV_GREEN: return gamma_[1];
    case VV_BLUE:  return gamma_[2];
    case VV_ALPHA: return gamma_[3];
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

bool vvRenderer::beginFrame(unsigned clearMask)
{
  if (renderTarget_->beginFrame(clearMask))
  {
    renderOpaqueGeometry();
    if (_fpsDisplay)
    {
      stopwatch_->start();
    }

    return true;
  }

  return false;
}

bool vvRenderer::endFrame()
{
  if (_fpsDisplay)
  {
    _lastRenderTime = stopwatch_->getTime();
  }

  return renderTarget_->endFrame();
}

bool vvRenderer::resize(int w, int h)
{
  return renderTarget_->resize(w, h);
}

bool vvRenderer::present() const
{
  return renderTarget_->displayColorBuffer();
}

void vvRenderer::renderFrame()
{
  GLint viewport[4] = {0};

  glGetIntegerv(GL_VIEWPORT, &viewport[0]);

  renderFrame(viewport[2], viewport[3]);
}

void vvRenderer::renderFrame(int w, int h)
{
  // TODO:
  // Error handling...

  resize(w, h);
  beginFrame(virvo::CLEAR_COLOR | virvo::CLEAR_DEPTH);
  renderVolumeGL();
  endFrame();
  present();

  renderHUD();
}

void vvRenderer::renderBoundingBox() const
{
  if (!_boundaries)
    return;

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);

  vvAABB bb(math::vec3f(0.0f), math::vec3f(0.0f));

  vd->getBoundingBox(bb);

  drawBoundingBox(bb.getMax() - bb.getMin(), vd->pos, _boundColor);
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
