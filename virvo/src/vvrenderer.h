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

#ifndef _VVRENDERER_H_
#define _VVRENDERER_H_

#include "vvexport.h"
#include "vvvecmath.h"
#include "vvvoldesc.h"
#include "vvrenderer.h"

//============================================================================
// Class Definition
//============================================================================

class VIRVOEXPORT vvRenderState
{
  public:
    vvVector3 _clipPoint;                         ///< point on clipping plane
    vvVector3 _clipNormal;                        ///< clipping plane normal
    float _clipColor[3];                          ///< clipping plane boundary color (R,G,B in [0..1])
    float _quality;                               ///< rendering image quality (0=minimum, 1=sampling rate, >1=oversampling)
    int   _mipMode;                               ///< min/maximum intensity projection (0=off, 1=max, 2=min)
	int	  _alphaMode;							  ///< calculation for alpha value (0=max of channel weights*values, 1=weighted avg)
    bool  _clipPerimeter;                         ///< true = render line around clipping plane
    bool  _boundaries;                            ///< true = display volume boundaries
    bool  _orientation;                           ///< true = display object orientation
    bool  _palette;                               ///< true = display transfer function palette
    bool  _qualityDisplay;                        ///< true = display rendering quality level
    bool  _clipMode;                              ///< true = clipping plane enabled, false=disabled
    bool  _clipSingleSlice;                       ///< true = use single slice in clipping mode
    bool  _clipOpaque;                            ///< true = make single slice opaque
    vvVector3 _roiSize;                           ///< size of roi in each dimension [0..1]
    bool  _isROIUsed;                             ///< true = use roi
    vvVector3 _roiPos;                            ///< object space coordinates of ROI midpoint [mm]
    int   _brickSize[3];                          ///< last bricksize in x/y/z
    bool  _showBricks;                            ///< true = show brick boundarys
    bool  _computeBrickSize;                      ///< true = calculate brick size
    int   _texMemorySize;                         ///< size of texture memory
    bool  _fpsDisplay;                            ///< true = show frame rate
    bool  _gammaCorrection;                       ///< true = gamma correction on
    float _gamma[4];                              ///< gamma correction value: 0=red, 1=green, 2=blue, 3=4th channel
    bool  _opacityWeights;                        ///< true = for multi-channel data sets only: allow weighted opacities in channels
    float _boundColor[3];                         ///< boundary color (R,G,B in [0..1])
    float _probeColor[3];                         ///< probe boundary color (R,G,B in [0..1])
	bool _showTexture;							  ///< true = show texture mapping, if applicable, added by Han, Feb 2008

    vvRenderState();
    void setClippingPlane(const vvVector3*, const vvVector3*);
    void getClippingPlane(vvVector3*, vvVector3*);
    void setClipColor(float, float, float);
    void getClipColor(float*, float*, float*);
    void setBoundariesColor(float, float, float);
    void getBoundariesColor(float*, float*, float*);
    void setProbeColor(float, float, float);
    void getProbeColor(float*, float*, float*);
};

/** Abstract volume rendering class.
  @author Juergen Schulze-Doebold (schulze@hlrs.de)
  @see vvSoftVR
  @see vvTexRend
  @see vvVolTex
  @see vvRenderVP
*/
class VIRVOEXPORT vvRenderer
{
  public:
    enum RendererType                             /// Current renderer
    {
      TEXREND = 0,                                ///< texture based renderer
      SOFTPAR,                                    ///< software based renderer for parallel projection
      SOFTPER,                                    ///< software based renderer for perspective projection
      VOLPACK,                                    ///< Phil Lacroute's VolPack renderer
      SIMIAN,                                     ///< Joe Kniss's Simian renderer
      IMGREND,                                    ///< 2D image renderer
      UNKNOWN,                                    ///< unknown renderer
      STINGRAY,                                    ///< Imedia's Stingray renderer
	  VIRTEXREND								  ///< virtualized texture memory using bricking + out-of-core
    };
    enum ParameterType                            ///  Names for rendering parameters
    {
      VV_OPCORR = 0,                              ///< opacity correction on/off
      VV_SLICEINT,
      VV_PREINT,                                  ///< pre-integration on/off
      VV_MIN_SLICE,                               ///< minimum slice index to render
      VV_MAX_SLICE,                               ///< maximum slice index to render
      VV_BINNING,                                 ///< binning type (linear, iso-value, opacity)
      VV_SLICEORIENT                              ///< slice orientation for planer 3d textures
    };
    enum BasicColorType                           /// basic colors
    {
      VV_RED = 0,
      VV_GREEN,
      VV_BLUE,
      VV_ALPHA
    };
    vvRenderState _renderState;                   ///< state of renderer

  protected:
    RendererType rendererType;                    ///< currently used renderer type
    vvVolDesc* vd;                                ///< volume description
    float      _channel4Color[3];                 ///< weights for visualization of 4th channel in RGB mode
    float      _opacityWeights[4];                ///< opacity weights for alpha blending with 4 channels

    virtual void init();                   		  ///< initialization routine

    // Class Methods:
  public:                                         // public methods will be inherited as public
    vvRenderer(vvVolDesc*, vvRenderState);
    virtual ~vvRenderer();
    float		_lastRenderTime;                   ///< time it took to render the previous frame (seconds)
	float		_lastComputeTime;
	float		_lastPlaneSortingTime;
	float		_lastGLdrawTime;

    // Static methods:
    static float adaptQuality(float, float, float, float);

    // Public methods that should be redefined by subclasses:
    virtual RendererType getRendererType();
    virtual void  renderVolumeGL();
    virtual void  renderVolumeRGB(int, int, uchar*);
	virtual void  renderMultipleVolume();
    virtual void  updateTransferFunction();
    virtual void  updateVolumeData();
    virtual int   getNumFrames();
    virtual int   getCurrentFrame();
    virtual void  setCurrentFrame(int);
    virtual float getLastRenderTime();
    virtual void  setPosition(const vvVector3*);
    virtual void  getPosition(vvVector3*);
    virtual void  renderCoordinates();
    virtual void  renderPalette();
    virtual void  renderQualityDisplay();
    virtual void  renderFPSDisplay();
    virtual void  drawBoundingBox(vvVector3*, vvVector3*, float*);
    virtual void  drawPlanePerimeter(vvVector3*, vvVector3*, vvVector3*, vvVector3*, float*);
    virtual bool  instantClassification();
    virtual void  setViewingDirection(const vvVector3*);
    virtual void  setObjectDirection(const vvVector3*);
    virtual void  setROIEnable(bool);
    virtual bool  isROIEnabled();
    virtual void  setProbePosition(const vvVector3*);
    virtual void  getProbePosition(vvVector3*);
    virtual void  setProbeSize(vvVector3*);
    virtual void  getProbeSize(vvVector3*);
    virtual void  getModelviewMatrix(vvMatrix*);
    virtual void  getProjectionMatrix(vvMatrix*);
    virtual void  setModelviewMatrix(vvMatrix*);
    virtual void  setProjectionMatrix(vvMatrix*);
    virtual void  getEyePosition(vvVector3*);
    virtual bool  isInVolume(const vvVector3*);
    virtual float getAlphaValue(float, float, float);
    virtual void  setParameter(ParameterType, float, char* = NULL);
    virtual float getParameter(ParameterType, char* = NULL);
    virtual void  profileStart();
    virtual void  profileStop();
    virtual void  setGamma(BasicColorType, float);
    virtual float getGamma(BasicColorType);
    virtual float gammaCorrect(float, BasicColorType);
    virtual void  setChannel4Color(BasicColorType, float);
    virtual float getChannel4Color(BasicColorType);
    virtual void  setOpacityWeight(BasicColorType, float);
    virtual float getOpacityWeight(BasicColorType);

	// added by Han Kim Feb. 2008
	virtual void setVolDesc(vvVolDesc*);
	virtual vvVolDesc* getVolDesc();
};
#endif

//============================================================================
// End of File
//============================================================================
