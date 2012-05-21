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

#ifndef _VVTEXMULTIREND_H_
#define _VVTEXMULTIREND_H_

// Glew:

// No circular dependencies between gl.h and glew.h
#ifndef GLEW_INCLUDED
#include <GL/glew.h>
#define GLEW_INCLUDED
#endif

#ifndef _WIN32
//#include "config.h"
#endif

// Virvo:
#include "../vvexport.h"
#include "../vvvoldesc.h"
#include "../vvrenderer.h"
#include "../vvtransfunc.h"
#include "../vvsllist.h"
#include "../vvshaderprogram.h"




class TexRendInfo
{
  public:
	vvMatrix  mv;                   ///< current modelview matrix
	float     glMV[16];				///< GL modelview matrix
	vvVector3 farthest, farWS;      ///< volume vertex farthest from the viewer
	vvVector3 delta;                ///< distance vector between textures [object space]
	vvVector3 normal;				///< normal vector of textures (in local & world space)
	vvVector3 probeMin, probeMax;   ///< probe min and max coordinates [object space]
	vvVector3 size2;				///< half object sizes
};


//============================================================================
// Class Definitions
//============================================================================

/** Volume rendering engine using a texture-based algorithm.
  Textures can be drawn as planes or spheres. In planes mode a rendering
  quality can be given (determining the number of texture slices used), and
  the texture normal can be set according to the application's needs.<P>
  The data points are located at the grid as follows:<BR>
  The outermost data points reside at the very edge of the drawn object,
  the other values are evenly distributed inbetween.
  @author Juergen Schulze (schulze@cs.brown.de)
  @author Martin Aumueller
  @see vvRenderer
*/
class VIRVOEXPORT vvTexMultiRend : public vvRenderer
{
  public:

    enum ErrorType                                /// Error Codes
    {
      OK,                                         ///< no error
      TRAM_ERROR,                                 ///< not enough texture memory
      NO3DTEX                                     ///< 3D textures not supported on this hardware
    };
    enum GeometryType                             /// Geometry textures are projected on
    {
      VV_VIEWPORT = 0                                ///< render planar slices using a 3D texture
    };
    enum VoxelType                                /// Internal data type used in textures
    {
	  VV_GLSL = 0                                   ///< OpenGL Shading Language
    };

    enum FeatureType                              /// Rendering features
    {
      VV_MIP                                      ///< maximum intensity projection
    };
    enum SliceOrientation                         /// Slice orientation for planar 3D textures
    {
      VV_VARIABLE = 0,                            ///< choose automatically
      VV_VIEWPLANE,                               ///< parallel to view plane
      VV_CLIPPLANE,                               ///< parallel to clip plane
      VV_VIEWDIR,                                 ///< perpendicular to viewing direction
      VV_OBJECTDIR,                               ///< perpendicular to line eye-object
      VV_ORTHO                                    ///< as in orthographic projection
    };

	vvVector3 translation;	// To manipulate each volume
	vvMatrix rotation;		// by changing modelview matrix
	TexRendInfo tr;

  protected:
    static const int NUM_PIXEL_SHADERS;           ///< number of pixel shaders used
    enum FragmentProgram
    {
      VV_FRAG_PROG_2D = 0,
      VV_FRAG_PROG_3D,
      VV_FRAG_PROG_PREINT,
      VV_FRAG_PROG_MAX                            // has always to be last in list
    };

	//========================================================================
	// copied from Chih's vvVolDesc
	float* rgbaTF;							///< density to RGBA conversion table, as created by TF [0..1]
    uchar* rgbaLUT;							///< final RGBA conversion table, as transferred to graphics hardware (includes opacity and gamma correction)

	uchar* preintTable;						///< lookup table for pre-integrated rendering, as transferred to graphics hardware
    float  lutDistance;						///< slice distance for which LUT was computed
	GLuint tfTexName;						///< name for transfer function texture
    GLuint* pixLUTName;

	GLuint* texNames;
	int   texels[3];						///< width, height and depth of volume, including empty space [texels]
    float texMin[3];						///< minimum texture value of object [0..1] (to prevent border interpolation)
    float texMax[3];						///< maximum texture value of object [0..1] (to prevent border interpolation)
    int   _ntextures;						///< number of textures stored in TRAM
	int   numSlices;
	float quality;
	float* chanWeight;						///< Weighted blending for data channels
	vvVector3* color;						///< Specified color for each channel
	float volWeight;						///< RGBA Weight for current volume

	enum TFMode								///< used for 2 and more channels
    {
      GAMMATF,
	  HIGHPASSTF,
	  HISTCDFTF
    };
	
	int tfmode;								///< Either gamma or high pass filter
	float* tfGamma;							///< Gamma adjustments for TF
	float* tfHPOrder;
	float* tfHPCutoff;						///< High Pass filter (order & cutoff)
	float* tfOffset;						///< Y-Offset from origin for TF
	uint* histCDF;

	//========================================================================


    int   texelsize;                        ///< number of bytes/voxel transferred to OpenGL (depending on rendering mode)
    GLint internalTexFormat;                ///< internal texture format (parameter for glTexImage...)
    GLenum texFormat;                       ///< texture format (parameter for glTexImage...)

    GeometryType geomType;                  ///< rendering geometry actually used
    VoxelType voxelType;                    ///< voxel type actually used
    bool extTex3d;                          ///< true = 3D texturing supported
    bool extMinMax;                         ///< true = maximum/minimum intensity projections supported
    bool extBlendEquation;                  ///< true = support for blend equation extension
    vvVector3 viewDir;                      ///< user's current viewing direction [object coordinates]
    vvVector3 objDir;                       ///< direction from viewer to object [object coordinates]
    bool interpolation;                     ///< interpolation mode: true=linear interpolation (default), false=nearest neighbor
    bool opacityCorrection;                 ///< true = opacity correction on
    int  minSlice, maxSlice;                ///< min/maximum slice to render [0..numSlices-1], -1 for no slice constraints

    SliceOrientation _sliceOrientation;     ///< slice orientation for planar 3d textures

  protected:
    ErrorType makeTextures();
    void makeLUTTexture();
    ErrorType makeTextures3D();
    ErrorType updateTextures3D(int, int, int, int, int, int, bool);
    void removeTextures();
    void renderTex3DPlanar(vvMatrix*);
    int  getLUTSize(int*);
    int  getPreintTableSize();

  public:
    vvTexMultiRend(vvVolDesc*, vvRenderState, GeometryType=VV_VIEWPORT, VoxelType=VV_GLSL);
    virtual ~vvTexMultiRend();

	void  init();

	void preRendering();
	void  renderMultipleVolume();
    void  renderVolumeGL();
    void  updateTransferFunction();
    void  updateVolumeData();
    void  updateVolumeData(int, int, int, int, int, int);
    void  setViewingDirection(const vvVector3*);
    void  setObjectDirection(const vvVector3*);
    virtual void setParameter(ParameterType param, const vvParam& value);
    virtual vvParam getParameter(ParameterType param) const;
    static bool isSupported(GeometryType);
    GeometryType getGeomType();
    VoxelType getVoxelType();
    void renderQualityDisplay();
    void printLUT();
	void updateChannelHistCDF(int, int, uchar*);	// from Chih's vvVolDesc
    void updateLUT(float);
  void enableLUTMode(vvShaderProgram* _glslShader);

	float getLUTDistance() { return lutDistance; }
	GLuint* getPixLUTName() { return pixLUTName; }
	GLuint* getTexNames() { return texNames; }
	float getVolWeight() { return volWeight; }
	float* getChanWeight() { return chanWeight; }
	float getChanWeight(int idx) { return chanWeight[idx]; }
	vvVector3* getColor() { return color; }
	vvVector3 getColor(int channel) { return color[channel]; }
	float getQuality() { return quality; }
	vvVector3* getTranslation() { return &translation; }
	void getTranslation(float* v1, float* v2, float* v3) { translation.get(v1, v2, v3); }
	vvMatrix* getRotation() { return &rotation; }
	void getRotation(float rot[16]) { rotation.getGL(rot); }
	float getTexMax(int idx) { assert(idx<3 && idx>=0); return (float)texMax[idx]; }
	float getTexMin(int idx) { assert(idx<3 && idx>=0); return (float)texMin[idx]; }
	float getTexels(int idx) { assert(idx<3 && idx>=0); return (float)texels[idx]; }
	int getNumSlices() { return numSlices; }
	int decreaseNumSlices() { return --numSlices; }
	int& getTFMode() { return tfmode; }
	float& getTFGamma(int chan) { return tfGamma[chan]; }
	float& getTFOffset(int chan) { return tfOffset[chan]; }
	float& getTFHPCutoff(int chan) { return tfHPCutoff[chan]; }
	float& getTFHPOrder(int chan) { return tfHPOrder[chan]; }

	void setTranslation(vvVector3 _translation) { translation = _translation; }
	void setTranslation(float _translation[3]) { translation.set(_translation[0], _translation[1], _translation[2]); }
	void addTranslation(float x, float y, float z) { translation.add(x, y, z); }
	void setRotation(vvMatrix _rotation) { rotation = _rotation; }
	void setRotation(float rot[16]) { rotation.setGL(rot); }
	void setNumSlices(int _numSlices) { numSlices = _numSlices; }
	void setTexMax(int idx, float val) { assert(idx<3 && idx>=0); texMax[idx]=val;} 
	void setTexMin(int idx, float val) { assert(idx<3 && idx>=0); texMin[idx]=val;} 
	void setTexels(int idx, int val) { assert(idx<3 && idx>=0); texels[idx]=val;} 
	void setChanWeight(int idx, float val) { chanWeight[idx] = val; }
	void setVolWeight(float val) { volWeight = val; }
	void setChannelColor(int channel, float val[3]) { color[channel].set(val[0], val[1], val[2]); }
	void setQuality(float val) { quality = val; }
	void setTFGamma(int channel, float gamma) { tfGamma[channel] = gamma; }
	void setTFOffset(int channel, float offset) { tfOffset[channel] = offset; }
	void setTFHPCutoff(int channel, float hpcutoff) { tfHPCutoff[channel] = hpcutoff; }
	void setTFHPOrder(int channel, float hporder) { tfHPOrder[channel] = hporder; }

};
#endif

//============================================================================
// End of File
//============================================================================

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
