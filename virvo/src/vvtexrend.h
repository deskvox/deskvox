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

#ifndef _VVTEXREND_H_
#define _VVTEXREND_H_

#ifndef NO_CONFIG_H
#ifndef _WIN32
#include "config.h"
#endif
#endif

#include "vvopengl.h"

// Cg:
#ifdef HAVE_CG
#include <Cg/cg.h>
#include <Cg/cgGL.h>
#endif

// Virvo:
#include "vvexport.h"
#include "vvvoldesc.h"
#include "vvrenderer.h"
#include "vvtransfunc.h"
#include "vvsllist.h"

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
  Make sure you define HAVE_CG in your compiler if you want to use Nvidia Cg.
  @author Juergen Schulze (schulze@cs.brown.de)
  @author Martin Aumueller
  @see vvRenderer
*/
class VIRVOEXPORT vvTexRend : public vvRenderer
{
  public:

    struct Brick
    {
      vvVector3 pos;                              ///< center position of brick
      vvVector3 min;                              ///< minimum position of brick
      vvVector3 max;                              ///< maximum position of brick
      int index;                                  ///< index for texture object
      int startOffset[3];                         ///< startvoxel of brick
      int texels[3];                              ///< number of texels in each dimension
      float dist;                                 ///< distance from plane given by eye and normal
    };

    enum ErrorType                                /// Error Codes
    {
      OK,                                         ///< no error
      TRAM_ERROR,                                 ///< not enough texture memory
      NO3DTEX                                     ///< 3D textures not supported on this hardware
    };
    enum GeometryType                             /// Geometry textures are projected on
    {
      VV_AUTO = 0,                                ///< automatically choose best
      VV_SLICES,                                  ///< render slices parallel to xy axis plane using 2D textures
      VV_CUBIC2D,                                 ///< render slices parallel to all axis planes using 2D textures
      VV_VIEWPORT,                                ///< render planar slices using a 3D texture
      VV_BRICKS,                                  ///< render volume using bricking
      VV_SPHERICAL                                ///< render spheres originating at viewer using a 3D texture
    };
    enum VoxelType                                /// Internal data type used in textures
    {
      VV_BEST = 0,                                ///< choose best
      VV_RGBA,                                    ///< transfer function look-up done in software
      VV_SGI_LUT,                                 ///< SGI color look-up table
      VV_PAL_TEX,                                 ///< OpenGL paletted textures
      VV_TEX_SHD,                                 ///< Nvidia texture shader
      VV_PIX_SHD,                                 ///< Nvidia pixel shader
      VV_FRG_PRG                                  ///< ARB fragment program
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

    static const int NUM_PIXEL_SHADERS;           ///< number of pixel shaders used
  private:
    enum AxisType                                 /// names for coordinate axes
    { X_AXIS, Y_AXIS, Z_AXIS };
    enum FragmentProgram
    {
      VV_FRAG_PROG_2D = 0,
      VV_FRAG_PROG_3D,
      VV_FRAG_PROG_PREINT,
      VV_FRAG_PROG_MAX                            // has always to be last in list
    };
    float* rgbaTF;                                ///< density to RGBA conversion table, as created by TF [0..1]
    uchar* rgbaLUT;                               ///< final RGBA conversion table, as transferred to graphics hardware (includes opacity and gamma correction)
    uchar* preintTable;                           ///< lookup table for pre-integrated rendering, as transferred to graphics hardware
    float  lutDistance;                           ///< slice distance for which LUT was computed
    int   texels[3];                              ///< width, height and depth of volume, including empty space [texels]
    float texMin[3];                              ///< minimum texture value of object [0..1] (to prevent border interpolation)
    float texMax[3];                              ///< maximum texture value of object [0..1] (to prevent border interpolation)
    int   textures;                               ///< number of textures stored in TRAM
    int   texelsize;                              ///< number of bytes/voxel transferred to OpenGL (depending on rendering mode)
    GLint internalTexFormat;                      ///< internal texture format (parameter for glTexImage...)
    GLenum texFormat;                             ///< texture format (parameter for glTexImage...)
    GLuint* texNames;                             ///< names of texture slices stored in TRAM
    GLuint pixLUTName;                            ///< name for transfer function texture
    GLuint fragProgName[VV_FRAG_PROG_MAX];        ///< names for fragment programs (for applying transfer function)
    GeometryType geomType;                        ///< rendering geometry actually used
    VoxelType voxelType;                          ///< voxel type actually used
    bool extTex3d;                                ///< true = 3D texturing supported
    bool extNonPower2;                            ///< true = NonPowerOf2 textures supported
    bool extColLUT;                               ///< true = SGI texture color lookup table supported
    bool extPalTex;                               ///< true = OpenGL 1.2 paletted textures supported
    bool extMinMax;                               ///< true = maximum/minimum intensity projections supported
    bool extTexShd;                               ///< true = Nvidia texture shader & texture shader 2 support
    bool extPixShd;                               ///< true = Nvidia pixel shader support (requires GeForce FX)
    bool extBlendEquation;                        ///< true = support for blend equation extension
    bool arbFrgPrg;                               ///< true = ARB fragment program support
    bool arbMltTex;                               ///< true = ARB multitexture support
    vvVector3 viewDir;                            ///< user's current viewing direction [object coordinates]
    vvVector3 objDir;                             ///< direction from viewer to object [object coordinates]
    bool preIntegration;                          ///< true = try to use pre-integrated rendering (planar 3d textures)
    bool usePreIntegration;                       ///< true = pre-integrated rendering is actually used
    bool interpolation;                           ///< interpolation mode: true=linear interpolation (default), false=nearest neighbor
    bool opacityCorrection;                       ///< true = opacity correction on
    int  minSlice, maxSlice;                      ///< min/maximum slice to render [0..numSlices-1], -1 for no slice constraints
    bool _areBricksCreated;                       ///< true after the first creation of the bricks
    vvSLList<vvSLList<Brick*>*> _brickList;       ///< contains all created bricks
    vvSLList<Brick*> _insideList;                 ///< contains all bricks inside the probe
    vvSLList<Brick*> _sortedList;                 ///< contains all bricks inside the probe in a sorted order (back to front)
    bool _useOnlyOneBrick;                        ///< true if whole data fits in texture memory
    vvVector4 _frustum[6];                        ///< current planes of view frustum
    SliceOrientation _sliceOrientation;           ///< slice orientation for planar 3d textures
    
#ifdef HAVE_CG
    CGcontext _cgContext;                         ///< context for running fragment program
    CGprogram* _cgProgram;                        ///< handles for fragment program
    CGprofile* _cgFragProfile;                    ///< profiles to determine which version of cg to compile with
    CGparameter _cgPixLUT;                        ///< fragment program input: RGBA look-up table
    CGparameter _cgChannel4Color;                 ///< fragment program input: color of 4th channel
    CGparameter _cgOpacityWeights;                ///< fragment program input: opacity of color channels
#endif
    int _currentShader;                           ///< ID of currently used fragment shader

#if defined(_WIN32)
    PFNGLTEXIMAGE3DEXTPROC glTexImage3DEXT;
    PFNGLTEXSUBIMAGE3DEXTPROC glTexSubImage3DEXT;
    PFNGLCOLORTABLESGIPROC glColorTableSGI;
    PFNGLCOLORTABLEEXTPROC glColorTableEXT;
    PFNGLBLENDEQUATIONEXTPROC glBlendEquationVV;
    PFNGLACTIVETEXTUREARBPROC glActiveTextureARB;
    PFNGLMULTITEXCOORD3FARBPROC glMultiTexCoord3fARB;
    PFNGLBINDPROGRAMARBPROC glBindProgramARB;
    PFNGLGENPROGRAMSARBPROC glGenProgramsARB;
    PFNGLDELETEPROGRAMSARBPROC glDeleteProgramsARB;
    PFNGLPROGRAMSTRINGARBPROC glProgramStringARB;
#else
    typedef void (glBlendEquationEXT_type)(GLenum);
    typedef void (glTexImage3DEXT_type)(GLenum, GLint, GLenum, GLsizei, GLsizei, GLsizei, GLint, GLenum, GLenum, const GLvoid*);
    typedef void (glTexSubImage3DEXT_type)(GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei, GLenum, GLenum, const GLvoid*);
    typedef void (glColorTableSGI_type)(GLenum, GLenum, GLsizei, GLenum, GLenum, const GLvoid*);
    typedef void (glColorTableEXT_type)(GLenum, GLenum, GLsizei, GLenum, GLenum, const GLvoid*);
    typedef void (glActiveTextureARB_type)(GLenum);
    typedef void (glMultiTexCoord3fARB_type)(GLenum, GLfloat, GLfloat, GLfloat);
    typedef void (glGenProgramsARB_type)(GLsizei, GLuint*);
    typedef void (glDeleteProgramsARB_type)(GLsizei, GLuint*);
    typedef void (glBindProgramARB_type)(GLenum, GLuint);
    typedef void (glProgramStringARB_type)(GLenum, GLenum, GLsizei, const GLvoid *);
    glBlendEquationEXT_type* glBlendEquationVV;
    glTexImage3DEXT_type* glTexImage3DEXT;
    glTexSubImage3DEXT_type* glTexSubImage3DEXT;
    glColorTableSGI_type* glColorTableSGI;
    glColorTableEXT_type* glColorTableEXT;
    glActiveTextureARB_type* glActiveTextureARB;
    glMultiTexCoord3fARB_type* glMultiTexCoord3fARB;
    glGenProgramsARB_type* glGenProgramsARB;
    glDeleteProgramsARB_type* glDeleteProgramsARB;
    glBindProgramARB_type* glBindProgramARB;
    glProgramStringARB_type* glProgramStringARB;
#endif

    // GL state variables:
    GLboolean glsCulling;                         ///< stores GL_CULL_FACE
    GLboolean glsBlend;                           ///< stores GL_BLEND
    GLboolean glsColorMaterial;                   ///< stores GL_COLOR_MATERIAL
    GLint glsBlendSrc;                            ///< stores glBlendFunc(source,...)
    GLint glsBlendDst;                            ///< stores glBlendFunc(...,destination)
    GLboolean glsLighting;                        ///< stores GL_LIGHTING
    GLboolean glsDepthTest;                       ///< stores GL_DEPTH_TEST
    GLint glsMatrixMode;                          ///< stores GL_MATRIX_MODE
    GLint glsDepthFunc;                           ///< stores glDepthFunc
    GLint glsBlendEquation;                       ///< stores GL_BLEND_EQUATION_EXT
    GLboolean glsDepthMask;                       ///< stores glDepthMask
    GLboolean glsTexColTable;                     ///< stores GL_TEXTURE_COLOR_TABLE_SGI
    GLboolean glsSharedTexPal;                    ///< stores GL_SHARED_TEXTURE_PALETTE_EXT

    void removeTextures();
    ErrorType makeTextures();
    void makeLUTTexture();
    ErrorType makeTextures2D(int axes);
    ErrorType makeTextureBricks();
    ErrorType makeTextures3D();
    ErrorType updateTextures3D(int, int, int, int, int, int, bool);
    ErrorType updateTextures2D(int, int, int, int, int, int, int);
    ErrorType updateTextureBricks(int, int, int, int, int, int);
    void enableLUTMode();
    void disableLUTMode();
    void setGLenvironment();
    void unsetGLenvironment();
    void renderTex3DSpherical(vvMatrix*);
    void renderTex3DPlanar(vvMatrix*);
    void renderTexBricks(vvMatrix*);
    void renderBricks(vvMatrix*);
    void renderTex2DSlices(float);
    void renderTex2DCubic(AxisType, float, float, float);
    VoxelType findBestVoxelType(VoxelType);
    GeometryType findBestGeometry(GeometryType, VoxelType);
    void updateLUT(float);
    int  getLUTSize(int*);
    int  getPreintTableSize();
    void enableNVShaders();
    void disableNVShaders();
    void enablePixelShaders();
    void disablePixelShaders();
    void enableFragProg();
    void disableFragProg();
    bool initPixelShaders();
    void enableTexture(GLenum target);
    void disableTexture(GLenum target);
    bool testBrickVisibility(Brick* brick, const vvMatrix& mvpMat);
    bool testBrickVisibility(Brick*);
    void updateFrustum();
    void getBricksInProbe(vvVector3, vvVector3);
    void sortBrickList(vvVector3, vvVector3, bool);
    void computeBrickSize();

  public:
    vvTexRend(vvVolDesc*, vvRenderState, GeometryType=VV_AUTO, VoxelType=VV_BEST);
    virtual ~vvTexRend();
    void  renderVolumeGL();
    void  updateTransferFunction();
    void  updateVolumeData();
    void updateVolumeData(int, int, int, int, int, int);
    void  activateClippingPlane();
    void  deactivateClippingPlane();
    void  setNumLights(int);
    bool  instantClassification();
    void  setViewingDirection(const vvVector3*);
    void  setObjectDirection(const vvVector3*);
    void  setParameter(ParameterType, float, char* = NULL);
    float getParameter(ParameterType, char* = NULL);
    static bool isSupported(GeometryType);
    static bool isSupported(VoxelType);
    bool isSupported(FeatureType);
    GeometryType getGeomType();
    VoxelType getVoxelType();
    int  getCurrentShader();
    void setCurrentShader(int);
    void renderQualityDisplay();
    void printLUT();
    void setFloatZoom(float, float);
    void updateBrickGeom();
    void setShowBricks(bool);
    bool getShowBricks();
    void setComputeBrickSize(bool);
    bool getComputeBrickSize();
    void setBrickSize(int);
    int getBrickSize();
    void setTexMemorySize(int);
    int getTexMemorySize();
    unsigned char* getHeightFieldData(float[4][3], int&, int&);
    float getManhattenDist(float[3], float[3]);
};
#endif

//============================================================================
// End of File
//============================================================================
