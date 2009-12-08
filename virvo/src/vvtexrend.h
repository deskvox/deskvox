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
#include "vvbsptree.h"

// Posix threads:
#include <pthread.h>

#ifdef __APPLE__
/* these are not available on OS X:
 * they are only used together with X11, so this should not matter too much */
typedef struct pthread_barrier_t_ { int dummy; } pthread_barrier_t;
typedef struct pthead_barrierattr_t_ { int dummy; } pthread_barrierattr_t;
static int pthread_barrier_wait(pthread_barrier_t *barrier) { return 1; }
static int pthread_barrier_init(pthread_barrier_t *barrier,
                    const pthread_barrierattr_t *attr, unsigned count) { return 1; }
static int pthread_barrier_destroy(pthread_barrier_t *barrier) { return 1; }
#endif

struct ThreadArgs;

class vvTexRend;

//============================================================================
// Class Definitions
//============================================================================

class Brick : public vvConvexObj
{
public:
  Brick()                                     ///< dflt. constructor (needed for C++ templates)
  {

  }

  Brick(Brick* rhs)                           ///< copy constructor (from ptr)
  {
    pos = vvVector3(&rhs->pos);
    min = vvVector3(&rhs->min);
    max = vvVector3(&rhs->max);
    minObj = vvVector3(&rhs->minObj);
    maxObj = vvVector3(&rhs->maxObj);
    minValue = rhs->minValue;
    maxValue = rhs->maxValue;
    visible = rhs->visible;
    index = rhs->index;
    startOffset[0] = rhs->startOffset[0];
    startOffset[1] = rhs->startOffset[1];
    startOffset[2] = rhs->startOffset[2];
    texels[0] = rhs->texels[0];
    texels[1] = rhs->texels[1];
    texels[2] = rhs->texels[2];
    dist = rhs->dist;
  }
#ifdef HAVE_CG
  void render(vvTexRend* renderer, const int numSlices, vvVector3& normal,
              const vvVector3& farthest, const vvVector3& delta,
              const vvVector3& probeMin, const vvVector3& probeMax,
              GLuint*& texNames, GLuint**& vertIndices, GLsizei*& elemCounts,
              CGparameter* cgVertices, CGparameter cgBrickMin,
              CGparameter cgBrickDimInv, CGparameter cgFrontIndex,
              CGparameter cgPlaneStart);
#else
  void render(vvTexRend* renderer, const int numSlices, vvVector3& normal,
              const vvVector3& farthest, const vvVector3& delta,
              const vvVector3& probeMin, const vvVector3& probeMax,
              GLuint*& texNames, GLuint**& vertIndices, GLsizei*& elemCounts);
#endif

  virtual vvAABB getAABB()
  {
    return vvAABB(minObj, maxObj);
  }

  ushort getFrontIndex(vvVector3* vertices,         ///< front index of the brick dependent upon the current model view
                       vvVector3& point,
                       vvVector3& normal,
                       float& minDot,
                       float& maxDot);
                                                    ///< and assuming that vertices are ordered back to front
  vvVector3 pos;                                    ///< center position of brick
  vvVector3 min;                                    ///< minimum position of brick
  vvVector3 max;                                    ///< maximum position of brick
  vvVector3 minObj;                                 ///< minimum position of brick in obj coords of the volume
  vvVector3 maxObj;                                 ///< maximum position of brick in obj coords of the volume
  int minValue;                                     ///< min scalar value after lut, needed for empty space leaping
  int maxValue;                                     ///< max scalar value after lut, needed for empty space leaping
  bool visible;                                     ///< if brick isn't visible, it won't be rendered at all
  int index;                                        ///< index for texture object
  int startOffset[3];                               ///< startvoxel of brick
  int texels[3];                                    ///< number of texels in each dimension
  float dist;                                       ///< distance from plane given by eye and normal
};

/** The content of each thread is rendered via the visitor pattern.
  The rendered results of each thread are managed using a bsp tree
  structure. Using the visitor pattern was a design decision so
  that the bsp tree doesn't need knowledge about the (rather specific)
  rendering code its half spaces utilize to display their results.
  This logic is supplied by the visitor which needs to be initialized
  once and passed to the bsp tree after initialization. Thus the bsp
  tree may be utilized in context not that specific as this one.
  @author stefan Zellmann
  @see vvVisitor
 */
class vvThreadVisitor : public vvVisitor
{
public:
  virtual ~vvThreadVisitor();
  virtual void visit(vvVisitable* obj);

  void setFrameBufferObjects(GLuint*& frameBufferObjects);
  void setDepthBuffers(GLuint*& depthBuffers);
  void setImageSpaceTextures(GLuint*& imageSpaceTextures);
  void setPixels(GLfloat**& pixels);
private:
  GLuint* _frameBufferObjects;
  GLuint* _depthBuffers;
  GLuint* _imageSpaceTextures;
  GLfloat** _pixels;
};

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
  @author Stefan Zellmann
  @see vvRenderer
*/
class VIRVOEXPORT vvTexRend : public vvRenderer
{
  friend class Brick;
  public:
    struct BrickSet
    {
      vvSLList<Brick*> bricks;
      int parentThreadId;

      vvVector3 center;
      float dist;

      inline bool operator<(const BrickSet& rhs) const      ///< compare bricks based upon dist to eye position
      {
        if (dist < rhs.dist)
        {
          return true;
        }
        else
        {
          return false;
        }
      }
    };
    unsigned int _numThreads;                     ///< thread count

    const char** _displayNames;                   ///< xorg display names, e.g. const char("host:0.0")
    unsigned int* _screens;                       ///< display name = :0.x ==> corresponding screen: x
    
    pthread_t* _threads;                          ///< worker threads
    ThreadArgs* _threadData;                      ///< args for each thread
    pthread_barrier_t _initBarrier;               ///< barrier assures that the render loop isn't entered before (pre)initialization
    pthread_barrier_t _distributeBricksBarrier;   ///< barrier assures that bricks are distributed before rendering and after (pre)initialization
    pthread_barrier_t _distributedBricksBarrier;  ///< barrier is passed when bricks are distributed eventually
    pthread_barrier_t _renderStartBarrier;        ///< barrier assures that the render loop doesn't resume until proper data is supplied
    pthread_barrier_t _compositingBarrier;        ///< barrier assures synchronization for compositing
    pthread_barrier_t _renderReadyBarrier;        ///< barrier assures that bricks aren't sorted until the previous frame was rendered
    pthread_mutex_t _makeTextureMutex;            ///< mutex ensuring that textures for each thread are build up synchronized

    GLuint* _frameBufferObjects;
    GLuint* _depthBuffers;
    GLuint* _imageSpaceTextures;                  ///< each thread will overlay the current output with a 2D texture of its px buffer
    bool _somethingChanged;                       ///< when smth changed (e.g. the transfer function, bricks will possibly be rearranged)
    vvBspTree* _bspTree;
    int _deviationExceedCnt;

    int _numBricks[3];                            ///< number of bricks for each dimension
    enum ErrorType                                /// Error Codes
    {
      OK = 0,                                     ///< no error
      TRAM_ERROR,                                 ///< not enough texture memory
      NO3DTEX,                                    ///< 3D textures not supported on this hardware
      NO_DISPLAYS_SPECIFIED,                      ///< no x-displays in _renderState, thus no multi-threading
      UNSUPPORTED                                 ///< general error code
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
    vvSLList<vvConvexObj*> _insideList;                 ///< contains all bricks inside the probe
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

    CGcontext _cgIsectContext;                    ///< context for gpu intersection program
    CGprofile _cgIsectProfile;                    ///< cg profile for gpu intersection program
    CGprogram _cgIsectProgram;                    ///< cg shader program for gpu intersection

    CGparameter _cgBrickMin;
    CGparameter _cgBrickDimInv;

    CGparameter _cgModelViewProj;
    CGparameter _cgPlaneStart;
    CGparameter _cgDelta;
    CGparameter _cgPlaneNormal;
    CGparameter _cgFrontIndex;

    CGparameter _cgVertexList;
    CGparameter* _cgVertices;

    GLuint _vbos[2];                              ///< vertex buffer objects for efficient slice transfer to gpu
#endif
    int _currentShader;                           ///< ID of currently used fragment shader

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

    void makeLUTTexture();
    ErrorType makeTextures2D(int axes);

    ErrorType setDisplayNames(const char** displayNames, const unsigned int numNames);
    ErrorType dispatchThreadedGLXContexts();
    ErrorType distributeBricks();
    static void* threadFuncTexBricks(void* threadargs);
    static void* threadFuncBricks(void* threadargs);
    void sortBrickList(const int, vvVector3, vvVector3, bool);
    void performLoadBalancing();

    ErrorType makeTextures(GLuint*& privateTexNames, int* numTextures);
    ErrorType makeTextureBricks(GLuint*& privateTexNames, int* numTextures);
#ifdef HAVE_CG
    bool initPixelShaders(CGcontext& cgContext, CGprogram*& cgProgram);
    void enableLUTMode(CGprogram*& cgProgram, CGparameter& cgPixLUT);
    void disableLUTMode(CGparameter& cgPixLUT);
    void enableIntersectionShader(CGprogram& cgIsectProgram, CGprofile& cgIsectProfile);
    void enablePixelShaders(CGprogram*& cgProgram, CGparameter& cgPixLUT);
    void disablePixelShaders(CGparameter& cgPixLUT);
    bool initIntersectionShader(CGcontext& cgIsectContext,
                                CGprofile& cgIsectProfile,
                                CGprogram& cgIsectProgram,
                                CGparameter& cgBrickMin, CGparameter& cgBrickDimInv,
                                CGparameter& cgModelViewProj, CGparameter& cgPlaneStart,
                                CGparameter& cgDelta, CGparameter& cgPlaneNormal,
                                CGparameter& cgFrontIndex, CGparameter& cgVertexList,
                                CGparameter*& cgVertices);
    void setupIntersectionCGParameters(CGprogram& cgIsectProgram,
                                       CGparameter& cgBrickMin, CGparameter& cgBrickDimInv,
                                       CGparameter& cgModelViewProj, CGparameter& cgPlaneStart,
                                       CGparameter& cgDelta, CGparameter& cgPlaneNormal,
                                       CGparameter& cgFrontIndex, CGparameter& cgVertexList,
                                       CGparameter*& cgVertices);
#else
    bool initPixelShaders();
    void enableLUTMode();
    void disableLUTMode();
    void enableIntersectionShader();
    void enablePixelShaders();
    void disablePixelShaders();
    void disableIntersectionShaders();
    bool initIntersectionShader();
#endif
    void disableIntersectionShader();

    ErrorType makeTextures3D();
    void removeTextures();
    ErrorType updateTextures3D(int, int, int, int, int, int, bool);
    ErrorType updateTextures2D(int, int, int, int, int, int, int);
    ErrorType updateTextureBricks(int, int, int, int, int, int);
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
    void enableFragProg();
    void disableFragProg();
    void enableTexture(GLenum target);
    void disableTexture(GLenum target);
    bool testBrickVisibility(Brick* brick, const vvMatrix& mvpMat);
    bool testBrickVisibility(Brick*);
    void updateFrustum();
    void calcProbeDims(vvVector3&, vvVector3&, vvVector3&, vvVector3&);
    void getBricksInProbe(vvVector3, vvVector3);
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
