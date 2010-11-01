//****************************************************************************
// Project Affiliation: Virvo (Virtual Reality Volume Renderer)
// Copyright:           (c) 2002 Juergen Schulze-Doebold. All rights reserved.
// Author's E-Mail:     schulze@hlrs.de
// Institution:         University of Stuttgart, Supercomputing Center (HLRS)
//****************************************************************************

#ifndef _VVSOFTVR_H_
#define _VVSOFTVR_H_

#include "vvexport.h"
#include "vvrenderer.h"
#include "vvsoftimg.h"
#include "vvimage.h"

/** A new rendering algorithm based on shear-warp factorization.
  This is the dispatcher class which chooses among the
  parallel (vvSoftPar) and perspective (vvSoftPer) projection variants,
  and it contains routines which are common to the above subclasses.

  The volume data are kept in memory three times, once for each principal
  axis. The permutations of coordinate axes are as follows:

  <PRE>
  Principal Axis    Coordinate System    Permutation Matrix
  ----------------------------------------------------------
z                / 0 1 0 0 \ 
|               |  0 0 1 0  |
X Axis              |___y           |  1 0 0 0  |
/                 \ 0 0 0 1 /
x
x                / 0 0 1 0 \ 
|               |  1 0 0 0  |
Y Axis              |___z           |  0 1 0 0  |
/                 \ 0 0 0 1 /
y
y                / 1 0 0 0 \ 
|               |  0 1 0 0  |
Z Axis              |___x           |  0 0 1 0  |
/                 \ 0 0 0 1 /
z

</PRE>

Compositing strategy: Bilinear resampling is performed within
the volume boundaries, so that there are always 4 voxel values
used for the resampling.

@author Juergen Schulze-Doebold (schulze@hlrs.de)
@see vvRenderer
@see vvSoftPar
@see vvSoftPer
*/
class VIRVOEXPORT vvSoftVR : public vvRenderer
{
   protected:
      enum AxisType                               /// type of coordinate axis
      {
         X_AXIS=0, Y_AXIS=1, Z_AXIS=2
      };
      enum WarpType                               /// possible warp techniques
      {
         SOFTWARE,                                ///< perform warp in software
         TEXTURE                                  ///< use 2D texturing hardware for warp
      };
      enum
      {
         PRE_INT_TABLE_SIZE = 256
      };
      vvSoftImg* outImg;                          ///< output image
      uchar* raw[3];                              ///< scalar voxel field for principle viewing axes (x, y, z)
      vvMatrix owView;                           ///< viewing transformation matrix from object space to world space
      vvMatrix osPerm;                           ///< permutation matrix
      vvMatrix wvConv;                           ///< conversion from world space to OpenGL viewport space
      vvMatrix oiShear;                          ///< shear matrix from object space to intermediate image space
      vvMatrix ivWarp;                           ///< warp object from intermediate image space to OpenGL viewport space
      vvMatrix iwWarp;                           ///< warp object from intermediate image space to world space
      int vWidth;                                 ///< OpenGL viewport width [pixels]
      int vHeight;                                ///< OpenGL viewport height [pixels]
      int len[3];                                 ///< volume dimensions in standard object space (x,y,z)
      AxisType principal;                         ///< principal viewing axis
      bool stacking;                              ///< slice stacking order; true=front to back
      WarpType warpMode;                          ///< current warp mode
      uchar rgbaConv[4096][4];                    ///< density to RGBA conversion table (max. 8 bit density supported) [scalar values][RGBA]
      vvVector3 xClipNormal;                      ///< clipping plane normal in permuted voxel coordinate system
      float xClipDist;                            ///< clipping plane distance in permuted voxel coordinate system
      uchar** rleStart[3];                        ///< pointer lists to line beginnings, for each principal viewing axis (x,y,z). If first entry is NULL, there is no RLE compressed volume data
      uchar* rle[3];                              ///< RLE encoded volume data for each principal viewing axis (x,y,z)
      int numProc;                                ///< number of processors in system
      bool compression;                           ///< true = use compressed volume data for rendering
      bool multiprocessing;                       ///< true = use multiprocessing where possible
      bool preIntegration;                        ///< true = use pre-integrated values for compositing
      bool sliceInterpol;                         ///< inter-slice interpolation mode: true=bilinear interpolation (default), false=nearest neighbor
      bool warpInterpol;                          ///< warp interpolation: true=bilinear, false=nearest neighbor
      bool sliceBuffer;                           ///< slice buffer: true=intermediate image aligned, false=slice aligned
      bool bilinLookup;                           ///< true=bilinear lookup in pre-integration table, false=nearest neighbor lookup
      bool opCorr;                                ///< true=opacity correction on
                                                  ///< size of pre-integrated LUT ([sf][sb][RGBA])
      uchar preIntTable[PRE_INT_TABLE_SIZE][PRE_INT_TABLE_SIZE][4];
      int earlyRayTermination;                    ///< counter for number of voxels which are skipped due to early ray termination
      bool _timing;
      vvVector3 _size;

      void setOutputImageSize();
      void findVolumeDimensions();
      void findAxisRepresentations();
      void encodeRLE();
      int  getLUTSize();
      void findViewMatrix();
      void findPermutationMatrix();
      void findViewportMatrix(int, int);
      void findSlicePosition(int, vvVector3*, vvVector3*);
      void findClipPlaneEquation();
      bool isVoxelClipped(int, int, int);
      void compositeOutline();
      virtual int  getCullingStatus(float);
      virtual void factorViewMatrix() = 0;

   public:
      vvSoftImg* intImg;                          ///< intermediate image

      vvSoftVR(vvVolDesc*, vvRenderState);
      virtual ~vvSoftVR();
      void     updateTransferFunction();
      void     makeLookupTextureCorrect(float = 1.0f);
      void     makeLookupTextureOptimized(float = 1.0f);
      void     updateVolumeData();
      bool     instantClassification();
      void     setWarpMode(WarpType);
      WarpType getWarpMode();
      void     setCurrentFrame(int);
      void     renderVolumeGL();
      void     getIntermediateImage(vvImage*);
      void     getWarpMatrix(vvMatrix*);
      bool     prepareRendering();
      void     setParameter(float, ParameterType, char* = NULL);
      float    getParameter(ParameterType, char* = NULL);
      virtual void compositeVolume(int = -1, int = -1) = 0;
      virtual void getIntermediateImageExtent(int*, int*, int*, int*);
};
#endif

//============================================================================
// End of File
//============================================================================
