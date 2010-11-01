//****************************************************************************
// Project Affiliation: Virvo (Virtual Reality Volume Renderer)
// Copyright:           (c) 2002 Juergen Schulze-Doebold. All rights reserved.
// Author's E-Mail:     schulze@hlrs.de
// Institution:         University of Stuttgart, Supercomputing Center (HLRS)
//****************************************************************************

#ifndef _VVSOFTPAR_H_
#define _VVSOFTPAR_H_

#include "vvrenderer.h"
#include "vvsoftimg.h"
#include "vvsoftvr.h"
#include "vvexport.h"

/** Parallel projection shear-warp algorithm.
  The algorithm was implemented according to the description in
  P. Lacroute's Ph.D. thesis (Stanford University).<BR>
  Coordinate systems (used in first two letters of matrix names (from-to) and
  first letter of vector names):
  <UL>
    <LI>o = object space</LI>
    <LI>s = standard object space</LI>
    <LI>i = intermediate image coordinates</LI>
    <LI>w = world coordinates</LI>
    <LI>v = OpenGL viewport space</LI>
</UL>

Terminology:<PRE>
glPM     = OpenGL projection matrix (perspective transformation matrix)
glMV     = OpenGL modelview matrix
owView   = combined view matrix from object space to world space (Lacroute: M_view)
wvConv   = conversion from world to OpenGL viewport
osPerm   = permutation from object space to standard object space (Lacroute: P)
siShear  = shear matrix from standard object space to intermediate image (Lacroute: M_shear)
oiShear  = shear matrix from object space to intermediate image space
iwWarp   = 2D warp from intermediate image to world (Lacroute: M_warp)
ivWarp   = 2D warp from intermediate image to OpenGL viewport
oViewDir = user's viewing direction [object space]
sViewDir = user's viewing direction [standard object space]
x        = matrix multiplication, order for multiple multiplications: from right to left
-1       = inverse of a matrix (e.g. owView-1 = woView)
</PRE>

Important equations:<PRE>
owView  = glPM x glMV
isShear = isShear(sViewDir, imgConv, objectSize)
osPerm  = depending on principal viewing axis
wvConv  = depending on OpenGL viewport size
oiShear = siShear x osPerm
iwWarp  = owView x soPerm x isShear
ivWarp  = wvConv x owView x soPerm x isShear
ovView  = wvConv x glMV x glPM
ovView  = wvConv x iwWarp x siShear x osPerm
</PRE>

@author Juergen Schulze-Doebold (schulze@hlrs.de)
@see vvRenderer
@see vvSoftVR
@see vvSoftImg
*/
class VIRVOEXPORT vvSoftPar : public vvSoftVR
{
   private:
      vvVector3 wViewDir;                         ///< viewing direction [world space]
      vvVector3 oViewDir;                         ///< viewing direction [object space]
      vvVector3 sViewDir;                         ///< viewing direction [standard object space]
      float* bufSlice[2];                         ///< buffer slices for preintegrated rendering
      int bufSliceLen[2];                         ///< size of buffer slices
      int readSlice;                              ///< index of buffer slice currently used for reading [0..1]
      enum
      {
         VV_OP_CORR_TABLE_SIZE = 1024
      };
      float opacityCorr[VV_OP_CORR_TABLE_SIZE];
      float colorCorr[VV_OP_CORR_TABLE_SIZE];

      void compositeSliceNearest(int, int = -1, int = -1);
      void compositeSliceBilinear(int);
      void compositeSliceCompressedNearest(int);
      void compositeSliceCompressedBilinear(int);
      void compositeSlicePreIntegrated(int, int);
      void findOViewingDirection();
      void findPrincipalAxis();
      void findSViewingDirection();
      void findShearMatrix();
      void findWarpMatrix();
      void factorViewMatrix();

   public:
      vvSoftPar(vvVolDesc*, vvRenderState);
      virtual ~vvSoftPar();
      void compositeVolume(int = -1, int = -1);
};
#endif

//============================================================================
// End of File
//============================================================================
