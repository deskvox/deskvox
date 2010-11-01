//****************************************************************************
// Project Affiliation: Virvo (Virtual Reality Volume Renderer)
// Copyright:           (c) 2002 Juergen Schulze-Doebold. All rights reserved.
// Author's E-Mail:     schulze@hlrs.de
// Institution:         University of Stuttgart, Supercomputing Center (HLRS)
//****************************************************************************

#ifndef _VVSOFTIMG_H_
#define _VVSOFTIMG_H_

#include "vvexport.h"
#include "vvopengl.h"
#include "vvrenderer.h"
#include "vvsoftimg.h"

/** Description of pixel image.
  This class was written for the software implementation of the
  shear-warp algorithm, but it can also be used for different purposes.
  Pixel (0|0) is bottom left, pixel (max_x|0) is bottom right,
  pixel (0|max_Y) is top left, and pixel (max_x|max_y) is top right.

  @author Juergen Schulze-Doebold (schulze@hlrs.de)
  @see vvSoftPar
  @see vvSoftPer
  @see vvSoftVR
*/
class VIRVOEXPORT vvSoftImg
{
   private:
      enum AlignType                              /// alignment of images
      {
         BOTTOM_LEFT
      };
#ifndef VV_REMOTE_RENDERING
      GLuint    texName;                          ///< name of warp texture
      GLboolean glsTexture2D;                     ///< state buffer for GL_TEXTURE_2D
      GLboolean glsBlend;                         ///< stores GL_BLEND
      GLboolean glsLighting;                      ///< stores GL_LIGHTING
      GLboolean glsCulling;                       ///< stores GL_CULL_FACE
      GLint     glsBlendSrc;                      ///< stores glBlendFunc(source,...)
      GLint     glsBlendDst;                      ///< stores glBlendFunc(...,destination)
      GLint     glsUnpackAlignment;               ///< stores glPixelStore(GL_UNPACK_ALIGNMENT,...)
      GLfloat   glsRasterPos[4];                  ///< current raster position (glRasterPos)
#endif
      bool      warpInterpolation;                ///< true = linear interpolation, false = nearest neighbor interpolation

   public:
      static const int PIXEL_SIZE;                ///< byte per pixel (3 for RGB, 4 for RGBA)
      int     width;                              ///< width in pixels
      int     height;                             ///< height in pixels
      uchar*  data;                               ///< Pointer to scalar image data. Data arrangement: origin is bottom left,
      ///< pixels to the right are first, then up. All RGB
      ///< components are saved in a row for each pixel.
      bool    deleteData;                         ///< true if image data pointer is to be deleted when not used anymore

      vvSoftImg(int=0, int=0);
      virtual ~vvSoftImg();
      void setSize(int, int);
      void zoom(vvSoftImg*);
      void copy(AlignType, vvSoftImg*);
      void overlay(vvSoftImg*);
      void draw();
      void clear();
      void fill(int, int, int, int);
      void drawBorder(int, int, int);
      void warp(vvMatrix*, vvSoftImg*);
      void warpTex(vvMatrix*);
      void putPixel(int, int, uint);
      void drawLine(int, int, int, int, int, int, int);
      void saveGLState();
      void restoreGLState();
      void setWarpInterpolation(bool);
      void setImageData(int, int, uchar*);
      void print(const char*);
};
#endif

//============================================================================
// End of File
//============================================================================
