// Virvo - Virtual Reality Volume Rendering
// Contact: Stefan Zellmann, zellmans@uni-koeln.de
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

#ifndef _VV_OFFSCREENBUFFER_H_
#define _VV_OFFSCREENBUFFER_H_

#include "vvexport.h"

namespace virvo
{
enum BufferPrecision
{
  Byte = 0,
  Short,
  Float
};
}

class VIRVOEXPORT vvOffscreenBuffer
{
public:
  vvOffscreenBuffer(float scale = 1.0f, virvo::BufferPrecision precision = virvo::Byte);
  vvOffscreenBuffer(int w, int h, float scale = 1.0f, virvo::BufferPrecision precision = virvo::Byte);
  virtual ~vvOffscreenBuffer();

  virtual void initForRender();
  /*!
   * \brief         Write data back to hardware framebuffer.
   *
   *                If not provided with width and height,
   *                this method will unbind the current buffer,
   *                get the viewport information from the
   *                hardware buffer and rebind the buffer again.
   *                In cases one knows width and height of the
   *                viewport, it is preferable to pass these
   *                to the function to avoid this behavior.
   * \param         w Width of the hardware buffer viewport.
   * \param         h Height of the hardware buffer viewport.
   */
  void writeBack(int w = -1, int h = -1);
  void resize(int w, int h);
  void clearBuffer();

  void bindFramebuffer() const;
  void unbindFramebuffer() const;
  void bindTexture() const;

  void setScale(float scale);
  void setPreserveDepthBuffer(bool preserveDepthBuffer);
  void setUseNVDepthStencil(bool useNVDepthStencil);
  void setPrecision(virvo::BufferPrecision precision);
  void setInterpolation(bool interpolation);

  int getBufferWidth() const;
  int getBufferHeight() const;
  float getScale() const;
  bool getPreserveFramebuffer() const;
  bool getUseNVDepthStencil() const;
  virvo::BufferPrecision getPrecision() const;
  bool getInterpolation() const;
private:
  struct GLData;
  GLData* _gldata;

  int _viewportWidth;
  int _viewportHeight;

  int _bufferWidth;
  int _bufferHeight;

  float _scale;
  bool _preserveDepthBuffer;
  bool _useNVDepthStencil;

  virvo::BufferPrecision _precision;

  bool _interpolation;

  unsigned char* _pixels;

  vvOffscreenBuffer* _scaledDepthBuffer;

  bool _updatePosted;

  void init(int w, int h, float scale, virvo::BufferPrecision precision);
  int getScaled(int v) const;
  void update();
  void storeColorBuffer();
  void storeDepthBuffer(int scaledWidth, int scaledHeight);
  void renderToViewAlignedQuad() const;
  void writeBackDepthBuffer() const;
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
