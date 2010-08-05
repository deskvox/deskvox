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

#include "vvrendertarget.h"

class VIRVOEXPORT vvOffscreenBuffer : public vvRenderTarget
{
public:
  vvOffscreenBuffer(const float scale, const BufferPrecision precision);
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
  virtual void writeBack(const int w = -1, const int h = -1);
  virtual void resize(const int w, const int h);
  virtual void clearBuffer();

  void bindFramebuffer() const;
  void unbindFramebuffer() const;
  void bindTexture() const;

  void setScale(const float scale);
  void setPreserveDepthBuffer(const bool preserveDepthBuffer);
  void setUseNVDepthStencil(const bool useNVDepthStencil);
  void setPrecision(const BufferPrecision& precision);
  void setInterpolation(const bool interpolation);

  int getBufferWidth() const;
  int getBufferHeight() const;
  float getScale() const;
  bool getPreserveFramebuffer() const;
  bool getUseNVDepthStencil() const;
  BufferPrecision getPrecision() const;
  bool getInterpolation() const;
private:
  int _viewportWidth;
  int _viewportHeight;

  int _bufferWidth;
  int _bufferHeight;

  float _scale;
  bool _preserveDepthBuffer;
  bool _useNVDepthStencil;

  BufferPrecision _precision;

  bool _interpolation;

  GLuint _frameBufferObject;
  GLuint _colorBuffer;
  GLuint _depthBuffer;
  GLuint _textureId;
  GLuint _depthTextureId;

  unsigned char* _pixels;

  /*!
   * \brief         This pointer is used if the depth buffer is stored as GL_FLOAT.
   */
  GLfloat* _depthPixelsF;
  /*!
   * \brief         This pointer is used with the GL_DEPTH_STENCIL_NV for storing the depth buffer.
   */
  GLuint* _depthPixelsNV;

  vvOffscreenBuffer* _scaledDepthBuffer;

  bool _updatePosted;

  void doScale();
  void update();
  void storeColorBuffer();
  void storeDepthBuffer();
  void renderToViewAlignedQuad() const;
  void writeBackDepthBuffer() const;
};

#endif
