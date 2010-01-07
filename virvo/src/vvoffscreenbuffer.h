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
  virtual void writeBack();
  virtual void resize(const int w, const int h);

  void setScale(const float scale);

  float getScale() const;
private:
  int _viewportWidth;
  int _viewportHeight;

  int _bufferWidth;
  int _bufferHeight;

  float _scale;

  BufferPrecision _precision;

  GLuint _frameBufferObject;
  GLuint _depthBuffer;
  GLuint _textureId;

  bool _updatePosted;

  void doScale();
  void update();
};

#endif
