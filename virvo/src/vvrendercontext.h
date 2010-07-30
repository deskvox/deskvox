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

#ifndef _VV_RENDERCONTEXT_H_
#define _VV_RENDERCONTEXT_H_

#include "vvrendertarget.h"

/*! Has to be defined in .cpp
 *  contains architecture specific variables
 */
struct ContextArchData;

class vvRenderContext : public vvRenderTarget
{
public:
  vvRenderContext(const bool debug = false);
  ~vvRenderContext();

  bool makeCurrent() const;
private:
  ContextArchData* _archData;

  bool _initialized;

  void init(const bool debug = false);
};

#endif // _VV_RENDERCONTEXT_H_
