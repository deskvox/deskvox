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

#include "vvshaderfactory.h"

// Include headers to your custom shader manager here rather than in the header.
#include "vvcg.h"
#include "vvglsl.h"

vvShaderManager* vvShaderFactory::provideShaderManager(const vvShaderManagerType& shaderManagerType)
{
  vvShaderManager* result = NULL;

  switch (shaderManagerType)
  {
#ifdef HAVE_CG
  case VV_CG_MANAGER:
    result = new vvCg();
    break;
#endif

  case VV_GLSL_MANAGER:
    result = new vvGLSL();
    break;
  default:
    break;
  }

  return result;
}
