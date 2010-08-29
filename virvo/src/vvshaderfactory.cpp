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

#include <algorithm>

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

std::vector<vvShaderManagerType> vvShaderFactory::c_supportedTypes = std::vector<vvShaderManagerType>();

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

const std::vector<vvShaderManagerType> &vvShaderFactory::getSupportedShaderManagers()
{
  if (!vvShaderFactory::c_supportedTypes.empty())
  {
    // Don't rebuild if not necessary.
    return vvShaderFactory::c_supportedTypes;
  }
#ifdef HAVE_CG
  vvShaderFactory::c_supportedTypes.push_back(VV_CG_MANAGER);
#endif

#if defined GL_VERSION_1_1 || defined GL_VERSION_1_2 \
  || defined GL_VERSION_1_3 || defined GL_VERSION_1_4 \
  || defined GL_VERSION_1_5 || defined GL_VERSION_2_0 \
  || defined GL_VERSION_3_0
  // Assume that even compilers that support higher gl versions
  // will know at least one of those listed here.
  vvShaderFactory::c_supportedTypes.push_back(VV_GLSL_MANAGER);
#endif
  return vvShaderFactory::c_supportedTypes;
}

bool vvShaderFactory::isSupported(const vvShaderManagerType& shaderManagerType)
{
  // If vector of supported types is empty, suspect that it wasn't built at all.
  if (vvShaderFactory::c_supportedTypes.empty())
  {
    getSupportedShaderManagers();
  }
  std::vector<vvShaderManagerType>::iterator result =
      std::find(vvShaderFactory::c_supportedTypes.begin(),
                vvShaderFactory::c_supportedTypes.end(),
                shaderManagerType);

  return result != vvShaderFactory::c_supportedTypes.end();
}
