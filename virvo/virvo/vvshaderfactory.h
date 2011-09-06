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

#ifndef _VV_SHADERFACTORY_H_
#define _VV_SHADERFACTORY_H_

#include "vvshaderprogram.h"
#include "vvexport.h"
#include "vvdebugmsg.h"

class VIRVOEXPORT vvShaderFactory
{
public:
  /** Creates a shader-program and tries to attach shaders with given name.
    GLSL-shaders will be prefered if both types (GLSL and CG) are available.
    @param name The filename of shader-tuple without suffix and file-extension
   */

  vvShaderFactory();
  vvShaderProgram* createProgram(const std::string& name);
  vvShaderProgram* createProgram(const std::string& vert, const std::string& geom, const std::string& frag);
  bool cgSupport();
  bool glslSupport();

private:
  const std::string getShaderDir();
  bool loadFileStrings();

  bool _cgSupport;
  std::string _shaderName[3];
  std::string _fileString[3];
  std::string _shaderDir;
};

#endif // _VV_SHADERFACTORY2_H_

//============================================================================
// End of File
//============================================================================
