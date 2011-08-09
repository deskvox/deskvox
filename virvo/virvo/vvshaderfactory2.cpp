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

#include "vvshaderfactory2.h"
#include "vvshaderprogram.h"
#include "vvglslprogram.h"
#include "vvcgprogram.h"
#include "vvtoolshed.h"

#include <cstdlib>

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

using std::string;
using std::cerr;
using std::endl;

string vvShaderFactory2::_shaderDir = string();
string vvShaderFactory2::_shaderName[3] = { string(), string(), string() };
string vvShaderFactory2::_fileString[3] = { string(), string(), string() };

vvShaderProgram* vvShaderFactory2::createProgram(const std::string& name)
{
  if(_shaderDir == "")
    _shaderDir = getShaderDir();

  vvShaderProgram *program = NULL;

  if(!glslSupport())
  {
    vvDebugMsg::msg(0, "vvShaderFactory2::createProgram: no GLSL support!");
  }

  _shaderName[0] = "vv_" + name + ".vsh";
  _shaderName[1] = "vv_" + name + ".gsh";
  _shaderName[2] = "vv_" + name + ".fsh";
  bool loaded = loadFileStrings();

  if(loaded && glslSupport())
  {
    cerr << "GLSL-shaders found:" << endl;

    for(int i=0;i<3;i++)
    {
      cerr << "try " << _shaderName[i] << endl;
      if(_fileString[i].length() > 0)
        cerr << _shaderName[i] << endl;
    }
    program = new vvGLSLProgram(_fileString[0], _fileString[1], _fileString[2]);
    if(!program->isValid())
    {
      delete program;
      program = NULL;
    }
  }
#ifdef HAVE_CG
  if(!program)
  {
    if(!cgSupport())
    {
      vvDebugMsg::msg(0, "vvShaderFactory2::createProgram: no CG support!");
    }

    _shaderName[0] = "vv_" + name + ".vert.cg";
    _shaderName[1] = "vv_" + name + ".geom.cg";
    _shaderName[2] = "vv_" + name + ".frag.cg";

    loaded = loadFileStrings();

    if(loaded && cgSupport())
    {
      cerr << "CG-shaders found:" << endl;

      for(int i=0;i<3;i++)
      {
        cerr << "try " << _shaderName[i] << endl;
        if(_fileString[i].length() > 0)
          cerr << _shaderName[i] << endl;
      }
      program = new vvCgProgram(_fileString[0], _fileString[1], _fileString[2]);
      if(!program->isValid())
      {
        delete program;
        program = NULL;
      }
    }
  }
#endif

  if(!program)
  {
    cerr << "No supported shaders with name " << name << " found!" << endl;
  }

  return program;
}

bool vvShaderFactory2::loadFileStrings()
{
  bool hit = false;
  for(int i=0;i<3;i++)
  {
    if(_shaderName[i].length() == 0)
      continue;

    std::string filePath = _shaderDir+_shaderName[i];
    cerr << "looking for shader: " << filePath << endl;
    char* tempString = vvToolshed::file2string(filePath.c_str());
    std::string fileString;
    if(tempString)
    {
      _fileString[i] = tempString;
      hit = true;
    }
    else
    {
      _fileString[i] = "";
    }
  }
  return hit;
}

const string vvShaderFactory2::getShaderDir()
{
  string result;

  const char* shaderEnv = "VV_SHADER_PATH";
  if (getenv(shaderEnv))
  {
    cerr << "Environment variable " << shaderEnv << " found: " << getenv(shaderEnv) << endl;
    result = getenv(shaderEnv);
  }
  else
  {
    cerr << "Warning: you should set the environment variable " << shaderEnv << " to point to your shader directory" << endl;
    static char shaderDir[256];
#ifdef _WIN32
    const char* primaryWin32ShaderDir = "..\\..\\..\\virvo\\shader";
    vvToolshed::getProgramDirectory(shaderDir, 256);
    strcat(shaderDir, primaryWin32ShaderDir);
    cerr << "Trying shader path: " << shaderDir << endl;
    if (!vvToolshed::isDirectory(shaderDir))
    {
       vvToolshed::getProgramDirectory(shaderDir, 256);
    }
    cerr << "Using shader path: " << shaderDir << endl;
    result = shaderDir;
#else
    const char* deskVoxShaderPath = "/..";
#ifdef SHADERDIR
    result = SHADERDIR;
#else
    vvToolshed::getProgramDirectory(shaderDir, 256);
    strcat(shaderDir, deskVoxShaderPath);
    result = shaderDir;
#endif
#endif
  }
#ifdef _WIN32
  result += "\\";
#else
  result += "/";
#endif

  return result;
}

bool vvShaderFactory2::cgSupport()
{
  #ifdef HAVE_CG
    return true;
  #else
    return false;
  #endif
}

bool vvShaderFactory2::glslSupport()
{
  #if defined GL_VERSION_1_1 || defined GL_VERSION_1_2 \
    || defined GL_VERSION_1_3 || defined GL_VERSION_1_4 \
    || defined GL_VERSION_1_5 || defined GL_VERSION_2_0 \
    || defined GL_VERSION_3_0
    // Assume that even compilers that support higher gl versions
    // will know at least one of those listed here.
    return true;
  #else
    return false;
  #endif
}
