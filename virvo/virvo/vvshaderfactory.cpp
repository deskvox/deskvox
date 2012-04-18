// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
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
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include "vvglew.h"
#include "vvcgprogram.h"
#include "vvdebugmsg.h"
#include "vvglslprogram.h"
#include "vvshaderfactory.h"
#include "vvshaderprogram.h"
#include "vvtoolshed.h"

#include <cstdlib>
#include <cstring>

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

using std::string;
using std::cerr;
using std::endl;

vvShaderFactory::vvShaderFactory()
{
  _shaderDir = string();
  for(int i=0;i<3;i++)
  {
    _shaderName[i] = string();
    _fileString[i] = string();
  }

  glewInit();
}

vvShaderProgram* vvShaderFactory::createProgram(const std::string& name)
{
  return createProgram(name, name, name);
}

vvShaderProgram* vvShaderFactory::createProgram(const std::string& vert, const std::string& geom, const std::string& frag)
{
  if(vert.empty() && geom.empty() && frag.empty())
    return NULL;

  _shaderName[0].clear();
  _shaderName[1].clear();
  _shaderName[2].clear();

  if(_shaderDir.empty())
    _shaderDir = getShaderDir();

  vvShaderProgram *program = NULL;

  if(!glslSupport())
  {
    vvDebugMsg::msg(0, "vvShaderFactory::createProgram: no GLSL support!");
  }
  else
  {
    if(!vert.empty()) _shaderName[0] = "vv_" + vert + ".vsh";
    if(!geom.empty()) _shaderName[1] = "vv_" + geom + ".gsh";
    if(!frag.empty()) _shaderName[2] = "vv_" + frag + ".fsh";
    bool loaded = loadFileStrings();

    if(loaded)
    {
      vvDebugMsg::msg(2, "GLSL-shaders found:");

      for(int i=0;i<3;i++)
        if(_fileString[i].length() > 0)
          vvDebugMsg::msg(2, _shaderName[i].c_str());

      program = new vvGLSLProgram(_fileString[0], _fileString[1], _fileString[2], _geoShaderArgs);
      if(!program->isValid())
      {
        delete program;
        program = NULL;
      }
    }
  }
#ifdef HAVE_CG
  if(!program)
  {
    if(!cgSupport())
    {
      vvDebugMsg::msg(0, "vvShaderFactory::createProgram: no CG support!");
    }
    else
    {
      if(!vert.empty()) _shaderName[0] = "vv_" + vert + ".vert.cg";
      if(!geom.empty()) _shaderName[1] = "vv_" + geom + ".geom.cg";
      if(!frag.empty()) _shaderName[2] = "vv_" + frag + ".frag.cg";

      bool loaded = loadFileStrings();

      if(loaded)
      {
        cerr << "CG-shaders found: ";

        for(int i=0;i<3;i++)
          if(_fileString[i].length() > 0)
            cerr << _shaderName[i] << " ";
        cerr << endl;

        program = new vvCgProgram(_fileString[0], _fileString[1], _fileString[2], _geoShaderArgs);
        if(!program->isValid())
        {
          delete program;
          program = NULL;
        }
      }
    }
  }
#endif

  if(!program)
  {
    string errmsg = "No supported shaders with name " + vert + " " + geom + " or " + frag + " found!";
    vvDebugMsg::msg(0, errmsg.c_str());
  }

  return program;
}

vvShaderProgram* vvShaderFactory::createProgram(const std::string& vert, const std::string& geom, const std::string& frag,
                                                const vvShaderProgram::GeoShaderArgs& geoShaderArgs)
{
  if (geom.empty())
  {
    vvDebugMsg::msg(0, "vvShaderFactory::createProgram(): Geometry shader args specified but no geometry shader supplied");
  }
  _geoShaderArgs = geoShaderArgs;
  return createProgram(vert, geom, frag);
}

bool vvShaderFactory::loadFileStrings()
{
  bool hit = true;
  for(int i=0;i<3;i++)
  {
    _fileString[i].clear();;

    if(_shaderName[i].empty())
      continue;

    std::string filePath = _shaderDir+_shaderName[i];
    char* tempString = vvToolshed::file2string(filePath.c_str());
    if(tempString)
      _fileString[i] = tempString;
    else
      hit = false;
  }
  return hit;
}

const string vvShaderFactory::getShaderDir()
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

bool vvShaderFactory::cgSupport()
{
  #ifdef HAVE_CG
    return true;
  #else
    return false;
  #endif
}

bool vvShaderFactory::glslSupport()
{
  #if defined GL_VERSION_1_1 || defined GL_VERSION_1_2 \
    || defined GL_VERSION_1_3 || defined GL_VERSION_1_4 \
    || defined GL_VERSION_1_5 || defined GL_VERSION_2_0 \
    || defined GL_VERSION_3_0
    // Assume that even compilers that support higher gl versions
    // will know at least one of those listed here.
    return glCreateProgram && glDeleteProgram && glUniform1f;
  #else
    return false;
  #endif
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
