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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include "vvcgprogram.h"
#include "vvgltools.h"
#include "vvopengl.h"
#include "vvdebugmsg.h"

#include <iostream>

using std::cerr;
using std::endl;
using std::string;

#ifdef HAVE_CG

vvCgProgram::vvCgProgram(const string& vert, const string& geom, const string& frag)
: vvShaderProgram(vert, geom, frag)
{
  for(int i=0; i<3;i++)
  {
    _shaderId[i] = 0;
    _profile[i] = CGprofile(0);
  }

  _shadersLoaded = loadShaders();
  if(_shadersLoaded)
  {
     vvDebugMsg::msg(1, "vvCgProgram::vvCgProgram() Loading Shaders failed!");
  }
}

vvCgProgram::~vvCgProgram()
{
  disableProgram();
  if (_program)
  {
    cgDestroyContext(_program);
  }
}

bool vvCgProgram::loadShaders()
{
  cgSetErrorHandler(cgErrorHandler, NULL);
  _program = cgCreateContext();

  if (_program == NULL)
  {
    cerr << "Can't create Cg context" << endl;
  }

  for(int i=0;i<3;i++)
  {
    if(_fileStrings[i].length() == 0)
      continue;

    _profile[i] = cgGLGetLatestProfile(toCgEnum(i));
    cgGLSetOptimalOptions(_profile[i]);
    _shaderId[i] = cgCreateProgram( _program, CG_SOURCE, _fileStrings[i].c_str(), _profile[i], NULL, NULL);

    if (_shaderId[i] == NULL)
    {
      cerr << "Couldn't load cg-shader!" << endl;
      return false;
    }
  }
  return true;
}

void vvCgProgram::enableProgram()
{
  for(int i=0;i<3;i++)
  {
    if(_shaderId[i] == 0)
      continue;

    cgGLLoadProgram(_shaderId[i]);
    cgGLEnableProfile(_profile[i]);
    cgGLBindProgram(_shaderId[i]);
  }
}

void vvCgProgram::disableProgram()
{
  for(int i=0;i<3;i++)
  {
    if(_profile[i] == 0)
      continue;

    cgGLDisableProfile(_profile[i]);
  }
}

void vvCgProgram::cgErrorHandler(CGcontext context, CGerror error, void*)
{
  if(error != CG_NO_ERROR)
    cerr << cgGetErrorString(error) << "(" << static_cast<int>(error) << ")" << endl;
  for(GLint glerr = glGetError(); glerr != GL_NO_ERROR; glerr = glGetError())
  {
    cerr << "GL error: " << gluErrorString(glerr) << endl;
  }
  if(context && error==CG_COMPILER_ERROR)
  {
     if(const char *listing = cgGetLastListing(context))
     {
        cerr << "last listing:" << endl;
        cerr << listing << endl;
     }
  }
}

CGGLenum vvCgProgram::toCgEnum(const int i) const
{
  CGGLenum result;
  switch(i)
  {
  case 0:
    result = CG_GL_VERTEX;
    break;
#if CG_VERSION_NUM >= 2000
  case 1:
    result = CG_GL_GEOMETRY;
    break;
#endif
  case 2:
    result = CG_GL_FRAGMENT;
    break;
  default:
    cerr << "toCgEnum() unknown ShaderType!" << endl;
    result = CG_GL_FRAGMENT;
    break;
  }
  return result;
}


bool vvCgProgram::initParameter(const string& parameterName)
{
  // check if already initialized
  if(_cgParameterNameMaps.find(parameterName) != _cgParameterNameMaps.end())
    return true;

  CGparameter paraFirst = 0;
  for(int i=0;i<3;i++)
  {
    if(_shaderId[i]==0)
      continue;

    CGparameter param = cgGetNamedParameter(_shaderId[i], parameterName.c_str());

    if(param != NULL && paraFirst == 0)
    {
      paraFirst = param;
    }
    else if(param != NULL && paraFirst != 0)
    {
      cgConnectParameter(paraFirst, param);
    }
  }

  if(paraFirst == 0)
  {
    cerr << "cgParameter ("<<parameterName<<")not found!"<<endl;
    return false;
  }
  else
  {
    _cgParameterNameMaps[parameterName] = paraFirst;
    return true;
  }
}

void vvCgProgram::setParameter1f(const string& parameterName, const float& f1)
{
  if(initParameter(parameterName))
    cgSetParameter1f(_cgParameterNameMaps[parameterName], f1);
}

void vvCgProgram::setParameter1i(const string& parameterName, const int& i1)
{
  if(initParameter(parameterName))
    cgSetParameter1i(_cgParameterNameMaps[parameterName], i1);
}

void vvCgProgram::setParameter3f(const string& parameterName, const float* array)
{
  if(initParameter(parameterName))
    cgSetParameter3fv(_cgParameterNameMaps[parameterName], array);
}

void vvCgProgram::setParameter3f(const string& parameterName,
                          const float& f1, const float& f2, const float& f3)
{
  if(initParameter(parameterName))
    cgSetParameter3f(_cgParameterNameMaps[parameterName], f1, f2, f3);
}

void vvCgProgram::setParameter4f(const string& parameterName, const float* array)
{
  if(initParameter(parameterName))
    cgSetParameter4fv(_cgParameterNameMaps[parameterName], array);
}

void vvCgProgram::setParameter4f(const string& parameterName,
                          const float& f1, const float& f2, const float& f3, const float& f4)
{
  if(initParameter(parameterName))
    cgSetParameter4f(_cgParameterNameMaps[parameterName], f1, f2, f3, f4);
}

void vvCgProgram::setParameterArray1i(const string& parameterName, const int* array, const int& count)
{
  if(initParameter(parameterName))
  {
    // transform integers to floats because CG doesn't support uniform integers
    float floats[count];
    for(int i=0;i<count;i++)
      floats[i] = float(array[i]);
    cgGLSetParameterArray1f(_cgParameterNameMaps[parameterName], 0, count, floats);
  }
}

void vvCgProgram::setParameterArray3f(const string& parameterName, const float* array, const int& count)
{
  if(initParameter(parameterName))
    cgGLSetParameterArray3f(_cgParameterNameMaps[parameterName], 0, 3*count, array);
}

void vvCgProgram::setMatrix4f(const string& parameterName, const float* mat)
{
  if(initParameter(parameterName))
    cgSetMatrixParameterfr(_cgParameterNameMaps[parameterName], mat);
}

void vvCgProgram::setParameterTexId(const string& parameterName, const unsigned int& ui1)
{
  if(initParameter(parameterName))
    cgGLSetTextureParameter(_cgParameterNameMaps[parameterName], ui1);
}

void vvCgProgram::setTextureId(const string& parameterName, const unsigned int& ui1)
{
  if(initParameter(parameterName))
   cgGLSetTextureParameter(_cgParameterNameMaps[parameterName], ui1);
}

void vvCgProgram::enableTexture(const string& parameterName)
{
  if(initParameter(parameterName))
   cgGLEnableTextureParameter(_cgParameterNameMaps[parameterName]);
}

void vvCgProgram::disableTexture(const string& parameterName)
{
  if(initParameter(parameterName))
    cgGLDisableTextureParameter(_cgParameterNameMaps[parameterName]);
}

#endif // ifdef HAVE_CG

//============================================================================
// End of File
//============================================================================
