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
     vvDebugMsg::msg(0, "vvCgProgram::vvCgProgram() Loading Shaders failed!");
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
    vvDebugMsg::msg(0, "Can't create Cg context");
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
      vvDebugMsg::msg(0, "Couldn't load cg-shader!");
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
    cerr << cgGetErrorString(error) << " (" << static_cast<int>(error) << ")" << endl;
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
    vvDebugMsg::msg(0, "toCgEnum() unknown ShaderType!");
    result = CG_GL_FRAGMENT;
    break;
  }
  return result;
}

vvCgProgram::ParameterIterator vvCgProgram::initParameter(const string& parameterName)
{
  // check if already initialized
  ParameterIterator paraIterator = _cgParameterNameMaps.find(parameterName);
  if(paraIterator != _cgParameterNameMaps.end())
    return paraIterator;

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

  _cgParameterNameMaps[parameterName] = paraFirst;

  if(paraFirst == 0)
  {
    string errmsg = "cgParameter (" + parameterName + ")not found!";
    vvDebugMsg::msg(2, errmsg.c_str());
    return _cgParameterNameMaps.end();
  }

  return _cgParameterNameMaps.find(parameterName);
}

void vvCgProgram::setParameter1f(const string& parameterName, const float& f1)
{
  ParameterIterator it = initParameter(parameterName);
  if(it->second != 0)
    cgSetParameter1f(it->second, f1);
}

void vvCgProgram::setParameter1i(const string& parameterName, const int& i1)
{
  ParameterIterator it = initParameter(parameterName);
  if(it->second != 0)
    cgSetParameter1i(it->second, i1);
}

void vvCgProgram::setParameter3f(const string& parameterName, const float* array)
{
  ParameterIterator it = initParameter(parameterName);
  if(it->second != 0)
    cgSetParameter3fv(it->second, array);
}

void vvCgProgram::setParameter3f(const string& parameterName,
                          const float& f1, const float& f2, const float& f3)
{
  ParameterIterator it = initParameter(parameterName);
  if(it->second != 0)
    cgSetParameter3f(it->second, f1, f2, f3);
}

void vvCgProgram::setParameter4f(const string& parameterName, const float* array)
{
  ParameterIterator it = initParameter(parameterName);
  if(it->second != 0)
    cgSetParameter4fv(it->second, array);
}

void vvCgProgram::setParameter4f(const string& parameterName,
                          const float& f1, const float& f2, const float& f3, const float& f4)
{
  ParameterIterator it = initParameter(parameterName);
  if(it->second != 0)
    cgSetParameter4f(it->second, f1, f2, f3, f4);
}

void vvCgProgram::setParameterArray1i(const string& parameterName, const int* array, const int& count)
{
  ParameterIterator it = initParameter(parameterName);
  if(it->second != 0)
  {
    // transform integers to floats because CG doesn't support uniform integers
    float floats[count];
    for(int i=0;i<count;i++)
      floats[i] = float(array[i]);
    cgGLSetParameterArray1f(it->second, 0, count, floats);
  }
}

void vvCgProgram::setParameterArray3f(const string& parameterName, const float* array, const int& count)
{
  ParameterIterator it = initParameter(parameterName);
  if(it->second != 0)
    cgGLSetParameterArray3f(it->second, 0, 3*count, array);
}

void vvCgProgram::setParameterMatrix4f(const string& parameterName, const float* mat)
{
  ParameterIterator it = initParameter(parameterName);
  if(it->second != 0)
    cgSetMatrixParameterfr(it->second, mat);
}

void vvCgProgram::setParameterTex1D(const string& parameterName, const unsigned int& ui)
{
  ParameterIterator it = initParameter(parameterName);
  if(it->second != 0)
  {
    cgGLSetTextureParameter(it->second, ui);
    cgGLEnableTextureParameter(it->second);
  }
}

void vvCgProgram::setParameterTex2D(const string& parameterName, const unsigned int& ui)
{
  setParameterTex1D(parameterName, ui);
}

void vvCgProgram::setParameterTex3D(const string& parameterName, const unsigned int& ui)
{
  setParameterTex1D(parameterName, ui);
}

#endif // ifdef HAVE_CG

//============================================================================
// End of File
//============================================================================
