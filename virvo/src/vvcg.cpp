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

#include "vvcg.h"

#include <assert.h>
#include <utility>

#ifdef HAVE_CG

vvCg::vvCg()
  : vvShaderManager()
{
  init();
}

vvCg::~vvCg()
{

}

void vvCg::loadShader(const char* shaderFileName, const ShaderType& shaderType)
{
  assert(shaderFileName != NULL);

  _shaderFileNames.push_back(shaderFileName);

  CGcontext cgContext = cgCreateContext();

  if (cgContext == NULL)
  {
    cerr << "Can't create Cg context" << endl;
  }

  CGprofile cgProfile = cgGLGetLatestProfile(toCgEnum(shaderType));
  cgGLSetOptimalOptions(cgProfile);
  CGprogram cgProgram = cgCreateProgramFromFile(cgContext, CG_SOURCE, shaderFileName,
                                                cgProfile, "main", 0);
  _cgPrograms.push_back(cgProgram);
  _cgProfiles.push_back(cgProfile);

  // Remember to initialize the parameters by calling initParameters
  // before using them. For efficiency's sake, the setParameterxx functions
  // won't check, but only assert if this is the case. So if the assert
  // macro is undefined, using parameters will result in unpredicted
  // behavior if they weren't initialized before.
  _parametersInitialized.push_back(false);

  if (cgProgram == NULL)
  {
    cerr << "Couldn't load shader program from file: " << shaderFileName << endl;
  }
}

void vvCg::loadShaderByString(const char*, const ShaderType&)
{
  throw "Not implemented yet";
}

void vvCg::enableShader(const int index)
{
  cgGLLoadProgram(_cgPrograms[index]);
  cgGLEnableProfile(_cgProfiles[index]);
  cgGLBindProgram(_cgPrograms[index]);
}

void vvCg::disableShader(const int index)
{
  cgGLDisableProfile(_cgProfiles.at(index));
}

void vvCg::initParameters(const int index,
                          const char** parameterNames,
                          const vvShaderParameterType* parameterTypes,
                          const int parameterCount)
{
  assert((parameterNames != NULL) && (parameterTypes != NULL));

  ParameterVector params(parameterCount);
  _cgParameterNameMaps.push_back(ParameterNameMap());

  for (int i = 0; i < parameterCount; ++i)
  {
    params[i] = vvCgParameter();
    params[i].setParameter(cgGetNamedParameter(_cgPrograms[index], parameterNames[i]));
    params[i].setType(parameterTypes[i]);
    params[i].setIdentifier(parameterNames[i]);

    // Insert the param to the name map in order to perform search for parameter by name.
    _cgParameterNameMaps[index][params[i].getIdentifier()] = params[i];
  }

  _cgParameters.push_back(params);
  _parametersInitialized[index] = true;
}

void vvCg::setParameter1f(const int programIndex, const char* parameterName,
                          const float& f1)
{
  assert(_parametersInitialized[programIndex] == true);

  vvCgParameter param = _cgParameterNameMaps[programIndex][parameterName];
  cgGLSetParameter1f(param.getParameter(), f1);
}

void vvCg::setParameter1i(const int programIndex, const char* parameterName,
                          const int& i1)
{
  assert(_parametersInitialized[programIndex] == true);

  vvCgParameter param = _cgParameterNameMaps[programIndex][parameterName];
  cgGLSetParameter1f(param.getParameter(), i1);
}

void vvCg::setParameter3f(const int programIndex, const char* parameterName,
                          const float& f1, const float& f2, const float& f3)
{
  assert(_parametersInitialized[programIndex] == true);

  vvCgParameter param = _cgParameterNameMaps[programIndex][parameterName];
  cgGLSetParameter3f(param.getParameter(), f1, f2, f3);
}

void vvCg::setParameter4f(const int programIndex, const char* parameterName,
                          const float& f1, const float& f2, const float& f3, const float& f4)
{
  assert(_parametersInitialized[programIndex] == true);

  vvCgParameter param = _cgParameterNameMaps[programIndex][parameterName];
  cgGLSetParameter4f(param.getParameter(), f1, f2, f3, f4);
}

void vvCg::setArrayParameter3f(const int programIndex, const char* parameterName, const int arrayIndex,
                              const float& f1, const float& f2, const float& f3)
{
  CGparameter array = cgGetNamedParameter(_cgPrograms[programIndex], parameterName);
  CGparameter element = cgGetArrayParameter(array, arrayIndex);

  // There exists no similar cg function for integers ==> use the float equivalent.
  cgGLSetParameter3f(element, f1, f2, f3);
}

void vvCg::setArrayParameter1i(const int programIndex, const char* parameterName, const int arrayIndex,
                              const int& i1)
{
  CGparameter array = cgGetNamedParameter(_cgPrograms[programIndex], parameterName);
  CGparameter element = cgGetArrayParameter(array, arrayIndex);

  // There exists no similar cg function for integers ==> use the float equivalent.
  cgGLSetParameter1f(element, i1);
}

void vvCg::setModelViewProj(const int programIndex, const char* parameterName)
{
  CGparameter modelViewProj = cgGetNamedParameter(_cgPrograms[programIndex], parameterName);
  cgGLSetStateMatrixParameter(modelViewProj, CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY);
}

void vvCg::init()
{

}

CGGLenum vvCg::toCgEnum(const ShaderType& shaderType)
{
  CGGLenum result;
  switch (shaderType)
  {
  case VV_FRAG_SHD:
    result = CG_GL_FRAGMENT;
    break;
#if 0 // TODO - find out if geometry shaders are supported.
  case VV_GEOM_SHD:
    result = CG_GL_GEOMETRY;
    break;
#endif
  case VV_VERT_SHD:
    result = CG_GL_VERTEX;
    break;
  default:
    result = CG_GL_FRAGMENT;
    break;
  }
  return result;
}

#endif
