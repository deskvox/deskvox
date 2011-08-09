// Virvo - Virtual Reality Volume Rendering
// Contact: Chih Liang, cliang@cs.ucsd.edu
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

#ifndef _VV_SHADERPROGRAM_H_
#define _VV_SHADERPROGRAM_H_

#include "vvexport.h"

#include <string>

using std::string;

enum ShaderType
{
  VV_VERT_SHD = 0,
  VV_GEOM_SHD,
  VV_FRAG_SHD
};

enum vvShaderParameterType2
{
  VV_SHD_TEXTURE_ID2 = 0,
  VV_SHD_VEC32,
  VV_SHD_VEC42,
  VV_SHD_SCALAR2,
  VV_SHD_ARRAY2
};

/** Parent Class for ShaderPrograms
  This class' pointers can be used to load shader programs
  from vvShaderFactory without taking care about the
  shading language itself.

  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
 */
class VIRVOEXPORT vvShaderProgram
{
public:
  vvShaderProgram(const string& vert, const string& geom, const string& frag);
  virtual ~vvShaderProgram();

  virtual void enableProgram() = 0;
  virtual void disableProgram() = 0;

  virtual void setParameter1f(const string& parameterName, const float& f1) = 0;
  virtual void setParameter1i(const string& parameterName, const int& i1) = 0;

  virtual void setParameter3f(const string& parameterName,
                              const float& f1, const float& f2, const float& f3) = 0;

  virtual void setParameter4f(const string& parameterName,
                              const float& f1, const float& f2, const float& f3, const float& f4) = 0;

  virtual void setParameterArray3f(const string& parameterName, const float* array) = 0;
  virtual void setParameterArrayf(const string& parameterName, const float* array, const int& count) = 0;
  virtual void setParameterArrayi(const string& parameterName, const int* array, const int& count) = 0;

  virtual void setMatrix4f(const string& parameterName, const float* mat) = 0;

  virtual void setTextureId(const string& parameterName, const unsigned int& ui1) = 0;
  virtual void enableTexture(const string& programIndex) = 0;
  virtual void disableTexture(const string& programIndex) = 0;
protected:
  bool _shadersLoaded;
  string _fileStrings[3];
};

#endif // _VV_SHADERPROGRAM_H_

//============================================================================
// End of File
//============================================================================
