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

enum ShaderType
{
  VV_VERT_SHD = 0,
  VV_GEOM_SHD,
  VV_FRAG_SHD
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
  /*!
    Create a program with combination of non-empty provided shader codes
  */
  vvShaderProgram(const std::string& vert, const std::string& geom, const std::string& frag);
  virtual ~vvShaderProgram();

  virtual bool isValid() const;

  virtual void enable() = 0;
  virtual void disable() = 0;

  virtual void setParameter1f(const std::string& parameterName, const float& f1) = 0;        ///< Set uniform floating variable
  virtual void setParameter1i(const std::string& parameterName, const int& i1) = 0;          ///< Set uniform integer variable

  virtual void setParameter3f(const std::string& parameterName, const float* array) = 0;     ///< Set uniform float-3D-vector stored in array
  virtual void setParameter3f(const std::string& parameterName,                              ///< Set uniform float-3D-vector
                              const float& f1, const float& f2, const float& f3) = 0;

  virtual void setParameter4f(const std::string& parameterName, const float* array) = 0;     ///< Set uniform float-4D-vector stored in array
  virtual void setParameter4f(const std::string& parameterName,                              ///< Set uniform float-4D-vector
                              const float& f1, const float& f2, const float& f3, const float& f4) = 0;

  /*!
    \brief Set uniform integer-array
    \param count number of intergers in array
  */
  virtual void setParameterArray1i(const std::string& parameterName, const int* array, const int& count) = 0;

  /*!
    \brief Set uniform float-array of 3D-vectors
    \param count number of vectors in array
  */
  virtual void setParameterArray3f(const std::string& parameterName, const float* array, const int& count) = 0;

  virtual void setParameterMatrix4f(const std::string& parameterName, const float* mat) = 0;          ///< set uniform 4x4-matrix float

  virtual void setParameterTex1D(const std::string& parameterName, const unsigned int& ui) = 0;
  virtual void setParameterTex2D(const std::string& parameterName, const unsigned int& ui) = 0;
  virtual void setParameterTex3D(const std::string& parameterName, const unsigned int& ui) = 0;

protected:
  bool _shadersLoaded;
  std::string _fileStrings[3];
};

#endif // _VV_SHADERPROGRAM_H_

//============================================================================
// End of File
//============================================================================
