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

#ifndef _VV_CC_H_
#define _VV_CG_H_

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include "vvshadermanager.h"

#include <map>
#include <vector>

#ifdef HAVE_CG

#ifdef _WIN32
#ifndef WIN32
#define WIN32 1
#endif
#include <windows.h>
#endif

#include <Cg/cg.h>
#include <Cg/cgGL.h>

class vvCgParameter
{
public:
  inline void setParameter(const CGparameter parameter) { _parameter = parameter; }
  inline void setType(const vvShaderParameterType type) { _type = type; }
  inline void setIdentifier(const char* identifier) { _identifier = identifier; }

  inline CGparameter getParameter() const { return _parameter; }
  inline vvShaderParameterType getType() const { return _type; }
  inline const char* getIdentifier() const { return _identifier; }
private:
  CGparameter _parameter;
  vvShaderParameterType _type;
  const char* _identifier;
};

class VIRVOEXPORT vvCg : public vvShaderManager
{
public:
  vvCg();
  virtual ~vvCg();

  virtual bool loadShader(const char* shaderFileName, const ShaderType& shaderType);
  virtual bool loadShaderByString(const char* shaderString, const ShaderType& shaderType);
  virtual void enableShader(const int programIndex);
  virtual void disableShader(const int programIndex);
  virtual void initParameters(const int index,
                              const char** parameterNames,
                              const vvShaderParameterType* parameterTypes,
                              const int parameterCount);
  virtual void printCompatibilityInfo() const;
  virtual const char* getShaderDir() const;

  virtual void enableTexture(const int programIndex, const char* textureParameterName);
  virtual void disableTexture(const int programIndex, const char* textureParameterName);

  /*!
   * \brief         Set a scalar cg float parameter.
   *
   *                Make sure to have called \ref initParameters() for this
   *                program before setting parameters.
   * \param         programIndex Index of the program on the program stack.
   * \param         parameterIndex Index of the param on the param stack for the program.
   */
  virtual void setParameter1f(const int programIndex, const char* parameterName,
                              const float& f1);

  virtual void setParameter1i(const int programIndex, const char* parameterName,
                              const int& i1);

  virtual void setParameter3f(const int programIndex, const char* parameterName,
                              const float& f1, const float& f2, const float& f3);

  virtual void setParameter4f(const int programIndex, const char* parameterName,
                              const float& f1, const float& f2, const float& f3, const float& f4);

  virtual void setArrayParameter3f(const int programIndex, const char* parameterName, const int arrayIndex,
                                   const float& f1, const float& f2, const float& f3);

  virtual void setArrayParameter1i(const int programIndex, const char* parameterName, const int arrayIndex,
                                   const int& i1);

  virtual void setParameterTexId(const int programIndex, const char* parameterName, const unsigned int& ui1);

  virtual void setModelViewProj(const int programIndex, const char* parameterName);
private:
  // Cg specific stuff.
  CGcontext _cgContext;
  std::vector<CGprofile> _cgProfiles;
  std::vector<CGprogram> _cgPrograms;
  typedef std::vector<vvCgParameter> ParameterVector;
  typedef std::map<std::string, vvCgParameter> ParameterNameMap;
  std::vector<ParameterVector> _cgParameters;
  std::vector<ParameterNameMap> _cgParameterNameMaps;

  void init();
  CGGLenum toCgEnum(const ShaderType& shaderType) const;

  static void cgErrorHandler(CGcontext context, CGerror error, void* appData);
};

#endif // HAVE_CG

#endif // _VV_CG_H_

//============================================================================
// End of File
//============================================================================

