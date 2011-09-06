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

#include "vvdebugmsg.h"
#include "vvglew.h"
#include "vvglslprogram.h"
#include "vvgltools.h"
#include "vvtoolshed.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

vvGLSLProgram::vvGLSLProgram(const string& vert, const string& geom, const string& frag)
: vvShaderProgram(vert, geom, frag)
{
  for(int i=0; i<3;i++)
    _shaderId[i] = 0;

  _shadersLoaded = loadShaders();
  if(!_shadersLoaded)
  {
    vvDebugMsg::msg(1, "vvGLSLProgram::vvGLSLProgram() Loading Shaders failed!");
  }

  _nTexture = 0;
}

vvGLSLProgram::~vvGLSLProgram()
{
  disableProgram();
  if(_programId)
  {
    glDeleteProgram(_programId);
  }
}

bool vvGLSLProgram::loadShaders()
{
  vvGLTools::printGLError("Enter vvGLSLProgram::loadShaders()");

  _programId = glCreateProgram();

  for(int i=0;i<3;i++)
  {
    if(_fileStrings[i].empty())
      continue;

    switch(i)
    {
    case 0:
      _shaderId[i] = glCreateShader(GL_VERTEX_SHADER);
      vvDebugMsg::msg(2, "glCreateShader(GL_VERTEX_SHADER)");
      break;
    case 1:
      _shaderId[i] = glCreateShader(GL_GEOMETRY_SHADER_EXT);
      vvDebugMsg::msg(2, "glCreateShader(GL_GEOMETRY_SHADER_EXT)");
      break;
    case 2:
      _shaderId[i] = glCreateShader(GL_FRAGMENT_SHADER);
      vvDebugMsg::msg(2, "glCreateShader(GL_FRAGMENT_SHADER)");
      break;
    }

    GLint size = (GLint)_fileStrings[i].size();
    const char* code = _fileStrings[i].c_str();
    glShaderSource(_shaderId[i], 1, (const GLchar**)&code, &size);
    glCompileShader(_shaderId[i]);

    GLint compiled;
    glGetShaderiv(_shaderId[i], GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
      GLint length;
      std::vector<GLchar> compileLog;
      glGetShaderiv(_shaderId[i], GL_INFO_LOG_LENGTH, &length);
      compileLog.resize(length);
      glGetShaderInfoLog(_shaderId[i], length, &length, &compileLog[0]);
      vvDebugMsg::msg(0, "glCompileShader failed: " , &compileLog[0]);
      return false;
    }

    glAttachShader(_programId, _shaderId[i]);
  }

  glLinkProgram(_programId);

  GLint linked;
  glGetProgramiv(_programId, GL_LINK_STATUS, &linked);
  if (!linked)
  {
    cerr << "glLinkProgram failed" << endl;
    return false;
  }

  _shadersLoaded = true;
  enableProgram();

  vvGLTools::printGLError("Leaving vvGLSLProgram::loadShaders()");

  return true;
}

void vvGLSLProgram::enableProgram()
{
  vvGLTools::printGLError("Enter vvGLSLProgram::enableProgram()");

  if(_shadersLoaded)
  {
    for(TextureMap::iterator i = _textureNameMaps.begin(); i != _textureNameMaps.end(); i++)
    {
      vvGLSLTexture* tex = i->second;
      glActiveTexture(GL_TEXTURE0+tex->_unit);
      switch(tex->_type)
      {
      case TEXTURE_1D:
        glEnable (GL_TEXTURE_1D);
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_TEXTURE_3D);
        glBindTexture(GL_TEXTURE_1D, tex->_id);
        break;
      case TEXTURE_2D:
        glDisable(GL_TEXTURE_1D);
        glEnable (GL_TEXTURE_2D);
        glDisable(GL_TEXTURE_3D);
        glBindTexture(GL_TEXTURE_2D, tex->_id);
        break;
      case TEXTURE_3D:
        glDisable(GL_TEXTURE_1D);
        glDisable(GL_TEXTURE_2D);
        glEnable (GL_TEXTURE_3D);
        glBindTexture(GL_TEXTURE_3D, tex->_id);
        break;
      default:
        // do nothing
        break;
      }
    }
    glUseProgramObjectARB(_programId);
    glActiveTexture(GL_TEXTURE0);
  }
  else
  {
    cerr << "vvGLSLProgram::enableProgram() Can't enable Programm: shaders not successfully loaded!" << endl;
  }

  vvGLTools::printGLError("Leaving vvGLSLProgram::enableProgram()");
}

void vvGLSLProgram::disableProgram()
{
  vvGLTools::printGLError("Enter vvGLSLProgram::disableProgram()");

  for(TextureMap::iterator i = _textureNameMaps.begin(); i != _textureNameMaps.end(); i++)
  {
    vvGLSLTexture* tex = i->second;
    glActiveTexture(GL_TEXTURE0+tex->_unit);
    switch(tex->_type)
    {
    case TEXTURE_1D:
      glDisable(GL_TEXTURE_1D);
      break;
    case TEXTURE_2D:
      glDisable(GL_TEXTURE_2D);
      break;
    case TEXTURE_3D:
      glDisable(GL_TEXTURE_3D);
      break;
    default:
      // do nothing
      break;
    }
  }
  glUseProgramObjectARB(0);
  glActiveTexture(GL_TEXTURE0);

  vvGLTools::printGLError("Leaving vvGLSLProgram::disableProgram()");
}

void vvGLSLProgram::setParameter1f(const string& parameterName, const float& f1)
{
  const GLint uniform = getUniform(parameterName, "setParameter1f");
  if(uniform != -1)
    glUniform1f(uniform, f1);
}

void vvGLSLProgram::setParameter1i(const string& parameterName, const int& i1)
{
  const GLint uniform = getUniform(parameterName, "setParameter1i");
  if(uniform != -1)
    glUniform1i(uniform, i1);
}

void vvGLSLProgram::setParameter3f(const string& parameterName, const float* array)
{
  const GLint uniform = getUniform(parameterName, "setParameter3f");
  if(uniform != -1)
    glUniform3fv(uniform, 3, array);
}

void vvGLSLProgram::setParameter3f(const string& parameterName,
                            const float& f1, const float& f2, const float& f3)
{
  const GLint uniform = getUniform(parameterName, "setParameter3f");
  if(uniform != -1)
    glUniform3f(uniform, f1, f2, f3);
}

void vvGLSLProgram::setParameter4f(const string& parameterName, const float* array)
{
  const GLint uniform = getUniform(parameterName, "setParameter4f");
  if(uniform != -1)
    glUniform4fv(uniform, 4, array);
}

void vvGLSLProgram::setParameter4f(const string& parameterName,
                            const float& f1, const float& f2, const float& f3, const float& f4)
{
  const GLint uniform = getUniform(parameterName, "setParameter4f");
  if(uniform != -1)
    glUniform4f(uniform, f1, f2, f3, f4);
}

void vvGLSLProgram::setParameterArray1i(const string& parameterName, const int* array, const int& count)
{
  const GLint uniform = getUniform(parameterName, "setParameterArray1i");
  if(uniform != -1)
    glUniform1iv(uniform, count, array);
}

void vvGLSLProgram::setParameterArray3f(const string& parameterName, const float* array, const int& count)
{
  const GLint uniform = getUniform(parameterName, "setParameterArray3f");
  if(uniform != -1)
    glUniform3fv(uniform, 3*count, array);
}

void vvGLSLProgram::setParameterMatrix4f(const string& parameterName, const float* mat)
{
  const GLint uniform = getUniform(parameterName, "setParameterMatrix4f");
  if(uniform != -1)
    glUniformMatrix4fv(uniform, 1, false, mat);
}

GLint vvGLSLProgram::getUniform(const string& parameterName, const string& parameterType)
{
  if(_parameterMaps.find(parameterName.c_str()) != _parameterMaps.end())
  {
    return _parameterMaps[parameterName.c_str()];
  }
  else
  {
    const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
    if(uniform == -1)
    {
      string errmsg;
      errmsg = parameterType + "(" + parameterName
               + ") does not correspond to an active uniform variable in program";
      vvDebugMsg::msg(1, errmsg.c_str());
    }

    _parameterMaps[parameterName.c_str()] = uniform;
    return uniform;
  }
}

vvGLSLProgram::vvGLSLTexture* vvGLSLProgram::getTexture(const string& parameterName, const string& parameterType)
{
  TextureIterator texIterator = _textureNameMaps.find(parameterName);
  if(texIterator != _textureNameMaps.end())
  {
    return texIterator->second;
  }
  else
  {
    vvGLSLTexture* newTex = new vvGLSLTexture;
    newTex->_unit = _nTexture++;
    newTex->_uniform = glGetUniformLocation(_programId, parameterName.c_str());
    if(newTex->_uniform == -1)
    {
      string errmsg;
      errmsg = parameterType + "(" + parameterName
               + ") does not correspond to an active uniform variable in program";
      vvDebugMsg::msg(1, errmsg.c_str());
    }
    _textureNameMaps[parameterName] = newTex;
    return newTex;
  }
}

void vvGLSLProgram::setParameterTex1D(const string& parameterName, const unsigned int& ui)
{
  vvGLSLTexture* tex = getTexture(parameterName, "setParameterTex1D");
  if(tex->_uniform != -1)
  {
    glUniform1i(tex->_uniform, tex->_unit);
    glActiveTexture(GL_TEXTURE0+tex->_unit);
    glEnable (GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_1D, ui);
    glActiveTexture(GL_TEXTURE0);
  }
}

void vvGLSLProgram::setParameterTex2D(const string& parameterName, const unsigned int& ui)
{
  vvGLSLTexture* tex = getTexture(parameterName, "setParameterTex2D");
  if(tex->_uniform != -1)
  {
    glUniform1i(tex->_uniform, tex->_unit);
    glActiveTexture(GL_TEXTURE0+tex->_unit);
    glDisable(GL_TEXTURE_1D);
    glEnable (GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_2D, ui);
    glActiveTexture(GL_TEXTURE0);
  }
}

void vvGLSLProgram::setParameterTex3D(const string& parameterName, const unsigned int& ui)
{
  vvGLSLTexture* tex = getTexture(parameterName, "setParameterTex3D");
  if(tex->_uniform != -1)
  {
    glUniform1i(tex->_uniform, tex->_unit);
    glActiveTexture(GL_TEXTURE0+tex->_unit);
    glDisable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glEnable (GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_3D, ui);
    glActiveTexture(GL_TEXTURE0);
  }
}

//============================================================================
// End of File
//============================================================================
