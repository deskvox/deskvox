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

#include "vvglslprogram.h"
#include "vvtoolshed.h"

#include "vvgltools.h"
#include "vvglew.h"
#include "vvdebugmsg.h"

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
    if(_fileStrings[i] == "")
      continue;

    switch(i)
    {
    case 0:
      _shaderId[i] = glCreateShader(GL_VERTEX_SHADER);
      cerr << "glCreateShader(GL_VERTEX_SHADER);" << endl;
      break;
    case 1:
      _shaderId[i] = glCreateShader(GL_GEOMETRY_SHADER_EXT);
      cerr << "glCreateShader(GL_GEOMETRY_SHADER_EXT);" << endl;
      break;
    case 2:
      _shaderId[i] = glCreateShader(GL_FRAGMENT_SHADER);
      cerr << "glCreateShader(GL_FRAGMENT_SHADER);" << endl;
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
      cerr << "glCompileShader failed: " << &compileLog[0] << endl;
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
      glActiveTexture(tex->_unit);
      switch(tex->_type)
      {
      case TEXTURE_1D:
        glActiveTexture(tex->_unit);
        glEnable (GL_TEXTURE_1D);
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_TEXTURE_3D);
        glBindTexture(GL_TEXTURE_1D, tex->_id);
        break;
      case TEXTURE_2D:
        glActiveTexture(tex->_unit);
        glDisable(GL_TEXTURE_1D);
        glEnable (GL_TEXTURE_2D);
        glDisable(GL_TEXTURE_3D);
        glBindTexture(GL_TEXTURE_1D, tex->_id);
        break;
      case TEXTURE_3D:
        glActiveTexture(tex->_unit);
        glDisable(GL_TEXTURE_1D);
        glDisable(GL_TEXTURE_2D);
        glEnable (GL_TEXTURE_3D);
        glBindTexture(GL_TEXTURE_1D, tex->_id);
        break;
      default:
        // do nothing
        break;
      }
    }

    glUseProgramObjectARB(_programId);
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
    glActiveTexture(tex->_unit);
    switch(tex->_type)
    {
    case TEXTURE_1D:
      glActiveTexture(tex->_unit);
      glDisable(GL_TEXTURE_1D);
      break;
    case TEXTURE_2D:
      glActiveTexture(tex->_unit);
      glDisable(GL_TEXTURE_2D);
      break;
    case TEXTURE_3D:
      glActiveTexture(tex->_unit);
      glDisable(GL_TEXTURE_3D);
      break;
    default:
      // do nothing
      break;
    }
  }
  glUseProgramObjectARB(0);

  vvGLTools::printGLError("Leaving vvGLSLProgram::disableProgram()");
}

void vvGLSLProgram::setParameter1f(const string& parameterName, const float& f1)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "setParameter1f("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  else
    glUniform1f(uniform, f1);
}

void vvGLSLProgram::setParameter1i(const string& parameterName, const int& i1)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "setParameter1i("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  else
    glUniform1i(uniform, i1);
}

void vvGLSLProgram::setParameter3f(const string& parameterName, const float* array)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "setParameter3f("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  else
    glUniform3fv(uniform, 3, array);
}

void vvGLSLProgram::setParameter3f(const string& parameterName,
                            const float& f1, const float& f2, const float& f3)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "setParameter3f("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  else
    glUniform3f(uniform, f1, f2, f3);
}

void vvGLSLProgram::setParameter4f(const string& parameterName, const float* array)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "setParameter4f("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  else
    glUniform4fv(uniform, 4, array);
}

void vvGLSLProgram::setParameter4f(const string& parameterName,
                            const float& f1, const float& f2, const float& f3, const float& f4)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "setParameter4f("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  else
    glUniform4f(uniform, f1, f2, f3, f4);
}

void vvGLSLProgram::setParameterArray1i(const string& parameterName, const int* array, const int& count)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "setParameterArray1i("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  else
    glUniform1iv(uniform, count, array);
}

void vvGLSLProgram::setParameterArray3f(const string& parameterName, const float* array, const int& count)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "setParameterArray3f("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  else
    glUniform3fv(uniform, 3*count, array);
}

void vvGLSLProgram::setParameterMatrix4f(const string& parameterName, const float* mat)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "setMatrix4f("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  else
    glUniformMatrix4fv(uniform, 1, false, mat);
}

vvGLSLProgram::vvGLSLTexture* vvGLSLProgram::getTexture(const string& parameterName)
{
  TextureIterator texIterator = _textureNameMaps.find(parameterName);
  if(texIterator != _textureNameMaps.end())
  {
    return texIterator->second;
  }
  else
  {
    vvGLSLTexture* newTex = new vvGLSLTexture;
    newTex->_unit = GL_TEXTURE0+_nTexture++;
    newTex->_uniform = glGetUniformLocation(_programId, parameterName.c_str());
    _textureNameMaps[parameterName] = newTex;
    return newTex;
  }
}

void vvGLSLProgram::setParameterTex1D(const std::string& parameterName, const unsigned int& ui)
{
  vvGLSLTexture* tex = getTexture(parameterName.c_str());

  if(tex->_uniform == -1)
  {
    cerr << "setTexture1D("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  else
  {
    glActiveTexture(tex->_unit);
    glUniform1i(tex->_uniform, tex->_unit);
    glEnable (GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_1D, ui);
  }
}

void vvGLSLProgram::setParameterTex2D(const std::string& parameterName, const unsigned int& ui)
{
  vvGLSLTexture* tex = getTexture(parameterName.c_str());

  if(tex->_uniform == -1)
  {
    cerr << "setTexture1D("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  else
  {
    glActiveTexture(tex->_unit);
    glUniform1i(tex->_uniform, tex->_unit);
    glDisable(GL_TEXTURE_1D);
    glEnable (GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_1D, ui);
  }
}

void vvGLSLProgram::setParameterTex3D(const std::string& parameterName, const unsigned int& ui)
{
  vvGLSLTexture* tex = getTexture(parameterName.c_str());

  if(tex->_uniform == -1)
  {
    cerr << "setTexture1D("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  else
  {
    glActiveTexture(tex->_unit);
    glUniform1i(tex->_uniform, tex->_unit);
    glDisable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
    glEnable (GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_1D, ui);
  }
}

//============================================================================
// End of File
//============================================================================
