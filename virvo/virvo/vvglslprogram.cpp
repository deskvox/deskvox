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
cerr << "GLSL-SOURCECODE "<<i<<": "<< _fileStrings[i]<<endl;

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
    glUseProgramObjectARB(_programId);
  else
    cerr << "vvGLSLProgram::enableProgram() Can't enable Programm: shaders not successfully loaded!" << endl;

  vvGLTools::printGLError("Leaving vvGLSLProgram::enableProgram()");
}

void vvGLSLProgram::disableProgram()
{
  vvGLTools::printGLError("Enter vvGLSLProgram::disableProgram()");

  glUseProgramObjectARB(0);

  vvGLTools::printGLError("Leaving vvGLSLProgram::disableProgram()");
}

void vvGLSLProgram::setParameter1f(const string& parameterName, const float& f1)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  glUniform1f(uniform, f1);
}

void vvGLSLProgram::setParameter1i(const string& parameterName, const int& i1)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "uniform1i("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  glUniform1i(uniform, i1);
}

void vvGLSLProgram::setParameter3f(const string& parameterName,
                            const float& f1, const float& f2, const float& f3)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  glUniform3f(uniform, f1, f2, f3);
}


void vvGLSLProgram::setParameter4f(const string& parameterName,
                            const float& f1, const float& f2, const float& f3, const float& f4)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  glUniform4f(uniform, f1, f2, f3, f4);
}

void vvGLSLProgram::setParameterArray3f(const string& parameterName, const float* array)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "uniformArray3f("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  glUniform3fv(uniform, 3, array);
}

void vvGLSLProgram::setParameterArrayf(const string& parameterName, const float* array, const int& count)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "uniformArrayf("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  glUniform1fv(uniform, count, array);
}

void vvGLSLProgram::setParameterArrayi(const string& parameterName, const int* array, const int& count)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "uniformArrayi("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  glUniform1iv(uniform, count, array);
}

void vvGLSLProgram::setMatrix4f(const string& parameterName, const float* mat)
{
  const GLint uniform = glGetUniformLocation(_programId, parameterName.c_str());
  if(uniform == -1)
  {
    cerr << "mat("<<parameterName
         <<") does not correspond to an active uniform variable in program"<< endl;
  }
  glUniformMatrix4fv(uniform, 1, false, mat);
}
