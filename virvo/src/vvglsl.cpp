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

#include "vvglew.h"
#include <iostream>
#include "vvtoolshed.h"
#include "vvglsl.h"
#include "vvgltools.h"
#include "vvdynlib.h"
#include <assert.h>

#define CHECK( funcname ) \
	if ( !funcname ) { std::cerr << "#funcname() not initialized\n"; return; }

/** OpenGL Shading Language
 */

vvGLSL::vvGLSL()
: vvShaderManager()
, nTexture(0)
, _isSupported(false)
, _uniformParameters(NULL)
{
  glewInit();

  if (!vvGLTools::isGLextensionSupported("GL_ARB_fragment_shader")
          && !vvGLTools(isGLVersionSupported(2,0,0)))
  {
	const char* ext = (char*) glGetString(GL_EXTENSIONS);

	for (int i = 0; ext != NULL && ext[i] != 0; i++)
	  if (ext[i]==' ') printf("\n");
	  else
		printf("%c", ext[i]);

	std::cerr << "OpenGL Shading Language NOT supported!\n";

	return;
  }
  //if (!vvGLTools::isGLextensionSupported("GL_ARB_texture_non_power_of_two"))
  //	std::cerr << "Texture with non-power-of-two NOT supported!\n";

  if (!vvGLTools::isGLextensionSupported("GL_ARB_multitexture")
          && !vvGLTools::isGLVersionSupported(1,3,0))
	  std::cerr << "ARB multitexture NOT supported!\n";

  CHECK(glCreateShader);
  CHECK(glShaderSource);
  CHECK(glCompileShader);
  CHECK(glGetShaderiv);
  CHECK(glGetShaderInfoLog);
  CHECK(glCreateProgram);
  CHECK(glAttachShader);
  CHECK(glLinkProgram);
  CHECK(glUseProgram);
  CHECK(glGetUniformLocation);
  CHECK(glUniform1i);
  CHECK(glUniform2i);
  CHECK(glUniform3i);
  CHECK(glUniform4i);
  CHECK(glUniform1iv);
  CHECK(glUniform2iv);
  CHECK(glUniform3iv);
  CHECK(glUniform4iv);
  CHECK(glUniform1f);
  CHECK(glUniform2f);
  CHECK(glUniform3f);
  CHECK(glUniform4f);
  CHECK(glUniform1fv);
  CHECK(glUniform2fv);
  CHECK(glUniform3fv);
  CHECK(glUniform4fv);
  CHECK(glActiveTexture);
  CHECK(glDetachShader);
  CHECK(glDeleteShader);
  CHECK(glDeleteProgram);
  CHECK(glUseProgramObjectARB);

  _uniformParameters = NULL;

  _isSupported = true;
}

vvGLSL::~vvGLSL()
{
  if(_isSupported)
  {
    glUseProgram(0);
    for (int n = 0; n < programArray.count(); n++)
    {
      glDetachShader(programArray[n], fragShaderArray[n]);
      glDeleteShader(fragShaderArray[n]);
      glDeleteProgram(programArray[n]);
    }
  }
  delete[] _uniformParameters;
}

bool vvGLSL::loadShader(const char* shaderFileName, const ShaderType& shaderType)
{
  assert(shaderFileName != NULL);

  if(!_isSupported)
     return false;

  _shaderFileNames.push_back(shaderFileName);
  _shaderTypes.push_back(shaderType);

  const char* fileString = vvToolshed::file2string(shaderFileName);

  if(fileString == NULL)
  {
    cerr << "vvToolshed::file2string error for: " << shaderFileName << endl;
    return false;
  }

  bool ok = loadShaderByString(fileString, shaderType);

  delete[] fileString;

  return ok;
}

bool vvGLSL::loadShaderByString(const char* shaderString, const ShaderType& shaderType)
{
  if(!_isSupported)
     return false;

  GLuint fragShader;
  GLuint fragProgram;

  if(shaderString == NULL)
  {
    cerr << "Shader string is NULL" << endl;
    return false;
  }

  fragShader = glCreateShader(toGLenum(shaderType));
  fragProgram = glCreateProgram();

  fragShaderArray.append( fragShader );
  programArray.append( fragProgram );

  glShaderSource(fragShader, 1, &shaderString, NULL);
  glCompileShader(fragShader);

  GLint compiled;
  glGetShaderiv(fragShader, GL_COMPILE_STATUS, &compiled);

  if(!compiled)
  {
    GLint length;
    GLchar* compileLog;
    glGetShaderiv(fragShader, GL_INFO_LOG_LENGTH, &length);
    compileLog = new GLchar[length];
    glGetShaderInfoLog(fragShader, length, &length, compileLog);
    cerr << "glCompileShader failed: " << compileLog << endl;
    return false;
  }

  glAttachShader(fragProgram, fragShader);
  glLinkProgram(fragProgram);

  return true;
}

void vvGLSL::enableShader(const int index)
{
  glUseProgramObjectARB(getFragProgramHandle(index));
}

void vvGLSL::disableShader(const int)
{
  glUseProgramObjectARB(0);
}

void vvGLSL::initParameters(const int index,
                            const char** parameterNames,
                            const vvShaderParameterType*,  const int parameterCount)
{
  delete[] _uniformParameters;
  _uniformParameters = new GLint[parameterCount];

  for (int i = 0; i < parameterCount; ++i)
  {
    _uniformParameters[i] = glGetUniformLocation(getFragProgramHandle(index), parameterNames[i]);
  }
}

void vvGLSL::printCompatibilityInfo() const
{

}

void vvGLSL::setParameter1f(const int programIndex, const int parameterIndex,
                            const float& f1)
{
  (void)programIndex;
  glUniform1f(_uniformParameters[parameterIndex], f1);
}

void vvGLSL::setParameter3f(const int programIndex, const char* parameterName,
                            const float& f1, const float& f2, const float& f3)
{
  const GLint uniform = glGetUniformLocation(getFragProgramHandle(programIndex), parameterName);
  glUniform3f(uniform, f1, f2, f3);
}

void vvGLSL::setParameter3f(const int programIndex, const int parameterIndex,
                            const float& f1, const float& f2, const float& f3)
{
  (void)programIndex;
  glUniform3f(_uniformParameters[parameterIndex], f1, f2, f3);
}

void vvGLSL::setParameter4f(const int programIndex, const int parameterIndex,
                            const float& f1, const float& f2, const float& f3, const float& f4)
{
  (void)programIndex;
  glUniform4f(_uniformParameters[parameterIndex], f1, f2, f3, f4);
}

void vvGLSL::setParameter1i(const int programIndex, const int parameterIndex,
                            const int& i1)
{
  (void)programIndex;
  glUniform1i(_uniformParameters[parameterIndex], i1);
}

void vvGLSL::setArray3f(const int programIndex, const int parameterIndex, const float* array, const int count)
{
  (void)programIndex;
  glUniform3fv(_uniformParameters[parameterIndex], count, array);
}

void vvGLSL::setArray1i(const int programIndex, const int parameterIndex, const int* array, const int count)
{
  (void)programIndex;
  glUniform1iv(_uniformParameters[parameterIndex], count, array);
}

GLuint vvGLSL::getFragProgramHandle(const int i)
{
  return *programArray.get(i);
}

GLuint vvGLSL::getFragProgramHandleLast()
{
  return *programArray.get(programArray.count()-1);
}

void vvGLSL::useProgram(GLuint program)
{
  //std::cerr << "using shader program: " << program << endl;
  nTexture = 0;
  glUseProgram(program);
}

void vvGLSL::disable()
{
  glUseProgram(0);
  nTexture = 0;
}

void vvGLSL::deleteProgram(GLuint program)
{
  for(int n = 0; n < programArray.count(); n++)
  {
	if(programArray[n] == program)
	{
	  glDetachShader(programArray[n], fragShaderArray[n]);
	  glDeleteShader(fragShaderArray[n]);
	  glDeleteProgram(programArray[n]);

	  return;
	}
  }

  // error handling
  std::cerr << "vvGLSL::deleteProgram(): there is no fragment shader program referenced by program: " << program << endl;
}

void vvGLSL::setValue(GLuint program, const char* name, int length, float* value)
{
  GLuint loc = glGetUniformLocation(program, name);
  if(loc == -1)
  {
	std::cerr << "vvGLSL::setValue(): the name does not correspond to a uniform variable in the shader program: " << name << std::endl;
  }

  switch(length)
  {
	case 1:
	  glUniform1f(loc, *value);
	  break;
	case 2:
	  glUniform2f(loc, value[0], value[1]);
	  break;
	case 3:
	  glUniform3f(loc, value[0], value[1], value[2]);
	  break;
	case 4:
	  glUniform4f(loc, value[0], value[1], value[2], value[3]);
	  break;
	default:
	  std::cerr << "vvGLSL::setVector the length of a vector has to be less than or equal to 4" << std::endl;
	  break;
  }
}

void vvGLSL::setValue(GLuint program, const char* name, int length, int* value)
{
  GLuint loc = glGetUniformLocation(program, name);

  if(loc == -1)
  {
	std::cerr << "vvGLSL::setValue(): the name does not correspond to a uniform variable in the shader program(" << program << "): " << name << std::endl;
  }
  switch(length)
  {
	case 1:
	  glUniform1i(loc, *value);
	  break;
	case 2:
	  glUniform2i(loc, value[0], value[1]);
	  break;
	case 3:
	  glUniform3i(loc, value[0], value[1], value[2]);
	  break;
	case 4:
	  glUniform4i(loc, value[0], value[1], value[2], value[3]);
	  break;
	default:
	  std::cerr << "vvGLSL::setVector(): the length of a vector has to be less than or equal to 4" << std::endl;
	  break;
  }
}

void vvGLSL::setValue(GLuint program, const char* name, int length, GLsizei count, float* value)
{
  GLuint loc = glGetUniformLocation(program, name);
  if(loc == -1)
  {
	std::cerr << "vvGLSL::setValue(): the name does not correspond to a uniform variable in the shader program: " << name << std::endl;
  }

  switch(length)
  {
	case 1:
	  glUniform1fv(loc, count, value);
	  break;
	case 2:
	  glUniform2fv(loc, count, value);
	  break;
	case 3:
	  glUniform3fv(loc, count, value);
	  break;
	case 4:
	  glUniform4fv(loc, count, value);
	  break;
	default:
	  std::cerr << "vvGLSL::setVector(): the length of a vector has to be less than or equal to 4" << std::endl;
	  break;
  }
}
void vvGLSL::setValue(GLuint program, const char* name, int length, GLsizei count, int* value)
{
  GLuint loc = glGetUniformLocation(program, name);
  if(loc == -1)
  {
	std::cerr << "vvGLSL::setVector(): the name does not correspond to a uniform variable in the shader program:" << name << std::endl;
  }

  switch(length)
  {
	case 1:
	  glUniform1iv(loc, count, value);
	  break;
	case 2:
	  glUniform2iv(loc, count, value);
	  break;
	case 3:
	  glUniform3iv(loc, count, value);
	  break;
	case 4:
	  glUniform4iv(loc, count, value);
	  break;
	default:
	  std::cerr << "vvGLSL::setVector(): the length of a vector has to be less than or equal to 4" << std::endl;
	  break;
  }
}


void vvGLSL::initializeMultiTexture1D(GLuint program, const char* varName, GLuint texName)
{
  GLuint loc = glGetUniformLocation(program, varName);
  glActiveTexture(GL_TEXTURE0+nTexture);
  glUniform1i(loc, nTexture++);
  glEnable(GL_TEXTURE_1D);
  glBindTexture(GL_TEXTURE_1D, texName);
}

void vvGLSL::initializeMultiTexture2D(GLuint program, const char* varName, GLuint texName)
{
  GLuint loc = glGetUniformLocation(program, varName);
  glActiveTexture(GL_TEXTURE0+nTexture);
  //std::cerr << "activate texture #" << nTexture << ", " << texName << endl;
  glUniform1i(loc, nTexture++);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, texName);
}

void vvGLSL::initializeMultiTexture3D(GLuint program, const char* varName, GLuint texName)
{
  GLuint loc = glGetUniformLocation(program, varName);
  glActiveTexture(GL_TEXTURE0+nTexture);
  glUniform1i(loc, nTexture++);
  glEnable(GL_TEXTURE_3D_EXT);
  glBindTexture(GL_TEXTURE_3D_EXT, texName);
}

void vvGLSL::disableMultiTexture1D()
{
  for(int i = nTexture - 1; i >= 0; i--)
  {
	glActiveTexture(GL_TEXTURE0+i);
	glDisable(GL_TEXTURE_1D);
  }
  nTexture = 0;
}

void vvGLSL::disableMultiTexture2D()
{
  for(int i = nTexture - 1; i >= 0; i--)
  {
	glActiveTexture(GL_TEXTURE0+i);
	glDisable(GL_TEXTURE_2D);
  }
  nTexture = 0;
}

void vvGLSL::disableMultiTexture3D()
{
  for(int i = nTexture - 1; i >= 0; i--)
  {
	glActiveTexture(GL_TEXTURE0+i);
	glDisable(GL_TEXTURE_3D_EXT);
  }
  nTexture = 0;
}

GLenum vvGLSL::toGLenum(const ShaderType& shaderType)
{
  GLenum result;
  switch (shaderType)
  {
  case VV_FRAG_SHD:
    result = GL_FRAGMENT_SHADER;
    break;
#if 0 // TODO - find out if geometry shaders are supported.
  case VV_GEOM_SHD:
    result = GL_GEOMETRY_SHADER_EXT;
    break;
#endif
  case VV_VERT_SHD:
    result = GL_VERTEX_SHADER;
    break;
  default:
    result = GL_FRAGMENT_SHADER;
    break;
  }
  return result;
}

//==================================================================
// old code
//==================================================================

#if 0
void vvGLSL::loadShader()
{
	const char *filename[] = {"glsl_1chan.frag", "glsl_2chan.frag", "glsl_3chan.frag", "glsl_multichan.frag"};

	for (unsigned int n = 0; n < sizeof(filename)/sizeof(char*); n++)
	{
		fragshader.append( glCreateShader(GL_FRAGMENT_SHADER) );
		program.append( glCreateProgram() );
		const char *filestring = vvToolshed::file2string(filename[n]);

		glShaderSource(fragshader[n], 1, &filestring, NULL);
		glCompileShader(fragshader[n]);
		glAttachShader(program[n], fragshader[n]);
		glLinkProgram(program[n]);

		//printf("%s\n", filestring);
		delete[] filestring;
	}

	//int loc = glGetUniformLocation(shader,"numchan");
	//glUniform1i(loc,6);
}




void vvGLSL::enableLUTMode(GLuint tfTex[], GLuint gl3dTex[],  float volweight, int numchan, float w[], vvVector3 color[], int alphaMode)
{
	//glUseProgram(0);
	//return;

	int n = (numchan < program.count())? numchan-1 : program.count()-1;
	//int loc = glGetUniformLocation(program[n], "numchan");
	//glUniform1i(loc, numchan);
	GLuint loc = 0;

	glUseProgram(program[n]);

	int i = 0;
	if (numchan == 1)
	{
		loc = glGetUniformLocation(program[n], "tfTex0");
		glActiveTexture(GL_TEXTURE0+i);
		glUniform1i(loc, i++);
		glBindTexture(GL_TEXTURE_2D, tfTex[0]);
	}
	else
	{
		loc = glGetUniformLocation(program[n], "tfTex[0]");
		glActiveTexture(GL_TEXTURE0+i);
		glUniform1i(loc, i++);
		glBindTexture(GL_TEXTURE_1D, tfTex[numchan]);

		for (int c = 0; c < numchan; c++)
		{
			char varName[20];
			sprintf(varName, "tfTex[%i]", c+1);
			loc = glGetUniformLocation(program[n], varName);
			glActiveTexture(GL_TEXTURE0+i);
			glUniform1i(loc, i++);
			glBindTexture(GL_TEXTURE_1D, tfTex[c]);
		}
	}
	
	for (int c = 0; c < numchan; c++)
	{
		char varName[20];
		sprintf(varName, "gl3dTex%i", c);
		loc = glGetUniformLocation(program[n], varName);
		glActiveTexture(GL_TEXTURE0+i);
		glUniform1i(loc, i++);
		glBindTexture(GL_TEXTURE_3D_EXT, gl3dTex[c]);
	}

	// hardware problem with uninitialized TU
	if (numchan >= program.count())
	for (int c = numchan; c < 7; c++)
	{
		char varName[20];
		sprintf(varName, "gl3dTex%i", c);
		loc = glGetUniformLocation(program[n], varName);
		glActiveTexture(GL_TEXTURE0+i);
		glUniform1i(loc, i++);
		glBindTexture(GL_TEXTURE_3D_EXT, 0);
	}

	float chanWeight[10];
	assert ( numchan  <= 10 );

	float maxWeight = 0.0f, sumWeight = 0.0f;
	// channel weights
	for (int c = 0; c < numchan; c++)
	{
		chanWeight[c] =  w[c] * volweight;
		maxWeight = max(chanWeight[c], maxWeight);
		sumWeight += chanWeight[c];
	}

	loc = glGetUniformLocation(program[n], "weight");
	glUniform1fv(loc, numchan, chanWeight);


	loc = glGetUniformLocation(program[n], "color");
	for (int c = 0; c < numchan; c++)
		glUniform3fv(loc+c, 1, color[c].e);

	loc = glGetUniformLocation(program[n], "numChan");
	glUniform1i(loc, numchan);

	loc = glGetUniformLocation(program[n], "normAlpha");
	alphaMode == 0? glUniform1f(loc, maxWeight): glUniform1f(loc, sumWeight);

	loc = glGetUniformLocation(program[n], "alphaMode");
	glUniform1i(loc, alphaMode);
}
#endif




