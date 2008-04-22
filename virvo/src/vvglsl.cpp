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

#include <iostream>
#include "vvtoolshed.h"
#include "vvglsl.h"
#include "vvgltools.h"
#include <assert.h>

#ifdef WIN32
#define DYNAMIC_BIND_NAME( funcname , type ) \
funcname = (type) wglGetProcAddress(#funcname); \
	if ( ! funcname ) std::cerr << "#funcname() not initialized\n";
#else
#include <GL/glx.h>
#define DYNAMIC_BIND_NAME( funcname , type ) \
funcname = (type) glXGetProcAddressARB((GLubyte*) #funcname); \
	if ( ! funcname ) std::cerr << "#funcname() not initialized\n";
#endif

/** OpenGL Shading Language
 */

vvGLSL::vvGLSL() 
{
	if (!vvGLTools::isGLextensionSupported("GL_ARB_fragment_shader"))
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

	if (!vvGLTools::isGLextensionSupported("GL_ARB_multitexture"))
		std::cerr << "ARB multitexture NOT supported!\n";

	DYNAMIC_BIND_NAME(glCreateShader, PFNGLCREATESHADERPROC );
	DYNAMIC_BIND_NAME(glShaderSource, PFNGLSHADERSOURCEPROC );
	DYNAMIC_BIND_NAME(glCompileShader, PFNGLCOMPILESHADERPROC);
	DYNAMIC_BIND_NAME(glCreateProgram, PFNGLCREATEPROGRAMPROC);
	DYNAMIC_BIND_NAME(glAttachShader, PFNGLATTACHSHADERPROC);
	DYNAMIC_BIND_NAME(glLinkProgram, PFNGLLINKPROGRAMPROC);
	DYNAMIC_BIND_NAME(glUseProgram, PFNGLUSEPROGRAMPROC);
	DYNAMIC_BIND_NAME(glGetUniformLocation, PFNGLGETUNIFORMLOCATIONPROC);
	DYNAMIC_BIND_NAME(glUniform1i, PFNGLUNIFORM1IPROC);
	DYNAMIC_BIND_NAME(glUniform1iv, PFNGLUNIFORM1IVPROC);
	DYNAMIC_BIND_NAME(glUniform1f, PFNGLUNIFORM1FPROC);
	DYNAMIC_BIND_NAME(glUniform1fv, PFNGLUNIFORM1FVPROC);
	DYNAMIC_BIND_NAME(glUniform3fv, PFNGLUNIFORM3FVPROC);
	DYNAMIC_BIND_NAME(glActiveTexture, PFNGLACTIVETEXTUREPROC);
	DYNAMIC_BIND_NAME(glDetachShader, PFNGLDETACHSHADERPROC);
	DYNAMIC_BIND_NAME(glDeleteShader, PFNGLDELETESHADERPROC );
	DYNAMIC_BIND_NAME(glDeleteProgram, PFNGLDELETEPROGRAMPROC );
}

vvGLSL::~vvGLSL()
{
	glUseProgram(0);
	for (int n = 0; n < program.count(); n++)
	{
		glDetachShader(program[n], fragshader[n]);

		glDeleteShader(program[n]);
		glDeleteProgram(fragshader[n]);
	}
}

void vvGLSL::loadShader()
{
	char *filename[] = {"glsl_1chan.frag", "glsl_2chan.frag", "glsl_3chan.frag", "glsl_multichan.frag"};

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

void vvGLSL::disable()
{
	glUseProgram(0);
}
