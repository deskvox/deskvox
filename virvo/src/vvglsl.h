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

#ifndef _VV_GLSL_H_
#define _VV_GLSL_H_

#include "vvopengl.h"
#include "vvglext.h"
#include "vvvecmath.h"
#include "vvarray.h"

/** OpenGL Shading Language
 */

class VIRVOEXPORT vvGLSL
{
public:
	vvGLSL();
	~vvGLSL();
	void loadShader();
	void enableLUTMode(GLuint tfTex[], GLuint gl3dTex[], float volweight, int numchan, float w[], vvVector3 color[], int alphaMode);
	void disable();
private:
	vvArray<GLuint> fragshader;
	vvArray<GLuint> program;

	PFNGLCREATESHADERPROC glCreateShader;
	PFNGLSHADERSOURCEPROC glShaderSource ;
	PFNGLCOMPILESHADERPROC glCompileShader ;
	PFNGLCREATEPROGRAMPROC glCreateProgram;
	PFNGLATTACHSHADERPROC glAttachShader;
	PFNGLLINKPROGRAMPROC glLinkProgram;
	PFNGLUSEPROGRAMPROC glUseProgram;
	PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
	PFNGLUNIFORM1IPROC glUniform1i;
	PFNGLUNIFORM1IVPROC glUniform1iv;
	PFNGLUNIFORM1FPROC glUniform1f;
	PFNGLUNIFORM1FVPROC glUniform1fv;
	PFNGLUNIFORM3FVPROC glUniform3fv;
	PFNGLACTIVETEXTUREPROC glActiveTexture;
	PFNGLDETACHSHADERPROC glDetachShader;
	PFNGLDELETESHADERPROC glDeleteShader;
	PFNGLDELETEPROGRAMPROC glDeleteProgram;
};
#endif
