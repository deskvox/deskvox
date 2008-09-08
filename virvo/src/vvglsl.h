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

//============================================================================
// Class Definition
//============================================================================

/** Wrapper Class for OpenGL Fragment Shading Language
  This class is a wrapper for fragment shading program APIs.
  With this class, one can initialize multiple fragment shader programs 
  and copy data in OpenGL applications to data structure accessible 
  in fragment shader programs.

  TODO: 
  There is no support for vertex shader programs such as attribute variables.
  Moreover, the APIs supported in this class is currently very limited.

  @author Han S Kim (hskim@cs.ucsd.edu)
  @author Chih Liang (cliang@cs.ucsd.edu)
  @author Jurgen P. Schulze (jschulze@ucsd.edu)
 */
class VIRVOEXPORT vvGLSL
{
  public:

	/** Creates a vvGLSL. Function pointers for shader programs are initialized
	*/
	vvGLSL();

	/** Deactivates and deletes shader programs that were generated and stored in this class
	*/
	~vvGLSL();

	/** Initializes, compiles, and links a shader program

	  @param shaderFileName The name of a fragment program source file
	  @return non-zero program ID if OK, 0 otherwise
	*/
	GLuint loadShader(const char* shaderFileName);

	/** Calls glUseProgram

	  @param program The handle of the program object whose executables are to be used as part of current rendering state.
	*/
	void useProgram(GLuint program);

	/** Calls glUseProgram(0);
	*/
	void disable();

	/** Deletes the fragment shader program referenced by <i>program</i>.

	  @param program the fragment shader program ID to be deleted.
	*/
	void deleteProgram(GLuint program);


	/** Copies one value to the fragment shader program specified by program.

	  @param program Fragment program id returned by loadShader()
	  @param name Variable name in fragment shader program
	  @param length Type of data structure, between 1 (scalar) and 4 (4D vector)
	  @param value Pointer to an array of floats/integers
	*/
	void setValue(GLuint program, const char* name, int length, float* value);
	void setValue(GLuint program, const char* name, int length, int* value);

	/** Copies an array of values to the fragment shader program specified by program.

	  @param program Fragment program ID returned by loadShader()
	  @param name Variable name in this fragment shader program
	  @param length Type of data structure, between 1 (scalar) and 4 (4D vector)
	  @param count The number of elements to be copied
	  @param value Pointer to an array of floats/integers
	*/
	void setValue(GLuint program, const char* name, int length, GLsizei count, float* value);
	void setValue(GLuint program, const char* name, int length, GLsizei count, int* value);

	/** Reset the number of activated textures
	*/
	void resetTextureCount() { nTexture = 0; }

	/** Activates a texture and binds a texture specified by texId
	  In order to add multiple textures, first you have to reset texture count by calling resetTextureCount().
	  Then subsequent initializeMultiTextureXD() calls bind each texture with unique texture, GL_TEXTUREX, 
	  where X denotes the texture number.

	  @param program Fragment program ID returned by loadShader()
	  @param name Uniform variable defined in this fragment shader program
	  @param texId Texture name
	*/
	void initializeMultiTexture1D(GLuint program, const char* name, GLuint texId);
	void initializeMultiTexture2D(GLuint program, const char* name, GLuint texId);
	void initializeMultiTexture3D(GLuint program, const char* name, GLuint texId);

	/** Disables multi-textures initialized by initializeMultiTextureXD()
	  This method needs to be called whenever the multiple textures are no longer used.
	*/
	void disableMultiTexture1D();
	void disableMultiTexture2D();
	void disableMultiTexture3D();

	int getProgramCount() { return programArray.count(); }

	/* OBSOLETE
	void loadShader();
	void enableLUTMode(GLuint tfTex[], GLuint gl3dTex[], float volweight, int numchan, float w[], vvVector3 color[], int alphaMode);
	*/

  private:
	vvArray<GLuint> fragShaderArray;			///< array of fragment shader IDs
	vvArray<GLuint> programArray;				///< array of fragment program IDs
	int nTexture;								///< the number of texture activated

	// function pointers
	PFNGLCREATESHADERPROC glCreateShader;
	PFNGLSHADERSOURCEPROC glShaderSource ;
	PFNGLCOMPILESHADERPROC glCompileShader ;
	PFNGLGETSHADERIVPROC glGetShaderiv ;
	PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog ;
	PFNGLCREATEPROGRAMPROC glCreateProgram;
	PFNGLATTACHSHADERPROC glAttachShader;
	PFNGLLINKPROGRAMPROC glLinkProgram;
	PFNGLUSEPROGRAMPROC glUseProgram;
	PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
	PFNGLUNIFORM1IPROC glUniform1i;
	PFNGLUNIFORM2IPROC glUniform2i;
	PFNGLUNIFORM3IPROC glUniform3i;
	PFNGLUNIFORM4IPROC glUniform4i;
	PFNGLUNIFORM1IVPROC glUniform1iv;
	PFNGLUNIFORM2IVPROC glUniform2iv;
	PFNGLUNIFORM3IVPROC glUniform3iv;
	PFNGLUNIFORM4IVPROC glUniform4iv;
	PFNGLUNIFORM1FPROC glUniform1f;
	PFNGLUNIFORM2FPROC glUniform2f;
	PFNGLUNIFORM3FPROC glUniform3f;
	PFNGLUNIFORM4FPROC glUniform4f;
	PFNGLUNIFORM1FVPROC glUniform1fv;
	PFNGLUNIFORM2FVPROC glUniform2fv;
	PFNGLUNIFORM3FVPROC glUniform3fv;
	PFNGLUNIFORM4FVPROC glUniform4fv;
	PFNGLACTIVETEXTUREPROC glActiveTexture;
	PFNGLDETACHSHADERPROC glDetachShader;
	PFNGLDELETESHADERPROC glDeleteShader;
	PFNGLDELETEPROGRAMPROC glDeleteProgram;
};
#endif

//============================================================================
// End of File
//============================================================================

