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

#ifndef _VV_GLSLPROGRAM_H_
#define _VV_GLSLPROGRAM_H_

#include "vvvecmath.h"
#include "vvshaderprogram.h"
#include "vvglew.h"

#include <vector>

using std::string;

/** Wrapper Class for OpenGL Shading Language
  This class loads a combination of up to three shaders
  (vertex-, geometry- and fragment-shaders) and
  manages all interaction with it.

  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
 */
class VIRVOEXPORT vvGLSLProgram : public vvShaderProgram
{
public:

  /** Creates a vvGLSLProgram and tries to attach the given shaders source code.
    @param vert Filestring of desired vertex-shader or emtpy/NULL.
    @param geom Filestring of desired geometry-shader or emtpy/NULL.
    @param frag Filestring of desired fragment-shader or emtpy/NULL.
   */
  vvGLSLProgram(const string& vert, const string& geom, const string& frag);

  /// Deactivates and deletes shader program that was generated in this class
  ~vvGLSLProgram();

  void enableProgram();   /// enables program with loaded shaders
  void disableProgram();  /// disables program with its shaders

  /**
    Set uniform parameter functions. Use parameters' names only.
    Parameters' ids are checked and connected between programs automatically.
   */
  void setParameter1f(const string& parameterName, const float& f1);
  void setParameter1i(const string& parameterName, const int& i1);

  void setParameter3f(const string& parameterName,
                              const float& f1, const float& f2, const float& f3);
  void setParameter4f(const string& parameterName,
                              const float& f1, const float& f2, const float& f3, const float& f4);

  void setParameterArray3f(const string& parameterName, const float* array);
  void setParameterArrayf(const string& parameterName, const float* array, const int& count);
  void setParameterArrayi(const string& parameterName, const int* array, const int& count);

  void setMatrix4f(const string& parameterName, const float* mat);

  void setTextureId(const string& parameterName, const unsigned int& ui1) {};
  void enableTexture(const string& parameterName) {};
  void disableTexture(const string& parameterName) {};

private:
  bool loadShaders();     /// Initializes, compiles, and links a shader program
  void deleteProgram();   /// deletes program with all shaders and frees memory

  GLuint _programId;
  GLuint _shaderId[3];
  int  _nTexture;         ///< the number of texture activated
  bool _isSupported;      ///< true if there is GLSL support

};
#endif // _VV_GLSLPROGRAM_H_

//============================================================================
// End of File
//============================================================================
