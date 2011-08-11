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

#ifndef _VV_CGPROGRAM_H_
#define _VV_CGPROGRAM_H_

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include "vvexport.h"
#include "vvshaderprogram.h"

#include <map>
#include <vector>
#include <iostream>

#ifdef HAVE_CG

#include <Cg/cg.h>
#include <Cg/cgGL.h>

using std::string;

/** Wrapper Class for Cg Shading Language
  This class loads a combination of up to three shaders
  (vertex-, geometry- and fragment-shaders) and
  manages all interaction with it.

  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
 */

class VIRVOEXPORT vvCgProgram : public vvShaderProgram
{
public:
  vvCgProgram();  /// trivial constructor

  /**
    Creates a vvCgProgram and tries to attach the given shaders source codes.
    @param vert Filestring of desired vertex-shader or emtpy/NULL.
    @param geom Filestring of desired geometry-shader or emtpy/NULL.
    @param frag Filestring of desired fragment-shader or emtpy/NULL.
   */
  vvCgProgram(const string& vert, const string& geom, const string& frag);

  /// Deactivates and deletes shader program that was generated in this class
  ~vvCgProgram();

  void enableProgram();   /// enables program with loaded shaders
  void disableProgram();  /// disables program with its shaders

  /**
    Set uniform parameter functions. Use parameters' names only.
    Parameters' ids are checked and connected between programs automatically.
   */
  void setParameter1f(const string& parameterName, const float& f1);
  void setParameter1i(const string& parameterName, const int& i1);

  void setParameter3f(const string& parameterName, const float* array);
  void setParameter3f(const string& parameterName,
                              const float& f1, const float& f2, const float& f3);

  void setParameter4f(const string& parameterName, const float* array);
  void setParameter4f(const string& parameterName,
                              const float& f1, const float& f2, const float& f3, const float& f4);

  void setParameterArray1i(const string& parameterName, const int* array, const int& count);

  void setParameterArray3f(const string& parameterName, const float* array, const int& count);

  void setMatrix4f(const string& parameterName, const float* mat);

  void setParameterTexId(const string& parameterName, const unsigned int& ui1);
  void setTextureId(const string& parameterName, const unsigned int& ui1);
  void enableTexture(const string& parameterName);
  void disableTexture(const string& parameterName);

private:
  bool loadShaders();                               /// Creates CgProgram and loads shaders into it.
  bool initParameter(const string& parameterName);  /// Looks for a parameter and connects it with all other active shaders

  static void cgErrorHandler(CGcontext context, CGerror error, void*);
  CGGLenum toCgEnum(const int i) const;

  // CG data
  CGcontext _program;     /// Id of the CgProgram (in Cg: context)
  CGprofile _profile[3];  /// Cg-specific profile-ids
  CGprogram _shaderId[3]; /// Shader-ids

  std::map<string, CGparameter> _cgParameterNameMaps;  /// Parameter name-id maps
};

#endif // HAVE_CG

#endif // _VV_CGPGROGRAM_H_

//============================================================================
// End of File
//============================================================================

