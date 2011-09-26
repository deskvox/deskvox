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

#ifdef HAVE_CG

#include <Cg/cg.h>
#include <Cg/cgGL.h>

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
  vvCgProgram(const std::string& vert, const std::string& geom, const std::string& frag);

  /// Deactivates and deletes shader program that was generated in this class
  ~vvCgProgram();

  void enable();   /// enables program with loaded shaders
  void disable();  /// disables program with its shaders

  /**
    Set uniform parameter functions. Use parameters' names only.
    Parameters' ids are checked and connected between programs automatically.
   */
  void setParameter1f(const std::string& parameterName, const float& f1);
  void setParameter1i(const std::string& parameterName, const int& i1);

  void setParameter3f(const std::string& parameterName, const float* array);
  void setParameter3f(const std::string& parameterName,
                              const float& f1, const float& f2, const float& f3);

  void setParameter4f(const std::string& parameterName, const float* array);
  void setParameter4f(const std::string& parameterName,
                              const float& f1, const float& f2, const float& f3, const float& f4);

  void setParameterArray1i(const std::string& parameterName, const int* array, const int& count);

  void setParameterArray3f(const std::string& parameterName, const float* array, const int& count);

  void setParameterMatrix4f(const std::string& parameterName, const float* mat);

  void setParameterTex1D(const std::string& parameterName, const unsigned int& ui);
  void setParameterTex2D(const std::string& parameterName, const unsigned int& ui);
  void setParameterTex3D(const std::string& parameterName, const unsigned int& ui);

private:
  typedef std::map<std::string, CGparameter> ParameterMap;
  typedef ParameterMap::iterator ParameterIterator;

  bool loadShaders();                               /// Creates CgProgram and loads shaders into it.
  //! Looks for a parameter and connects it with all other active shaders
  ParameterIterator initParameter(const std::string& parameterName);

  static void cgErrorHandler(CGcontext context, CGerror error, void*);
  CGGLenum toCgEnum(const int i) const;

  // CG data
  CGcontext _program;     /// Id of the CgProgram (in Cg: context)
  CGprofile _profile[3];  /// Cg-specific profile-ids
  CGprogram _shaderId[3]; /// Shader-ids

  ParameterMap _cgParameterNameMaps;  /// Parameter name-id maps
};

#endif // HAVE_CG

#endif // _VV_CGPGROGRAM_H_

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
