// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
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
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef _VV_SHADERFACTORY_H_
#define _VV_SHADERFACTORY_H_

#include "vvshaderprogram.h"
#include "vvexport.h"

/** Class to create complete shader programs with all related shaders

  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
 */
class VIRVOEXPORT vvShaderFactory
{
public:
  /// Constructor initiating factory and call glewInit()
  vvShaderFactory();

  /** Create Program and try to attach shaders with given name
    \param name name contained in all shaders names with standard pattern
    \note the prefix, suffix, extensions etc are added to the names internally
    */
  vvShaderProgram* createProgram(const std::string& name);
    /** Create Program and try to attach shaders with givens names
    \param vert name of vertex shader
    \param geom name of geometry shader
    \param frag name of fragment shader
    \note the prefix, suffix, extensions etc are added to the names internally
    */
  vvShaderProgram* createProgram(const std::string& vert, const std::string& geom, const std::string& frag);

  bool cgSupport();     ///< returns if CG is supported
  bool glslSupport();   ///< returns if GLSL is supported

private:
  const std::string getShaderDir();
  bool loadFileStrings();

  bool _cgSupport;
  std::string _shaderName[3];
  std::string _fileString[3];
  std::string _shaderDir;
};

#endif // _VV_SHADERFACTORY2_H_

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
