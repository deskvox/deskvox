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

#ifndef _VV_SHADERFACTORY_H_
#define _VV_SHADERFACTORY_H_

#include "vvexport.h"
#include "vvshadermanager.h"

enum vvShaderManagerType
{
  VV_CG_MANAGER = 0,
  VV_GLSL_MANAGER
};

/*!
 * \brief           Class creating shader managers using the factory pattern.
 *
 *                  In the best case, renderers and applications using shaders
 *                  have no knowledge of what kind of shader they are using.
 *                  Shaders are encapsulating by shader managers (\see
 *                  vvShaderManager) which administer multiple shader programs.
 *                  If one wants to use a not yet implemented shading language,
 *                  one simply has to implement a new shader manager. After that,
 *                  one will have to provide a new literal to the \ref vvShaderManagerType
 *                  enumeration describing his shader manager, as well as to
 *                  extend the \ref provideShaderManager() function of this
 *                  class to return an instance of his class given the enum
 *                  literal. That way, renderers only have to communicate
 *                  with the factory using the enum to obtain the desired
 *                  shader, without knowledge of internals of the specific shader
 *                  class.
 */
class VIRVOEXPORT vvShaderFactory
{
public:
  /*!
   * \brief         Given a desired type, return the appropriate shader manager.
   *
   *                Factory function returning the shader manager desired by
   *                evaluating the specified type.
   * \param         shaderManagerType The desired shader manager type.
   * \return        A shader manager instance or NULL.
   */
  static vvShaderManager* provideShaderManager(const vvShaderManagerType& shaderManagerType);
};

#endif // _VV_SHADERFACTORY_H_
