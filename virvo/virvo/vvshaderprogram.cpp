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

#include "vvshaderprogram.h"

#include <string>

using std::string;

vvShaderProgram::vvShaderProgram(const string& vert, const string& geom, const string& frag)
{
  _shadersLoaded = false;
  _fileStrings[0] = vert;
  _fileStrings[1] = geom;
  _fileStrings[2] = frag;
}

vvShaderProgram::~vvShaderProgram()
{
}

bool vvShaderProgram::isValid() const
{
  return _shadersLoaded;
}

//============================================================================
// End of File
//============================================================================
