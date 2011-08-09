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
