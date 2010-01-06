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

#include "vvshadermanager.h"

vvShaderManager::vvShaderManager()
{
  init();
}

vvShaderManager::~vvShaderManager()
{

}

void vvShaderManager::printCompatibilityInfo()
{
  throw "Function printCompatibilityInfo not implemented by inherited class";
}

const char* vvShaderManager::getShaderDir()
{
  throw "Function getShaderDir not implemented by inherited class";
}

void vvShaderManager::enableTexture(const int, const char*)
{
  throw "Function setParameter1f not implemented by inherited class";
}

void vvShaderManager::disableTexture(const int, const char*)
{
  throw "Function setParameter1f not implemented by inherited class";
}

void vvShaderManager::setParameter1f(const int, const char*,
                                     const float&)
{
  throw "Function setParameter1f not implemented by inherited class";
}

void vvShaderManager::setParameter2f(const int, const char*,
                                     const float&, const float&)
{
  throw "Function setParameter2f not implemented by inherited class";
}

void vvShaderManager::setParameter3f(const int, const char*,
                                     const float&, const float&, const float&)
{
  throw "Function setParameter3f not implemented by inherited class";
}

void vvShaderManager::setParameter4f(const int, const char*,
                                     const float&, const float&, const float&, const float&)
{
  throw "Function setParameter4f not implemented by inherited class";
}

void vvShaderManager::setParameter1i(const int, const char*,
                                     const int&)
{
  throw "Function setParameter1i not implemented by inherited class";
}

void vvShaderManager::setParameter2i(const int, const char*,
                                     const int&, const int&)
{
  throw "Function setParameter2i not implemented by inherited class";
}

void vvShaderManager::setParameter3i(const int, const char*,
                                     const int&, const int&, const int&)
{
  throw "Function setParameter3i not implemented by inherited class";
}

void vvShaderManager::setParameter4i(const int, const char*,
                                     const int&, const int&, const int&, const int&)
{
  throw "Function setParameter4i not implemented by inherited class";
}

void vvShaderManager::setParameter1fv(const int, const char*,
                                      const float*&)
{
  throw "Function setParameter1fv not implemented by inherited class";
}

void vvShaderManager::setParameter2fv(const int, const char*,
                                      const float*&)
{
  throw "Function setParameter2fv not implemented by inherited class";
}

void vvShaderManager::setParameter3fv(const int, const char*,
                                      const float*&)
{
  throw "Function setParameter3fv not implemented by inherited class";
}

void vvShaderManager::setParameter4fv(const int, const char*,
                                      const float*&)
{
  throw "Function setParameter4fv not implemented by inherited class";
}

void vvShaderManager::setParameter1iv(const int, const char*,
                                      const int*&)
{
  throw "Function setParameter1iv not implemented by inherited class";
}

void vvShaderManager::setParameter2iv(const int, const char*,
                                      const int*&)
{
  throw "Function setParameter2iv not implemented by inherited class";
}

void vvShaderManager::setParameter3iv(const int, const char*,
                                      const int*&)
{
  throw "Function setParameter3iv not implemented by inherited class";
}

void vvShaderManager::setParameter4iv(const int, const char*,
                                      const int*&)
{
  throw "Function setParameter4iv not implemented by inherited class";
}

void vvShaderManager::setArrayParameter1f(const int, const char*, const int,
                                          const float&)
{
  throw "Function setArrayParameter1f not implemented by inherited class";
}

void vvShaderManager::setArrayParameter2f(const int, const char*, const int,
                                          const float&, const float&)
{
  throw "Function setArrayParameter2f not implemented by inherited class";
}

void vvShaderManager::setArrayParameter3f(const int, const char*, const int,
                                          const float&, const float&, const float&)
{
  throw "Function setArrayParameter3f not implemented by inherited class";
}

void vvShaderManager::setArrayParameter4f(const int, const char*, const int,
                                          const float&, const float&, const float&, const float&)
{
  throw "Function setArrayParameter4f not implemented by inherited class";
}

void vvShaderManager::setArrayParameter1i(const int, const char*, const int,
                                          const int&)
{
  throw "Function setArrayParameter1i not implemented by inherited class";
}

void vvShaderManager::setArrayParameter2i(const int, const char*, const int,
                                          const int&, const int&)
{
  throw "Function setArrayParameter2i not implemented by inherited class";
}

void vvShaderManager::setArrayParameter3i(const int, const char*, const int,
                                          const int&, const int&, const int&)
{
  throw "Function setArrayParameter3i not implemented by inherited class";
}

void vvShaderManager::setArrayParameter4i(const int, const char*, const int,
                                          const int&, const int&, const int&, const int&)
{
  throw "Function setArrayParameter4i not implemented by inherited class";
}

void vvShaderManager::setParameterTexId(const int, const char*, const unsigned int&)
{
  throw "Function setParameterTexId not implemented by inherited class";
}

void vvShaderManager::setModelViewProj(int, const char*)
{
  throw "Function setModelViewProj not implemented by inherited class";
}

void vvShaderManager::init()
{

}
