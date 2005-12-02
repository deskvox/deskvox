// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, schulze@cs.brown.edu
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

#include <math.h>
#include "vvtfwidget.h"
#include "vvtoolshed.h"

//============================================================================
// Class vvTFWidget and related classes
//============================================================================

vvColor::vvColor()
{
  for (int i=0; i<3; ++i) _col[i] = 1.0f;         // default color is white to be visible on typically black screens
}

vvColor::vvColor(float r, float g, float b)
{
  _col[0] = r;
  _col[1] = g;
  _col[2] = b;
}

/// Overload RHS subscription operator.
const float vvColor::operator[](const int index) const
{
  return _col[index];
}

/// Overload LHS subscription operator.
float& vvColor::operator[](const int index)
{
  return _col[index];
}

/** Add two colors by using the maximum intensity of each channel.
 */
vvColor vvColor::operator+ (const vvColor operand) const
{
  vvColor tmp;
  int i;

  for (i=0; i<3; ++i)
  {
    tmp._col[i] = ts_max(this->_col[i], operand._col[i]);
  }
  return tmp;
}

/** Sets the color according to the RGB color model.
  @param r,g,b color components [0..1]. Any component that is negative remains the same.
*/
void vvColor::setRGB(float r, float g, float b)
{
  if (r>=0.0f) _col[0] = r;
  if (g>=0.0f) _col[1] = g;
  if (b>=0.0f) _col[2] = b;
}

/** Sets the color according to the HSB color model.
  @param h,s,b color components [0..1]. Any component that is negative remains the same.
*/
void vvColor::setHSB(float h, float s, float b)
{
  float hOld, sOld, bOld;
  vvToolshed::RGBtoHSB(_col[0], _col[1], _col[2], &hOld, &sOld, &bOld);
  if (h<0.0f) h = hOld;
  if (s<0.0f) s = sOld;
  if (b<0.0f) b = bOld;
  vvToolshed::HSBtoRGB(h, s, b, &_col[0], &_col[1], &_col[2]);
}

void vvColor::getRGB(float& r, float& g, float& b)
{
  r = _col[0];
  g = _col[1];
  b = _col[2];
}

void vvColor::getHSB(float& h, float& s, float& b)
{
  vvToolshed::RGBtoHSB(_col[0], _col[1], _col[2], &h, &s, &b);
}

//============================================================================

const char* vvTFWidget::NO_NAME = "UNNAMED";

vvTFWidget::vvTFWidget()
{
  int i;

  _name = NULL;
  for (i=0; i<3; ++i)
  {
    _col[i] = 1.0f;                               // default is white
    _pos[i] = 0.5f;                               // default is center of TF space
  }
}

vvTFWidget::vvTFWidget(vvTFWidget* src)
{
  int i;

  _name = NULL;
  setName(src->_name);
  for (i=0; i<3; ++i)
  {
    _col[i] = src->_col[i];
    _pos[i] = src->_pos[i];
  }
}

vvTFWidget::vvTFWidget(vvColor col, float x, float y, float z)
{
  _name = NULL;
  _col = col;
  _pos[0] = x;
  _pos[1] = y;
  _pos[2] = z;
}

vvTFWidget::~vvTFWidget()
{
  delete[] _name;
}

void vvTFWidget::setName(const char* newName)
{
  delete[] _name;
  if (newName)
  {
    _name = new char[strlen(newName) + 1];
    strcpy(_name, newName);
  }
  else _name = NULL;
}

const char* vvTFWidget::getName()
{
  return _name;
}

void vvTFWidget::readName(FILE* fp)
{
  char tmpName[128];
  fscanf(fp, " %s", tmpName);
  setName(tmpName);
}

float vvTFWidget::getOpacity(float, float, float)
{
  return 0.0f;
}

bool vvTFWidget::getColor(vvColor&, float, float, float)
{
  return false;
}

//============================================================================

vvTFBell::vvTFBell() : vvTFWidget()
{
  int i;

  _opacity = 1.0f;
  _ownColor = true;
  for (i=0; i<3; ++i)
  {
    _size[i] = 0.2f;                              // default with: smaller rather than bigger
  }
}

vvTFBell::vvTFBell(vvTFBell* src) : vvTFWidget(src)
{
  int i;

  _opacity = src->_opacity;
  _ownColor = src->_ownColor;
  for (i=0; i<3; ++i)
  {
    _size[i] = src->_size[i];
  }
}

vvTFBell::vvTFBell(vvColor col, bool ownColor, float opacity,
float x, float w, float y, float h, float z, float d) : vvTFWidget(col, x, y, z)
{
  _ownColor = ownColor;
  _opacity = opacity;
  _size[0] = w;
  _size[1] = h;
  _size[2] = d;
}

vvTFBell::vvTFBell(FILE* fp) : vvTFWidget()
{
  readName(fp);
  fscanf(fp, " %g %g %g %g %g %g %g %g %g %d %g\n",
    &_pos[0], &_pos[1], &_pos[2], &_size[0], &_size[1], &_size[2],
    &_col[0], &_col[1], &_col[2], &_ownColor, &_opacity);
}

void vvTFBell::write(FILE* fp)
{
  fprintf(fp, "TF_BELL %s %g %g %g %g %g %g %g %g %g %d %g\n", (_name) ? _name : NO_NAME,
    _pos[0], _pos[1], _pos[2], _size[0], _size[1], _size[2],
    _col[0], _col[1], _col[2], (_ownColor) ? 1 : 0, _opacity);
}

/** The 2D gaussian function can be found at:
  http://mathworld.wolfram.com/GaussianFunction.html
*/
float vvTFBell::getOpacity(float x, float y, float z)
{
  const float WIDTH_ADJUST = 5.0f;
  const float HEIGHT_ADJUST = 0.1f;
  float stdev[3];                                 // standard deviation in x,y,z
  float exponent = 0.0f;
  float factor = 1.0f;
  float opacity;
  float bMin[3], bMax[3];                         // widget boundary
  float p[3];                                     // sample point position
  float sqrt2pi;                                  // square root of 2 PI
  int dim;
  int i;

  p[0] = x;
  p[1] = y;
  p[2] = z;

  // Determine dimensionality of transfer function:
  dim = 1;
  if (z>-1.0f) dim = 3;
  else if (y>-1.0f) dim = 2;

  // Determine widget boundaries:
  for (i=0; i<dim; ++i)
  {
    bMin[i] = _pos[i] - _size[i] / 2.0f;
    bMax[i] = _pos[i] + _size[i] / 2.0f;
  }

  // First find out if point lies within bell:
  if (x < bMin[0] || x > bMax[0] ||
    (dim>1 && (y < bMin[1] || y > bMax[1])) ||
    (dim>2 && (z < bMin[2] || z > bMax[2])))
  {
    return 0.0f;
  }

  for (i=0; i<dim; ++i)
  {
    stdev[i] = _size[i] / WIDTH_ADJUST;
  }

  sqrt2pi = sqrtf(2.0f * TS_PI);
  for (i=0; i<dim; ++i)
  {
    exponent += (p[i] - _pos[0]) * (p[i] - _pos[i]) / (2.0f * stdev[i] * stdev[i]);
    factor *= sqrt2pi * stdev[i];
  }
  opacity = ts_min(HEIGHT_ADJUST * _opacity * expf(-exponent) / factor, 1.0f);
  return opacity;
}

/** @return true if coordinate is within bell
  @param color returned color value
  @param x,y,z probe coordinates
*/
bool vvTFBell::getColor(vvColor& color, float x, float y, float z)
{
  float bMin[3], bMax[3];                         // widget boundary
  int i, dim;

  // Determine dimensionality of transfer function:
  dim = 1;
  if (z>-1.0f) dim = 3;
  else if (y>-1.0f) dim = 2;

  // Determine widget boundaries:
  for (i=0; i<dim; ++i)
  {
    bMin[i] = _pos[i] - _size[i] / 2.0f;
    bMax[i] = _pos[i] + _size[i] / 2.0f;
  }

  // First find out if point lies within bell:
  if (x < bMin[0] || x > bMax[0] ||
    (dim>1 && (y < bMin[1] || y > bMax[1])) ||
    (dim>2 && (z < bMin[2] || z > bMax[2])))
  {
    return false;
  }

  color = _col;
  return true;
}

bool vvTFBell::hasOwnColor()
{
  return _ownColor;
}

void vvTFBell::setOwnColor(bool own)
{
  _ownColor = own;
}

//============================================================================

vvTFPyramid::vvTFPyramid() : vvTFWidget()
{
  int i;

  _opacity = 1.0f;
  _ownColor = true;
  for (i=0; i<3; ++i)
  {
    _top[i] = 0.2f;                               // default with: smaller rather than bigger
    _bottom[i] = 0.4f;
  }
}

vvTFPyramid::vvTFPyramid(vvTFPyramid* src) : vvTFWidget(src)
{
  int i;

  _opacity = src->_opacity;
  _ownColor = src->_ownColor;
  for (i=0; i<3; ++i)
  {
    _top[i] = src->_top[i];
    _bottom[i] = src->_bottom[i];
  }
}

vvTFPyramid::vvTFPyramid(vvColor col, bool ownColor, float opacity, float x, float wb, float wt,
float y, float hb, float ht, float z, float db, float dt) : vvTFWidget(col, x, y, z)
{
  _ownColor = ownColor;
  _opacity = opacity;
  _top[0]    = wt;
  _bottom[0] = wb;
  _top[1]    = ht;
  _bottom[1] = hb;
  _top[2]    = dt;
  _bottom[2] = db;
}

vvTFPyramid::vvTFPyramid(FILE* fp) : vvTFWidget()
{
  readName(fp);
  fscanf(fp, " %g %g %g %g %g %g %g %g %g %g %g %g %d %g\n",
    &_pos[0], &_pos[1], &_pos[2], &_bottom[0], &_bottom[1], &_bottom[2],
    &_top[0], &_top[1], &_top[2], &_col[0], &_col[1], &_col[2], &_ownColor, &_opacity);
}

void vvTFPyramid::write(FILE* fp)
{
  fprintf(fp, "TF_PYRAMID %s %g %g %g %g %g %g %g %g %g %g %g %g %d %g\n", (_name) ? _name : NO_NAME,
    _pos[0], _pos[1], _pos[2], _bottom[0], _bottom[1], _bottom[2],
    _top[0], _top[1], _top[2], _col[0], _col[1], _col[2], (_ownColor) ? 1 : 0, _opacity);
}

float vvTFPyramid::getOpacity(float x, float y, float z)
{
  float p[3];                                     // sample point position
  float outMin[3], outMax[3];                     // outer boundaries of pyramid
  float inMin[3], inMax[3];                       // inner boundaries of pyramid
  int dim, i;

  p[0] = x;
  p[1] = y;
  p[2] = z;

  // Determine dimensionality of transfer function:
  dim = 1;
  if (z>-1.0f) dim = 3;
  else if (y>-1.0f) dim = 2;

  // Calculate inner and outer boundaries of pyramid:
  for (i=0; i<dim; ++i)
  {
    outMin[i] = _pos[i] - _bottom[i] / 2.0f;
    outMax[i] = _pos[i] + _bottom[i] / 2.0f;
    inMin[i]  = _pos[i] - _top[i] / 2.0f;
    inMax[i]  = _pos[i] + _top[i] / 2.0f;
  }

  // First find out if point lies within pyramid:
  if (x < outMin[0] || x > outMax[0] ||
    (dim>1 && (y < outMin[1] || y > outMax[1])) ||
    (dim>2 && (z < outMin[2] || z > outMax[2])))
  {
    return 0.0f;
  }

  // Now check if point is within the pyramid's plateau:
  if (x >= inMin[0] && x <= inMax[0])
  {
    if (dim<2) return _opacity;
    if (y >= inMin[1] && y <= inMax[1])
    {
      if (dim<3) return _opacity;
      if (z >= inMin[2] && z <= inMax[2]) return _opacity;
    }
    else return _opacity;
  }

  // Now it's clear that the point is on one of the pyramid's flanks, so
  // we interpolate the value along the flank:
  switch (dim)
  {
    case 1:
      if (p[0] < inMin[0]) return vvToolshed::interpolateLinear(outMin[0], 0.0f, inMin[0], _opacity, p[0]);
      if (p[0] > inMax[0]) return vvToolshed::interpolateLinear(inMax[0], _opacity, outMax[0], 0.0f, p[0]);
      break;
    case 2:                                       // TODO: use diagonal lines
      if (p[0] < inMin[0]) return vvToolshed::interpolateLinear(outMin[0], 0.0f, inMin[0], _opacity, p[0]);
      if (p[0] > inMax[0]) return vvToolshed::interpolateLinear(inMax[0], _opacity, outMax[0], 0.0f, p[0]);
      if (p[1] < inMin[1]) return vvToolshed::interpolateLinear(outMin[1], 0.0f, inMin[1], _opacity, p[1]);
      if (p[1] > inMax[1]) return vvToolshed::interpolateLinear(inMax[1], _opacity, outMax[1], 0.0f, p[1]);
      break;
    case 3:
      break;
    default: assert(0); break;
  }
  return 0.0f;
}

/** @return true if coordinate is within pyramid
  @param color returned color value
  @param x,y,z probe coordinates
*/
bool vvTFPyramid::getColor(vvColor& color, float x, float y, float z)
{
  float outMin[3], outMax[3];                     // outer boundaries of pyramid
  int i, dim;

  // Determine dimensionality of transfer function:
  dim = 1;
  if (z>-1.0f) dim = 3;
  else if (y>-1.0f) dim = 2;

  // Calculate inner and outer boundaries of pyramid:
  for (i=0; i<dim; ++i)
  {
    outMin[i] = _pos[i] - _bottom[i] / 2.0f;
    outMax[i] = _pos[i] + _bottom[i] / 2.0f;
  }

  // First find out if point lies within pyramid:
  if (x < outMin[0] || x > outMax[0] ||
    (dim>1 && (y < outMin[1] || y > outMax[1])) ||
    (dim>2 && (z < outMin[2] || z > outMax[2])))
  {
    return false;
  }
  color = _col;
  return true;
}

bool vvTFPyramid::hasOwnColor()
{
  return _ownColor;
}

void vvTFPyramid::setOwnColor(bool own)
{
  _ownColor = own;
}

//============================================================================

vvTFColor::vvTFColor() : vvTFWidget()
{
}

vvTFColor::vvTFColor(vvTFColor* src) : vvTFWidget(src)
{
}

vvTFColor::vvTFColor(vvColor col, float x, float y, float z) : vvTFWidget(col, x, y, z)
{
}

vvTFColor::vvTFColor(FILE* fp) : vvTFWidget()
{
  readName(fp);
  fscanf(fp, " %g %g %g %g %g %g\n", &_pos[0], &_pos[1], &_pos[2], &_col[0], &_col[1], &_col[2]);
}

void vvTFColor::write(FILE* fp)
{
  bool running=true;

  fprintf(fp, "TF_COLOR %s %g %g %g %g %g %g\n", (_name) ? _name : NO_NAME,
    _pos[0], _pos[1], _pos[2], _col[0], _col[1], _col[2]);
}

//============================================================================
// End of File
//============================================================================
