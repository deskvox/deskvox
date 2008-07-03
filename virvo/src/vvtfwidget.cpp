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

#include <math.h>
#include "vvtfwidget.h"
#include "vvtoolshed.h"

//============================================================================
// Class vvTFWidget and related classes
//============================================================================

vvTFPoint::vvTFPoint()
{
  for (int i=0; i<3; ++i)
  {
    _pos[i] = 0.0f;
  }
  _opacity = 1.0f;
}

vvTFPoint::vvTFPoint(float op, float x, float y, float z)
{
  _pos[0] = x;
  _pos[1] = y;
  _pos[2] = z;
  _opacity = op;
}

//============================================================================

const char* vvTFWidget::NO_NAME = "UNNAMED";

/** Default constructor.
*/
vvTFWidget::vvTFWidget()
{
  int i;

  _name = NULL;
  for (i=0; i<3; ++i)
  {
    _pos[i] = 0.5f;                               // default is center of TF space
  }
}

/** Copy constructor.
*/
vvTFWidget::vvTFWidget(vvTFWidget* src)
{
  int i;

  _name = NULL;
  setName(src->_name);
  for (i=0; i<3; ++i)
  {
    _pos[i] = src->_pos[i];
  }
}

/** Constructor with parameter initialization.
*/
vvTFWidget::vvTFWidget(float x, float y, float z)
{
  _name = NULL;
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
    _col[i] = 1.0f;                               // default is white
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
    _col[i] = src->_col[i];
    _size[i] = src->_size[i];
  }
}

vvTFBell::vvTFBell(vvColor col, bool ownColor, float opacity,
float x, float w, float y, float h, float z, float d) : vvTFWidget(x, y, z)
{
  _col = col;
  _ownColor = ownColor;
  _opacity = opacity;
  _size[0] = w;
  _size[1] = h;
  _size[2] = d;
}

vvTFBell::vvTFBell(FILE* fp) : vvTFWidget()
{
  readName(fp);
  int ownColorInt;
  fscanf(fp, " %g %g %g %g %g %g %g %g %g %d %g\n",
    &_pos[0], &_pos[1], &_pos[2], &_size[0], &_size[1], &_size[2],
    &_col[0], &_col[1], &_col[2], &ownColorInt, &_opacity);
  _ownColor = ownColorInt;
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
    exponent += (p[i] - _pos[i]) * (p[i] - _pos[i]) / (2.0f * stdev[i] * stdev[i]);
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
    _col[i] = 1.0f;                               // default is white
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
    _col[i] = src->_col[i];
    _top[i] = src->_top[i];
    _bottom[i] = src->_bottom[i];
  }
}

vvTFPyramid::vvTFPyramid(vvColor col, bool ownColor, float opacity, float x, float wb, float wt,
float y, float hb, float ht, float z, float db, float dt) : vvTFWidget(x, y, z)
{
  _col = col;
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
  int ownColorInt;
  readName(fp);
  fscanf(fp, " %g %g %g %g %g %g %g %g %g %g %g %g %d %g\n",
    &_pos[0], &_pos[1], &_pos[2], &_bottom[0], &_bottom[1], &_bottom[2],
    &_top[0], &_top[1], &_top[2], &_col[0], &_col[1], &_col[2], &ownColorInt, &_opacity);
  _ownColor = ownColorInt;
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
    //else return _opacity;
  }

  // Now it's clear that the point is on one of the pyramid's flanks, so
  // we interpolate the value along the flank:
  switch (dim)
  {
    case 1:
      if (p[0] < inMin[0]) return vvToolshed::interpolateLinear(outMin[0], 0.0f, inMin[0], _opacity, p[0]);
      if (p[0] > inMax[0]) return vvToolshed::interpolateLinear(inMax[0], _opacity, outMax[0], 0.0f, p[0]);
      break;
    case 2:   
       {
          // new: uses bilinear interpolation
          float x1, y1, x2, y2;
          float r2, val;

          //outMin[i] = _pos[i] - _bottom[i] / 2.0f;
          //outMax[i] = _pos[i] + _bottom[i] / 2.0f;
          //inMin[i]  = _pos[i] - _top[i] / 2.0f;
          //inMax[i]  = _pos[i] + _top[i] / 2.0f;

          // decide for x
          if (p[0] > _pos[0])
          {
             if (p[0] > inMax[0]) 
             {
                x1 = inMax[0];
                x2 = outMax[0]; 
                r2 = (x2 - p[0]) / (x2 - x1);
             }
             else
             {
                r2 = 1.0f;
             }
          }
          else
          {
             if (p[0] < inMin[0]) 
             {
                x1 = inMin[0];
                x2 = outMin[0];  
                r2 = (x2 - p[0]) / (x2 - x1);
             }
             else
             {
                r2 = 1.0f;
             }
          }

          //y
          if (p[1] > _pos[1])
          {
             if (p[1] > inMax[1])
             {
               y1 = outMax[1];
               y2 = inMax[1];

               val = ((p[1] - y1)/(y2 - y1)) * r2;
             }
             else
             {
                val = r2;
             }
          }
          else
          {
             if (p[1] < inMin[1])
             {
               y1 = outMin[1];
               y2 = inMin[1];

               val = ((p[1] - y1)/(y2 - y1)) * r2;
             }
             else
             {
                val = r2;
             }
          }
          return val;          
       }
      break;
    case 3:
      // TODO: implement for 3D pyramids
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
  int i;
  
  for(i=0; i<3; ++i)
  {
    _col[i] = 1.0f;                               // default is white
  }
}

vvTFColor::vvTFColor(vvTFColor* src) : vvTFWidget(src)
{
  int i;
  
  for(i=0; i<3; ++i)
  {
    _col[i] = src->_col[i];
  }
}

vvTFColor::vvTFColor(vvColor col, float x, float y, float z) : vvTFWidget(x, y, z)
{
  _col = col;
}

vvTFColor::vvTFColor(FILE* fp) : vvTFWidget()
{
  readName(fp);
  fscanf(fp, " %g %g %g %g %g %g\n", &_pos[0], &_pos[1], &_pos[2], &_col[0], &_col[1], &_col[2]);
}

void vvTFColor::write(FILE* fp)
{
  fprintf(fp, "TF_COLOR %s %g %g %g %g %g %g\n", (_name) ? _name : NO_NAME,
    _pos[0], _pos[1], _pos[2], _col[0], _col[1], _col[2]);
}

//============================================================================

/** Default constructor.
  The skip widget defines a value range in which the voxels should always be transparent.
*/
vvTFSkip::vvTFSkip() : vvTFWidget()
{
  int i;
  
  for (i=0; i<3; ++i)
  {
    _size[i] = 0.0f;
  }
}

/** Copy constructor
*/
vvTFSkip::vvTFSkip(vvTFSkip* src) : vvTFWidget(src)
{
  int i;
  
  for (i=0; i<3; ++i)
  {
    _size[i] = src->_size[i];
  }
}

/** Constructor with parameter initialization.
  @param xpos,ypos,zpos position of center of skipped area
  @param xsize,ysize,zsize width, height, depth of skipped area
*/
vvTFSkip::vvTFSkip(float xpos, float xsize, float ypos, float ysize, float zpos, float zsize) : vvTFWidget(xpos, ypos, zpos)
{
  _size[0] = xsize;
  _size[1] = ysize;
  _size[2] = zsize;
}

vvTFSkip::vvTFSkip(FILE* fp) : vvTFWidget()
{
  readName(fp);
  fscanf(fp, " %g %g %g %g %g %g\n", &_pos[0], &_pos[1], &_pos[2], &_size[0], &_size[1], &_size[2]);
}

void vvTFSkip::write(FILE* fp)
{
  fprintf(fp, "TF_SKIP %s %g %g %g %g %g %g\n", (_name) ? _name : NO_NAME,
    _pos[0], _pos[1], _pos[2], _size[0], _size[1], _size[2]);
}

/** @return 0 if x/y/z point is within skipped area, otherwise -1
*/
float vvTFSkip::getOpacity(float x, float y, float z)
{
  float _min[3];
  float _max[3];
  int i, dim;
  
  for (i=0; i<3; ++i)
  {
    _min[i] = _pos[i] - _size[i] / 2.0f;
    _max[i] = _pos[i] + _size[i] / 2.0f;
  }
  
  // Determine dimensionality of transfer function:
  dim = 1;
  if (z>-1.0f) dim = 3;
  else if (y>-1.0f) dim = 2;

  // Now find out if point lies within skip area:
  if (x>=_min[0] && x<=_max[0] &&
      (dim<2 || (y>=_min[1] && y<=_max[1])) &&
      (dim<3 || (z>=_min[2] && z<=_max[2]))) return 0.0f;
  else return -1.0f;
}

//============================================================================

/** Default constructor.
  The custom widget defines an area where users can define a TF with 
  a series of control points.
*/
vvTFCustom::vvTFCustom() : vvTFWidget()
{
  int i;
  
  for (i=0; i<3; ++i)
  {
    _size[i] = 0.0f;
  }
  _currentPoint = NULL;
}

/** Copy constructor
*/
vvTFCustom::vvTFCustom(vvTFCustom* src) : vvTFWidget(src)
{
   int i;

   for (i=0; i<3; ++i)
   {
      _size[i] = src->_size[i];
   }

   // deep copy the list too!
   _currentPoint = NULL;

   list<vvTFPoint*>::iterator iter;
   for(iter = src->_points.begin(); iter != src->_points.end(); iter++) 
   {
      vvTFPoint* newPoint = new vvTFPoint((*iter)->_opacity, 
         (*iter)->_pos[0], (*iter)->_pos[1], (*iter)->_pos[2]);

      // check for the current point
      if ((*iter) == src->_currentPoint)
         _currentPoint = newPoint;

      this->_points.push_back(newPoint);
   }
}

/** Constructor with parameter initialization.
  @param xpos,ypos,zpos position of center of control point area
  @param xsize,ysize,zsize width, height, depth of control point area
*/
vvTFCustom::vvTFCustom(float xpos, float xsize, float ypos, float ysize, float zpos, float zsize) : vvTFWidget(xpos, ypos, zpos)
{
  _size[0] = xsize;
  _size[1] = ysize;
  _size[2] = zsize;
  _currentPoint = NULL;
}

/** Constructor reading parameters from file.
*/
vvTFCustom::vvTFCustom(FILE* fp) : vvTFWidget()
{
  list<vvTFPoint*>::iterator iter;
  vvTFPoint* point;
  float op, x, y, z;
  int numPoints;
  int i;
  
  readName(fp);
  fscanf(fp, " %d %g %g %g\n", &numPoints, &_size[0], &_size[1], &_size[2]);

  for(i=0; i<numPoints; ++i) 
  {
    fscanf(fp, "%g %g %g %g\n", &op, &x, &y, &z);
    point = new vvTFPoint(op, x, y, z);
    _points.push_back(point);
  }  
  _currentPoint = NULL;
}

// Destructor
vvTFCustom::~vvTFCustom()
{
  list<vvTFPoint*>::iterator iter;

  // Remove point instances from list:
  for(iter=_points.begin(); iter!=_points.end(); iter++) 
  {
    delete *iter;
  }
  _points.clear();
}

/** Write class parameters to file.
*/
void vvTFCustom::write(FILE* fp)
{
  list<vvTFPoint*>::iterator iter;

  fprintf(fp, "TF_CUSTOM %s %g %g %g %d\n", (_name) ? _name : NO_NAME,
    _size[0], _size[1], _size[2], (int)_points.size());

  for(iter=_points.begin(); iter!=_points.end(); iter++) 
  {
    fprintf(fp, "%g %g %g %g\n", (*iter)->_opacity, (*iter)->_pos[0], (*iter)->_pos[1], (*iter)->_pos[2]);
  }  
}

/** @return opacity of a value in the TF, as defined by this widget
  @param x,y,z point in TF space, not widget space [0..1]
*/
float vvTFCustom::getOpacity(float x, float y, float z)
{
  list<vvTFPoint*>::iterator iter;
  vvTFPoint* prev;    // previous point in list
  float _min[3];
  float _max[3];
  int i, dim;
  float xTF;    // x position in widget space
  
  for (i=0; i<3; ++i)
  {
    _min[i] = _pos[i] - _size[i] / 2.0f;
    _max[i] = _pos[i] + _size[i] / 2.0f;
  }
  
  // Determine dimensionality of transfer function:
  dim = 1;
  if (z>-1.0f) dim = 3;
  else if (y>-1.0f) dim = 2;

  // Now find out if point lies within defined area:
  if (x>=_min[0] && x<=_max[0] &&
      (dim<2 || (y>=_min[1] && y<=_max[1])) &&
      (dim<3 || (z>=_min[2] && z<=_max[2])))
  {
    xTF = x - _pos[0]; // transform x to widget space
    prev = NULL;
    if (_points.size()==0) return 0.0f; // no control points specified
    for(iter=_points.begin(); iter!=_points.end(); iter++) 
    {
      if (xTF < (*iter)->_pos[0])
      {
        if (iter==_points.begin())  // x is between left edge and first control point
        {
          return vvToolshed::interpolateLinear(-_size[0]/2.0f, 0.0f, (*iter)->_pos[0], (*iter)->_opacity, xTF);
        }
        else  // x is between two control points (iter and iter-1)
        {
          return vvToolshed::interpolateLinear(prev->_pos[0], prev->_opacity, (*iter)->_pos[0], (*iter)->_opacity, xTF);
        }
      }
      prev = *iter;
    }
    // x is between last control point and the right edge
    return vvToolshed::interpolateLinear(prev->_pos[0], prev->_opacity, _size[0]/2.0f, 0.0f, xTF);
  }
  else return -1.0f;
}

/** Add a new control point, use opacity of desired pin position.
  @param x,y,z position of new point, relative to global transfer function space
  @return pointer to newly added point
*/
vvTFPoint* vvTFCustom::addPoint(float x, float y, float z)
{
  list<vvTFPoint*>::iterator iter;
  vvTFPoint* newPoint;
  
  newPoint = new vvTFPoint(getOpacity(x, y, z), x - _pos[0], y - _pos[1], z - _pos[2]);
  _points.push_back(newPoint);
  sortPoints();
  return newPoint;  
}

/** Remove current control point from list.
*/
void vvTFCustom::removeCurrentPoint()
{
  if (_currentPoint)
  {
    _points.remove(_currentPoint);
    _currentPoint = NULL;
  }
}

/** Set current point to the one closest to x/y/z.
  @param o opacity [0..1]
  @param ot opacity tolerance radius [0..1]
  @param x,y,z coordinates in TF space, not widget space [0..1]
  @param xt,yt,zt x/y/z tolerance radii [0..1]
  @return pointer to new current point, or NULL if no point within tolerance
*/
vvTFPoint* vvTFCustom::selectPoint(float o, float ot, float x, float xt, float y, float yt, float z, float zt)
{
  (void)y;
  (void)yt;
  (void)z;
  (void)zt;

  float global[3];  // global x/y/z coordinates of control point
  list<vvTFPoint*>::iterator iter;
  int i;
  
  for(iter=_points.begin(); iter!=_points.end(); iter++) 
  {
    for (i=0; i<3; ++i)
    {
      global[i] = _pos[i] + (*iter)->_pos[i];
    }
    if (fabs(global[0] - x) <= xt && 
        fabs((*iter)->_opacity - o) <= ot) 
    {
      _currentPoint = *iter;
      return *iter;
    }
  }
  return NULL;
}

/** Set position and opacity of current point.
  @param opacity [0..1]
  @param x,y,z coordinates in TF space, not widget space
*/
void vvTFCustom::setCurrentPoint(float opacity, float x, float y, float z)
{
  (void)opacity;

  if (_currentPoint)
  {
    if (x!=-1.0f) _currentPoint->_pos[0] = x - _pos[0];
    if (y!=-1.0f) _currentPoint->_pos[1] = y - _pos[1];
    if (z!=-1.0f) _currentPoint->_pos[2] = z - _pos[2];
  }
  sortPoints();
}

/** Move position and change opacity of current point.
  @param opacity [0..1]
  @param x,y,z coordinates in TF space, not widget space
*/
void vvTFCustom::moveCurrentPoint(float dop, float dx, float dy, float dz)
{
  int i;
  
  if (_currentPoint)
  {
    _currentPoint->_pos[0]  += dx;
    _currentPoint->_pos[1]  += dy;
    _currentPoint->_pos[2]  += dz;
    _currentPoint->_opacity += dop;
    
    // Constrain point position to limits:
    _currentPoint->_opacity = ts_clamp(_currentPoint->_opacity, 0.0f, 1.0f);
    for (i=0; i<3; ++i)
    {
      if (_size[i] > 0.0f)
      {
        _currentPoint->_pos[i] = ts_clamp(_currentPoint->_pos[i], -_size[i] / 2.0f, _size[i] / 2.0f);
      }
    }
    sortPoints();
  }
}

/** Sort control points to make x positions ascending.
*/
void vvTFCustom::sortPoints()
{
  list<vvTFPoint*>::iterator iter;
  vvTFPoint* prev;
  vvTFPoint* point;
  bool done=false;
  
  if (_points.size()<=1) return; // no sorting necessary  
  while (!done)
  {
    prev = NULL;
    for(iter=_points.begin(); iter!=_points.end(); iter++) 
    {
      if (prev==NULL) 
      {
        prev = *iter;
        continue;
      }
      if ((*iter)->_pos[0] < prev->_pos[0]) 
      {
        point = prev;
        _points.remove(prev);
        _points.push_back(point);
        break;
      }
      prev = *iter;
    }
    if (iter==_points.end()) done = true;
  }
}

/** Set widget size.
*/
void vvTFCustom::setSize(float x, float y, float z)
{
  list<vvTFPoint*>::iterator iter;
  int i;

  if (x!=-1.0f) _size[0] = x;
  if (y!=-1.0f) _size[1] = y;
  if (z!=-1.0f) _size[2] = z;
  
  for(iter=_points.begin(); iter!=_points.end(); iter++) 
  {
    for (i=0; i<3; ++i)
    {
      if (_size[i]!=-1.0f)
      {
        (*iter)->_pos[i] = ts_clamp((*iter)->_pos[i], -_size[i]/2.0f, _size[i]/2.0f);
      }
    }
  }
}


// 
// free draw support
//
template<class T>
inline T sqr(T a) { return a * a; }

template<class T>
inline T linearInterpolation(T x1, T y1, T x2, T y2, T x)
{
   if (x1 == x2) return std::max(y1, y2);              // on equal x values: return maximum value
   if (x1 > x2)                                    // make x1 less than x2
   {
      std::swap(x1, x2);
      std::swap(y1, y2);
   }
   return (y2 - y1) * (x - x1) / (x2 - x1) + y1;
}


//
//
//
vvTFCustom2D::vvTFCustom2D(bool extrude, float opacity, float xCenter, float yCenter)
{
   this->_ownColor = false;
   this->_mapDirty = true;
   this->_map = NULL;
   this->_size[0] = 0.0f;
   this->_size[1] = 0.0f;

   this->_extrude = extrude;
   this->_opacity = opacity;
   this->_centralPoint = new vvTFPoint(xCenter, yCenter);
   this->_centralPoint->_opacity = this->_opacity;
}

vvTFCustom2D::vvTFCustom2D(vvTFCustom2D* other) : vvTFWidget(other)
{
   this->_ownColor = other->_ownColor;
   if (_ownColor)
   {
      this->_col._col[0] = other->_col._col[0];
      this->_col._col[1] = other->_col._col[1];
      this->_col._col[2] = other->_col._col[2];
   }

   this->_opacity = other->_opacity;
   this->_extrude = other->_extrude;
   
   this->_centralPoint = new vvTFPoint(this->_centralPoint->_pos[0], this->_centralPoint->_pos[1]);
   this->_centralPoint->_opacity = this->_opacity;
   
   this->_mapDirty = true;
   this->_map = NULL;
   this->_size[0] = 0.0f;
   this->_size[1] = 0.0f;

   list<vvTFPoint*>::iterator it;
   for (it = other->_points.begin(); it != other->_points.end(); ++it)
   {
      vvTFPoint* p = new vvTFPoint((**it)._pos[0], (**it)._pos[1], (**it)._pos[2]);
      p->_opacity = (**it)._opacity;
      this->_points.push_back(p);
   }
}


vvTFCustom2D::~vvTFCustom2D()
{
   if (_centralPoint)
      delete _centralPoint;

   list<vvTFPoint*>::iterator it;
   for (it = _points.begin(); it != _points.end(); ++it)
   {
      delete (*it);
   }

   if (_map != NULL)
      delete[] _map;
}

void vvTFCustom2D::write(FILE*)
{
   //TODO!!
}

float vvTFCustom2D::getOpacity(float x, float y, float)
{
   if (_mapDirty) 
   {
      //points were added, or position changed.

      //recompute size
      this->_size[0] = 0.0f;
      this->_size[1] = 0.0f;
      list<vvTFPoint*>::iterator it;
      for (it = _points.begin(); it != _points.end(); ++it)
      {
         _size[0] = std::max(_size[0], (**it)._pos[0]);
         _size[1] = std::max(_size[1], (**it)._pos[1]);
      }

      //recompute the map
      if (_map != NULL)
         delete[] _map;

      int xDim = int(this->_size[0] * 255) + 1;
      int yDim = int(this->_size[1] * 255) + 1;
      _map = new float[xDim * yDim];

      drawFreeContour();
      uniformFillFreeArea();

      _mapDirty = false;
   }

   int xDim = int(this->_size[0] * 255);
   int idx = int(y * 255) * xDim + int(x * 255);
   return (float)(_map[idx]) / 255.0f;
}

vvTFPoint* vvTFCustom2D::addPoint(float opacity, float x, float y)
{
   vvTFPoint* newPoint;

   newPoint = new vvTFPoint(opacity, x, y);
   _points.push_back(newPoint);

   _mapDirty = true;

   return newPoint;  
}

void vvTFCustom2D::addPoint(vvTFPoint* newPoint)
{
   _points.push_back(newPoint);
   _mapDirty = true;
}

void vvTFCustom2D::addMapPoint(int x, int y, float value)
{
   int xDim = int(this->_size[0] * 255);
   int idx = y * xDim + x;
   _map[idx] = value; 
}

void vvTFCustom2D::midPointLine(float* /*map*/, int x0, int y0, int x1, int y1, float alpha0, float alpha1)
{  
   int dx=x1-x0;
   int dy=y1-y0;
   
   int xIncr, yIncr, adx, ady;

   if (dy < 0)
   {
      ady = -dy;
      yIncr = -1;
   }
   else
   {
      ady = dy;
      yIncr = 1;
   }

   if (dx < 0)
   {
      adx = -dx;
      xIncr = -1;
   }
   else
   {
      adx = dx;
      xIncr = 1;
   }


   int x=x0;
   int y=y0;

   //alpha will be interpolated
   float dist = sqrt(float(sqr(x0 + x1) + sqr(y0 + y1)));
   float da = (alpha1 - alpha0) / dist;
   float value = alpha0;

   //handle special case: vertical lines
   if (dx == 0) // m = inf
   {
      addMapPoint(x, y, value);
      if (dy > 0)
      {
         while (y < y1)
         {
            ++y;
            value = value + da;
            addMapPoint(x, y, value);            
         }
      }
      else
      {
         while (y > y1)
         {
            --y;
            value += da;
            addMapPoint(x, y, value);
         }
      }
   }
   else if (adx > ady) // m > 1 or m < -1
   {
      int d = 2 * ady - adx;
      int incrE = 2 * ady;
      int incrNE = 2 * (ady - adx);

      addMapPoint(x, y, value);
      if (dx > 0)
      {
         while(x < x1)
         {
            if(d <= 0)
            {
               d += incrE;
               ++x;
            }
            else
            {
               d += incrNE;
               ++x;
               y += yIncr;
            }         
            value += da;
            addMapPoint(x, y, value);            
         }
      }
      else
      {
         while(x > x1)
         {
            if(d >= 0)
            {
               d -= incrE;
               --x;
            }
            else
            {
               d -= incrNE;
               --x;
               y += yIncr;
            }         
            value += da;
            addMapPoint(x, y, value);            
         }
      }
   }
   else //if (dx < dy)
   {
      int d = 2 * adx - ady;
      int incrE = 2 * adx;
      int incrNE = 2 * (adx-ady);

      addMapPoint(x, y, value);
      if (dy > 0)
      {
         while(y < y1)
         {
            if(d <= 0)
            {
               d += incrE;
               ++y;
            }
            else
            {
               d += incrNE;
               ++y;
               x += xIncr;
            }         
            value += da;
            addMapPoint(x, y, value);
         }
      }
      else
      {
         while(y > y1)
         {
            if(d >= 0)
            {
               d -= incrE;
               --y;
            }
            else
            {
               d -= incrNE;
               --y;
               x += xIncr;
            }         
            value += da;
            addMapPoint(x, y, value);
         }
      }
   }
}

void vvTFCustom2D::internalFloodFill(float* map, int x, int y, int xDim, int yDim, float oldV, float newV)
{
   int fillL, fillR, i;
   int in_line = 1;
   //unsigned char c = src_c, fillC = dst_c;

   /* find left side, filling along the way */
   fillL = fillR = x;
   while( in_line )
   {
      map[y * xDim + fillL] = newV;
      fillL--;
      in_line = (fillL < 0) ? 0 : (map[y * xDim + fillL] == oldV);
   }
   fillL++;
   /* find right side, filling along the way */
   in_line = 1;
   while( in_line )
   {
      map[y * xDim + fillR] = newV;
      fillR++;
      in_line = (fillR > yDim-1) ? 0 : (map[y * xDim + fillR] == oldV);
   }
   fillR--;
   /* search top and bottom */
   for(i = fillL; i <= fillR; i++)
   {
      if( y > 0 && map[(y - 1) * xDim + i] == oldV)
         internalFloodFill(map, i, y - 1, xDim, yDim, oldV, newV);
      if( y < xDim-1 && map[(y + 1) * xDim + i] == oldV )
         internalFloodFill(map, i, y + 1, xDim, yDim, oldV, newV);
   }
}

void vvTFCustom2D::uniformFillFreeArea()
{
   int x = int(_centralPoint->_pos[0] * 255);
   int y = int(_centralPoint->_pos[1] * 255); 
   int xDim = int(_size[0] * 255);  
   int yDim = int(_size[1] * 255);
   internalFloodFill(_map, x, y, xDim, yDim, _map[y * xDim + x], _opacity);
}

void vvTFCustom2D::drawFreeContour()
{
   int xDim = int(this->_size[0] * 255) + 1;
   int yDim = int(this->_size[1] * 255) + 1;
   // firts, clear map
   memset((void*)_map, 0, xDim * yDim * sizeof(float));

   // write to map the boundary points
   if (_points.size() > 1)
   {
      list<vvTFPoint*>::iterator prev_it = _points.begin();
      list<vvTFPoint*>::iterator next_it = ++prev_it;
      for (int i = 1; i < _points.size(); ++i)
      {
         int x1 = int((**prev_it)._pos[0] * 255);
         int y1 = int((**prev_it)._pos[1] * 255);
         int x2 = int((**next_it)._pos[0] * 255);
         int y2 = int((**next_it)._pos[1] * 255);
         midPointLine(_map, x1, y1, x2, y2, this->_opacity, this->_opacity);

         prev_it = next_it;
         ++next_it;
      }         
   }
}

bool vvTFCustom2D::getColor(vvColor& col, float, float, float)
{
   if (_ownColor)
   {
      col._col[0] = this->_col._col[0];
      col._col[1] = this->_col._col[1];
      col._col[2] = this->_col._col[2];
      return true;
   }
   return false;
}

bool vvTFCustom2D::hasOwnColor()
{
   return _ownColor;
}

void vvTFCustom2D::setOwnColor(bool flag)
{
   _ownColor = flag;
}


//
//
//
vvTFCustomMap::vvTFCustomMap(float x, float w, float y, float h, float z, float d)
: vvTFWidget(x, y, z)
{
   _size[0] = w;
   _size[1] = h;
   _size[2] = d;

   this->_ownColor = false;
   
   int dim;
   _dim[0] = int(256 * w);
   dim = _dim[0];

   if (h > 0.0f)
   {
      _dim[1] = int(256 * h);
      dim *= _dim[1];
   }
   else
   {
      _dim[1] = 1;
   }

   if (d > 0.0f)
   {
      _dim[2] = int(256 * d);
      dim *= _dim[2];
   }
   else
   {
      _dim[2] = 1;
   }

   _map = new float[dim];
}

vvTFCustomMap::vvTFCustomMap(vvColor col, bool ownColor, float x, float w, float y, float h, float z, float d) 
: vvTFWidget(x, y, z)
{
  _col = col;
  _ownColor = ownColor;
  _size[0] = w;
  _size[1] = h;
  _size[2] = d;

   this->_ownColor = false;
   
   int dim;
   _dim[0] = int(256 * w);
   dim = _dim[0];

   if (h > 0.0f)
   {
      _dim[1] = int(256 * h);
      dim *= _dim[1];
   }
   else
   {
      _dim[1] = 1;
   }

   if (d > 0.0f)
   {
      _dim[2] = int(256 * d);
      dim *= _dim[2];
   }
   else
   {
      _dim[2] = 1;
   }

   _map = new float[dim];
}

vvTFCustomMap::vvTFCustomMap(vvTFCustomMap* other) 
: vvTFWidget(other)
{
   this->_size[0] = other->_size[0];
   this->_size[1] = other->_size[1];
   this->_size[2] = other->_size[2];

   this->_ownColor = other->_ownColor;
   if (_ownColor)
   {
      this->_col._col[0] = other->_col._col[0];
      this->_col._col[1] = other->_col._col[1];
      this->_col._col[2] = other->_col._col[2];
   }

   this->_dim[0] = other->_dim[0];
   this->_dim[1] = other->_dim[1];
   this->_dim[2] = other->_dim[2];

   int dim = this->_dim[0] * this->_dim[1] * this->_dim[2];
   this->_map = new float[dim];
   memcpy(this->_map, other->_map, dim * sizeof(float));
}

vvTFCustomMap::~vvTFCustomMap()
{
   delete[] _map;
}

void vvTFCustomMap::write(FILE*)
{
   //TODO!!
}

// Given x, y, z in volume space, find the index in the map 
// based on _pos and _size
int vvTFCustomMap::computeIdx(float x, float y, float z)
{
   int idx = -1;

   // Determine dimensionality of transfer function:
   int dim = 1;
   if (z>-1.0f) dim = 3;
   else if (y>-1.0f) dim = 2;

   float half[3], lb[3], ub[3];

   // compute half sizes, upper and lower bounds
   for (int i = 0; i < dim; ++i)
   {
      half[i] = _size[i] / 2.0f;
      lb[i] = std::max(0.0f, _pos[i] - half[i]);
      ub[i] = std::min(1.0f, _pos[i] + half[i]);
   }

   if (x < lb[0] || x > ub[0])
      return -1;
   
   if (dim > 1)
   {
      if (y < lb[1] || y > ub[1])
         return -1;
   }

   if (dim > 2)
   {
      if (z < lb[2] || z > ub[2])
         return -1;
   }

   switch (dim)
   {
   case 1:
      idx = int((x - lb[0]) * 256);
      if (idx >= _dim[0])
         idx = _dim[0] - 1;
      break;

   case 2:
      {
         int yPos, xPos;

         yPos = int((y - lb[1]) * 256);
         if (yPos >= _dim[1])
            yPos = _dim[1] - 1;


         xPos = int((x - lb[0]) * 256);
         if (yPos >= _dim[0])
            yPos = _dim[0] - 1;
         
         idx = yPos * _dim[0] + xPos;
      }
      break;

   case 3:
      {
         int yPos, xPos, zPos;

         zPos = int((z - lb[2]) * 256);
         if (zPos >= _dim[2])
            zPos = _dim[2] - 1;

         yPos = int((y - lb[1]) * 256);
         if (yPos >= _dim[1])
            yPos = _dim[1] - 1;


         xPos = int((x - lb[0]) * 256);
         if (yPos >= _dim[0])
            yPos = _dim[0] - 1;
         
         idx = zPos * _dim[0] * _dim[1] + yPos * _dim[0] + xPos;
      }
      break;
   }
   return idx;
}

float vvTFCustomMap::getOpacity(float x, float y, float z)
{
   int idx = computeIdx(x, y, z);
   if (idx >= 0)   
      return _map[idx];   
   else
      return 0.0f;
}

void vvTFCustomMap::setOpacity(float val, float x, float y, float z)
{
   int idx = computeIdx(x, y, z);
   assert(idx >= 0 && idx < (_dim[0] * _dim[1] * _dim[2]));
   _map[idx] = val;
}

bool vvTFCustomMap::getColor(vvColor& col, float, float, float)
{
   if (_ownColor)
   {
      col._col[0] = this->_col._col[0];
      col._col[1] = this->_col._col[1];
      col._col[2] = this->_col._col[2];
      return true;
   }
   return false;
}

bool vvTFCustomMap::hasOwnColor()
{
   return _ownColor;
}

void vvTFCustomMap::setOwnColor(bool flag)
{
   _ownColor = flag;
}




//============================================================================
// End of File
//============================================================================
