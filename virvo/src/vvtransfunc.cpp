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

#ifdef _WIN32
#include <float.h>
#endif
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvdebugmsg.h"
#include "vvvecmath.h"
#include "vvtransfunc.h"

//----------------------------------------------------------------------------
/// Constructor
vvTransFunc::vvTransFunc()
{
  _nextBufferEntry = 0;
  _bufferUsed      = 0;
  _discreteColors  = 0;
}

// Copy Constructor
vvTransFunc::vvTransFunc(vvTransFunc* tf)
{
  tf->_widgets.first();
  for (int i = 0; i < tf->_widgets.count(); i++)
  {
    vvTFWidget* oldW = tf->_widgets.getData();

    vvTFColor* c;
    vvTFPyramid* p;
    vvTFBell* b;
    vvTFSkip* s;
    if ((c = dynamic_cast<vvTFColor*>(oldW)) != NULL)
      _widgets.append(new vvTFColor(c), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    else if ((p = dynamic_cast<vvTFPyramid*>(oldW)) != NULL)
      _widgets.append(new vvTFPyramid(p), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    else if ((b = dynamic_cast<vvTFBell*>(oldW)) != NULL)
      _widgets.append(new vvTFBell(b), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    else if ((s = dynamic_cast<vvTFSkip*>(oldW)) != NULL)
      _widgets.append(new vvTFSkip(s), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    else assert(0);
    tf->_widgets.next();
  }

  _nextBufferEntry = tf->_nextBufferEntry;
  _bufferUsed      = tf->_bufferUsed;
  _discreteColors  = tf->_discreteColors;

}

//----------------------------------------------------------------------------
/// Destructor
vvTransFunc::~vvTransFunc()
{
  _widgets.removeAll();
}

//----------------------------------------------------------------------------
/** Delete all pins of given pin type from the list.
  @param wt widget type to delete
*/
void vvTransFunc::deleteWidgets(WidgetType wt)
{
  vvTFWidget* w;
  bool done = false;

  _widgets.first();
  while (!done && !_widgets.isEmpty())
  {
    w = _widgets.getData();
    if ((wt==TF_COLOR && dynamic_cast<vvTFColor*>(w)) ||
      (wt==TF_PYRAMID && dynamic_cast<vvTFPyramid*>(w)) ||
      (wt==TF_BELL    && dynamic_cast<vvTFBell*>(w)) ||
      (wt==TF_SKIP    && dynamic_cast<vvTFSkip*>(w)))
    {
      _widgets.remove();
      _widgets.first();
    }
    else if (!_widgets.next()) done = true;
  }
}

//----------------------------------------------------------------------------
/** @return true if the transfer function contains no widgets.
 */
bool vvTransFunc::isEmpty()
{
  return (_widgets.count()==0);
}

//----------------------------------------------------------------------------
/** Set default color values in the global color transfer function.
 All previous color widgets are deleted, other widgets are not affected.
*/
void vvTransFunc::setDefaultColors(int index)
{
  deleteWidgets(TF_COLOR);
  switch (index)
  {
    case 0:                                       // bright colors
    default:
      // Set RGBA table to bright colors (range: blue->green->red):
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 1.0f), 0.0f),  vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(0.0f, 1.0f, 1.0f), 0.33f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 1.0f, 0.0f), 0.67f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 0.0f, 0.0f), 1.0f),  vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;

    case 1:                                       // hue gradient
      // Set RGBA table to maximum intensity and value HSB colors:
      _widgets.append(new vvTFColor(vvColor(1.0f, 0.0f, 0.0f), 0.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 1.0f, 0.0f), 0.2f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(0.0f, 1.0f, 0.0f), 0.4f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(0.0f, 1.0f, 1.0f), 0.6f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 1.0f), 0.8f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 0.0f, 1.0f), 1.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;

    case 2:                                       // grayscale ramp
      // Set RGBA table to grayscale ramp (range: black->white).
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 0.0f), 0.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 1.0f, 1.0f), 1.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;

    case 3:                                       // white
      // Set RGBA table to all white values:
      _widgets.append(new vvTFColor(vvColor(1.0f, 1.0f, 1.0f), 0.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 1.0f, 1.0f), 1.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;

    case 4:                                       // red ramp
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 0.0f), 0.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 0.0f, 0.0f), 1.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;

    case 5:                                       // green ramp
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 0.0f), 0.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(0.0f, 1.0f, 0.0f), 1.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;

    case 6:                                       // blue ramp
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 0.0f), 0.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 1.0f), 1.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;
  }
}

//----------------------------------------------------------------------------
/// Returns the number of default color schemes.
int vvTransFunc::getNumDefaultColors()
{
  return 7;
}

//----------------------------------------------------------------------------
/** Set default alpha values in the transfer function.
 The previous alpha pins are deleted, color pins are not affected.
*/
void vvTransFunc::setDefaultAlpha(int index)
{
  vvDebugMsg::msg(2, "vvTransFunc::setDefaultAlpha()");

  deleteWidgets(TF_PYRAMID);
  deleteWidgets(TF_BELL);
  switch (index)
  {
    case 0:                                       // ascending (0->1)
    default:
      _widgets.append(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, 1.0f, 2.0f, 0.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;
    case 1:                                       // descending (1->0)
      _widgets.append(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, 0.0f, 2.0f, 0.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;
    case 2:                                       // opaque (all 1)
      _widgets.append(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, 0.5f, 1.0f, 1.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;
  }
}

//----------------------------------------------------------------------------
/// Returns the number of default alpha schemes.
int vvTransFunc::getNumDefaultAlpha()
{
  return 3;
}

//----------------------------------------------------------------------------
/** Calculate background color for pixel in TF space by interpolating
  in RGB color space. Only TF_COLOR widgets contribute to this color.
  Currently, only the x coordinate of color widgets is considered,
  so the resulting color field varies only along the x axis.
*/
vvColor vvTransFunc::computeBGColor(float x, float, float)
{
  vvColor col;
  vvTFWidget* w;
  vvTFColor* wBefore = NULL;
  vvTFColor* wAfter = NULL;
  vvTFColor* cw;
  int c, i;
  int numNodes;

  numNodes = _widgets.count();
  _widgets.first();
  for (i=0; i<numNodes; ++i)
  {
    w = _widgets.getData();
    if ((cw = dynamic_cast<vvTFColor*>(w)) != NULL)
    {
      if (cw->_pos[0] <= x)
      {
        if (wBefore==NULL || wBefore->_pos[0] < cw->_pos[0]) wBefore = cw;
      }
      if (cw->_pos[0] > x)
      {
        if (wAfter==NULL || wAfter->_pos[0] > cw->_pos[0]) wAfter = cw;
      }
    }
    _widgets.next();
  }

  if (wBefore==NULL && wAfter==NULL) return col;
  if (wBefore==NULL) col = wAfter->_col;
  else if (wAfter==NULL) col = wBefore->_col;
  else
  {
    for (c=0; c<3; ++c)
    {
      col[c] = vvToolshed::interpolateLinear(wBefore->_pos[0], wBefore->_col[c], wAfter->_pos[0], wAfter->_col[c], x);
    }
  }
  return col;
}

//----------------------------------------------------------------------------
/** Compute the color of a point in transfer function space. By definition
  the color is copied from the first non-TF_COLOR widget found. If no
  non-TF_COLOR widget is found, the point is colored according to the
  background color.
*/
vvColor vvTransFunc::computeColor(float x, float y, float z)
{
  vvColor col;
  vvColor resultCol(0,0,0);
  vvTFWidget* w;
  vvTFPyramid* pw;
  vvTFBell* bw;
  int numNodes, i;
  int currentRange;
  float rangeWidth;
  bool hasOwn = false;

  if (_discreteColors>0)
  {
    rangeWidth = 1.0f / _discreteColors;
    currentRange = int(x * _discreteColors);
    if (currentRange >= _discreteColors)          // constrain range to valid ranges
    {
      currentRange = _discreteColors - 1;
    }
    x = currentRange * rangeWidth + (rangeWidth / 2.0f);
  }

  numNodes = _widgets.count();
  _widgets.first();
  for (i=0; i<numNodes; ++i)
  {
    w = _widgets.getData();
    if (pw = dynamic_cast<vvTFPyramid*>(w))
    {
      if (pw->hasOwnColor() && pw->getColor(col, x, y, z))
      {
        hasOwn = true;
        resultCol = resultCol + col;
      }
    }
    else if (bw = dynamic_cast<vvTFBell*>(w))
    {
      if (bw->hasOwnColor() && bw->getColor(col, x, y, z))
      {
        hasOwn = true;
        resultCol = resultCol + col;
      }
    }
    _widgets.next();
  }
  if (!hasOwn) resultCol = computeBGColor(x, y, z);
  return resultCol;
}

//----------------------------------------------------------------------------
/** Goes through all widgets and returns the highest opacity value any widget
  has at the point.
*/
float vvTransFunc::computeOpacity(float x, float y, float z)
{
  vvTFWidget* w;
  float opacity = 0.0f;
  int numNodes, i;

  numNodes = _widgets.count();
  _widgets.first();
  for (i=0; i<numNodes; ++i)
  {
    w = _widgets.getData();
    if (dynamic_cast<vvTFSkip*>(w) && w->getOpacity(x, y, z)==0.0f) return 0.0f;  // skip widget is dominant
    else opacity = ts_max(opacity, w->getOpacity(x, y, z));
    _widgets.next();
  }
  return opacity;
}

//----------------------------------------------------------------------------
/** Discretize transfer function values and write to a float array.
 Order of components: RGBARGBARGBA...
 @param w,h,d  number of array entries in each dimension
 @param array  _allocated_ float array in which to store computed values [0..1]
               Space for w*h*d float values must be provided.
*/
void vvTransFunc::computeTFTexture(int w, int h, int d, float* array)
{
  vvColor col;
  int x, y, z, index;
  float norm[3];                                  // normalized 3D position

  index = 0;
  for (z=0; z<d; ++z)
  {
    norm[2] = (d==1) ? -1.0f : (float(z) / float(d-1));
    for (y=0; y<h; ++y)
    {
      norm[1] = (h==1) ? -1.0f : (float(y) / float(h-1));
      for (x=0; x<w; ++x)
      {
        norm[0] = float(x) / float(w-1);
        col = computeColor(norm[0], norm[1], norm[2]);
        array[index]   = col[0];
        array[index+1] = col[1];
        array[index+2] = col[2];
        array[index+3] = computeOpacity(norm[0], norm[1], norm[2]);
        index += 4;
      }
    }
  }
}

//----------------------------------------------------------------------------
/** Returns RGBA texture values for a color preview bar: without opacity at the top,
  with opacity at the bottom.
@param width  width of color bar [pixels]
@param colors pointer to _allocated_ memory providing space for num x 2 x 4 bytes.
              Byte quadruple 0 will be considered to correspond with scalar
              value 0.0, quadruple num-1 will be considered to correspond with
              scalar value 1.0. The resulting RGBA values are stored in the
              following order: RGBARGBARGBA...
*/
void vvTransFunc::makeColorBar(int width, uchar* colors)
{
  float* rgba;                                    // component values
  int c, x, index;

  assert(colors);

  // Compute color components:
  rgba = new float[width * 4];                    // four values per pixel
  computeTFTexture(width, 1, 1, rgba);

  // Convert to uchar:
  for (x=0; x<width; ++x)
  {
    for (c=0; c<4; ++c)
    {
      index = x * 4 + c;
      if (c<3) colors[index] = uchar(rgba[index] * 255.0f);
      else colors[index] = (uchar)255;
      colors[index + width * 4] = uchar(rgba[index] * 255.0f);
    }
  }
  delete[] rgba;
}

//----------------------------------------------------------------------------
/** Returns RGBA texture values for the alpha functino of the 1D transfer function.
 Order of components: RGBARGBARGBA...
 @param width,height size of texture [pixels]
 @param texture  _allocated_ array in which to store texture values.
                 Space for width*height*4 bytes must be provided.
*/
void vvTransFunc::makeAlphaTexture(int width, int height, uchar* texture)
{
  const int GRAY_LEVEL = 160;
  int x, y, index1D, index2D, barHeight;

  float* rgba = new float[width * 4];
  computeTFTexture(width, 1, 1, rgba);
  memset(texture, 255, width * height * 4);

  for (x=0; x<width; ++x)
  {
    index1D = 4 * x + 3;                          // alpha component of TF
    barHeight = int(rgba[index1D] * float(height));
    for (y=0; y<barHeight; ++y)
    {
      index2D = 4 * (x + (height - y - 1) * width);
      texture[index2D]     = GRAY_LEVEL;
      texture[index2D + 1] = GRAY_LEVEL;
      texture[index2D + 2] = GRAY_LEVEL;
      texture[index2D + 3] = 255;                 // alpha = opaque
    }
  }
  delete[] rgba;
}

//----------------------------------------------------------------------------
/** Returns RGBA texture values for the 2D transfer function.
 Order of components: RGBARGBARGBA...
 @param width,height size of texture [pixels]
 @param texture  _allocated_ array in which to store texture values.
                 Space for width*height*4 bytes must be provided.
*/
void vvTransFunc::make2DTFTexture(int width, int height, uchar* texture)
{
  int x, y, index;

  float* rgba = new float[width * height * 4];
  computeTFTexture(width, height, 1, rgba);

  for (y=0; y<height; ++y)
  {
    for (x=0; x<width; ++x)
    {
      index = 4 * (x + y * width);
      texture[index]     = uchar(rgba[index]     * 255.0f);
      texture[index + 1] = uchar(rgba[index + 1] * 255.0f);
      texture[index + 2] = uchar(rgba[index + 2] * 255.0f);
      texture[index + 3] = uchar(rgba[index + 3] * 255.0f);
    }
  }
  delete[] rgba;
}

//----------------------------------------------------------------------------
/** Create a look-up table of 8-bit integer values from current transfer
  function.
  @param width number of LUT entries (typically 256 or 4096, depending on bpv)
  @param lut     _allocated_ space with space for entries*4 bytes
*/
void vvTransFunc::make8bitLUT(int width, uchar* lut)
{
  float* rgba;                                    // temporary LUT in floating point format
  int i, c;

  rgba = new float[4 * width];

  // Generate arrays from pins:
  computeTFTexture(width, 1, 1, rgba);

  // Copy RGBA values to internal array:
  for (i=0; i<width; ++i)
  {
    for (c=0; c<4; ++c)
    {
      *lut = uchar(rgba[i * 4 + c] * 255.0f);
      ++lut;
    }
  }
  delete[] rgba;
}

//----------------------------------------------------------------------------
/** Make a deep copy of the widget list.
  @param dst destination list
  @param src source list
*/
void vvTransFunc::copy(vvSLList<vvTFWidget*>* dst, vvSLList<vvTFWidget*>* src)
{
  vvTFWidget* w;
  vvTFPyramid* pw;
  vvTFBell* bw;
  vvTFColor* cw;
  vvTFSkip* sw;
  int numNodes, i;

  dst->removeAll();
  numNodes = src->count();
  src->first();
  for (i=0; i<numNodes; ++i)
  {
    w = src->getData();
    if (pw = dynamic_cast<vvTFPyramid*>(w))
    {
      dst->append(new vvTFPyramid(pw), src->getDeleteType());
    }
    else if (bw = dynamic_cast<vvTFBell*>(w))
    {
      dst->append(new vvTFBell(bw), src->getDeleteType());
    }
    else if (cw = dynamic_cast<vvTFColor*>(w))
    {
      dst->append(new vvTFColor(cw), src->getDeleteType());
    }
    else if (sw = dynamic_cast<vvTFSkip*>(w))
    {
      dst->append(new vvTFSkip(sw), src->getDeleteType());
    }
    else assert(0);
    src->next();
  }
}

//----------------------------------------------------------------------------
/// Store the current pin list in the undo ring buffer.
void vvTransFunc::putUndoBuffer()
{
  copy(&_buffer[_nextBufferEntry], &_widgets);
  if (_bufferUsed < BUFFER_SIZE) ++_bufferUsed;
  if (_nextBufferEntry < BUFFER_SIZE-1) ++_nextBufferEntry;
  else _nextBufferEntry = 0;
}

//----------------------------------------------------------------------------
/** Restore the latest element from the undo ring buffer to the current pin list.
  If the ring buffer is empty, nothing happens.
*/
void vvTransFunc::getUndoBuffer()
{
  int bufferEntry;

  if (_bufferUsed==0) return;                     // ring buffer is empty
  if (_nextBufferEntry > 0) bufferEntry = _nextBufferEntry - 1;
  else bufferEntry = BUFFER_SIZE - 1;
  copy(&_widgets, &_buffer[bufferEntry]);
  _nextBufferEntry = bufferEntry;
  --_bufferUsed;
}

//----------------------------------------------------------------------------
/// Clear the undo ring buffer.
void vvTransFunc::clearUndoBuffer()
{
  _bufferUsed      = 0;
  _nextBufferEntry = 0;
}

//----------------------------------------------------------------------------
/** Set the number of discrete colors to use for color interpolation.
  @param numColors number of discrete colors (use 0 for smooth colors)
*/
void vvTransFunc::setDiscreteColors(int numColors)
{
  assert(numColors >= 0);
  _discreteColors = numColors;
}

//----------------------------------------------------------------------------
/** @return the number of discrete colors used for color interpolation.
            0 means smooth colors.
*/
int vvTransFunc::getDiscreteColors()
{
  return _discreteColors;
}

//----------------------------------------------------------------------------
/** Creates the look-up table for pre-integrated rendering.
  This version of the code runs rather slow compared to
  makeLookupTextureOptimized because it does a correct applications of
  the volume rendering integral.
  This method is
 * Copyright (C) 2001  Klaus Engel   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
@author Klaus Engel

I would like to thank Martin Kraus who helped in the adaptation
of the pre-integration method to Virvo.
@param thickness  distance of two volume slices in the direction
of the principal viewing axis (defaults to 1.0)
*/
void vvTransFunc::makePreintLUTCorrect(int width, uchar *preIntTable, float thickness)
{
  const int minLookupSteps = 2;
  const int addLookupSteps = 1;
  double r=0,g=0,b=0,tau=0;
  double rc=0,gc=0,bc=0,tauc=0;
  double stepWidth;

  vvDebugMsg::msg(1, "vvTransFunc::makePreintLUTCorrect()");

  // Generate arrays from pins:
  float *rgba = new float[width * 4];
  computeTFTexture(width, 1, 1, rgba);

  // cerr << "Calculating dependent texture - Please wait ...";
  vvToolshed::initProgress(width);
  for (int sb=0;sb<width;sb++)
  {
    for (int sf=0;sf<width;sf++)
    {
      int n=minLookupSteps+addLookupSteps*abs(sb-sf);
      stepWidth = 1./n;
      r=0;g=0;b=0;tau=0;
      for (int i=0;i<n;i++)
      {
        double s = sf+(sb-sf)*(double)i/n;
        int is = (int)s;
        tauc = thickness*stepWidth*(rgba[is+3*width]*(s-floor(s))+rgba[(is+1)+3*width]*(1.0-s+floor(s)));
#ifdef STANDARD
        /* standard optical model: r,g,b densities are multiplied with opacity density */
        rc = exp(-tau)*tauc*(rgba[is+0*width]*(s-floor(s))+rgba[(is+1)+0*width]*(1.0-s+floor(s)));
        gc = exp(-tau)*tauc*(rgba[is+1*width]*(s-floor(s))+rgba[(is+1)+1*width]*(1.0-s+floor(s)));
        bc = exp(-tau)*tauc*(rgba[is+2*width]*(s-floor(s))+rgba[(is+1)+2*width]*(1.0-s+floor(s)));

#else
        /* Willhelms, Van Gelder optical model: r,g,b densities are not multiplied */
        rc = exp(-tau)*stepWidth*(rgba[is+0*width]*(s-floor(s))+rgba[(is+1)+0*width]*(1.0-s+floor(s)));
        gc = exp(-tau)*stepWidth*(rgba[is+1*width]*(s-floor(s))+rgba[(is+1)+1*width]*(1.0-s+floor(s)));
        bc = exp(-tau)*stepWidth*(rgba[is+2*width]*(s-floor(s))+rgba[(is+1)+2*width]*(1.0-s+floor(s)));
#endif

        r = r+rc;
        g = g+gc;
        b = b+bc;
        tau = tau + tauc;
      }
      if (r>1.)
        r = 1.;
      preIntTable[sf*width*4+sb*4+0] = uchar(r*255.99);
      if (g>1.)
        g = 1.;
      preIntTable[sf*width*4+sb*4+1] = uchar(g*255.99);
      if (b>1.)
        b = 1.;
      preIntTable[sf*width*4+sb*4+2] = uchar(b*255.99);
      preIntTable[sf*width*4+sb*4+3] = uchar((1.- exp(-tau))*255.99);
    }
    vvToolshed::printProgress(sb);
  }
  delete[] rgba;
  // cerr << "done." << endl;
}

//----------------------------------------------------------------------------
/** Creates the look-up table for pre-integrated rendering.
  This version of the code runs much faster than makeLookupTextureCorrect
  due to some minor simplifications of the volume rendering integral.
  This method is
 * Copyright (C) 2001  Klaus Engel   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
@author Klaus Engel

I would like to thank Martin Kraus who helped in the adaptation
of the pre-integration method to Virvo.
@param thickness  distance of two volume slices in the direction
of the principal viewing axis (defaults to 1.0)
*/
void vvTransFunc::makePreintLUTOptimized(int width, uchar *preIntTable, float thickness)
{
  float *rInt = new float[width];
  float *gInt = new float[width];
  float *bInt = new float[width];
  float *aInt = new float[width];

  vvDebugMsg::msg(1, "vvTransFunc::makePreintLUTOptimized()");

  // Generate arrays from pins:
  float *rgba = new float[width * 4];
  computeTFTexture(width, 1, 1, rgba);

  // cerr << "Calculating optimized dependent texture" << endl;
  int rcol=0, gcol=0, bcol=0, acol=0;
  rInt[0] = 0.f;
  gInt[0] = 0.f;
  bInt[0] = 0.f;
  aInt[0] = 0.f;
  for (int i=1;i<width;i++)
  {
#ifdef STANDARD
    /* standard optical model: r,g,b densities are multiplied with opacity density */
    // accumulated values
    float tauc = (int(rgba[(i-1)+3*width]) + int(rgba[i+3*width])) * .5f;
    rInt[i] = rInt[i-1] + (int(255.99f*rgba[(i-1)+0*width]) + int(255.99f*rgba[i+0*width])) * .5f * tauc;
    gInt[i] = gInt[i-1] + (int(255.99f*rgba[(i-1)+1*width]) + int(255.99f*rgba[i+1*width])) * .5f * tauc;
    bInt[i] = bInt[i-1] + (int(255.99f*rgba[(i-1)+2*width]) + int(255.99f*rgba[i+2*width])) * .5f * tauc;
    aInt[i] = aInt[i-1] + tauc;

    // diagonal for lookup texture
    rcol = int(rgba[i+0*width] * rgba[i+3*width] * thickness * 255.99f);
    gcol = int(rgba[i+1*width] * rgba[i+3*width] * thickness * 255.99f);
    bcol = int(rgba[i+2*width] * rgba[i+3*width] * thickness * 255.99f);
    acol = int((1.f - expf(- rgba[i+3*width] * thickness)) * 255.99f);
#else
    /* Willhelms, Van Gelder optical model: r,g,b densities are not multiplied */
    // accumulated values
    rInt[i] = rInt[i-1] + (rgba[(i-1)+0*width] + rgba[i+0*width]) * .5f * 255.99f;
    gInt[i] = gInt[i-1] + (rgba[(i-1)+1*width] + rgba[i+1*width]) * .5f * 255.99f;
    bInt[i] = bInt[i-1] + (rgba[(i-1)+2*width] + rgba[i+2*width]) * .5f * 255.99f;
    aInt[i] = aInt[i-1] + (rgba[(i-1)+3*width] + rgba[i+3*width]) * .5f;

    // diagonal for lookup texture
    rcol = int(255.99f*rgba[i+0*width]);
    gcol = int(255.99f*rgba[i+1*width]);
    bcol = int(255.99f*rgba[i+2*width]);
    acol = int((1.f - expf(-rgba[i+3*width] * thickness)) * 255.99f);
#endif

    preIntTable[i*width*4+i*4+0] = uchar(rcol);
    preIntTable[i*width*4+i*4+1] = uchar(gcol);
    preIntTable[i*width*4+i*4+2] = uchar(bcol);
    preIntTable[i*width*4+i*4+3] = uchar(acol);
  }

  for (int sb=0;sb<width;sb++)
  {
    for (int sf=0;sf<sb;sf++)
    {
      bool opaque = false;
      for (int s = sf; s <= sb; s++)
      {
        if (rgba[s+3*width] == 1.f)
        {
          rcol = int(rgba[s+0*width]*255.99f);
          gcol = int(rgba[s+1*width]*255.99f);
          bcol = int(rgba[s+2*width]*255.99f);
          acol = int(255);
          opaque = true;
          break;
        }
      }

      if(opaque)
      {
        preIntTable[sb*width*4+sf*4+0] = uchar(rcol);
        preIntTable[sb*width*4+sf*4+1] = uchar(gcol);
        preIntTable[sb*width*4+sf*4+2] = uchar(bcol);
        preIntTable[sb*width*4+sf*4+3] = uchar(acol);

        for (int s = sb; s >= sf; s--)
        {
          if (rgba[s+3*width] == 1.f)
          {
            rcol = int(rgba[s+0*width]*255.99f);
            gcol = int(rgba[s+1*width]*255.99f);
            bcol = int(rgba[s+2*width]*255.99f);
            acol = int(255);
            break;
          }
        }
        preIntTable[sf*width*4+sb*4+0] = uchar(rcol);
        preIntTable[sf*width*4+sb*4+1] = uchar(gcol);
        preIntTable[sf*width*4+sb*4+2] = uchar(bcol);
        preIntTable[sf*width*4+sb*4+3] = uchar(acol);
        continue;
      }

      float scale = 1.f/(sb-sf);
      rcol = int((rInt[sb] - rInt[sf])*scale);
      gcol = int((gInt[sb] - gInt[sf])*scale);
      bcol = int((bInt[sb] - bInt[sf])*scale);
      acol = int((1.f - expf(-(aInt[sb]-aInt[sf])*scale * thickness)) * 255.99f);

      if (rcol > 255)
        rcol = 255;
      if (gcol > 255)
        gcol = 255;
      if (bcol > 255)
        bcol = 255;
      if (acol > 255)
        acol = 255;

      preIntTable[sf*width*4+sb*4+0] = uchar(rcol);
      preIntTable[sf*width*4+sb*4+1] = uchar(gcol);
      preIntTable[sf*width*4+sb*4+2] = uchar(bcol);
      preIntTable[sf*width*4+sb*4+3] = uchar(acol);

      preIntTable[sb*width*4+sf*4+0] = uchar(rcol);
      preIntTable[sb*width*4+sf*4+1] = uchar(gcol);
      preIntTable[sb*width*4+sf*4+2] = uchar(bcol);
      preIntTable[sb*width*4+sf*4+3] = uchar(acol);

#if 0
      if (sb%16==0 && sf%16==0)
      {
        std::cerr << "preIntTable " << sf << " " << sb << "[0]=" << int(preIntTable[sf*width*4+sb*4+0]) << std::endl;
        std::cerr << "preIntTable " << sf << " " << sb << "[1]=" << int(preIntTable[sf*width*4+sb*4+1]) << std::endl;
        std::cerr << "preIntTable " << sf << " " << sb << "[2]=" << int(preIntTable[sf*width*4+sb*4+2]) << std::endl;
        std::cerr << "preIntTable " << sf << " " << sb << "[3]=" << int(preIntTable[sf*width*4+sb*4+3]) << std::endl;
      }
#endif

    }
  }

  delete[] rInt;
  delete[] gInt;
  delete[] bInt;
  delete[] aInt;
}

//----------------------------------------------------------------------------
/** Save transfer function to a disk file in Meshviewer format:
  Example:
  <pre>
  ColorMapKnots: 3
  Knot:  0.0  1.0  0.0  0.0
  Knot: 50.0  0.0  1.0  0.0
  Knot: 99.0  0.0  0.0  1.0
  OpacityMapPoints: 3
  Point:  0.0   0.00
  Point: 50.0   0.05
  Point: 99.0   0.00

  Syntax:
  Knot: <float_data_value> <red 0..1> <green> <blue>
  Point: <float_data_value> <opacity 0..1>

  - numbers are floating point with any number of mantissa digits
  - '#' allowed for comments
  </pre>
  @return 1 if successful, 0 if not
*/
int vvTransFunc::saveMeshviewer(const char* filename)
{
  vvTFWidget* w;
  vvTFColor* cw;
  vvTFBell* bw;
  vvTFPyramid* pw;
  FILE* fp;
  int i;

  if ( (fp = fopen(filename, "wb")) == NULL)
  {
    cerr << "Error: Cannot create file." << endl;
    return 0;
  }
  
  // Write color pins to file:
  fprintf(fp, "ColorMapKnots: %d\n", getNumWidgets(TF_COLOR));
  _widgets.first();
  for (i=0; i<_widgets.count(); ++i)
  { 
    w = _widgets.getData();
    if ((cw=dynamic_cast<vvTFColor*>(w)))
    {
      fprintf(fp, "Knot: %f %f %f %f\n", cw->_pos[0], cw->_col[0], cw->_col[1], cw->_col[2]);
    }
    _widgets.next();
  }
  
  // Write opacity pins to file:
  fprintf(fp, "OpacityMapPoints: %d\n", getNumWidgets(TF_BELL) + getNumWidgets(TF_PYRAMID));
  _widgets.first();
  for (i=0; i<_widgets.count(); ++i)
  { 
    w = _widgets.getData();
    if ((bw=dynamic_cast<vvTFBell*>(w)))
    {
      fprintf(fp, "Point: %f %f\n", bw->_pos[0], bw->_opacity);
    }
    else if ((pw=dynamic_cast<vvTFPyramid*>(w)))
    {
      fprintf(fp, "Point: %f %f\n", pw->_pos[0], pw->_opacity);
    }
    _widgets.next();
  }

  // Wrap up:
  fclose(fp);
  cerr << "Wrote transfer function file: " << filename << endl;
  return 1;
}

//----------------------------------------------------------------------------
/** Load transfer function from a disk file in Meshviewer format.
  @see vvTransFunc::saveMeshViewer
  @return 1 if successful, 0 if not
*/
int vvTransFunc::loadMeshviewer(const char* filename)
{
  vvTFColor* cw;
  vvTFPyramid* pw;
  FILE* fp;
  int i;
  int numColorWidgets, numOpacityPoints;
  float pos, col[3], opacity;

  if ( (fp = fopen(filename, "rb")) == NULL)
  {
    cerr << "Error: Cannot open file." << endl;
    return 0;
  }
  
  // Remove all existing widgets:
  _widgets.removeAll();
  
  // Read color pins from file:
  fscanf(fp, "ColorMapKnots: %d\n", &numColorWidgets);
  for (i=0; i<numColorWidgets; ++i)
  { 
    fscanf(fp, "Knot: %f %f %f %f\n", &pos, &col[0], &col[1], &col[2]);
    cw = new vvTFColor();
    cw->_pos[0] = pos;
    cw->_col[0] = col[0];
    cw->_col[1] = col[1];
    cw->_col[2] = col[2];
    _widgets.append(cw, vvSLNode<vvTFWidget*>::NORMAL_DELETE);
  }
  
  // Read opacity pins from file:
  fscanf(fp, "OpacityMapPoints: %d\n", &numOpacityPoints);
  for (i=0; i<numOpacityPoints; ++i)
  { 
    fscanf(fp, "Point: %f %f\n", &pos, &opacity);
    pw = new vvTFPyramid();
    pw->_pos[0] = pos;
    pw->_opacity = opacity;
    pw->setOwnColor(false);
    _widgets.append(pw, vvSLNode<vvTFWidget*>::NORMAL_DELETE);
  }

  // Wrap up:
  fclose(fp);
  cerr << "Loaded transfer function from file: " << filename << endl;
  return 1;
}

//----------------------------------------------------------------------------
/** @return the number of widgets of a given type
*/
int vvTransFunc::getNumWidgets(WidgetType wt)
{
  vvTFWidget* w;
  int num = 0;
  int i;
  
  _widgets.first();
  for (i=0; i<_widgets.count(); ++i)
  {
    w = _widgets.getData();
    switch(wt)
    {
      case TF_COLOR:   if (dynamic_cast<vvTFColor*>(w))   ++num; break;
      case TF_PYRAMID: if (dynamic_cast<vvTFPyramid*>(w)) ++num; break;
      case TF_BELL:    if (dynamic_cast<vvTFBell*>(w))    ++num; break;
      case TF_SKIP:    if (dynamic_cast<vvTFSkip*>(w))    ++num; break;
    }
    _widgets.next();
  }
  return num;
}

//----------------------------------------------------------------------------
/** @set the transfer function type to LUT_1D and set LUT
 */
/*
void vvTransFunc::setLUT(int numEntries, const uchar *rgba)
{
   vvDebugMsg::msg(1, "vvTransFunc::setLUT()");
   lutEntries = numEntries;
   type = LUT_1D;
   delete[] rgbaLUT;
   rgbaLUT = new uchar[4*lutEntries];
   memcpy(rgbaLUT, rgba, 4*lutEntries);
}
*/
//============================================================================
// End of File
//============================================================================
