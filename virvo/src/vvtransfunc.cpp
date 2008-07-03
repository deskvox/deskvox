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
    vvTFCustom* cu;
    vvTFCustom2D* c2;
    vvTFCustomMap* cm;

    if ((c = dynamic_cast<vvTFColor*>(oldW)) != NULL)
      _widgets.append(new vvTFColor(c), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    else if ((p = dynamic_cast<vvTFPyramid*>(oldW)) != NULL)
      _widgets.append(new vvTFPyramid(p), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    else if ((b = dynamic_cast<vvTFBell*>(oldW)) != NULL)
      _widgets.append(new vvTFBell(b), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    else if ((s = dynamic_cast<vvTFSkip*>(oldW)) != NULL)
      _widgets.append(new vvTFSkip(s), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    else if ((cu = dynamic_cast<vvTFCustom*>(oldW)) != NULL)
      _widgets.append(new vvTFCustom(cu), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    else if ((c2 = dynamic_cast<vvTFCustom2D*>(oldW)) != NULL)
      _widgets.append(new vvTFCustom2D(c2), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    else if ((cm = dynamic_cast<vvTFCustomMap*>(oldW)) != NULL)
      _widgets.append(new vvTFCustomMap(cm), vvSLNode<vvTFWidget*>::NORMAL_DELETE);

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
    if ((wt==TF_COLOR   && dynamic_cast<vvTFColor*>(w)) ||
      (wt==TF_PYRAMID   && dynamic_cast<vvTFPyramid*>(w)) ||
      (wt==TF_BELL      && dynamic_cast<vvTFBell*>(w)) ||
      (wt==TF_SKIP      && dynamic_cast<vvTFSkip*>(w)) ||
      (wt==TF_CUSTOM    && dynamic_cast<vvTFCustom*>(w)) ||
      (wt==TF_CUSTOM_2D && dynamic_cast<vvTFCustom2D*>(w)) ||
      (wt==TF_MAP       && dynamic_cast<vvTFCustomMap*>(w)))
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
 @param index color scheme
 @param min,max data range for color scheme
*/
void vvTransFunc::setDefaultColors(int index, float min, float max)
{
  deleteWidgets(TF_COLOR);
  switch (index)
  {
    case 0:                                       // bright colors
    default:
      // Set RGBA table to bright colors (range: blue->green->red):
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 1.0f), min),  vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(0.0f, 1.0f, 1.0f), (max-min) * 0.33f + min), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 1.0f, 0.0f), (max-min) * 0.67f + min), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 0.0f, 0.0f), max),  vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;

    case 1:                                       // hue gradient
      // Set RGBA table to maximum intensity and value HSB colors:
      _widgets.append(new vvTFColor(vvColor(1.0f, 0.0f, 0.0f), min), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 1.0f, 0.0f), (max-min) * 0.2f + min), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(0.0f, 1.0f, 0.0f), (max-min) * 0.4f + min), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(0.0f, 1.0f, 1.0f), (max-min) * 0.6f + min), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 1.0f), (max-min) * 0.8f + min), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 0.0f, 1.0f), max), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;

    case 2:                                       // grayscale ramp
      // Set RGBA table to grayscale ramp (range: black->white).
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 0.0f), min), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 1.0f, 1.0f), max), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;

    case 3:                                       // white
      // Set RGBA table to all white values:
      _widgets.append(new vvTFColor(vvColor(1.0f, 1.0f, 1.0f), min), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 1.0f, 1.0f), max), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;

    case 4:                                       // red ramp
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 0.0f), min), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(1.0f, 0.0f, 0.0f), max), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;

    case 5:                                       // green ramp
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 0.0f), min), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(0.0f, 1.0f, 0.0f), max), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;

    case 6:                                       // blue ramp
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 0.0f), min), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      _widgets.append(new vvTFColor(vvColor(0.0f, 0.0f, 1.0f), max), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
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
 @param min,max data range for alpha scheme
*/
void vvTransFunc::setDefaultAlpha(int index, float min, float max)
{
  vvDebugMsg::msg(2, "vvTransFunc::setDefaultAlpha()");

  deleteWidgets(TF_PYRAMID);
  deleteWidgets(TF_BELL);
  deleteWidgets(TF_CUSTOM);
  deleteWidgets(TF_CUSTOM_2D);
  deleteWidgets(TF_MAP);
  deleteWidgets(TF_SKIP);
  switch (index)
  {
    case 0:                                       // ascending (0->1)
    default:
      _widgets.append(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, max, 2.0f * (max-min), 0.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;
    case 1:                                       // descending (1->0)
      _widgets.append(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, min, 2.0f * (max-min), 0.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
      break;
    case 2:                                       // opaque (all 1)
      _widgets.append(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, (max-min)/2.0f+min, max-min, max-min), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
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
    if (vvTFPyramid *pw = dynamic_cast<vvTFPyramid*>(w))
    {
      if (pw->hasOwnColor() && pw->getColor(col, x, y, z))
      {
        hasOwn = true;
        resultCol = resultCol + col;
      }
    }
    else if (vvTFBell *bw = dynamic_cast<vvTFBell*>(w))
    {
      if (bw->hasOwnColor() && bw->getColor(col, x, y, z))
      {
        hasOwn = true;
        resultCol = resultCol + col;
      }
    }
    else if (vvTFCustom2D *cw = dynamic_cast<vvTFCustom2D*>(w))
    {
      if (cw->hasOwnColor() && cw->getColor(col, x, y, z))
      {
        hasOwn = true;
        resultCol = resultCol + col;
      }
    }
    else if (vvTFCustomMap *cmw = dynamic_cast<vvTFCustomMap*>(w))
    {
      if (cmw->hasOwnColor() && cmw->getColor(col, x, y, z))
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
               Space for w*h*d*4 float values must be provided.
 @param min,max min/max values to create texture for               
*/
void vvTransFunc::computeTFTexture(int w, int h, int d, float* array, 
  float minX, float maxX, float minY, float maxY, float minZ, float maxZ)
{
  vvColor col;
  int x, y, z, index;
  float norm[3];    // normalized 3D position

  index = 0;
  for (z=0; z<d; ++z)
  {
    norm[2] = (d==1) ? -1.0f : ((float(z) / float(d-1)) * (maxZ - minZ) + minZ);
    for (y=0; y<h; ++y)
    {
      norm[1] = (h==1) ? -1.0f : ((float(y) / float(h-1)) * (maxY - minY) + minY);
      for (x=0; x<w; ++x)
      {
        norm[0] = (float(x) / float(w-1)) * (maxX - minX) + minX;
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
// 1st channel in contiguous block; last for opacity channel
void vvTransFunc::computeTFTextureGamma(int w, float* dest, float minX, float maxX, 
										int numchan, float gamma[], float offset[])
{
  int index = 0;
  for (int c = 0; c < numchan+1; c++)
  {
	  for (int i=0; i<w; ++i)
	  {
		  float x = (float(i) / float(w-1)) * (maxX - minX) + minX;	  
		  dest[index++] = ts_clamp((1-offset[c])*powf(x, gamma[c])+offset[c], 0.0f, 1.0f);
	  }
  }
}

void vvTransFunc::computeTFTextureHighPass(int w, float* dest, float minX, float maxX, 
										int numchan, float cutoff[], float order[], float offset[])
{
  int index = 0;
  for (int c = 0; c < numchan+1; c++)
  {
	  for (int i=0; i<w; ++i)
	  {
		  float x = (float(i) / float(w-1)) * (maxX - minX) + minX;
		  float filter = 0.0f;
		  if (x != 0.0f) filter = 1.0f / (1 + powf(cutoff[c]/x, 2*order[c]));
		  dest[index++] = ts_clamp((1-offset[c])*filter+offset[c], 0.0f, 1.0f);
	  }
  }
}

void vvTransFunc::computeTFTextureHistCDF(int w, float* dest, float minX, float maxX, 
										int numchan, int frame, uint* histCDF, float gamma[], float offset[])
{
  int index = 0;
  
  //int alphaCDF[256];
  //memset(alphaCDF, 0, 256*sizeof(int));
  for (int c = 0; c < numchan; c++)
  {	
	  uint* hist = histCDF + (numchan*frame + c)*256;

	  float min = float(hist[0]), max = float(hist[w-1]);
	  for (int i=0; i<w; ++i)
	  {
		  //alphaCDF[i] += hist[i];
		  float x = (float(hist[i])-min)/(max-min);
		  dest[index++] = ts_clamp((1-offset[c])*x+offset[c], 0.0f, 1.0f); 
	  }
  }	

  for (int i=0; i<w; ++i)
  {
	  float x = (float(i) / float(w-1)) * (maxX - minX) + minX;
	  //x = ts_clamp((1-offset[numchan])*float(alphaCDF[i])/float(alphaCDF[w-1])+offset[numchan], 0.0f, 1.0f); 
	  dest[index++] = ts_clamp((1-offset[numchan])*powf(x, gamma[numchan])+offset[numchan], 0.0f, 1.0f);
	  //ts_clamp((1-offset[numchan]+1)* x + (offset[numchan]-1), 0.0f, 1.0f);
  }
}

//----------------------------------------------------------------------------
/** Returns RGBA texture values for a color preview bar consisting of 3 rows:
  top: just the color, ignores opacity
  middle: color and opacity
  bottom: opacity only as grayscale
  @param width  width of color bar [pixels]
  @param colors pointer to _allocated_ memory providing space for num x 3 x 4 bytes.
               Byte quadruple 0 will be considered to correspond with scalar
                value 0.0, quadruple num-1 will be considered to correspond with
                scalar value 1.0. The resulting RGBA values are stored in the
               following order: RGBARGBARGBA...
  @param min,max data range for which color bar is to be created. Use 0..1 for integer data types.
  @param invertAlpha Setting for opacity only color bar: false=high opacity is white; true=high opacity is black
*/
void vvTransFunc::makeColorBar(int width, uchar* colors, float min, float max, bool invertAlpha)
{
  float* rgba;                                    // component values
  int c, x, index;
  float alpha;

  assert(colors);

  // Compute color components:
  rgba = new float[width * 4];                    // four values per pixel
  computeTFTexture(width, 1, 1, rgba, min, max);

  // Convert to uchar:
  for (x=0; x<width; ++x)
  {
    for (c=0; c<4; ++c)
    {
      index = x * 4 + c;
      if (c<3) colors[index] = uchar(rgba[index] * 255.0f);
      else colors[index] = (uchar)255;
      colors[index + width * 4] = uchar(rgba[index] * 255.0f);
      alpha = rgba[x * 4 + 3];
      if (invertAlpha) alpha = 1.0f - alpha;
      colors[index + 2 * width * 4] = (c<3) ? (uchar(alpha * 255.0f)) : 255;
    }
  }
  delete[] rgba;
}

//----------------------------------------------------------------------------
/** Returns RGBA texture values for the alpha function of the 1D transfer function.
 Order of components: RGBARGBARGBA...
 @param width,height size of texture [pixels]
 @param texture  _allocated_ array in which to store texture values.
                 Space for width*height*4 bytes must be provided.
 @param min,max data range for which alpha texture is to be created. Use 0..1 for integer data types.
*/
void vvTransFunc::makeAlphaTexture(int width, int height, uchar* texture, float min, float max)
{
  const int RGBA = 4;
  const int GRAY_LEVEL = 160;
  const int ALPHA_LEVEL = 230;
  int x, y, index1D, index2D, barHeight;

  float* rgba = new float[width * RGBA];
  computeTFTexture(width, 1, 1, rgba, min, max);
  memset(texture, 0, width * height * RGBA); // make black and transparent

  for (x=0; x<width; ++x)
  {
    index1D = RGBA * x + 3;                          // alpha component of TF
    barHeight = int(rgba[index1D] * float(height));
    for (y=0; y<barHeight; ++y)
    {
      index2D = RGBA * (x + (height - y - 1) * width);
      texture[index2D]     = GRAY_LEVEL;
      texture[index2D + 1] = GRAY_LEVEL;
      texture[index2D + 2] = GRAY_LEVEL;
      texture[index2D + 3] = ALPHA_LEVEL;
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
void vvTransFunc::make2DTFTexture(int width, int height, uchar* texture, float minX, float maxX, float minY, float maxY)
{
  int x, y, index;

  float* rgba = new float[width * height * 4];
  computeTFTexture(width, height, 1, rgba, minX, maxX, minY, maxY);

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
/** Returns BGRA texture values for the 2D transfer function.
 Order of components: BGRABGRABGRA...
 Texture is flipped along Y axis, to be displayed on windows managers (Qt)
 @param width,height size of texture [pixels]
 @param texture  _allocated_ array in which to store texture values.
                 Space for width*height*4 bytes must be provided.
*/
void vvTransFunc::make2DTFTexture2(int width, int height, uchar* texture, float minX, float maxX, float minY, float maxY)
{
  int x, y, index1, index2;

  float* rgba = new float[width * height * 4];
  computeTFTexture(width, height, 1, rgba, minX, maxX, minY, maxY);

  for (y=0; y<height; ++y)
  {
    for (x=0; x<width; ++x)
    {
      index1 = 4 * (x + y * width);
      index2 = 4 * (x + (height - 1 - y) * width);
      texture[index1]     = uchar(rgba[index2 + 2] * 255.0f);
      texture[index1 + 1] = uchar(rgba[index2 + 1] * 255.0f);
      texture[index1 + 2] = uchar(rgba[index2]     * 255.0f);
      texture[index1 + 3] = uchar(rgba[index2 + 3] * 255.0f);
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
void vvTransFunc::make8bitLUT(int width, uchar* lut, float min, float max)
{
  float* rgba;                                    // temporary LUT in floating point format
  int i, c;

  rgba = new float[4 * width];

  // Generate arrays from pins:
  computeTFTexture(width, 1, 1, rgba, min, max);

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
  int numNodes, i;

  dst->removeAll();
  numNodes = src->count();
  src->first();
  for (i=0; i<numNodes; ++i)
  {
    w = src->getData();
    if (vvTFPyramid *pw = dynamic_cast<vvTFPyramid*>(w))
    {
      dst->append(new vvTFPyramid(pw), src->getDeleteType());
    }
    else if (vvTFBell *bw = dynamic_cast<vvTFBell*>(w))
    {
      dst->append(new vvTFBell(bw), src->getDeleteType());
    }
    else if (vvTFColor *cw = dynamic_cast<vvTFColor*>(w))
    {
      dst->append(new vvTFColor(cw), src->getDeleteType());
    }
    else if (vvTFCustom *cuw = dynamic_cast<vvTFCustom*>(w))
    {
      dst->append(new vvTFCustom(cuw), src->getDeleteType());
    }
    else if (vvTFSkip *sw = dynamic_cast<vvTFSkip*>(w))
    {
      dst->append(new vvTFSkip(sw), src->getDeleteType());
    }
    else if (vvTFCustomMap *cmw = dynamic_cast<vvTFCustomMap*>(w))
    {
      dst->append(new vvTFCustomMap(cmw), src->getDeleteType());
    }
    else if (vvTFCustom2D *c2w = dynamic_cast<vvTFCustom2D*>(w))
    {
      dst->append(new vvTFCustom2D(c2w), src->getDeleteType());
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
void vvTransFunc::makePreintLUTCorrect(int width, uchar *preIntTable, float thickness, float min, float max)
{
  const int minLookupSteps = 2;
  const int addLookupSteps = 1;

  vvDebugMsg::msg(1, "vvTransFunc::makePreintLUTCorrect()");

  // Generate arrays from pins:
  float *rgba = new float[width * 4];
  computeTFTexture(width, 1, 1, rgba, min, max);

  // cerr << "Calculating dependent texture - Please wait ...";
  vvToolshed::initProgress(width);
  for (int sb=0;sb<width;sb++)
  {
    for (int sf=0;sf<width;sf++)
    {
      int n=minLookupSteps+addLookupSteps*abs(sb-sf);
      double stepWidth = 1./n;
      double r=0., g=0., b=0., tau=0.;
      for (int i=0;i<n;i++)
      {
        double s = sf+(sb-sf)*(double)i/n;
        int is = (int)s;
        double tauc = thickness*stepWidth*(rgba[is*4+3]*(s-floor(s))+rgba[(is+1)*4+3]*(1.0-s+floor(s)));
#ifdef STANDARD
        /* standard optical model: r,g,b densities are multiplied with opacity density */
        double rc = exp(-tau)*tauc*(rgba[is*4+0]*(s-floor(s))+rgba[(is+1)*4+0]*(1.0-s+floor(s)));
        double gc = exp(-tau)*tauc*(rgba[is*4+1]*(s-floor(s))+rgba[(is+1)*4+1]*(1.0-s+floor(s)));
        double bc = exp(-tau)*tauc*(rgba[is*4+2]*(s-floor(s))+rgba[(is+1)*4+2]*(1.0-s+floor(s)));

#else
        /* Willhelms, Van Gelder optical model: r,g,b densities are not multiplied */
        double rc = exp(-tau)*stepWidth*(rgba[is*4+0]*(s-floor(s))+rgba[(is+1)*4+0]*(1.0-s+floor(s)));
        double gc = exp(-tau)*stepWidth*(rgba[is*4+1]*(s-floor(s))+rgba[(is+1)*4+1]*(1.0-s+floor(s)));
        double bc = exp(-tau)*stepWidth*(rgba[is*4+2]*(s-floor(s))+rgba[(is+1)*4+2]*(1.0-s+floor(s)));
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
void vvTransFunc::makePreintLUTOptimized(int width, uchar *preIntTable, float thickness, float min, float max)
{
  float *rInt = new float[width];
  float *gInt = new float[width];
  float *bInt = new float[width];
  float *aInt = new float[width];

  vvDebugMsg::msg(1, "vvTransFunc::makePreintLUTOptimized()");

  // Generate arrays from pins:
  float *rgba = new float[width * 4];
  computeTFTexture(width, 1, 1, rgba, min, max);

  // cerr << "Calculating optimized dependent texture" << endl;
  int rcol=0, gcol=0, bcol=0, acol=0;
  rInt[0] = 0.f;
  gInt[0] = 0.f;
  bInt[0] = 0.f;
  aInt[0] = 0.f;
  preIntTable[0] = int(rgba[0]);
  preIntTable[1] = int(rgba[1]);
  preIntTable[2] = int(rgba[2]);
  preIntTable[3] = int((1.f - expf(-rgba[3]*thickness)) * 255.99f);
  for (int i=1;i<width;i++)
  {
#ifdef STANDARD
    /* standard optical model: r,g,b densities are multiplied with opacity density */
    // accumulated values
    float tauc = (int(rgba[(i-1)*4+3]) + int(rgba[i*4+3])) * .5f;
    rInt[i] = rInt[i-1] + (int(255.99f*rgba[(i-1)*4+0]) + int(255.99f*rgba[i*4+0])) * .5f * tauc;
    gInt[i] = gInt[i-1] + (int(255.99f*rgba[(i-1)*4+1]) + int(255.99f*rgba[i*4+1])) * .5f * tauc;
    bInt[i] = bInt[i-1] + (int(255.99f*rgba[(i-1)*4+2]) + int(255.99f*rgba[i*4+2])) * .5f * tauc;
    aInt[i] = aInt[i-1] + tauc;

    // diagonal for lookup texture
    rcol = int(rgba[i*4+0] * rgba[i*4+3] * thickness * 255.99f);
    gcol = int(rgba[i*4+1] * rgba[i*4+3] * thickness * 255.99f);
    bcol = int(rgba[i*4+2] * rgba[i*4+3] * thickness * 255.99f);
    acol = int((1.f - expf(- rgba[i*4+3] * thickness)) * 255.99f);
#else
    /* Willhelms, Van Gelder optical model: r,g,b densities are not multiplied */
    // accumulated values
    rInt[i] = rInt[i-1] + (rgba[(i-1)*4+0] + rgba[i*4+0]) * .5f * 255;
    gInt[i] = gInt[i-1] + (rgba[(i-1)*4+1] + rgba[i*4+1]) * .5f * 255;
    bInt[i] = bInt[i-1] + (rgba[(i-1)*4+2] + rgba[i*4+2]) * .5f * 255;
    aInt[i] = aInt[i-1] + (rgba[(i-1)*4+3] + rgba[i*4+3]) * .5f;

    // diagonal for lookup texture
    rcol = int(255.99f*rgba[i*4+0]);
    gcol = int(255.99f*rgba[i*4+1]);
    bcol = int(255.99f*rgba[i*4+2]);
    acol = int((1.f - expf(-rgba[i*4+3] * thickness)) * 255.99f);
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
        if (rgba[s*4+3] >= .996f)
        {
          rcol = int(rgba[s*4+0]*255.99f);
          gcol = int(rgba[s*4+1]*255.99f);
          bcol = int(rgba[s*4+2]*255.99f);
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
          if (rgba[s*4+3] >= .996f)
          {
            rcol = int(rgba[s*4+0]*255.99f);
            gcol = int(rgba[s*4+1]*255.99f);
            bcol = int(rgba[s*4+2]*255.99f);
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
        std::cerr << "preIntTable(" << sf << "," << sb << ") = ("
          << int(preIntTable[sf*width*4+sb*4+0]) << " "
          << int(preIntTable[sf*width*4+sb*4+1]) << " "
          << int(preIntTable[sf*width*4+sb*4+2]) << " "
          << int(preIntTable[sf*width*4+sb*4+3]) << ")" << std::endl;
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

  - only Color and Custom transfer function widgets are supported, not Bell or Pyramid!
  - numbers are floating point with any number of mantissa digits
  - '#' allowed for comments
  </pre>
  @return 1 if successful, 0 if not
*/
int vvTransFunc::saveMeshviewer(const char* filename)
{
  vvTFWidget* w;
  vvTFColor* cw;
  vvTFCustom* cuw;
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
  _widgets.first();
  for (i=0; i<_widgets.count(); ++i)
  { 
    w = _widgets.getData();
    if ((cuw=dynamic_cast<vvTFCustom*>(w)))
    {
      fprintf(fp, "OpacityMapPoints: %d\n", (int)cuw->_points.size() + 2);   // add two points for edges of TF space
      fprintf(fp, "Point: %f %f\n", cuw->_pos[0] - cuw->_size[0]/2.0f, 0.0f);
      list<vvTFPoint*>::iterator iter;
      for(iter=cuw->_points.begin(); iter!=cuw->_points.end(); iter++) 
      {
        fprintf(fp, "Point: %f %f\n", (*iter)->_pos[0] + cuw->_pos[0], (*iter)->_opacity);
      }
      fprintf(fp, "Point: %f %f\n", cuw->_pos[0] + cuw->_size[0]/2.0f, 0.0f);
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
  vvTFCustom* cuw;
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
  if (numOpacityPoints>0) 
  {
    float begin=0., end=0.;
    cuw = new vvTFCustom(0.5f, 1.0f);
    _widgets.append(cuw, vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    for (i=0; i<numOpacityPoints; ++i)
    { 
      fscanf(fp, "Point: %f %f\n", &pos, &opacity);
      if (i>0 && i<numOpacityPoints-1)  // skip start and end point (will be determined by widget position and width)
      {
        cuw->_points.push_back(new vvTFPoint(opacity, pos));
      }
      else 
      {
        if (i==0) begin = pos;
        else if (i==numOpacityPoints-1) end = pos;
      }
    }
    
    // Adjust widget size:
    cuw->_size[0] = end - begin;
    cuw->_pos[0] = (begin + end) / 2.0f;

    // Adjust point positions:    
    list<vvTFPoint*>::iterator iter;
    for(iter=cuw->_points.begin(); iter!=cuw->_points.end(); iter++) 
    {
      (*iter)->_pos[0] -= cuw->_pos[0];
    }
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
      case TF_CUSTOM:  if (dynamic_cast<vvTFCustom*>(w))  ++num; break;

      case TF_CUSTOM_2D: if (dynamic_cast<vvTFCustom2D*>(w))  ++num; break;
      case TF_MAP:       if (dynamic_cast<vvTFCustomMap*>(w)) ++num; break;
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
