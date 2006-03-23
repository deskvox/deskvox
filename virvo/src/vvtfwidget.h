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

#ifndef _VVTFWIDGET_H_
#define _VVTFWIDGET_H_

#include <stdio.h>
#include "vvexport.h"
#include "vvsllist.h"

/** Creates a general color class for RGB colors.
 */
class VIRVOEXPORT vvColor
{
  public:
    float _col[3];                                ///< RGB color [0..1]

    vvColor();
    vvColor(float, float, float);
    const float operator[](const int) const;
    float& operator[](const int);
    vvColor operator+(const vvColor) const;
    void setRGB(float, float, float);
    void setHSB(float, float, float);
    void getRGB(float&, float&, float&);
    void getHSB(float&, float&, float&);
};

/** Base class of transfer function widgets.
  @author Jurgen P. Schulze (schulze@cs.brown.de)
  @see vvTransFunc
*/
class VIRVOEXPORT vvTFWidget
{
  protected:
    static const char* NO_NAME;
    char* _name;                                  ///< widget name (bone, soft tissue, etc)

  public:
    float _pos[3];                                ///< position of widget's center in transfer function space [0..1 is inside TF space, other values are valid but outside]

    vvTFWidget();
    vvTFWidget(float, float, float);
    vvTFWidget(vvTFWidget*);
    virtual ~vvTFWidget();
    virtual void setName(const char*);
    virtual const char* getName();
    virtual void readName(FILE*);
    virtual void write(FILE*) = 0;
    virtual float getOpacity(float, float=-1.0f, float=-1.0f);
    virtual bool getColor(vvColor&, float, float=-1.0f, float=-1.0f);
};

/** Transfer function widget shaped like a Gaussian bell.
 */
class VIRVOEXPORT vvTFBell : public vvTFWidget
{
  protected:
    bool _ownColor;                               ///< true = use widget's own color for TF; false=use background color for TF

  public:
    vvColor _col;                                 ///< RGB color
    float _size[3];                               ///< width, height, depth of bell's bounding box [TF space is 0..1]
    float _opacity;                               ///< maximum opacity [0..1]

    vvTFBell();
    vvTFBell(vvTFBell*);
    vvTFBell(vvColor, bool, float, float, float, float=0.5f, float=1.0f, float=0.5f, float=1.0f);
    vvTFBell(FILE*);
    virtual void write(FILE*);
    virtual bool getColor(vvColor&, float, float=-1.0f, float=-1.0f);
    virtual float getOpacity(float, float=-1.0f, float=-1.0f);
    virtual bool hasOwnColor();
    virtual void setOwnColor(bool);
};

/** Pyramid-shaped transfer function widget:
  the pyramid has four sides and its tip can be flat (frustum).
*/
class VIRVOEXPORT vvTFPyramid : public vvTFWidget
{
  protected:
    bool _ownColor;                               ///< true = use widget's own color for TF; false=use background color for TF

  public:
    vvColor _col;                                 ///< RGB color
    float _top[3];                                ///< width at top [0..1]
    float _bottom[3];                             ///< width at bottom of pyramid [0..1]
    float _opacity;                               ///< maximum opacity [0..1]

    vvTFPyramid();
    vvTFPyramid(vvTFPyramid*);
    vvTFPyramid(vvColor, bool, float, float, float, float, float=0.5f, float=1.0f, float=0.0f, float=0.5f, float=1.0f, float=0.0f);
    vvTFPyramid(FILE*);
    virtual void write(FILE*);
    virtual bool getColor(vvColor&, float, float=-1.0f, float=-1.0f);
    virtual float getOpacity(float, float=-1.0f, float=-1.0f);
    virtual bool hasOwnColor();
    virtual void setOwnColor(bool);
};

/** Transfer function widget specifying a color point in TF space.
 */
class VIRVOEXPORT vvTFColor : public vvTFWidget
{
  public:
    vvColor _col;                                 ///< RGB color

    vvTFColor();
    vvTFColor(vvTFColor*);
    vvTFColor(vvColor, float, float=0.0f, float=0.0f);
    vvTFColor(FILE*);
    virtual void write(FILE*);
};

/** Transfer function widget to skip an area of the transfer function when rendering.
 */
class VIRVOEXPORT vvTFSkip : public vvTFWidget
{
  public:
    float _size[3];          ///< width, height, depth of skipped area [TF space is 0..1]

    vvTFSkip();
    vvTFSkip(vvTFSkip*);
    vvTFSkip(float, float, float=0.5f, float=0.0f, float=0.5f, float=0.0f);
    vvTFSkip(FILE*);
    virtual void write(FILE*);
    virtual float getOpacity(float, float=-1.0f, float=-1.0f);
};

/** Transfer function widget to specify a function freehandedly. The widget defines a rectangular area
  in which the user can specify an opacity value for each element, defined by resolution.
 */
class VIRVOEXPORT vvTFFreehand : public vvTFWidget
{
  public:
    int _resolution[3];     ///< granularity of freehand grid for each dimension [# array elements, e.g., 256]
    float _size[3];         ///< width, height, depth of freehand area [TF space is 0..1]

    vvTFFreehand();
    vvTFFreehand(vvTFFreehand*);
    vvTFFreehand(float, float, int, float=0.5f, float=0.0f, int=0, float=0.5f, float=0.0f, int=0);
    vvTFFreehand(FILE*);
    virtual void write(FILE*);
    virtual float getOpacity(float, float=-1.0f, float=-1.0f);
};

#endif

//============================================================================
// End of File
//============================================================================
