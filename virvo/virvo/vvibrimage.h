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

#ifndef _VVIBRIMAGE_H_
#define _VVIBRIMAGE_H_

#include "vvimage.h"
#include "vvvecmath.h"
#include "vvgltools.h"

class VIRVOEXPORT vvIbrImage : public vvImage
{
  public:
    vvIbrImage(short w, short h, uchar *data, int depthPrecision,
               int uncertaintyPrecision);
    vvIbrImage();
    virtual ~vvIbrImage();

    virtual int encode(short ct, short sh=-1, short eh=-1, short sw=-1, short ew=-1);
    virtual int decode();

    uchar *getPixelDepth() const;
    uchar *getCodedDepth() const;
    uchar *getUncertainty() const;
    uchar *getCodedUncertainty() const;
    int  getDepthSize() const;
    void  setDepthSize(int size);
    int  getUncertaintySize() const;
    void  setUncertaintySize(int size);
    void    alloc_pd();
    void setNewDepthPtr(uchar *depth);
    void setNewUncertaintyPtr(uchar* uncertainty);

    void setDepthPrecision(int dp);
    int getDepthPrecision() const;
    void setUncertaintyPrecision(int up);
    int getUncertaintyPrecision() const;
    CodeType getDepthCodetype() const;
    void setDepthCodetype(CodeType ct);
    CodeType getUncertaintyCodetype() const;
    void setUncertaintyCodetype(CodeType ct);

    void setModelViewMatrix(const vvMatrix &mv);
    vvMatrix getModelViewMatrix() const;
    void setProjectionMatrix(const vvMatrix &pm);
    vvMatrix getProjectionMatrix() const;
    void setViewport(const vvGLTools::Viewport &vp);
    vvGLTools::Viewport getViewport() const;
    void setDepthRange(float depthMin, float depthMax);
    void getDepthRange(float *depthMin, float *depthMax) const;

    vvMatrix getReprojectionMatrix() const;

  private:
    int  _depthPrecision; // 8: 8 bit int, 16: 16 bit int, 32: 32 bit float
    CodeType  _depthCodeType;
    int _codedDepthSize;
    int  _uncertaintyPrecision; // 8: 8 bit int, 16: 16 bit int, 32: 32 bit float
    CodeType  _uncertaintyCodeType;
    int _codedUncertaintySize;
    uchar*          _pixeldepth;
    uchar*          _codeddepth;
    uchar*          _uncertainty;
    uchar*          _codeduncertainty;

    vvMatrix        _modelViewMatrix;
    vvMatrix        _projectionMatrix;
    vvGLTools::Viewport _viewport;
    float           _depthMin;
    float           _depthMax;
};

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
