// DeskVOX - Volume Exploration Utility for the Desktop
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
// 
// This file is part of DeskVOX.
//
// DeskVOX is free software; you can redistribute it and/or
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

#ifndef _VV_SLICEVIEWER_H_
#define _VV_SLICEVIEWER_H_

// FOX:
#include <fx.h>

// C++:
#include <iostream>

// Local:
#include "vvshell.h"
#include "vvcanvas.h"

class VVSliceViewer : public FXDialogBox
{
  FXDECLARE(VVSliceViewer)
  
  protected:
    VVSliceViewer(){}
    VVSliceViewer(const VVSliceViewer&){};
  
  public:
    const static int SLICE_WIDTH;
    const static int SLICE_HEIGHT;
    enum
    {
      ID_SLICE=FXDialogBox::ID_LAST,
      ID_BEGINNING,
      ID_BACK,
      ID_FORWARD,
      ID_END,
      ID_SLICE_CANVAS,
      ID_MIRROR_X,
      ID_MIRROR_Y,
      ID_X_AXIS,
      ID_Y_AXIS,
      ID_Z_AXIS,
      ID_LAST
    };
    vox::vvCanvas* _canvas;
    VVShell* _shell;
    FXCheckButton* _mirrorX;
    FXCheckButton* _mirrorY;
    FXCanvas* _sliceCanvas;
    FXSlider* _sliceSlider;
    FXLabel* _resolutionLabel;
    FXLabel* _sliceLabel;
    FXRadioButton* _xAxisButton;
    FXRadioButton* _yAxisButton;
    FXRadioButton* _zAxisButton;
    vvVolDesc::AxisType _axis;

    VVSliceViewer(FXWindow*, vox::vvCanvas*);
    virtual ~VVSliceViewer();
    long onCmdMirrorX(FXObject*,FXSelector,void*);
    long onCmdMirrorY(FXObject*,FXSelector,void*);
    long onCmdBeginning(FXObject*,FXSelector,void*);
    long onCmdBack(FXObject*,FXSelector,void*);
    long onCmdForward(FXObject*,FXSelector,void*);
    long onCmdEnd(FXObject*,FXSelector,void*);
    long onCmdXAxis(FXObject*,FXSelector,void*);
    long onCmdYAxis(FXObject*,FXSelector,void*);
    long onCmdZAxis(FXObject*,FXSelector,void*);
    long onChngSlice(FXObject*,FXSelector,void*);
    long onPaint(FXObject*,FXSelector,void*);
    void selectSlice(int);
    void showSlice(int);
    void updateValues();
};

#endif
