// DeskVOX - Volume Exploration Utility for the Desktop
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, schulze@cs.brown.edu
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

#ifndef _VV_TRANSWINDOW_H_
#define _VV_TRANSWINDOW_H_

// FOX:
#include <fx.h>

// Virvo:
#include <vvtransfunc.h>

// Compiler:
#include <iostream>
#include <string.h>

// Local:
#include "vvshell.h"
#include "vvcanvas.h"

class vvTransFunc;

class VVTransferWindow : public FXDialogBox
{
  FXDECLARE(VVTransferWindow)
  
  public:
    enum
    {
      ID_PYRAMID=FXDialogBox::ID_LAST,
      ID_BELL,
      ID_COLOR,
      ID_SKIP,
      ID_P_TOP_X,
      ID_P_BOTTOM_X,
      ID_P_TOP_Y,
      ID_P_BOTTOM_Y,
      ID_P_MAX,
      ID_OWN_COLOR,
      ID_B_WIDTH,
      ID_B_HEIGHT,
      ID_B_MAX,
      ID_S_WIDTH,
      ID_S_HEIGHT,
      ID_TF_CANVAS_1D,
      ID_TF_CANVAS_2D,
      ID_DELETE,
      ID_UNDO,
      ID_PICK_COLOR,
      ID_COLOR_PICKER,
      ID_COLOR_COMBO,
      ID_ALPHA_COMBO,
      ID_INSTANT,
      ID_HIST_ALL,
      ID_HIST_FIRST,
      ID_HIST_NONE,
      ID_APPLY,
      ID_DIS_COLOR,
      ID_TF_BOOK,
      ID_PIN_BOOK,
      ID_NORMALIZATION,
      ID_LAST
    };
    enum WidgetType
    {
      COLOR,
      PYRAMID,
      BELL,
      SKIP
    };

    VVTransferWindow(FXWindow*, vox::vvCanvas*);
    virtual ~VVTransferWindow();

    long onCmdColor(FXObject*,FXSelector,void*);
    long onCmdPyramid(FXObject*,FXSelector,void*);
    long onCmdBell(FXObject*,FXSelector,void*);
    long onCmdSkip(FXObject*,FXSelector,void*);
    long onCmdDelete(FXObject*,FXSelector,void*);
    long onCmdUndo(FXObject*,FXSelector,void*);
    long onCmdApply(FXObject*,FXSelector,void*);
    long onCmdColorCombo(FXObject*,FXSelector,void*);
    long onCmdAlphaCombo(FXObject*,FXSelector,void*);
    long onCmdInstant(FXObject*,FXSelector,void*);
    long onCmdOwnColor(FXObject*,FXSelector,void*);
    long onCmdHist(FXObject*,FXSelector,void*);
    long onChngPyramid(FXObject*,FXSelector,void*);
    long onChngBell(FXObject*,FXSelector,void*);
    long onChngSkip(FXObject*,FXSelector,void*);
    long onChngPos(FXObject*,FXSelector,void*);
    long onMouseDown1D(FXObject*,FXSelector,void*);
    long onMouseUp1D(FXObject*,FXSelector,void*);
    long onMouseMove1D(FXObject*,FXSelector,void*);
    long onMouseDown2D(FXObject*,FXSelector,void*);
    long onMouseUp2D(FXObject*,FXSelector,void*);
    long onMouseMove2D(FXObject*,FXSelector,void*);
    long onChngDisColors(FXObject*,FXSelector,void*);
    long onTFCanvasPaint(FXObject*,FXSelector,void*);
    long onCmdHistAll(FXObject*,FXSelector,void*);
    long onCmdHistFirst(FXObject*,FXSelector,void*);
    long onCmdHistNone(FXObject*,FXSelector,void*);
    long onCmdPickColor(FXObject*,FXSelector,void*);
    long onCmdNormalization(FXObject*,FXSelector,void*);
    long onChngPickerColor(FXObject*,FXSelector,void*);
    void updateValues();

  protected:
    static const FXColor BLACK;
    static const FXColor WHITE;
    static const float CLICK_TOLERANCE;
    static const int COLORBAR_HEIGHT;
    const static int TF_WIDTH;
    const static int TF_HEIGHT;
    FXGLVisual* _glVisual1D;
    FXGLCanvas* _glCanvas1D;
    FXGLVisual* _glVisual2D;
    FXGLCanvas* _glCanvas2D;
    FXComboBox* _colorCombo;
    FXComboBox* _alphaCombo;
    FXSlider* _disColorSlider;
    FXRealSlider* _pTopXSlider;
    FXRealSlider* _pBottomXSlider;
    FXRealSlider* _pTopYSlider;
    FXRealSlider* _pBottomYSlider;
    FXRealSlider* _pMaxSlider;
    FXRealSlider* _bWidthSlider;
    FXRealSlider* _bHeightSlider;
    FXRealSlider* _bMaxSlider;
    FXRealSlider* _sWidthSlider;
    FXRealSlider* _sHeightSlider;
    FXLabel* _pTopXLabel;
    FXLabel* _pBottomXLabel;
    FXLabel* _pTopYLabel;
    FXLabel* _pBottomYLabel;
    FXLabel* _pMaxLabel;
    FXLabel* _bWidthLabel;
    FXLabel* _bHeightLabel;
    FXLabel* _bMaxLabel;
    FXLabel* _sWidthLabel;
    FXLabel* _sHeightLabel;
    FXLabel* _realMinLabel;
    FXLabel* _realPosLabel;
    FXLabel* _realMaxLabel;
    FXLabel* _disColorLabel;
    FXRadioButton* _histAll;
    FXRadioButton* _histFirst;
    FXRadioButton* _histNone;
    FXCheckButton* _instantButton;
    FXCheckButton* _pColorButton;
    FXCheckButton* _bColorButton;
    FXCheckButton* _cbNorm;
    FXSwitcher* _pinSwitcher;
    FXColorDialog* _colorPicker;
    FXTabBook* _tfBook;

    VVShell* _shell;
    vox::vvCanvas* _canvas;
    vvTFWidget* _currentWidget;
    uchar* _histoTexture1D;
    uchar* _histoTexture2D;

    VVTransferWindow(){}
    VVTransferWindow(const VVTransferWindow&){};
    void initGL();
    void updateLabels();
    void computeHistogram();
    void updateTransFunc();
    void newWidget(WidgetType);
    void drawTF();
    void draw1DTF();
    void draw2DTF();
    void drawColorTexture();
    void drawAlphaTexture();
    void drawHistogram();
    void draw2DTFTexture();
    void draw2DTFWidgets();
    void draw2DWidget(vvTFWidget*);
    void drawPinLines();
    void drawPinBackground();
    void drawPinLine(vvTFWidget*);
    vvTFWidget* closestWidget(float, float, float);
    float getRealPinPos(float);
};
#endif
