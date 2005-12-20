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

#pragma warning(disable: 4244)    // disable warning about conversion from int to short
#pragma warning(disable: 4512)    // disable warning: assignment operator could not be generated
#pragma warning(disable: 4800)    // disable warning about forcing value to bool

// Virvo:
#include <vvdebugmsg.h>
#include <vvfileio.h>

// Local:
#include "vvtranswindow.h"
#include "vvcanvas.h"
#include "vvshell.h"

using namespace vox;

const FXColor VVTransferWindow::BLACK = FXRGB(0,0,0);
const FXColor VVTransferWindow::WHITE = FXRGB(255,255,255);
const float VVTransferWindow::CLICK_TOLERANCE = 0.05f; // [TF space]
const int VVTransferWindow::TF_WIDTH  = 512;
const int VVTransferWindow::TF_HEIGHT = 256;
const int VVTransferWindow::COLORBAR_HEIGHT = 20;

/*******************************************************************************/
FXDEFMAP(VVTransferWindow) VVTransferWindowMap[]=
{
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_PYRAMID,       VVTransferWindow::onCmdPyramid),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_BELL,          VVTransferWindow::onCmdBell),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_SKIP,          VVTransferWindow::onCmdSkip),
  FXMAPFUNC(SEL_CHANGED,          VVTransferWindow::ID_P_TOP_X,       VVTransferWindow::onChngPyramid),
  FXMAPFUNC(SEL_CHANGED,          VVTransferWindow::ID_P_BOTTOM_X,    VVTransferWindow::onChngPyramid),
  FXMAPFUNC(SEL_CHANGED,          VVTransferWindow::ID_P_TOP_Y,       VVTransferWindow::onChngPyramid),
  FXMAPFUNC(SEL_CHANGED,          VVTransferWindow::ID_P_BOTTOM_Y,    VVTransferWindow::onChngPyramid),
  FXMAPFUNC(SEL_CHANGED,          VVTransferWindow::ID_P_MAX,         VVTransferWindow::onChngPyramid),
  FXMAPFUNC(SEL_CHANGED,          VVTransferWindow::ID_B_WIDTH,       VVTransferWindow::onChngBell),
  FXMAPFUNC(SEL_CHANGED,          VVTransferWindow::ID_B_HEIGHT,      VVTransferWindow::onChngBell),
  FXMAPFUNC(SEL_CHANGED,          VVTransferWindow::ID_B_MAX,         VVTransferWindow::onChngBell),
  FXMAPFUNC(SEL_CHANGED,          VVTransferWindow::ID_S_WIDTH,       VVTransferWindow::onChngSkip),
  FXMAPFUNC(SEL_CHANGED,          VVTransferWindow::ID_S_HEIGHT,      VVTransferWindow::onChngSkip),
  FXMAPFUNC(SEL_CHANGED,          VVTransferWindow::ID_DIS_COLOR,     VVTransferWindow::onChngDisColors),
  FXMAPFUNC(SEL_PAINT,            VVTransferWindow::ID_TF_CANVAS_1D,  VVTransferWindow::onTFCanvasPaint),
  FXMAPFUNC(SEL_PAINT,            VVTransferWindow::ID_TF_CANVAS_2D,  VVTransferWindow::onTFCanvasPaint),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_DELETE,        VVTransferWindow::onCmdDelete),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_UNDO,          VVTransferWindow::onCmdUndo),
  FXMAPFUNC(SEL_LEFTBUTTONPRESS,  VVTransferWindow::ID_TF_CANVAS_1D,  VVTransferWindow::onMouseDown1D),
  FXMAPFUNC(SEL_LEFTBUTTONRELEASE,VVTransferWindow::ID_TF_CANVAS_1D,  VVTransferWindow::onMouseUp1D),
  FXMAPFUNC(SEL_MOTION,           VVTransferWindow::ID_TF_CANVAS_1D,  VVTransferWindow::onMouseMove1D),
  FXMAPFUNC(SEL_LEFTBUTTONPRESS,  VVTransferWindow::ID_TF_CANVAS_2D,  VVTransferWindow::onMouseDown2D),
  FXMAPFUNC(SEL_LEFTBUTTONRELEASE,VVTransferWindow::ID_TF_CANVAS_2D,  VVTransferWindow::onMouseUp2D),
  FXMAPFUNC(SEL_MOTION,           VVTransferWindow::ID_TF_CANVAS_2D,  VVTransferWindow::onMouseMove2D),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_COLOR_COMBO,   VVTransferWindow::onCmdColorCombo),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_ALPHA_COMBO,   VVTransferWindow::onCmdAlphaCombo),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_INSTANT,       VVTransferWindow::onCmdInstant),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_OWN_COLOR,     VVTransferWindow::onCmdOwnColor),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_APPLY,         VVTransferWindow::onCmdApply),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_IMPORT,        VVTransferWindow::onCmdImport),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_COLOR,         VVTransferWindow::onCmdColor),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_HIST_ALL,      VVTransferWindow::onCmdHistAll),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_HIST_FIRST,    VVTransferWindow::onCmdHistFirst),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_HIST_NONE,     VVTransferWindow::onCmdHistNone),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_PICK_COLOR,    VVTransferWindow::onCmdPickColor),
  FXMAPFUNC(SEL_COMMAND,          VVTransferWindow::ID_NORMALIZATION, VVTransferWindow::onCmdNormalization),
  FXMAPFUNC(SEL_CHANGED,          VVTransferWindow::ID_COLOR_PICKER,  VVTransferWindow::onChngPickerColor),
};

FXIMPLEMENT(VVTransferWindow,FXDialogBox,VVTransferWindowMap,ARRAYNUMBER(VVTransferWindowMap))

// Construct a dialog box
VVTransferWindow::VVTransferWindow(FXWindow* owner, vvCanvas* c) :
  FXDialogBox(owner, "Transfer Function", DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE, 100, 100)
{
  _canvas = c;
  _shell = (VVShell*)owner;

  FXVerticalFrame* master = new FXVerticalFrame(this,LAYOUT_FILL_X|LAYOUT_FILL_Y);
  _tfBook = new FXTabBook(master,this,ID_TF_BOOK,PACK_UNIFORM_WIDTH|PACK_UNIFORM_HEIGHT|LAYOUT_FILL_X|LAYOUT_FILL_Y|LAYOUT_RIGHT);
  
  // Tab page 1:
  FXTabItem* tab1=new FXTabItem(_tfBook,"&1D Transfer Function",NULL);
  FXVerticalFrame* page1 = new FXVerticalFrame(_tfBook,FRAME_THICK|FRAME_RAISED|LAYOUT_FILL_X|LAYOUT_FILL_Y);

  FXVerticalFrame* glpanel = new FXVerticalFrame(page1, FRAME_SUNKEN|LAYOUT_SIDE_LEFT|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT, 0,0,TF_WIDTH,160);
  _glVisual1D = new FXGLVisual(getApp(), VISUAL_DOUBLEBUFFER);
  _glCanvas1D = new FXGLCanvas(glpanel, _glVisual1D, this, ID_TF_CANVAS_1D, LAYOUT_FILL_X|LAYOUT_FILL_Y|LAYOUT_TOP|LAYOUT_LEFT);

  FXHorizontalFrame* legendFrame = new FXHorizontalFrame(page1, LAYOUT_FILL_X);
  _realMinLabel = new FXLabel(legendFrame, "Min = 0", NULL, LABEL_NORMAL | LAYOUT_LEFT);
  _realPosLabel = new FXLabel(legendFrame, "", NULL, LABEL_NORMAL | LAYOUT_CENTER_X);
  _realMaxLabel = new FXLabel(legendFrame, "Max = 1", NULL, LABEL_NORMAL | LAYOUT_RIGHT);

  // Tab page 2:
  FXTabItem* tab2=new FXTabItem(_tfBook,"&2D Transfer Function",NULL);
  FXVerticalFrame* page2 = new FXVerticalFrame(_tfBook,FRAME_THICK|FRAME_RAISED|LAYOUT_FILL_X|LAYOUT_FILL_Y);
  _glVisual2D = new FXGLVisual(getApp(), VISUAL_DOUBLEBUFFER);
  _glCanvas2D = new FXGLCanvas(page2, _glVisual2D, this, ID_TF_CANVAS_2D, FRAME_SUNKEN | LAYOUT_FIX_HEIGHT | LAYOUT_FIX_WIDTH, 0, 0, TF_WIDTH,TF_HEIGHT);

  // Common elements:
  FXHorizontalFrame* buttonFrame = new FXHorizontalFrame(master, LAYOUT_FILL_Y | LAYOUT_CENTER_X | PACK_UNIFORM_WIDTH);
  new FXButton(buttonFrame,"Color",      NULL,this,ID_COLOR,  FRAME_RAISED|FRAME_THICK,0,0,0,0,20,20);   // sets width for all buttons
  new FXButton(buttonFrame,"Pyramid",    NULL,this,ID_PYRAMID,FRAME_RAISED|FRAME_THICK);
  new FXButton(buttonFrame,"Gaussian",   NULL,this,ID_BELL,   FRAME_RAISED|FRAME_THICK);
  new FXButton(buttonFrame,"Skip Range", NULL,this,ID_SKIP,   FRAME_RAISED|FRAME_THICK);
  new FXButton(buttonFrame,"Delete",     NULL,this,ID_DELETE, FRAME_RAISED|FRAME_THICK);
  new FXButton(buttonFrame,"Undo",       NULL,this,ID_UNDO,   FRAME_RAISED|FRAME_THICK);

  FXHorizontalFrame* controlFrame=new FXHorizontalFrame(master,LAYOUT_FILL_X);
  FXVerticalFrame* comboFrame=new FXVerticalFrame(controlFrame,LAYOUT_FILL_Y);

  FXGroupBox *colorComboGP=new FXGroupBox(comboFrame,"Preset Colors", FRAME_GROOVE | LAYOUT_FILL_X);
  _colorCombo=new FXComboBox(colorComboGP,5,this,ID_COLOR_COMBO,COMBOBOX_INSERT_LAST|COMBOBOX_STATIC|FRAME_SUNKEN|FRAME_THICK|LAYOUT_FILL_X);
  _colorCombo->appendItem("Bright Colors");
  _colorCombo->appendItem("Hue Gradient");
  _colorCombo->appendItem("Grayscale");
  _colorCombo->appendItem("White");
  _colorCombo->appendItem("Red");
  _colorCombo->appendItem("Green");
  _colorCombo->appendItem("Blue");
  _colorCombo->setNumVisible(_colorCombo->getNumItems());
  _colorCombo->setCurrentItem(0);

  FXGroupBox *alphaComboGP=new FXGroupBox(comboFrame,"Preset Alpha", FRAME_GROOVE | LAYOUT_FILL_X);
  _alphaCombo=new FXComboBox(alphaComboGP,5,this,ID_ALPHA_COMBO,COMBOBOX_INSERT_LAST|COMBOBOX_STATIC|FRAME_SUNKEN|FRAME_THICK|LAYOUT_FILL_X);
  _alphaCombo->appendItem("Ascending");
  _alphaCombo->appendItem("Descending");
  _alphaCombo->appendItem("Opaque");
  _alphaCombo->setNumVisible(_alphaCombo->getNumItems());
  _alphaCombo->setCurrentItem(0);

  _pinSwitcher = new FXSwitcher(controlFrame, LAYOUT_FILL_X | LAYOUT_FILL_Y);

  // Switcher state #0:
  new FXVerticalFrame(_pinSwitcher, LAYOUT_FILL_X | LAYOUT_FILL_Y);

  // Switcher state #1:
  FXGroupBox* pinGroup1 = new FXGroupBox(_pinSwitcher,"Color settings",FRAME_GROOVE|LAYOUT_FILL_X|LAYOUT_FILL_Y);
  FXVerticalFrame* colorFrame = new FXVerticalFrame(pinGroup1, LAYOUT_FILL_X);
  new FXButton(colorFrame,"Select color",NULL,this,ID_PICK_COLOR, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);

  // Switcher state #2:
  FXGroupBox* pinGroup2 = new FXGroupBox(_pinSwitcher,"Pyramid settings",FRAME_GROOVE|LAYOUT_FILL_X|LAYOUT_FILL_Y);
  FXVerticalFrame* sliderFrame1 = new FXVerticalFrame(pinGroup2, LAYOUT_FILL_X);

  FXMatrix* pTopXMat = new FXMatrix(sliderFrame1, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(pTopXMat, "Top width X: ",NULL,LABEL_NORMAL);
  _pTopXLabel = new FXLabel(pTopXMat, "0",NULL,LABEL_NORMAL);
  _pTopXSlider=new FXRealSlider(sliderFrame1,this,ID_P_TOP_X,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _pTopXSlider->setRange(0.0f, 2.0f);
  _pTopXSlider->setValue(0.0f);
  _pTopXSlider->setTickDelta(.01);

  FXMatrix* pBottomXMat = new FXMatrix(sliderFrame1, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(pBottomXMat, "Bottom width X: ",NULL,LABEL_NORMAL);
  _pBottomXLabel = new FXLabel(pBottomXMat, "0",NULL,LABEL_NORMAL);
  _pBottomXSlider=new FXRealSlider(sliderFrame1,this,ID_P_BOTTOM_X, SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _pBottomXSlider->setRange(0.0f, 2.0f);
  _pBottomXSlider->setValue(0.0f);
  _pBottomXSlider->setTickDelta(.01);

  FXMatrix* pTopYMat = new FXMatrix(sliderFrame1, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(pTopYMat, "Top width Y: ",NULL,LABEL_NORMAL);
  _pTopYLabel = new FXLabel(pTopYMat, "0",NULL,LABEL_NORMAL);
  _pTopYSlider=new FXRealSlider(sliderFrame1,this,ID_P_TOP_Y,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _pTopYSlider->setRange(0.0f, 2.0f);
  _pTopYSlider->setValue(0.0f);
  _pTopYSlider->setTickDelta(.01);

  FXMatrix* pBottomYMat = new FXMatrix(sliderFrame1, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(pBottomYMat, "Bottom width Y: ",NULL,LABEL_NORMAL);
  _pBottomYLabel = new FXLabel(pBottomYMat, "0",NULL,LABEL_NORMAL);
  _pBottomYSlider=new FXRealSlider(sliderFrame1,this,ID_P_BOTTOM_Y, SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _pBottomYSlider->setRange(0.0f, 2.0f);
  _pBottomYSlider->setValue(0.0f);
  _pBottomYSlider->setTickDelta(.01);

  FXMatrix* pMaxMat = new FXMatrix(sliderFrame1, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(pMaxMat, "Maximum opacity: ",NULL,LABEL_NORMAL);
  _pMaxLabel = new FXLabel(pMaxMat, "0",NULL,LABEL_NORMAL);
  _pMaxSlider=new FXRealSlider(sliderFrame1,this,ID_P_MAX,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _pMaxSlider->setRange(0.0f, 1.0f);
  _pMaxSlider->setValue(0.0f);
  _pMaxSlider->setTickDelta(.01);

  FXHorizontalFrame* pColorFrame = new FXHorizontalFrame(sliderFrame1);
  _pColorButton = new FXCheckButton(pColorFrame,"Has own color",this,ID_OWN_COLOR,ICON_BEFORE_TEXT|LAYOUT_LEFT);
  new FXButton(pColorFrame,"Pick color",NULL,this,ID_PICK_COLOR, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);

  // Switcher state #3:
  FXGroupBox* pinGroup3 = new FXGroupBox(_pinSwitcher,"Gaussian settings",FRAME_GROOVE|LAYOUT_FILL_X|LAYOUT_FILL_Y);
  FXVerticalFrame* sliderFrame2 = new FXVerticalFrame(pinGroup3, LAYOUT_FILL_X);

  FXMatrix* _bWidthMat = new FXMatrix(sliderFrame2, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(_bWidthMat, "Width: ",NULL,LABEL_NORMAL);
  _bWidthLabel = new FXLabel(_bWidthMat, "0",NULL,LABEL_NORMAL);
  _bWidthSlider=new FXRealSlider(sliderFrame2,this,ID_B_WIDTH,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _bWidthSlider->setRange(0.0f, 1.0f);
  _bWidthSlider->setValue(0.0f);
  _bWidthSlider->setTickDelta(.01);

  FXMatrix* _bHeightMat = new FXMatrix(sliderFrame2, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(_bHeightMat, "Height: ",NULL,LABEL_NORMAL);
  _bHeightLabel = new FXLabel(_bHeightMat, "0",NULL,LABEL_NORMAL);
  _bHeightSlider=new FXRealSlider(sliderFrame2,this,ID_B_HEIGHT,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _bHeightSlider->setRange(0.0f, 1.0f);
  _bHeightSlider->setValue(0.0f);
  _bHeightSlider->setTickDelta(.01);

  FXMatrix* _bMaxMat = new FXMatrix(sliderFrame2, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(_bMaxMat, "Maximum value: ",NULL,LABEL_NORMAL);
  _bMaxLabel = new FXLabel(_bMaxMat, "0",NULL,LABEL_NORMAL);
  _bMaxSlider=new FXRealSlider(sliderFrame2,this,ID_B_MAX,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _bMaxSlider->setRange(0.0f, 5.0f);
  _bMaxSlider->setValue(0.0f);
  _bMaxSlider->setTickDelta(.01);

  FXHorizontalFrame* bColorFrame = new FXHorizontalFrame(sliderFrame2);
  _bColorButton = new FXCheckButton(bColorFrame,"Has own color",this,ID_OWN_COLOR,ICON_BEFORE_TEXT|LAYOUT_LEFT);
  new FXButton(bColorFrame,"Pick color",NULL,this,ID_PICK_COLOR, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);

  // Switcher state #4:
  FXGroupBox* pinGroup4 = new FXGroupBox(_pinSwitcher,"Skip Range Settings",FRAME_GROOVE|LAYOUT_FILL_X|LAYOUT_FILL_Y);
  FXVerticalFrame* sliderFrame3 = new FXVerticalFrame(pinGroup4, LAYOUT_FILL_X);

  FXMatrix* _sWidthMat = new FXMatrix(sliderFrame3, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(_sWidthMat, "Width: ",NULL,LABEL_NORMAL);
  _sWidthLabel = new FXLabel(_sWidthMat, "0",NULL,LABEL_NORMAL);
  _sWidthSlider=new FXRealSlider(sliderFrame3,this,ID_S_WIDTH,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _sWidthSlider->setRange(0.0f, 1.0f);
  _sWidthSlider->setValue(0.0f);
  _sWidthSlider->setTickDelta(.01);

  FXMatrix* _sHeightMat = new FXMatrix(sliderFrame3, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);
  new FXLabel(_sHeightMat, "Height: ",NULL,LABEL_NORMAL);
  _sHeightLabel = new FXLabel(_sHeightMat, "0",NULL,LABEL_NORMAL);
  _sHeightSlider=new FXRealSlider(sliderFrame3,this,ID_S_HEIGHT,SLIDER_HORIZONTAL|SLIDER_ARROW_DOWN|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
  _sHeightSlider->setRange(0.0f, 1.0f);
  _sHeightSlider->setValue(0.0f);
  _sHeightSlider->setTickDelta(.01);

  // Continue with pin independent widgets:
  _cbNorm = new FXCheckButton(master, "Logarithmic histogram normalization", this, ID_NORMALIZATION, ICON_BEFORE_TEXT|LAYOUT_LEFT);
  _cbNorm->setCheck(true);

  FXGroupBox* histoGroup = new FXGroupBox(master,"Display",FRAME_GROOVE | LAYOUT_FILL_X);
  FXHorizontalFrame* histoFrame = new FXHorizontalFrame(histoGroup, LAYOUT_CENTER_X | PACK_UNIFORM_WIDTH);
  _histNone  = new FXRadioButton(histoFrame,"Opacity",this,ID_HIST_NONE, ICON_BEFORE_TEXT, 0, 0, 0, 0, 20, 20);
  _histFirst = new FXRadioButton(histoFrame,"Histogram for first time step",this,ID_HIST_FIRST, ICON_BEFORE_TEXT);
  _histAll   = new FXRadioButton(histoFrame,"Histogram for all time steps",this,ID_HIST_ALL, ICON_BEFORE_TEXT);
  _histNone->setCheck(true);

  FXHorizontalFrame* disColorFrame = new FXHorizontalFrame(master,LAYOUT_FILL_X);
  new FXLabel(disColorFrame, "Discrete Colors:",NULL,LABEL_NORMAL);
  _disColorSlider = new FXSlider(disColorFrame,this, ID_DIS_COLOR, SLIDER_HORIZONTAL | SLIDER_ARROW_DOWN | LAYOUT_FILL_X);
  _disColorSlider->setRange(0,64);
  _disColorSlider->setValue(0);
  _disColorSlider->setTickDelta(1);
  _disColorLabel = new FXLabel(disColorFrame, "",NULL,LABEL_NORMAL);

  FXHorizontalFrame* endFrame=new FXHorizontalFrame(master, LAYOUT_FILL_X | PACK_UNIFORM_WIDTH);
  _instantButton = new FXCheckButton(endFrame,"Instant Classification",this,ID_INSTANT,ICON_BEFORE_TEXT|LAYOUT_LEFT);
  new FXButton(endFrame,"Import TF",NULL,this,ID_IMPORT, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);
  new FXButton(endFrame,"Apply",NULL,this,ID_APPLY, FRAME_RAISED | FRAME_THICK | LAYOUT_CENTER_X, 0, 0, 0, 0, 20, 20);
  new FXButton(endFrame,"Close",NULL,this,ID_ACCEPT, FRAME_RAISED | FRAME_THICK | LAYOUT_RIGHT);

  // Initialize color picker:
  _colorPicker  = new FXColorDialog((FXWindow*)this, "Pin Color",DECOR_TITLE|DECOR_BORDER, 50,50);
  _colorPicker->setTarget(this);
  _colorPicker->setSelector(ID_COLOR_PICKER);
  _colorPicker->setOpaqueOnly(true);

  _currentWidget = NULL;
  _histoTexture1D = NULL;
  _histoTexture2D = NULL;
}

// Must delete the menus
VVTransferWindow::~VVTransferWindow()
{
  delete _glVisual1D;
}

void VVTransferWindow::initGL()
{
  glDrawBuffer(GL_BACK);         // set draw buffer to front in order to read image data
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 1.0f, -1.0f);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

long VVTransferWindow::onTFCanvasPaint(FXObject*,FXSelector,void*)
{
  if (_glCanvas1D->makeCurrent())
  {
    initGL();
    _glCanvas1D->makeNonCurrent();
  }
  if (_glCanvas2D->makeCurrent())
  {
    initGL();
    _glCanvas2D->makeNonCurrent();
  }
  computeHistogram();
  drawTF();
  return 1;
}

long VVTransferWindow::onCmdColor(FXObject*,FXSelector,void*)
{
  newWidget(COLOR);
  return 1;
}

long VVTransferWindow::onCmdPyramid(FXObject*,FXSelector,void*)
{
  newWidget(PYRAMID);
  return 1;
}

long VVTransferWindow::onCmdBell(FXObject*,FXSelector,void*)
{
  newWidget(BELL);
  return 1;
}

long VVTransferWindow::onCmdSkip(FXObject*,FXSelector,void*)
{
  newWidget(SKIP);
  return 1;
}

long VVTransferWindow::onCmdPickColor(FXObject*,FXSelector,void*)
{
  float r=0.0f, g=0.0f, b=0.0f;
  vvTFColor* cw=NULL;
  vvTFPyramid* pw=NULL;
  vvTFBell* bw=NULL;

  if(!_currentWidget) return 1;
  
  // Find out widget type:
  if ((cw = dynamic_cast<vvTFColor*>(_currentWidget))   != NULL ||
      (pw = dynamic_cast<vvTFPyramid*>(_currentWidget)) != NULL ||
      (bw = dynamic_cast<vvTFBell*>(_currentWidget))    != NULL)
  {
    // Get widget color:
    if (cw) cw->_col.getRGB(r,g,b);
    else if (pw) pw->_col.getRGB(r,g,b);
    else if (bw) bw->_col.getRGB(r,g,b);
    else assert(0);

    // Set color picker color:   
    _colorPicker->setRGBA(FXRGBA(r*255,g*255,b*255,255));
    if (_colorPicker->execute() == 0)   // has picker exited with 'cancel'?
    {
      // Undo changes to color:
      if (cw) cw->_col.setRGB(r,g,b);
      else if (pw) pw->_col.setRGB(r,g,b);
      else if (bw) bw->_col.setRGB(r,g,b);
       drawTF();
      if(_instantButton->getCheck()) updateTransFunc();
    }
  }
  return 1;
}

long VVTransferWindow::onChngPickerColor(FXObject*,FXSelector,void*)
{
  vvTFColor* cw=NULL;
  vvTFPyramid* pw=NULL;
  vvTFBell* bw=NULL;

  if(!_currentWidget) return 1;
  
  // Find out widget type:
  if ((cw = dynamic_cast<vvTFColor*>(_currentWidget))   != NULL ||
      (pw = dynamic_cast<vvTFPyramid*>(_currentWidget)) != NULL ||
      (bw = dynamic_cast<vvTFBell*>(_currentWidget))    != NULL)
  {
    FXColor col = _colorPicker->getRGBA();
    float r = float(FXREDVAL(col))   / 255.0f;
    float g = float(FXGREENVAL(col)) / 255.0f;
    float b = float(FXBLUEVAL(col))  / 255.0f;
    if (cw) cw->_col.setRGB(r,g,b);
    else if (pw) pw->_col.setRGB(r,g,b);
    else if (bw) bw->_col.setRGB(r,g,b);
    drawTF();
    if(_instantButton->getCheck()) updateTransFunc();
  }  
 
  return 1;
}

long VVTransferWindow::onChngPyramid(FXObject*,FXSelector,void*)
{
  vvTFPyramid* pw;

  _pTopXLabel->setText(FXStringFormat("%.2f", _pTopXSlider->getValue()));
  _pTopYLabel->setText(FXStringFormat("%.2f", _pTopYSlider->getValue()));
  _pBottomXLabel->setText(FXStringFormat("%.2f", _pBottomXSlider->getValue()));
  _pBottomYLabel->setText(FXStringFormat("%.2f", _pBottomYSlider->getValue()));
  _pMaxLabel->setText(FXStringFormat("%.2f",_pMaxSlider->getValue()));
  if(!_currentWidget) return 1;
  assert((pw=dynamic_cast<vvTFPyramid*>(_currentWidget))!=NULL);
  pw->_top[0] = _pTopXSlider->getValue();
  pw->_top[1] = _pTopYSlider->getValue();
  pw->_bottom[0] = _pBottomXSlider->getValue();
  pw->_bottom[1] = _pBottomYSlider->getValue();
  pw->_opacity = _pMaxSlider->getValue();
  drawTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onChngBell(FXObject*,FXSelector,void*)
{
  vvTFBell* bw;

  _bWidthLabel->setText(FXStringFormat("%.2f", _bWidthSlider->getValue()));
  _bHeightLabel->setText(FXStringFormat("%.2f", _bHeightSlider->getValue()));
  _bMaxLabel->setText(FXStringFormat("%.2f", _bMaxSlider->getValue()));
  if(!_currentWidget) return 1;
  assert((bw=dynamic_cast<vvTFBell*>(_currentWidget))!=NULL);
  bw->_size[0] = _bWidthSlider->getValue();
  bw->_size[1] = _bHeightSlider->getValue();
  bw->_opacity = _bMaxSlider->getValue();
  drawTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onChngSkip(FXObject*,FXSelector,void*)
{
  vvTFSkip* sw;

  _sWidthLabel->setText(FXStringFormat("%.2f", _sWidthSlider->getValue()));
  _sHeightLabel->setText(FXStringFormat("%.2f", _sHeightSlider->getValue()));
  if(!_currentWidget) return 1;
  assert((sw=dynamic_cast<vvTFSkip*>(_currentWidget))!=NULL);
  sw->_size[0] = _sWidthSlider->getValue();
  sw->_size[1] = _sHeightSlider->getValue();
  drawTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

float VVTransferWindow::getRealPinPos(float sliderVal)
{
  return _canvas->_vd->real[0] + sliderVal * (_canvas->_vd->real[1] - _canvas->_vd->real[0]);
}

long VVTransferWindow::onMouseDown1D(FXObject*,FXSelector,void* ptr)
{
  if(!_canvas || _canvas->_vd->tf._widgets.count() == 0)
  {
    _currentWidget = NULL;
    return 1;
  }
  _canvas->_vd->tf.putUndoBuffer();
  _glCanvas1D->grab();
  FXEvent* ev = (FXEvent*)ptr;
  _currentWidget = closestWidget(float(ev->win_x) / float(_glCanvas1D->getWidth()), 
                                 1.0f - float(ev->win_y) / float(_glCanvas1D->getHeight()), 
                                 -1.0f);
  updateLabels();
  drawTF();
  return 1;
}

long VVTransferWindow::onMouseUp1D(FXObject*,FXSelector,void*)
{
  _glCanvas1D->ungrab();
  return 1;
}

long VVTransferWindow::onMouseMove1D(FXObject*, FXSelector, void* ptr)
{
  if (!_glCanvas1D->grabbed()) return 1;
  if(!_currentWidget || !_canvas) return 1;
  if(_canvas->_vd->tf._widgets.count() == 0) return 1;
  FXEvent* ev = (FXEvent*)ptr;
  float pos = ts_clamp(float(ev->win_x) / float(_glCanvas1D->getWidth()), 0.0f, 1.0f);
  _realPosLabel->setText(FXStringFormat("Pin = %.5g", getRealPinPos(pos)));
  _currentWidget->_pos[0] = pos;
  drawTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onMouseDown2D(FXObject*,FXSelector,void* ptr)
{
  if(!_canvas || _canvas->_vd->tf._widgets.count() == 0)
  {
    _currentWidget = NULL;
    return 1;
  }
  _canvas->_vd->tf.putUndoBuffer();
  _glCanvas2D->grab();
  FXEvent* ev = (FXEvent*)ptr;
  _currentWidget = closestWidget(float(ev->win_x) / float(_glCanvas1D->getWidth()), 
                                 1.0f - float(ev->win_y) / float(_glCanvas2D->getHeight()), 
                                 -1.0f);
  updateLabels();
  draw2DTF();
  return 1;
}

long VVTransferWindow::onMouseUp2D(FXObject*,FXSelector,void*)
{
  _glCanvas2D->ungrab();
  return 1;
}

long VVTransferWindow::onMouseMove2D(FXObject*, FXSelector, void* ptr)
{
  if (!_glCanvas2D->grabbed()) return 1;
  if(!_currentWidget || !_canvas) return 1;
  if(_canvas->_vd->tf._widgets.count() == 0) return 1;
  FXEvent* ev = (FXEvent*)ptr;
  float xPos = ts_clamp(float(ev->win_x) / float(_glCanvas2D->getWidth()),  0.0f, 1.0f);
  float yPos = ts_clamp(1.0f - float(ev->win_y) / float(_glCanvas2D->getHeight()), 0.0f, 1.0f);
  _currentWidget->_pos[0] = xPos;
  _currentWidget->_pos[1] = yPos;
  draw2DTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

void VVTransferWindow::updateTransFunc()
{
  _shell->_glcanvas->makeCurrent();
  _canvas->_renderer->updateTransferFunction();
  _shell->_glcanvas->makeNonCurrent();
}

long VVTransferWindow::onCmdDelete(FXObject*,FXSelector,void*)
{
  if(_canvas->_vd->tf._widgets.count() == 0 || _currentWidget == NULL) return 1;
  _canvas->_vd->tf.putUndoBuffer();
  if (_canvas->_vd->tf._widgets.find(_currentWidget)) _canvas->_vd->tf._widgets.remove();
  _currentWidget = NULL;
  drawTF();
  updateLabels();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdUndo(FXObject*,FXSelector,void*)
{
  _canvas->_vd->tf.getUndoBuffer();
  _currentWidget = NULL;
  drawTF();
  updateLabels();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdColorCombo(FXObject*,FXSelector,void*)
{
  _canvas->_vd->tf.putUndoBuffer();
  _canvas->_vd->tf.setDefaultColors(_colorCombo->getCurrentItem());
  _currentWidget = NULL;
  drawTF();
  updateLabels();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdAlphaCombo(FXObject*,FXSelector,void*)
{
  _canvas->_vd->tf.putUndoBuffer();
  _canvas->_vd->tf.setDefaultAlpha(_alphaCombo->getCurrentItem());
  _currentWidget = NULL;
  drawTF();
  updateLabels();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdInstant(FXObject*,FXSelector,void* ptr)
{
  if(ptr != 0) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdOwnColor(FXObject*,FXSelector,void* ptr)
{
  vvTFPyramid* pw;
  vvTFBell* bw;

  if ((pw = dynamic_cast<vvTFPyramid*>(_currentWidget)) != NULL)
  {
    pw->setOwnColor(ptr != NULL);
  }
  else if ((bw = dynamic_cast<vvTFBell*>(_currentWidget)) != NULL)
  {
    bw->setOwnColor(ptr != NULL);
  }
  
  drawTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdApply(FXObject*,FXSelector,void*)
{
  updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdImport(FXObject*,FXSelector,void*)
{
  const FXchar patterns[]="XVF files (*.xvf)\nAll Files (*.*)";
  FXString filename = _shell->getOpenFilename("Import Transfer Function", patterns);
  if(filename.length() > 0)
  {
    vvFileIO* fio = new vvFileIO();
    if (fio->importTF(_canvas->_vd, filename.text())==vvFileIO::OK)
    {
      _currentWidget = NULL;
      updateTransFunc();
      drawTF();
      updateLabels();
    }
    delete fio;
  }
  return 1;
}

long VVTransferWindow::onCmdNormalization(FXObject*,FXSelector,void*)
{
  computeHistogram();
  drawTF();
  return 1;
}

void VVTransferWindow::newWidget(WidgetType wt)
{
  vvColor col;
  vvTFWidget* widget = NULL;

  if(!_canvas) return;
  _canvas->_vd->tf.putUndoBuffer();
  switch(wt)
  {
    case COLOR:
      widget = new vvTFColor(col, 0.5f);
      break;
    case PYRAMID:
      widget = new vvTFPyramid(col, false, 1.0f, 0.5f, 0.4f, 0.2f);
      break;
    case BELL:
      widget = new vvTFBell(col, false, 1.0f, 0.5f, 0.2f);
      break;
    case SKIP:
      widget = new vvTFSkip(0.5f, 0.2f);
      break;
    default: return;
  }
  _canvas->_vd->tf._widgets.append(widget, vvSLNode<vvTFWidget*>::NORMAL_DELETE);
  _currentWidget = widget;
  if(_instantButton->getCheck()) updateTransFunc();
  drawTF();
  updateLabels();
}

void VVTransferWindow::drawHistogram()
{
  switch (_tfBook->getCurrent())
  {
    case 0:
      if (_glCanvas1D->makeCurrent())
      {
        glRasterPos2f(0.0f, 0.0f); 
        glPixelZoom(1.0f, 1.0f);
        glDrawPixels(_glCanvas1D->getWidth(), _glCanvas1D->getHeight() - COLORBAR_HEIGHT, 
          GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)_histoTexture1D);
        _glCanvas1D->makeNonCurrent();
      }
      break;
    case 1:
      if (_glCanvas2D->makeCurrent())
      {
        glRasterPos2f(0.0f, 0.0f); 
        glPixelZoom(1.0f, 1.0f);
        glDrawPixels(_glCanvas2D->getWidth(), _glCanvas2D->getHeight(), 
          GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)_histoTexture2D);
        _glCanvas2D->makeNonCurrent();
      }
      break;
    default: break;
  }
}

void VVTransferWindow::computeHistogram()
{
  int size[2];

  switch (_tfBook->getCurrent())
  {
    case 0:
      delete[] _histoTexture1D;
      size[0] = _glCanvas1D->getWidth();
      size[1] = _glCanvas1D->getHeight() - COLORBAR_HEIGHT;
      _histoTexture1D = new uchar[size[0] * size[1] * 4];
      _canvas->_vd->makeHistogramTexture((_histAll->getCheck()) ? -1 : 0, 0, 1, size, _histoTexture1D, 
        (_cbNorm->getCheck()) ? vvVolDesc::VV_LOGARITHMIC : vvVolDesc::VV_LINEAR);
      break;
    case 1:
      delete[] _histoTexture2D;
      size[0] = _glCanvas2D->getWidth();
      size[1] = _glCanvas2D->getHeight();
      _histoTexture2D = new uchar[size[0] * size[1] * 4];
      _canvas->_vd->makeHistogramTexture((_histAll->getCheck()) ? -1 : 0, 0, 2, size, _histoTexture2D,
        (_cbNorm->getCheck()) ? vvVolDesc::VV_LOGARITHMIC : vvVolDesc::VV_LINEAR);
      break;
    default: break;
  }
}

void VVTransferWindow::drawTF()
{
  switch (_tfBook->getCurrent())
  {
    case 0:
      draw1DTF();
      break;
    case 1:
      draw2DTF();
      break;
    default: break;
  }
}

void VVTransferWindow::draw1DTF()
{
  float r,g,b;

  if (_glCanvas1D->makeCurrent())
  {
    _canvas->getBackgroundColor(r, g, b);
    glClearColor(r, g, b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    _glCanvas1D->makeNonCurrent();
  }
  drawColorTexture();
  drawPinBackground();
  drawPinLines();
  if (_glCanvas1D->makeCurrent())
  {
    if(_glVisual1D->isDoubleBuffer())
    {
      _glCanvas1D->swapBuffers();
    }
    _glCanvas1D->makeNonCurrent();
  }
}

void VVTransferWindow::draw2DTF()
{
  float r,g,b;

  if (_glCanvas2D->makeCurrent())
  {
    _canvas->getBackgroundColor(r, g, b);
    glClearColor(r, g, b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    _glCanvas2D->makeNonCurrent();
  }
  if (_histAll->getCheck() || _histFirst->getCheck())
  {
    drawHistogram();
  }
  draw2DTFTexture();
  draw2DTFWidgets();
  if (_glCanvas2D->makeCurrent())
  {
    if(_glVisual2D->isDoubleBuffer())
    {
      _glCanvas2D->swapBuffers();
    }
    _glCanvas2D->makeNonCurrent();
  }
}

void VVTransferWindow::draw2DTFWidgets()
{
  _canvas->_vd->tf._widgets.first();
  for(int j=0; j<_canvas->_vd->tf._widgets.count(); ++j)
  {
    draw2DWidget(_canvas->_vd->tf._widgets.getData());
    _canvas->_vd->tf._widgets.next();
  }
}

void VVTransferWindow::draw2DWidget(vvTFWidget* w)
{
  float xHalf=0.0f, yHalf=0.0f; // half width and height
  vvTFPyramid* pw;
  vvTFBell* bw;
  bool selected;

  if ((pw = dynamic_cast<vvTFPyramid*>(w)) != NULL)
  {
    xHalf = pw->_bottom[0] / 2.0f;
    yHalf = pw->_bottom[1] / 2.0f;
  }
  else if ((bw = dynamic_cast<vvTFBell*>(w)) != NULL)
  {
    xHalf = bw->_size[0] / 2.0f;
    yHalf = bw->_size[1] / 2.0f;
  }  
  selected = (w == _currentWidget);
  if (_glCanvas2D->makeCurrent())
  {
    glColor3f(1.0f, 1.0f, 1.0f);
    if (selected) glLineWidth(4.0f);
    else glLineWidth(2.0f);

    glBegin(GL_LINE_STRIP);
      glVertex2f(w->_pos[0] - xHalf, w->_pos[1] - yHalf);   // bottom left
      glVertex2f(w->_pos[0] + xHalf, w->_pos[1] - yHalf);   // bottom right
      glVertex2f(w->_pos[0] + xHalf, w->_pos[1] + yHalf);   // top right
      glVertex2f(w->_pos[0] - xHalf, w->_pos[1] + yHalf);   // top left
      glVertex2f(w->_pos[0] - xHalf, w->_pos[1] - yHalf);   // bottom left
    glEnd();

    _glCanvas2D->makeNonCurrent();
  }
}

void VVTransferWindow::drawPinBackground()
{
  if (_histAll->getCheck() || _histFirst->getCheck())
  {
    drawHistogram();
  }
  else
  {
    drawAlphaTexture();
  }
}

void VVTransferWindow::drawPinLines()
{
  if(!_canvas) return;

  _canvas->_vd->tf._widgets.first();
  for(int j=0; j<_canvas->_vd->tf._widgets.count(); ++j)
  {
    drawPinLine(_canvas->_vd->tf._widgets.getData());
    _canvas->_vd->tf._widgets.next();
  }
}

void VVTransferWindow::drawPinLine(vvTFWidget* w)
{
  float xPos, yTop, height;
  bool selected;

  if (dynamic_cast<vvTFColor*>(w) != NULL)
  { 
    yTop = 1.0f;
    height = float(COLORBAR_HEIGHT) / float(_glCanvas1D->getHeight());
  }
  else if ((dynamic_cast<vvTFPyramid*>(w) != NULL) ||
           (dynamic_cast<vvTFBell*>(w) != NULL) ||
           (dynamic_cast<vvTFSkip*>(w) != NULL))
  {
    yTop = 1.0f - float(COLORBAR_HEIGHT) / float(_glCanvas1D->getHeight());
    height = float(_glCanvas1D->getHeight() - COLORBAR_HEIGHT) / float(_glCanvas1D->getHeight());
  }
  else return;

  selected = (w == _currentWidget);
  xPos = w->_pos[0];
  if (_glCanvas1D->makeCurrent())
  {
    glColor3f(0.0f, 0.0f, 0.0f);
    if (selected) glLineWidth(4.0f);
    else glLineWidth(2.0f);
    glBegin(GL_LINES);
      glVertex2f(xPos, yTop);
      glVertex2f(xPos, yTop - height);
    glEnd();
    _glCanvas1D->makeNonCurrent();
  }
}

/** Returns the pointer to the closest widget to a specific point in TF space.
  @param x,y,z  query position in TF space [0..1]. -1 if undefined
*/
vvTFWidget* VVTransferWindow::closestWidget(float x, float y, float z)
{
  vvTFWidget* w = NULL;
  vvTFWidget* temp = NULL;
  float dist, xDist, yDist;     // [TF space]
  float minDist = FLT_MAX;      // [TF space]
  int i;
  bool isColor;

  if (!_canvas) return NULL;
  int numWidgets = _canvas->_vd->tf._widgets.count();
  _canvas->_vd->tf._widgets.first();
  for(i=0; i<numWidgets; ++i)
  {
    temp = _canvas->_vd->tf._widgets.getData();
    switch (_tfBook->getCurrent())
    {
      case 0: 
        isColor = (y > 1.0f - float(COLORBAR_HEIGHT) / float(_glCanvas1D->getHeight())) ? true : false;
        if ((isColor && dynamic_cast<vvTFColor*>(temp)) || (!isColor && dynamic_cast<vvTFColor*>(temp)==NULL)) 
        {
          dist = fabs(x - temp->_pos[0]);
          if(dist < minDist && dist <= CLICK_TOLERANCE)
          {
            minDist = dist;
            w = temp;
          }
        }
        break;
      case 1:
        if (!dynamic_cast<vvTFColor*>(temp))
        {
          xDist = x - temp->_pos[0];
          yDist = y - temp->_pos[1];
          dist = float(sqrt(xDist * xDist + yDist * yDist));
          if (dist < minDist && dist <= CLICK_TOLERANCE)
          {
            minDist = dist;
            w = temp;
          }
        }
        break;
      default: assert(0); break;
    }
    _canvas->_vd->tf._widgets.next();
  }
  return w;
}

void VVTransferWindow::updateLabels()
{
  vvTFColor* cw;
  vvTFPyramid* pw;
  vvTFBell* bw;
  vvTFSkip* sw;

  if ((cw=dynamic_cast<vvTFColor*>(_currentWidget))!=NULL)
  {
    _pinSwitcher->setCurrent(1);
  }
  else if ((pw=dynamic_cast<vvTFPyramid*>(_currentWidget))!=NULL)
  {
    _pinSwitcher->setCurrent(2);
    _pTopXSlider->setValue(pw->_top[0]);
    _pTopYSlider->setValue(pw->_top[1]);
    _pTopXLabel->setText(FXStringFormat("%.2f", _pTopXSlider->getValue()));
    _pTopYLabel->setText(FXStringFormat("%.2f", _pTopYSlider->getValue()));
    _pBottomXSlider->setValue(pw->_bottom[0]);
    _pBottomYSlider->setValue(pw->_bottom[1]);
    _pBottomXLabel->setText(FXStringFormat("%.2f", _pBottomXSlider->getValue()));
    _pBottomYLabel->setText(FXStringFormat("%.2f", _pBottomYSlider->getValue()));
    _pMaxSlider->setValue(pw->_opacity);
    _pMaxLabel->setText(FXStringFormat("%.2f", _pMaxSlider->getValue()));
    _pColorButton->setCheck(pw->hasOwnColor());
  }
  else if ((bw=dynamic_cast<vvTFBell*>(_currentWidget))!=NULL)
  {
    _pinSwitcher->setCurrent(3);
    _bWidthSlider->setValue(bw->_size[0]);
    _bWidthLabel->setText(FXStringFormat("%.2f", _bWidthSlider->getValue()));
    _bHeightSlider->setValue(bw->_size[1]);
    _bHeightLabel->setText(FXStringFormat("%.2f", _bHeightSlider->getValue()));
    _bMaxSlider->setValue(bw->_opacity);
    _bMaxLabel->setText(FXStringFormat("%.2f", _bMaxSlider->getValue()));
    _bColorButton->setCheck(bw->hasOwnColor());
  }
  else if ((sw=dynamic_cast<vvTFSkip*>(_currentWidget))!=NULL)
  {
    _pinSwitcher->setCurrent(4);
    _sWidthSlider->setValue(sw->_size[0]);
    _sWidthLabel->setText(FXStringFormat("%.2f", _sWidthSlider->getValue()));
    _sHeightSlider->setValue(sw->_size[1]);
    _sHeightLabel->setText(FXStringFormat("%.2f", _sWidthSlider->getValue()));
  }
  else    // no current widget
  {
    _pinSwitcher->setCurrent(0);
  }
  if (_currentWidget) _realPosLabel->setText(FXStringFormat("Pin = %.5g", getRealPinPos(_currentWidget->_pos[0])));
  else _realPosLabel->setText("");
}

void VVTransferWindow::drawColorTexture()
{
  const int WIDTH = 256;
  static uchar* colorBar = new uchar[WIDTH * 4 * 2];
  
  _canvas->_vd->tf.makeColorBar(WIDTH, colorBar);
  if (_glCanvas1D->makeCurrent())
  {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glRasterPos2f(0.0f, 1.0f);  // pixmap origin is bottom left corner of output window
    glPixelZoom(float(_glCanvas1D->getWidth()) / float(WIDTH), -10.0f);
    glDrawPixels(WIDTH, 2, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)colorBar);
    _glCanvas1D->makeNonCurrent();
  }
}

void VVTransferWindow::drawAlphaTexture()
{
  static uchar* tfTexture = new uchar[_glCanvas1D->getWidth() * (_glCanvas1D->getHeight() - COLORBAR_HEIGHT) * 4];
  
  if (_glCanvas1D->makeCurrent())
  {
    _canvas->_vd->tf.makeAlphaTexture(_glCanvas1D->getWidth(), _glCanvas1D->getHeight() - COLORBAR_HEIGHT, tfTexture);
    glRasterPos2f(0.0f, 1.0f - float(COLORBAR_HEIGHT) / float(_glCanvas1D->getHeight())); 
    glPixelZoom(1.0f, -1.0f);
    glDrawPixels(_glCanvas1D->getWidth(), _glCanvas1D->getHeight() - COLORBAR_HEIGHT, 
      GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)tfTexture);
    _glCanvas1D->makeNonCurrent();
  }
}

void VVTransferWindow::draw2DTFTexture()
{
  static int WIDTH  = 128;
  static int HEIGHT = 128;
  static uchar* tfTexture = new uchar[WIDTH * HEIGHT * 4];

  if (_glCanvas2D->makeCurrent())
  {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    _canvas->_vd->tf.make2DTFTexture(WIDTH, HEIGHT, tfTexture);
    glRasterPos2f(0.0f, 0.0f); 
    glPixelZoom(float(_glCanvas2D->getWidth()) / float(WIDTH), float(_glCanvas2D->getHeight()) / float(HEIGHT));
    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)tfTexture);
    _glCanvas2D->makeNonCurrent();
  }
}

long VVTransferWindow::onChngDisColors(FXObject*,FXSelector,void*)
{
  _disColorLabel->setText(FXStringFormat("%d",_disColorSlider->getValue()));
  _canvas->_vd->tf.setDiscreteColors(_disColorSlider->getValue());
  drawTF();
  if(_instantButton->getCheck()) updateTransFunc();
  return 1;
}

long VVTransferWindow::onCmdHistAll(FXObject*,FXSelector,void*)
{
  _histNone->setCheck(false);
  _histFirst->setCheck(false);
  computeHistogram();
  drawTF();
  return 1;
}

long VVTransferWindow::onCmdHistFirst(FXObject*,FXSelector,void*)
{
  _histNone->setCheck(false);
  _histAll->setCheck(false);
  computeHistogram();
  drawTF();
  return 1;
}

long VVTransferWindow::onCmdHistNone(FXObject*,FXSelector,void*)
{
  _histAll->setCheck(false);
  _histFirst->setCheck(false);
  computeHistogram();
  drawTF();
  return 1;
}

void VVTransferWindow::updateValues()
{
  _currentWidget = NULL;
  _realMinLabel->setText(FXStringFormat("min = %.5g", _canvas->_vd->real[0]));
  _realMaxLabel->setText(FXStringFormat("max = %.5g", _canvas->_vd->real[1]));
  updateLabels();
  _disColorSlider->setValue(_canvas->_vd->tf.getDiscreteColors());
  _disColorLabel->setText(FXStringFormat("%d", _disColorSlider->getValue()));
  computeHistogram();
  drawTF();
}

