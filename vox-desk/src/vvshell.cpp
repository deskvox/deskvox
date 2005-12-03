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
#pragma warning(disable: 4511)    // disable warning: copy constructor could not be generated
#pragma warning(disable: 4512)    // disable warning: assignment operator could not be generated

// Virvo:
#include <vvvoldesc.h>
#include <vvgltools.h>
#include <vvdebugmsg.h>
#include <vvvirvo.h>

// Local:
#include "vvshell.h"
#include "vvdialogs.h"
#include "vvprefwindow.h"
#include "vvtranswindow.h"
#include "vvsliceviewer.h"
#include "vvcanvas.h"
#include "vvmovie.h"
#include "vvfileio.h"
#include "vvstopwatch.h"

using namespace vox;
using namespace std;

//----------------------------------------------------------------------------
/** Message Map
  Unhandled events:
    SEL_MAP: sent when the window is mapped to the screen; the message data is an FXEvent instance.  
    SEL_UNMAP: sent when the window is unmapped; the message data is an FXEvent instance.  
    SEL_CONFIGURE: sent when the window’s size changes; the message data is an FXEvent instance.  
    SEL_ENTER: sent when the mouse cursor enters this window  
    SEL_LEAVE: sent when the mouse cursor leaves this window  
    SEL_FOCUSIN: sent when this window gains the focus  
    SEL_FOCUSOUT: sent when this window loses the focus  
    SEL_UPDATE: sent when this window needs an update  
    SEL_UNGRABBED: sent when this window loses the mouse grab (or capture)  
*/
FXDEFMAP(VVShell) VVShellMap[]=
{
  //________Message_Type_____________________ID________________________Message_Handler_______
  FXMAPFUNC(SEL_PAINT,               VVShell::ID_CANVAS,        VVShell::onExpose),
  FXMAPFUNC(SEL_LEFTBUTTONPRESS,     VVShell::ID_CANVAS,        VVShell::onLeftMouseDown),
  FXMAPFUNC(SEL_LEFTBUTTONRELEASE,   VVShell::ID_CANVAS,        VVShell::onLeftMouseUp),
  FXMAPFUNC(SEL_MIDDLEBUTTONPRESS,   VVShell::ID_CANVAS,        VVShell::onMidMouseDown),
  FXMAPFUNC(SEL_MIDDLEBUTTONRELEASE, VVShell::ID_CANVAS,        VVShell::onMidMouseUp),
  FXMAPFUNC(SEL_RIGHTBUTTONPRESS,    VVShell::ID_CANVAS,        VVShell::onRightMouseDown),
  FXMAPFUNC(SEL_RIGHTBUTTONRELEASE,  VVShell::ID_CANVAS,        VVShell::onRightMouseUp),
  FXMAPFUNC(SEL_MOTION,              VVShell::ID_CANVAS,        VVShell::onMouseMove),
  FXMAPFUNC(SEL_MOUSEWHEEL,          VVShell::ID_CANVAS,        VVShell::onMouseWheel),
  FXMAPFUNC(SEL_CONFIGURE,           VVShell::ID_CANVAS,        VVShell::onConfigure),
  FXMAPFUNC(SEL_KEYPRESS,            VVShell::ID_CANVAS,        VVShell::onKeyPress),
  FXMAPFUNC(SEL_KEYRELEASE,          VVShell::ID_CANVAS,        VVShell::onKeyRelease),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_PREFERENCES,   VVShell::onCmdPrefs),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_TRANSFER,      VVShell::onCmdTrans),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_VIS_INFO,      VVShell::onCmdVisInfo),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_ABOUT,         VVShell::onCmdAbout),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_KEYS,          VVShell::onCmdKeys),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_CAMERA,        VVShell::onCmdCameraSettings),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_GAMMA,         VVShell::onCmdGammaSettings),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_CHANNEL4,      VVShell::onCmdChannel4Settings),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_OPACITY,       VVShell::onCmdOpacitySettings),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_FLOAT_RANGE,   VVShell::onCmdFloatRange),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_CLIP_PLANE,    VVShell::onCmdClipping),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_ROI,           VVShell::onCmdROI),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_BG_COLOR,      VVShell::onCmdBGColor),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_DIMENSIONS,    VVShell::onCmdDimensions),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_DRAW,          VVShell::onCmdDraw),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_DIAGRAMS,      VVShell::onCmdDiagrams),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_DATA_TYPE,     VVShell::onCmdDataType),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_EDIT_VOXELS,   VVShell::onCmdEditVoxels),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_HEIGHT_FIELD,  VVShell::onCmdMakeHeightField),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_GL_SETTINGS,   VVShell::onCmdGLSettings),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_SCREEN_SHOT,   VVShell::onCmdScreenShot),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_MOVIE_SCRIPT,  VVShell::onCmdMovie),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_TIME_STEPS,    VVShell::onCmdTimeSteps),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_LOAD_VOLUME,   VVShell::onCmdLoadVolume),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_SAVE_VOLUME,   VVShell::onCmdSaveVolume),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_RELOAD_VOLUME, VVShell::onCmdReloadVolume),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_MERGE,         VVShell::onCmdMerge),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_SERVER,        VVShell::onCmdServerRequest),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_LOAD_CAMERA,   VVShell::onCmdLoadCamera),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_SAVE_CAMERA,   VVShell::onCmdSaveCamera),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_SLICE_VIEWER,  VVShell::onCmdSliceViewer),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_ORIENTATION,   VVShell::onDispOrientChange),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_BOUNDARIES,    VVShell::onDispBoundsChange),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_PALETTE,       VVShell::onDispPaletteChange),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_QUALITY,       VVShell::onDispQualityChange),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_FPS,           VVShell::onDispFPSChange),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_SPIN,          VVShell::onDispSpinChange),
  FXMAPFUNC(SEL_CHANGED,             VVShell::ID_COLOR_PICKER,  VVShell::pickerColorChanged),
  FXMAPFUNC(SEL_COMMAND,             VVShell::ID_COLOR_PICKER,  VVShell::pickerColorChanged),
  FXMAPFUNC(SEL_TIMEOUT,             VVShell::ID_ART_TIMER,     VVShell::onARToolkitTimerEvent),
  FXMAPFUNC(SEL_TIMEOUT,             VVShell::ID_ANIM_TIMER,    VVShell::onAnimTimerEvent),
  FXMAPFUNC(SEL_TIMEOUT,             VVShell::ID_SPIN_TIMER,    VVShell::onSpinTimerEvent),
  FXMAPFUNCS(SEL_UPDATE,             MINKEY,MAXKEY,             VVShell::onAllUpdate),
};

// Macro for the class hierarchy implementation
FXIMPLEMENT(VVShell, FXMainWindow, VVShellMap, ARRAYNUMBER(VVShellMap))

//----------------------------------------------------------------------------
VVShell::VVShell(FXApp* a) : FXMainWindow(a,"DeskVOX",NULL,NULL,DECOR_ALL,0,0,600,600)
{
  vvDebugMsg::msg(1, "VVShell::VVShell(FXApp*)");
  FXVerticalFrame* glcanvasFrame;
  FXComposite* glpanel;
  FXMenuBar* menubar;
  FXMenuPane* filemenu;
  FXMenuPane* setmenu;
  FXMenuPane* editmenu;
  FXMenuPane* viewmenu;
  FXMenuPane* helpmenu;

  lmdFlag = mmdFlag = rmdFlag = 0;

  setTitle(FXString("DeskVOX v") + FXString(VV_VERSION) + FXString(".") + FXString(VV_RELEASE));

  // Menubar
  menubar=new FXMenuBar(this,LAYOUT_SIDE_TOP|LAYOUT_FILL_X);

  filemenu=new FXMenuPane(this);
  initFileMenu(filemenu);
  new FXMenuTitle(menubar,"File",NULL,filemenu);

  setmenu=new FXMenuPane(this);
  initSettingsMenu(setmenu);
  new FXMenuTitle(menubar,"Settings",NULL,setmenu);

  editmenu=new FXMenuPane(this);
  initEditMenu(editmenu);
  new FXMenuTitle(menubar,"Edit",NULL,editmenu);

  viewmenu=new FXMenuPane(this);
  initViewMenu(viewmenu);
  new FXMenuTitle(menubar,"View",NULL,viewmenu);

  helpmenu=new FXMenuPane(this);
  initHelpMenu(helpmenu);
  new FXMenuTitle(menubar,"Help",NULL,helpmenu);

  // LEFT pane to contain the _glcanvas
  glcanvasFrame = new FXVerticalFrame(this, LAYOUT_FILL_X | LAYOUT_FILL_Y);

  // Drawing _glcanvas
  glpanel = new FXVerticalFrame(glcanvasFrame,FRAME_SUNKEN | FRAME_THICK | LAYOUT_FILL_X | LAYOUT_FILL_Y, 0,0,0,0,0,0,0,0);

  // A Visual to draw OpenGL
  _glvisual = new FXGLVisual(getApp(), VISUAL_DOUBLEBUFFER | VISUAL_STEREO);

  // Drawing _glcanvas
  _glcanvas = new FXGLCanvas(glpanel, _glvisual,this,ID_CANVAS,LAYOUT_FILL_X|LAYOUT_FILL_Y|LAYOUT_TOP|LAYOUT_LEFT);

  _statusBar = new FXLabel(glpanel, "Welcome to DeskVOX!");

  _canvas = new vvCanvas();
  _movie = new vvMovie(_canvas);
  initDialogs();
}

//----------------------------------------------------------------------------
VVShell::~VVShell()
{
  vvDebugMsg::msg(1, "VVShell::~VVShell()");
  stopAnimTimer();
  stopARToolkitTimer();
  delete _glvisual;
  delete volumeDialog;
  delete prefWindow;
  delete transWindow;
  delete _sliceViewer;
  delete _cameraDialog;
  delete gammaDialog;
  delete _channel4Dialog;
  delete _opacityDialog;
  delete _floatRangeDialog;
  delete _clipDialog;
  delete _roiDialog;
  delete _colorPicker;
  delete _dimDialog;
  delete _drawDialog;
  delete _mergeDialog;
  delete _serverDialog;
  delete screenshotDialog;
  delete movieDialog;
  delete tsDialog;
  delete _diagramDialog;
  delete _dataTypeDialog;
  delete _editVoxelsDialog;
  delete _heightFieldDialog;

  if(_canvas) delete _canvas;
}

//----------------------------------------------------------------------------
void VVShell::parseCommandline()
{
  if (getApp()->getArgc()==2)
  {
    FXString filename = getApp()->getArgv()[1];
    cerr << "Loading volume file: " << filename.text() << endl;
    loadVolumeFile(filename);
  }
}

//----------------------------------------------------------------------------
bool VVShell::initFileMenu(FXMenuPane* filemenu)
{
  vvDebugMsg::msg(1, "VVShell::initFileMenu()");

  new FXMenuCommand(filemenu,"Load Volume...",NULL,this,ID_LOAD_VOLUME);
  new FXMenuCommand(filemenu,"Reload Volume", NULL,this,ID_RELOAD_VOLUME);
  new FXMenuCommand(filemenu,"Save Volume As...",NULL,this,ID_SAVE_VOLUME);
  new FXMenuCommand(filemenu,"Merge Files...",NULL,this,ID_MERGE);
  new FXMenuCommand(filemenu,"Server Request...",NULL,this,ID_SERVER);

  new FXHorizontalSeparator(filemenu,LAYOUT_SIDE_TOP|LAYOUT_FILL_X|SEPARATOR_GROOVE);

  new FXMenuCommand(filemenu,"Load Camera...",NULL,this,ID_LOAD_CAMERA);
  new FXMenuCommand(filemenu,"Save Camera As...",NULL,this,ID_SAVE_CAMERA);

  new FXHorizontalSeparator(filemenu,LAYOUT_SIDE_TOP|LAYOUT_FILL_X|SEPARATOR_GROOVE);

  new FXMenuCommand(filemenu,"Screen Shot...",NULL,this,ID_SCREEN_SHOT);
  new FXMenuCommand(filemenu,"Movie Script...",NULL,this,ID_MOVIE_SCRIPT);

  new FXHorizontalSeparator(filemenu,LAYOUT_SIDE_TOP|LAYOUT_FILL_X|SEPARATOR_GROOVE);

  new FXMenuCommand(filemenu,"Preferences...",NULL,this,ID_PREFERENCES);

  new FXHorizontalSeparator(filemenu,LAYOUT_SIDE_TOP|LAYOUT_FILL_X|SEPARATOR_GROOVE);

  new FXMenuCommand(filemenu,"Quit",NULL,getApp(),FXApp::ID_QUIT,0);

  return true;
}

//----------------------------------------------------------------------------
bool VVShell::initSettingsMenu(FXMenuPane* setmenu)
{
  vvDebugMsg::msg(1, "VVShell::initSettingsMenu()");

  new FXMenuCommand(setmenu,"Transfer Function...",    NULL, this, ID_TRANSFER);
  new FXMenuCommand(setmenu,"Gamma Correction...",     NULL, this, ID_GAMMA);
  new FXMenuCommand(setmenu,"Channel 4...",            NULL, this, ID_CHANNEL4);
  new FXMenuCommand(setmenu,"Opacity Weights...",      NULL, this, ID_OPACITY);
  new FXMenuCommand(setmenu,"Floating Point Range...", NULL, this, ID_FLOAT_RANGE);
  new FXMenuCommand(setmenu,"Clipping Plane...",       NULL, this, ID_CLIP_PLANE);
  new FXMenuCommand(setmenu,"Region of Interest...",   NULL, this, ID_ROI);
  new FXMenuCommand(setmenu,"Background Color...",     NULL, this, ID_BG_COLOR);

  return true;
}

//----------------------------------------------------------------------------
bool VVShell::initEditMenu(FXMenuPane* editmenu)
{
  vvDebugMsg::msg(1, "VVShell::initEditMenu()");

  new FXMenuCommand(editmenu,"Sample Distances...",NULL,this,ID_DIMENSIONS);
  new FXMenuCommand(editmenu,"Draw...",NULL,this,ID_DRAW);
  new FXMenuCommand(editmenu,"Data Format...",NULL,this,ID_DATA_TYPE);
  new FXMenuCommand(editmenu,"Geometry...",NULL,this,ID_EDIT_VOXELS);
  new FXMenuCommand(editmenu,"Make Height Field...",NULL,this,ID_HEIGHT_FIELD);
  return true;
}

//----------------------------------------------------------------------------
bool VVShell::initViewMenu(FXMenuPane* viewmenu)
{
  vvRenderState dummyState;
  vvDebugMsg::msg(1, "VVShell::initViewMenu()");

  _orientItem = new FXMenuCheck(viewmenu,   "Show Orientation", this, ID_ORIENTATION);
  _orientItem->setCheck(dummyState._orientation);
  _boundaryItem = new FXMenuCheck(viewmenu, "Show Boundaries", this, ID_BOUNDARIES);
  _boundaryItem->setCheck(dummyState._boundaries);
  _paletteItem = new FXMenuCheck(viewmenu, "Show Palette", this, ID_PALETTE);
  _paletteItem->setCheck(dummyState._palette);
  _qualityItem = new FXMenuCheck(viewmenu, "Show #Textures", this, ID_QUALITY);
  _qualityItem->setCheck(dummyState._qualityDisplay);
  _fpsItem = new FXMenuCheck(viewmenu, "Show Frame Rate", this, ID_FPS);
  _fpsItem->setCheck(false);
  _spinItem = new FXMenuCheck(viewmenu, "Auto rotation", this, ID_SPIN);
  _spinItem->setCheck(false);

  new FXMenuCommand(viewmenu, "Camera...",NULL,this,ID_CAMERA);
  new FXMenuCommand(viewmenu, "Volume Information...",NULL, this, ID_VIS_INFO);
  new FXMenuCommand(viewmenu, "Slice Viewer...",NULL, this, ID_SLICE_VIEWER);
  new FXMenuCommand(viewmenu, "Time Steps...",NULL, this, ID_TIME_STEPS);
  new FXMenuCommand(viewmenu, "Diagrams...",NULL,this,ID_DIAGRAMS);
  return true;
}

//----------------------------------------------------------------------------
bool VVShell::initHelpMenu(FXMenuPane* helpmenu)
{
  vvDebugMsg::msg(1, "VVShell::initHelpMenu()");

  new FXMenuCommand(helpmenu,"Keyboard Commands...",NULL,this,ID_KEYS);
  new FXMenuCommand(helpmenu,"OpenGL Settings...",NULL,this,ID_GL_SETTINGS);
  new FXMenuCommand(helpmenu,"About DeskVOX...",NULL,this,ID_ABOUT);
  return true;
}

//----------------------------------------------------------------------------
void VVShell::initDialogs()
{
  vvDebugMsg::msg(1, "VVShell::initDialogs()");

  volumeDialog = new VVVolumeDialog((FXWindow*)this, _canvas);
  prefWindow   = new VVPreferenceWindow((FXWindow*)this, _canvas);
  transWindow  = new VVTransferWindow((FXWindow*)this, _canvas);
  _sliceViewer = new VVSliceViewer((FXWindow*)this, _canvas);
  _cameraDialog = new VVCameraSetDialog((FXWindow*)this, _canvas);
  gammaDialog  = new VVGammaDialog((FXWindow*)this, _canvas);
  _channel4Dialog = new VVChannel4Dialog((FXWindow*)this, _canvas);
  _opacityDialog = new VVOpacityDialog((FXWindow*)this, _canvas);
  _floatRangeDialog = new VVFloatRangeDialog((FXWindow*)this, _canvas);
  _clipDialog   = new VVClippingDialog((FXWindow*)this, _canvas);
  _roiDialog    = new VVROIDialog((FXWindow*)this, _canvas);
  _colorPicker  = new FXColorDialog((FXWindow*)this, "Background Color",DECOR_TITLE|DECOR_BORDER, 50,50);
  _colorPicker->setTarget(this);
  _colorPicker->setSelector(ID_COLOR_PICKER);
  _colorPicker->setOpaqueOnly(true);
  _dimDialog   = new VVDimensionDialog((FXWindow*)this, _canvas);
  _drawDialog   = new VVDrawDialog((FXWindow*)this, _canvas);
  _mergeDialog = new VVMergeDialog((FXWindow*)this, _canvas);
  _serverDialog = new VVServerDialog((FXWindow*)this, _canvas);
  screenshotDialog = new VVScreenshotDialog((FXWindow*)this, _canvas);
  movieDialog  = new VVMovieDialog((FXWindow*)this, _canvas);
  tsDialog     = new VVTimeStepDialog((FXWindow*)this, _canvas);
  _diagramDialog = new VVDiagramDialog((FXWindow*)this, _canvas);
  _dataTypeDialog = new VVDataTypeDialog((FXWindow*)this, _canvas);
  _editVoxelsDialog = new VVEditVoxelsDialog((FXWindow*)this, _canvas);
  _heightFieldDialog = new VVHeightFieldDialog((FXWindow*)this, _canvas);
}

//----------------------------------------------------------------------------
void VVShell::create()
{
  vvDebugMsg::msg(1, "VVShell::create()");

  FXMainWindow::create();
  show(PLACEMENT_SCREEN);
}

//----------------------------------------------------------------------------
// Widget has been resized
long VVShell::onConfigure(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onConfigure()");

  if(_glcanvas->makeCurrent())
  {
    if(_canvas != NULL)
    {
      _canvas->resize(_glcanvas->getWidth(), _glcanvas->getHeight());
      _canvas->draw();
    }
    _glcanvas->makeNonCurrent();
  }

  FXString canvasSize = FXStringFormat("%d x %d", _glcanvas->getWidth(), _glcanvas->getHeight());
  screenshotDialog->_sizeLabel->setText(canvasSize);
  movieDialog->_sizeLabel->setText(canvasSize);

  return 1;
}

//----------------------------------------------------------------------------
// Widget needs repainting
long VVShell::onExpose(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onExpose()");

  if(_canvas == NULL) drawScene();
  return 1;
}

/*************** Mouse interaction *****************/

//----------------------------------------------------------------------------
long VVShell::onLeftMouseDown(FXObject*,FXSelector,void* ptr)
{
  vvDebugMsg::msg(1, "VVShell::onLeftMouseDown()");

  FXEvent *ev=(FXEvent*)ptr;
  if (_canvas) _canvas->mousePressed(ev->win_x, ev->win_y, vvCanvas::LEFT_BUTTON);
  if (_spinItem->getCheck()) stopSpinTimer();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onLeftMouseUp(FXObject*,FXSelector,void* ptr)
{
  vvDebugMsg::msg(1, "VVShell::onLeftMouseUp()");

  FXEvent *ev=(FXEvent*)ptr;
  if(_canvas) _canvas->mouseReleased(ev->win_x, ev->win_y, vvCanvas::NO_BUTTON);
  if (_spinItem->getCheck()) startSpinTimer();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onMidMouseDown(FXObject*,FXSelector,void* ptr)
{
  vvDebugMsg::msg(1, "VVShell::onMidMouseDown()");

  FXEvent *ev=(FXEvent*)ptr;
  if(_canvas != NULL) _canvas->mousePressed(ev->win_x, ev->win_y, vvCanvas::MIDDLE_BUTTON);
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onMidMouseUp(FXObject*,FXSelector,void* ptr)
{
  vvDebugMsg::msg(1, "VVShell::onMidMouseUp()");

  FXEvent *ev=(FXEvent*)ptr;
  if(_canvas != NULL) _canvas->mouseReleased(ev->win_x, ev->win_y, vvCanvas::NO_BUTTON);
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onRightMouseDown(FXObject*,FXSelector,void* ptr)
{
  vvDebugMsg::msg(1, "VVShell::onRightMouseDown()");

  FXEvent *ev=(FXEvent*)ptr;
  if(_canvas != NULL)
    _canvas->mousePressed(ev->win_x, ev->win_y, vvCanvas::RIGHT_BUTTON);
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onRightMouseUp(FXObject*,FXSelector,void* ptr)
{
  vvDebugMsg::msg(1, "VVShell::onRightMouseUp()");

  FXEvent *ev=(FXEvent*)ptr;
  if(_canvas != NULL)
    _canvas->mouseReleased(ev->win_x, ev->win_y, vvCanvas::NO_BUTTON);
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onMouseMove(FXObject*,FXSelector,void* ptr)
{
  vvDebugMsg::msg(3, "VVShell::onMouseMove()");

  FXEvent *ev=(FXEvent*)ptr;
  if (_glcanvas->makeCurrent())
  {
    if(_canvas != NULL)  _canvas->mouseDragged(ev->win_x, ev->win_y);
    if(_glvisual->isDoubleBuffer())
    {
     _glcanvas->swapBuffers();
    }
    _glcanvas->makeNonCurrent();
  }
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onMouseWheel(FXObject*, FXSelector, void*)
{
  vvDebugMsg::msg(1, "VVShell::onMouseWheel()");

  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onKeyPress(FXObject*, FXSelector, void* ptr)
{
  vvDebugMsg::msg(1, "VVShell::onKeyPress()");

  FXEvent* ev = (FXEvent*)ptr;
  const char* txt = ev->text.text();
  switch(txt[0])
  {
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9': break;
    case '-': prefWindow->scaleQuality(0.95f); break;
    case '+':
    case '=': prefWindow->scaleQuality(1.05f); break;
    case 'a': tsDialog->playback(); break;
    case 'b': toggleBounds(); break;
    case 'c': togglePalette(); break;
    case 'd': break;
    case 'e': benchmarkTest(); break;
    case 'f': toggleFPS(); break;
    case 'H': _dimDialog->scaleZ(1.1f); break;
    case 'h': _dimDialog->scaleZ(0.9f); break;
    case 'i': prefWindow->toggleInterpol(); break;
    case 'j': break;
    case 'J': break;
    case 'k': break;
    case 'K': break;
    case 'l': break;
    case 'L': break;
    case 'm': toggleSpin(); break;
    case 'n': tsDialog->stepForward(); break;
    case 'N': tsDialog->stepBack(); break;
    case 'o': toggleOrientation(); break;
    case 'p': _canvas->setPerspectiveMode(!_canvas->getPerspectiveMode()); break;
    case 27:  // Escape
    case 'q': getApp()->exit(); break;
    case 'r': _cameraDialog->reset(); break;
    case 's': tsDialog->scaleSpeed(0.9f); break;
    case 'S': tsDialog->scaleSpeed(1.1f); break;
    case 't': toggleQualityDisplay(); break;
    case 'u': break;
    case 'v': break;
    case 'w': break;
    case 'x': prefWindow->toggleMIP(); break;
    case 'z': if (isMaximized()) restore();
              else maximize(); break;
    case '<': break;
    case '>': break;
    default: 
    {
      switch(ev->state)   // check for control keys
      {
        case 17: break;     // shift
        case 20: break;     // ctrl
        case 24: break;     // alt
        default:
          cerr << "key '" << txt[0] << "' " << " (state: " << ev->state << ") has no function'" << endl; 
          break;
      }
    }
  }
  return 1;
}

long VVShell::onKeyRelease(FXObject*, FXSelector, void*)
{
  vvDebugMsg::msg(1, "VVShell::onKeyRelease()");
  return 1;
}

/********************************************************/
/*************** menu interactions **********************/
/********************************************************/

//----------------------------------------------------------------------------
void VVShell::loadDefaultVolume(int algorithm, int w, int h, int s)
{
  vvVolDesc* vd = new vvVolDesc();
  vd->vox[0] = w;
  vd->vox[1] = h;
  vd->vox[2] = s;
  vd->frames = algorithm;
  vvFileIO* fio = new vvFileIO();
  fio->loadVolumeData(vd, vvFileIO::ALL_DATA);    // load default volume
  delete fio;
  if (vd->tf.isEmpty())
  {
    vd->tf.setDefaultAlpha(0);
    vd->tf.setDefaultColors((vd->chan==1) ? 0 : 2);
  }
  if (vd->bpc==4 && vd->real[0]==0.0f && vd->real[1]==1.0f) vd->setDefaultRealMinMax();
  setCanvasRenderer(vd, 0, _canvas->_currentGeom);
  cerr << "default" << vd->frames << endl;
}

//----------------------------------------------------------------------------
/** Load a volume from disk.
*/
long VVShell::onCmdLoadVolume(FXObject*, FXSelector, void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdLoadVolume()");

  std::cerr << "onCmdLoadVolume" << endl;

  if(_canvas == NULL) return 1;
  const FXchar patterns[]="All Volume Files (*.rvf,*.xvf,*.avf,*.tif,*.tiff,*.hdr)\n3D TIF Files (*.tif,*.tiff)\nASCII Volume Files (*.avf)\nExtended Volume Files (*.xvf)\nRaw Volume Files (*.rvf)\nAll Files (*.*)";
  FXString filename = getOpenFilename("Load Volume File", patterns);
  if(filename.length() == 0) return 1;
  loadVolumeFile(filename);
  return 1;
}

//----------------------------------------------------------------------------
void VVShell::loadVolumeFile(FXString& filename)
{
  FXString message;

  // Load file:
  vvVolDesc* vd = new vvVolDesc(filename.text());
  vvFileIO* fio = new vvFileIO();
  switch (fio->loadVolumeData(vd, vvFileIO::ALL_DATA))
  {
    case vvFileIO::OK:
      vvDebugMsg::msg(2, "Loaded file: ", filename.text());
      // Use default TF if none stored:
      if (vd->tf.isEmpty())
      {
        vd->tf.setDefaultAlpha(0);
        vd->tf.setDefaultColors((vd->chan==1) ? 0 : 2);
      }
      if (vd->bpc==4 && vd->real[0]==0.0f && vd->real[1]==1.0f) vd->setDefaultRealMinMax();
      setCanvasRenderer(vd, 0, _canvas->_currentGeom);
      vd->printInfoLine();
      break;
    case vvFileIO::FILE_NOT_FOUND:
      vvDebugMsg::msg(2, "File not found: ", filename.text());
      message = "File Not Found: " + filename;
      FXMessageBox::error((FXWindow*)this, MBOX_OK, "Error", message.text());
      delete vd;
      break;
    default:
      vvDebugMsg::msg(2, "Cannot load file: ", filename.text());
      message = "Cannot load: " + filename;
      FXMessageBox::error((FXWindow*)this, MBOX_OK, "Error", message.text());
      delete vd;
      break;
  }
  delete fio;
}

//----------------------------------------------------------------------------
long VVShell::onCmdSaveVolume(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdSaveVolume()");

  const FXchar patterns[]="All Volume Files (*.xvf,*.rvf,*.avf)\nExtended Volume Files (*.xvf)\nRaw Volume Files (*.rvf)\nASCII Volume Files (*.avf)\nAll Files (*.*)";
  FXString filename = getSaveFilename("Save Volume", _canvas->_vd->getFilename(), patterns);
  if(filename.length() == 0) return 1;
  if(vvToolshed::isFile(filename.text()))
  {
    int over = FXMessageBox::question((FXWindow*)this, MBOX_OK_CANCEL, "Warning", "Overwrite existing file?");
    if(over == FX::MBOX_CLICKED_CANCEL) return 1;
  }
  vvFileIO* fio;
  const char* str = filename.text();
  _canvas->_vd->setFilename(str);
  fio = new vvFileIO();
  switch (fio->saveVolumeData(_canvas->_vd, true))
  {
    case vvFileIO::OK:
      cerr << "Volume saved as " << _canvas->_vd->getFilename() << endl;
      break;
    case vvFileIO::FILE_EXISTS:
      cerr << "Error: file exists despite previous check" << endl;
      assert(0);
      break;
    default:
      break;
  }
  delete fio;

  cerr << "icon: " << _canvas->_vd->iconSize << endl;

  drawScene();

  return 1;
}

//----------------------------------------------------------------------------
/** Reload a volume from disk to update in case it has changed.
*/
long VVShell::onCmdReloadVolume(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdReloadVolume()");

  vvVolDesc* vd = new vvVolDesc(_canvas->_vd->getFilename());
  vvFileIO* fio = new vvFileIO();

  FXString msgString;
  switch (fio->loadVolumeData(vd, vvFileIO::ALL_DATA))
  {
    case vvFileIO::OK:
      vvDebugMsg::msg(2, "Loaded file: ", vd->getFilename());
      // Use previous pin list if loaded dataset has no pins:
      if (vd->tf.isEmpty()) vd->tf.copy(&vd->tf._widgets, &_canvas->_vd->tf._widgets);
      setCanvasRenderer(vd, 0, _canvas->_currentGeom, _canvas->_currentVoxels);
      break;
    case vvFileIO::FILE_NOT_FOUND:
      vvDebugMsg::msg(2, "File not found: ", vd->getFilename());
      msgString = "File Not Found:   ";
      msgString += vd->getFilename();
      FXMessageBox::error((FXWindow*)this, MBOX_OK, "Error", msgString.text());
      delete vd;
      break;
    default:
      vvDebugMsg::msg(2, "Cannot load file: ", vd->getFilename());
      msgString = "Cannot Load File:   ";
      msgString += vd->getFilename();
      FXMessageBox::error((FXWindow*)this, MBOX_OK, "Error", msgString.text());
      delete vd;
      break;
  }
  delete fio;
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdServerRequest(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdServerRequest()");

  _serverDialog->updateValues();
  _serverDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdMerge(FXObject*,FXSelector,void*)
{
  vvVolDesc::MergeType mergeType = vvVolDesc::VV_MERGE_SLABS2VOL;

  vvDebugMsg::msg(1, "VVShell::onCmdMerge()");

  _mergeDialog->updateValues();
  if (_mergeDialog->execute())
  {
    FXString filename = _mergeDialog->_fileTField->getText();
    int numFiles = FXIntVal(_mergeDialog->_numberTField->getText());
    int increment = FXIntVal(_mergeDialog->_incrementTField->getText());
    if (_mergeDialog->_slices2volButton->getCheck())    mergeType = vvVolDesc::VV_MERGE_SLABS2VOL;
    else if (_mergeDialog->_vol2animButton->getCheck()) mergeType = vvVolDesc::VV_MERGE_VOL2ANIM;
    else if (_mergeDialog->_chan2volButton->getCheck()) mergeType = vvVolDesc::VV_MERGE_CHAN2VOL;
    else assert(0);
    mergeFiles(filename.text(), numFiles, increment, mergeType);
  }
  return 1;
}

//----------------------------------------------------------------------------
/** Load a volume from slices.
  @param firstFile name of first slice file
  @param num number of slices, 0 for all that can be found
  @param increment number increment from slice to slice, default: 1
  @param mergeType way to merge files
*/
void VVShell::mergeFiles(const char* firstFile, int num, int increment, vvVolDesc::MergeType mergeType)
{
  vvDebugMsg::msg(1, "VVShell::mergeFiles()");

  vvFileIO* fio;
  vvVolDesc* vd;
  FXString msgString;

  vd = new vvVolDesc(firstFile);
  fio = new vvFileIO();
  switch (fio->mergeFiles(vd, num, increment, mergeType))
  {
    case vvFileIO::OK:
      vvDebugMsg::msg(2, "Loaded slice sequence: ", vd->getFilename());
      // Use previous pin list if loaded dataset has no pins:
      if (vd->tf.isEmpty())
      {
        vd->tf.setDefaultAlpha(0);
        vd->tf.setDefaultColors((vd->chan==1) ? 0 : 2);
      }
      if (vd->bpc==4 && vd->real[0]==0.0f && vd->real[1]==1.0f) vd->setDefaultRealMinMax();
      setCanvasRenderer(vd, 0, _canvas->_currentGeom, _canvas->_currentVoxels);
      break;
    case vvFileIO::FILE_NOT_FOUND:
      vvDebugMsg::msg(2, "File not found: ", vd->getFilename());
      msgString = "File Not Found:   ";
      msgString += vd->getFilename();
      FXMessageBox::error((FXWindow*)this, MBOX_OK, "Error", msgString.text());
      delete vd;
      break;
    default:
      vvDebugMsg::msg(2, "Cannot merge file: ", vd->getFilename());
      msgString = "Cannot merge file:   ";
      msgString += vd->getFilename();
      FXMessageBox::error((FXWindow*)this, MBOX_OK, "Error", msgString.text());
      delete vd;
      break;
  }
  delete fio;
}

//----------------------------------------------------------------------------
FXString VVShell::getOpenFilename(const FXString& caption, const FXString& patterns)
{
  FXString path = getApp()->reg().readStringEntry("Settings", "CurrentDirectory", "");
  FXString filename = FXFileDialog::getOpenFilename(this, caption, path, patterns);
  if (filename.length() > 0)
  {
    char* dirname = new char[strlen(filename.text()) + 1];
    vvToolshed::extractDirname(dirname, filename.text());
    getApp()->reg().writeStringEntry("Settings", "CurrentDirectory", dirname);
    getApp()->reg().write();  // update registry
    vvToolshed::setCurrentDirectory(dirname);
    delete[] dirname;
  }
  return filename;
}

//----------------------------------------------------------------------------
FXString VVShell::getSaveFilename(const FXString& caption, const FXString& defaultName, 
  const FXString& patterns)
{
  FXString path = getApp()->reg().readStringEntry("Settings", "CurrentDirectory", "");
  path += defaultName;
  FXString filename = FXFileDialog::getSaveFilename(this, caption, path, patterns);
  if (filename.length() > 0)
  {
    char* dirname = new char[strlen(filename.text()) + 1];
    vvToolshed::extractDirname(dirname, filename.text());
    getApp()->reg().writeStringEntry("Settings", "CurrentDirectory", dirname);
    getApp()->reg().write();  // update registry
    vvToolshed::setCurrentDirectory(dirname);
    delete[] dirname;
  }
  return filename;
}

//----------------------------------------------------------------------------
FXString VVShell::getOpenDirectory(const FXString& caption)
{
  FXString path = getApp()->reg().readStringEntry("Settings", "CurrentDirectory", "");
  FXString filename = FXFileDialog::getOpenDirectory(this, caption, path);
  if (filename.length() > 0)
  {
    FXString directory = filename;
#ifdef WIN32
    directory += '\\';
#else
    directory += '/';
#endif
    getApp()->reg().writeStringEntry("Settings", "CurrentDirectory", directory.text());
    getApp()->reg().write();  // update registry
    vvToolshed::setCurrentDirectory(directory.text());
  }
  return filename;
}

//----------------------------------------------------------------------------
long VVShell::onCmdLoadCamera(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdLoadCamera()");

  const FXchar patterns[]="Camera Files (*.cam)\nAll Files (*.*)";
  FXString filename = getOpenFilename("Load Camera File", patterns);
  if(filename.length() > 0)
  {
    _canvas->_ov.loadCamera(filename.text());
    drawScene();
    _cameraDialog->updateValues();
  }
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdSaveCamera(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdSaveCamera()");

  const FXchar patterns[]="Camera Files (*.cam)\nAll Files (*.*)";
  FXString filename = getSaveFilename ("Save Camera to File", "camera.cam", patterns);
  
  if(filename.length() > 0)
  {
    if(vvToolshed::isFile(filename.text()))
    {
      int over = FXMessageBox::question((FXWindow*)this, MBOX_OK_CANCEL, "Warning", "Overwrite existing camera file?");
      if(over == FXMessageBox::ID_CLICKED_CANCEL) return 1;
    }
    _canvas->_ov.saveCamera(filename.text());
  }
  return 1;
}

//----------------------------------------------------------------------------
/// Pop preferences window
long VVShell::onCmdPrefs(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdPrefs()");

  prefWindow->updateValues();
  prefWindow->show();
  return 1;
}

//----------------------------------------------------------------------------
/// Pop tranfer function window
long VVShell::onCmdTrans(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdTrans()");

  transWindow->updateValues();
  transWindow->show();
  return 1;
}

//----------------------------------------------------------------------------
/// Pop slice viewer window
long VVShell::onCmdSliceViewer(FXObject*, FXSelector, void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdSliceViewer()");

  _sliceViewer->updateValues();
  _sliceViewer->show();
  return 1;
}

//----------------------------------------------------------------------------
/// Pop a dialog showing volume info
long VVShell::onCmdVisInfo(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdVisInfo()");

  if(_canvas == NULL)return 1;
  volumeDialog->updateValues();
  volumeDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
/// Pop the about dialog
long VVShell::onCmdAbout(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdAbout()");

  FXString name = "DeskVOX - Desktop VOlume eXplorer\n";
  FXString version = FXString("Version ") + FXString(VV_VERSION) + FXString(".") + FXString(VV_RELEASE) + FXString("\n\n");
  FXString info = name + version +
    "(c) 2004-2005 Brown University, Providence, RI\n" \
    "Jürgen P. Schulze (schulze@cs.brown.edu)\n" \
    "Alexander C. Rice (acrice@cs.brown.edu)\n\n" \
    "DeskVOX comes with ABSOLUTELY NO WARRANTY.\n" \
    "It is free software, and you are welcome to redistribute it under\n" \
    "the LGPL license. See the file 'license.txt' in the program directory.\n\n" \
    "http://www.cs.brown.edu/people/schulze/projects/vox/";

  FXMessageBox::information((FXWindow*)this, MBOX_OK, "About DeskVOX", info.text());

  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdKeys(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdKeys()");

  FXString info = 
    "-/+/=: Change rendering quality\n" \
    "a: Animate: start/stop playback of time series\n" \
    "b: Boundary box on/off\n" \
    "c: Color bar on/off\n" \
    "e: Run benchmark test\n" \
    "f: Frame rate display on/off\n" \
    "H/h: Change data set thickness\n" \
    "i: Slice interpolation on/off\n" \
    "N/n: Previous/next animation step\n" \
    "o: Orientation display on/off\n" \
    "p: Perspective projection on/off\n" \
    "q: Quit\n" \
    "r: Reset viewing parameters\n" \
    "S/s: Change animation speed\n" \
    "t: #Texture display on/off\n" \
    "x: Maximum intensity projection on/off\n" \
    "z: Maximize window on/off";
  FXMessageBox::information((FXWindow*)this, MBOX_OK, "DeskVOX Keyboard Commands", info.text());

  return 1;
}

//----------------------------------------------------------------------------
/// Pop camera settings dialog
long VVShell::onCmdCameraSettings(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdCameraSettings()");

  _cameraDialog->updateValues();
  _cameraDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
/// Pop channel settings dialog
long VVShell::onCmdGammaSettings(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdGammaSettings()");

  gammaDialog->updateValues();
  gammaDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdChannel4Settings(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdChannel4Settings()");

  _channel4Dialog->show();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdOpacitySettings(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdOpacitySettings()");

  _opacityDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdFloatRange(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdFloatRange()");

  _floatRangeDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
/// Pop clip plane dialog
long VVShell::onCmdClipping(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdClipping()");

  _clipDialog->updateValues();
  _clipDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdROI(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdROI()");

  _roiDialog->updateValues();
  _roiDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdBGColor(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdBGColor()");

  float r,g,b;
  _canvas->getBackgroundColor(r, g, b);
  if (_colorPicker->execute() == 0)   // has picker exited with 'cancel'?
  {
    _canvas->setBackgroundColor(r, g, b);  // then undo changes to BG color
  }
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::pickerColorChanged(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::pickerColorChanged()");

  FXColor c = _colorPicker->getRGBA();
  float r = float(FXREDVAL(c))   / 255.0f;
  float g = float(FXGREENVAL(c)) / 255.0f;
  float b = float(FXBLUEVAL(c))  / 255.0f;
  _canvas->setBackgroundColor(r, g, b);
  return 1;
}

//----------------------------------------------------------------------------
/// Pop dimensions dialog
long VVShell::onCmdDimensions(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdDimensions()");

  _dimDialog->updateValues();
  _dimDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdDraw(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdDraw()");

  if(_drawDialog->_canvas == NULL)
  {
    _drawDialog->_canvas = _canvas;
  }
  _drawDialog->updateValues();
  _drawDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
/// Pop diagram dialog
long VVShell::onCmdDiagrams(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdDiagrams()");

  if (_diagramDialog->_canvas == NULL)
  {
    _diagramDialog->_canvas = _canvas;
  }
  _diagramDialog->updateValues();
  _diagramDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdDataType(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdDataType()");

  _dataTypeDialog->updateValues();
  _dataTypeDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdEditVoxels(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdEditVoxels()");

  _editVoxelsDialog->updateValues();
  _editVoxelsDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdMakeHeightField(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdHeightField()");

  _heightFieldDialog->updateValues();
  _heightFieldDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onCmdGLSettings(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdGLSettings()");

  if (_glcanvas->makeCurrent())
  {
    VVGLSettingsDialog dialog((FXWindow*)this);
    _glcanvas->makeNonCurrent();
    dialog.execute();
  }

  return 1;
}

//----------------------------------------------------------------------------
/// Pop screen shot dialog
long VVShell::onCmdScreenShot(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdScreenShot()");
  screenshotDialog->updateValues();
  screenshotDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
/// Pop movie dialog
long VVShell::onCmdMovie(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdMovie()");

  movieDialog->updateValues();
  movieDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
/// Pop time step dialog
long VVShell::onCmdTimeSteps(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onCmdTimeSteps()");

  tsDialog->updateValues();
  tsDialog->show();
  return 1;
}

//----------------------------------------------------------------------------
void VVShell::drawScene()
{
  vvDebugMsg::msg(3, "VVShell::drawScene()");

  static bool firstTime = true;
  if (_glcanvas->makeCurrent())
  {
    if(firstTime)
    {
      _canvas->initCanvas();
      _canvas->resize(_glcanvas->getWidth(), _glcanvas->getHeight());
      prefWindow->updateValues();
    }
    _canvas->draw();

    if (firstTime)
    {
      firstTime = false;
      parseCommandline();
    }

    // Swap if it is double-buffered
    if (_glvisual->isDoubleBuffer())
    {
      _glcanvas->swapBuffers();
    }

    // Make context non-current
    _glcanvas->makeNonCurrent();
  }  
}

//----------------------------------------------------------------------------
long VVShell::onAllUpdate(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(3, "VVShell::onAllUpdate()");

  drawScene();
  return 1;
}

//----------------------------------------------------------------------------
/** Set or change the renderer. Required when data set changes.
  @param vd new volume, NULL if no change to volume data
  @param algorithm 0=no change, 1=textures, 2=Stingray, -1=suppress rendering
*/
void VVShell::setCanvasRenderer(vvVolDesc* vd, int algorithm, vvTexRend::GeometryType gt, vvTexRend::VoxelType vt)
{
  char string[256];

  vvDebugMsg::msg(1, "VVShell::setCanvasRenderer()");

  if (_glcanvas->makeCurrent())
  {
    if (vd)
    {
      delete _canvas->_vd;
      _canvas->_vd = vd;
    }
    else vd = _canvas->_vd;
  
    // Override geometry and voxel type if single slice:
    if (vd->vox[2]==1) 
    {
      gt = vvTexRend::VV_SLICES;
      vt = vvTexRend::VV_RGBA;    // FIXME: all rendering algorithms should be able to handle 1 voxel thick data sets
    }

    _canvas->setRenderer(algorithm, gt, vt);

    if (vd->chan>1 && _canvas->_renderer->_renderState._mipMode==0) prefWindow->toggleMIP();
    else if (vd->chan==1 && _canvas->_renderer->_renderState._mipMode > 0) prefWindow->toggleMIP();
    vd->makeInfoString(string);
    _statusBar->setText(string);
    volumeDialog->updateValues();
    transWindow->updateValues();
    prefWindow->updateValues();
    _dataTypeDialog->updateValues();
    _editVoxelsDialog->updateValues();
    _dimDialog->updateValues();
    _dimDialog->initDefaultDistances();
  }
  else cerr << "VVShell: Cannot set OpenGL context." << endl;
}

//----------------------------------------------------------------------------
void VVShell::updateRendererVolume()
{
  vvDebugMsg::msg(1, "VVShell::updateRendererVolume()");

  if (_glcanvas->makeCurrent())
  {
    _canvas->_renderer->updateVolumeData();
    _glcanvas->makeNonCurrent();
  }
}

//----------------------------------------------------------------------------
/** Make a copy of the OpenGL canvas and save to TIF file on disk.
*/
void VVShell::takeScreenshot(const char* fname, int imgWidth, int imgHeight)
{
  vvFileIO* fio;
  vvVolDesc* imgVD;
  char* currentFile;                              // current file name
  uchar* image = NULL;                            // image data
  int index = 0;

  vvDebugMsg::msg(1, "VVShell::takeScreenshot()");

  if (!_canvas) return;
  if (imgWidth<=0 || imgHeight<=0) return;

  cerr << "filename = " << fname << endl;

  image = new uchar[imgWidth * imgHeight * 3];          // reserve RGB image space

  // Render screenshot to memory:
  if (_glcanvas->makeCurrent())
  {
    _canvas->draw();
    _canvas->_renderer->renderVolumeRGB(imgWidth, imgHeight, image);
    _glcanvas->makeNonCurrent();
  }

  // Search for unused file name:
  currentFile = new char[strlen(fname) + 1 + 20];
  do
  {
    sprintf(currentFile, "%s-%05d.tif", fname, index);
    ++index;
  } while (vvToolshed::isFile(currentFile) && index < 100000);

  // Write screenshot to file:
  imgVD = new vvVolDesc(currentFile, imgWidth, imgHeight, image);
  cerr << "Writing screenshot to file: " << currentFile << endl;
  fio = new vvFileIO();
  fio->saveVolumeData(imgVD, false);

  // Clean up:
  delete fio;
  delete imgVD;
  delete[] image;
  delete[] currentFile;
}

//----------------------------------------------------------------------------
void VVShell::startARToolkitTimer()
{
  vvDebugMsg::msg(1, "VVShell::startARToolkitTimer()");

  const float TIME_STEP = 0.2f;   // time step in seconds
  getApp()->addTimeout(this, ID_ART_TIMER, int(TIME_STEP * 1000.0f), NULL);
}

//----------------------------------------------------------------------------
void VVShell::stopARToolkitTimer()
{
  vvDebugMsg::msg(1, "VVShell::stopARToolkitTimer()");

  getApp()->removeTimeout(this, ID_ART_TIMER);
}

//----------------------------------------------------------------------------
long VVShell::onARToolkitTimerEvent(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onARToolkitTimerEvent()");
  
  if(_glcanvas->makeCurrent())
  {
    _canvas->artTimerEvent();
    _glcanvas->makeNonCurrent();
  }
  startARToolkitTimer();  // trigger next event for continuous events
  return 1;
}

//----------------------------------------------------------------------------
void VVShell::startAnimTimer()
{
  vvDebugMsg::msg(1, "VVShell::startAnimTimer()");

  float delay = fabs(_canvas->_vd->dt * 1000.0f);
  getApp()->addTimeout(this, ID_ANIM_TIMER, int(delay), NULL);
}

//----------------------------------------------------------------------------
void VVShell::stopAnimTimer()
{
  vvDebugMsg::msg(1, "VVShell::stopAnimTimer()");

  getApp()->removeTimeout(this, ID_ANIM_TIMER);
}

//----------------------------------------------------------------------------
long VVShell::onAnimTimerEvent(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onAnimTimerEvent()");

  if (_canvas->_vd->dt > 0.0f) tsDialog->stepForward();
  else tsDialog->stepBack();
  startAnimTimer();  // trigger next event for continuous events
  return 1;
}

//----------------------------------------------------------------------------
void VVShell::startSpinTimer()
{
  const float SPIN_DELAY = 0.05f;  // delay between spin events [seconds]

  vvDebugMsg::msg(1, "VVShell::startSpinTimer()");

  float delay = fabs(SPIN_DELAY * 1000.0f);
  getApp()->addTimeout(this, ID_SPIN_TIMER, int(delay), NULL);
}

//----------------------------------------------------------------------------
void VVShell::stopSpinTimer()
{
  vvDebugMsg::msg(1, "VVShell::stopSpinTimer()");

  getApp()->removeTimeout(this, ID_SPIN_TIMER);
}

//----------------------------------------------------------------------------
long VVShell::onSpinTimerEvent(FXObject*,FXSelector,void*)
{
  vvDebugMsg::msg(1, "VVShell::onSpinTimerEvent()");
  if (_canvas) _canvas->repeatMouseDrag();
  startSpinTimer();  // trigger next event for continuous events
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onDispOrientChange(FXObject*,FXSelector,void* ptr)
{
  _canvas->_renderer->_renderState._orientation = (ptr != NULL);
  drawScene();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onDispBoundsChange(FXObject*,FXSelector,void* ptr)
{
  _canvas->_renderer->_renderState._boundaries = (ptr != NULL);
  drawScene();
  return 1;
}

//----------------------------------------------------------------------------
void VVShell::toggleBounds()
{
  bool newState = !_boundaryItem->getCheck();
  _boundaryItem->setCheck(newState);
  onDispBoundsChange(this, ID_BOUNDARIES, (void*)newState);
}

//----------------------------------------------------------------------------
void VVShell::toggleOrientation()
{
  bool newState = !_orientItem->getCheck();
  _orientItem->setCheck(newState);
  onDispOrientChange(this, ID_ORIENTATION, (void*)newState);
}

//----------------------------------------------------------------------------
void VVShell::togglePalette()
{
  bool newState = !_paletteItem->getCheck();
  _paletteItem->setCheck(newState);
  onDispPaletteChange(this, ID_PALETTE, (void*)newState);
}

//----------------------------------------------------------------------------
void VVShell::toggleQualityDisplay()
{
  bool newState = !_qualityItem->getCheck();
  _qualityItem->setCheck(newState);
  onDispQualityChange(this, ID_QUALITY, (void*)newState);
}

//----------------------------------------------------------------------------
long VVShell::onDispPaletteChange(FXObject*,FXSelector,void* ptr)
{
  _canvas->_renderer->_renderState._palette = (ptr != NULL);
  drawScene();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onDispQualityChange(FXObject*,FXSelector,void* ptr)
{
  _canvas->_renderer->_renderState._qualityDisplay = (ptr != NULL);
  drawScene();
  return 1;
}

//----------------------------------------------------------------------------
long VVShell::onDispFPSChange(FXObject*,FXSelector,void* ptr)
{
  _canvas->_renderer->_renderState._fpsDisplay = (ptr != NULL);
  drawScene();
  return 1;
}

//----------------------------------------------------------------------------
void VVShell::toggleFPS()
{
  bool newState = !_fpsItem->getCheck();
  _fpsItem->setCheck(newState);
  onDispFPSChange(this, ID_FPS, (void*)newState);
}

//----------------------------------------------------------------------------
long VVShell::onDispSpinChange(FXObject*,FXSelector,void* ptr)
{
  if (ptr == NULL) stopSpinTimer();
  return 1;
}

//----------------------------------------------------------------------------
void VVShell::toggleSpin()
{
  bool newState = !_spinItem->getCheck();
  _spinItem->setCheck(newState);
  onDispSpinChange(this, ID_SPIN, (void*)newState);
}

//----------------------------------------------------------------------------
void VVShell::benchmarkTest()
{
  vvDebugMsg::msg(1, "vvView::benchmarkTest()");
  
  vvStopwatch* totalTime;
  char  onOffMode[2][8] = {"off","on"};
  const int HOST_NAME_LEN = 80;
  char  localHost[HOST_NAME_LEN];
  float step = 2.0f * VV_PI / 180.0f;
  GLint viewport[4];        // OpenGL viewport information (position and size)
  int   angle;
  int   framesRendered = 0;

  // Prepare test:
  totalTime = new vvStopwatch();

  _canvas->_renderer->_renderState._quality = 1.0f;
  _canvas->_ov.reset();
  if(_glcanvas->makeCurrent())
  {
    glGetIntegerv(GL_VIEWPORT, viewport);
    drawScene();
    _glcanvas->makeNonCurrent();
  }
  if (gethostname(localHost, HOST_NAME_LEN-1)) strcpy(localHost, "n/a");

  // Print profiling info:
  cerr.setf(ios::fixed, ios::floatfield);
  cerr.precision(3);
  cerr << "*******************************************************************************" << endl;
  cerr << "Local host........................................" << localHost << endl;
  cerr << "Volume file name.................................." << _canvas->_vd->getFilename() << endl;
  cerr << "Volume size [voxels].............................." << _canvas->_vd->vox[0] << " x " << _canvas->_vd->vox[1] << " x " << _canvas->_vd->vox[2] << endl;
  cerr << "Output image size [pixels]........................" << viewport[2] << " x " << viewport[3] << endl;
  cerr << "Image quality....................................." << _canvas->_renderer->_renderState._quality << endl;
  cerr << "Gamma correction.................................." << onOffMode[_canvas->_renderer->_renderState._gammaCorrection] << endl;

  // Perform test:
  totalTime->start();
  for (angle=0; angle<360; angle+=2)
  {
    _canvas->_ov._camera.rotate(step, 0.0f, 1.0f, 0.0f);   // rotate model view matrix
    if(_glcanvas->makeCurrent())
    {
      drawScene();
      _glcanvas->makeNonCurrent();
    }
    ++framesRendered;
  }

  cerr << "Total profiling time [sec]........................" << totalTime->getTime() << endl;
  cerr << "Frames rendered..................................." << framesRendered << endl;
  cerr << "Average time per frame [sec]......................" << (float(totalTime->getTime()/framesRendered)) << endl;
  cerr << "*******************************************************************************" << endl;

  delete totalTime;  
}

/**************************************************************/

/** @return 0 if OK, 1 if program can't continue.
*/
int checkSystemAssumptions()
{
  if (sizeof(uchar)!=1) { cerr << "Unsigned char must be 1 byte." << endl;  return 1; }
  if (sizeof(short)!=2) { cerr << "Short must be 2 bytes." << endl;         return 1; }
  if (sizeof(int)!=4)   { cerr << "Int must be 4 bytes." << endl;           return 1; }
  if (sizeof(long)<4)   { cerr << "Long must be at least 4 bytes." << endl; return 1; }
  return 0;
}

//----------------------------------------------------------------------------
int main(int argc,char *argv[])
{
  vvDebugMsg::setDebugLevel(1); // set global debug level here
  vvDebugMsg::msg(1, "main()");

  if (checkSystemAssumptions()) return 1;

  // Make application
  FXString name;
  name = "DeskVOX " + FXString(VV_VERSION) + FXString(".") + FXString(VV_RELEASE);
  FXApp application(name, "BrownUniversity");    // these values are used in the registry system

  // Open the display
  application.init(argc,argv);

  // Make window
  new VVShell(&application);

  // Create the application's windows
  application.create();
  
  // Run the application
  application.run();

  // Write registry values back to system and exit:
  application.exit();

  cerr << "Bye bye!" << endl;

  return 0;
}

// EOF
