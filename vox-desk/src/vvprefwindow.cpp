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

#pragma warning(disable: 4244)                    // disable warning about conversion from int to short
#pragma warning(disable: 4512)                    // disable warning: assignment operator could not be generated

#ifdef WIN32
#include <typeinfo.h>
#endif

#include "vvprefwindow.h"
#include "vvcanvas.h"
#include "vvshell.h"

using namespace vox;

/*******************************************************************************/
FXDEFMAP(VVPreferenceWindow) VVPreferenceWindowMap[]=
{
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_STEREO,         VVPreferenceWindow::onStereoChange),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_ARTOOLKIT,      VVPreferenceWindow::onARToolkitChange),
  FXMAPFUNC(SEL_CHANGED,     VVPreferenceWindow::ID_EYE_DIST,       VVPreferenceWindow::onEyeChange),
  FXMAPFUNC(SEL_CHANGED,     VVPreferenceWindow::ID_QUALITY,        VVPreferenceWindow::onQualityChanging),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_VOXELTYPE,      VVPreferenceWindow::onVTChange),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_GEOMTYPE,       VVPreferenceWindow::onGTChange),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_PIXEL_SHADER,   VVPreferenceWindow::onPSChange),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_BRICK_SIZE,     VVPreferenceWindow::onBSChange),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_QUALITY,        VVPreferenceWindow::onQualityChange),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_MIP,            VVPreferenceWindow::onMIPSelect),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_LINTERP,        VVPreferenceWindow::onInterpolationSelect),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_SHOWBRICKS,     VVPreferenceWindow::onShowBricksSelect),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_COMPUTE_BRICKSIZE, VVPreferenceWindow::onComputeBricksizeSelect),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_TEX_MEMORY,     VVPreferenceWindow::onTexMemoryChange),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_DEF_VOL,        VVPreferenceWindow::onDefaultVolume),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_SUPPRESS,       VVPreferenceWindow::onSuppressRendering),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_SWAP_EYES,      VVPreferenceWindow::onSwapEyes),
  FXMAPFUNC(SEL_COMMAND,     VVPreferenceWindow::ID_QUALITY_TEXT,   VVPreferenceWindow::onQualityTextChange),
};

FXIMPLEMENT(VVPreferenceWindow,FXDialogBox,VVPreferenceWindowMap,ARRAYNUMBER(VVPreferenceWindowMap))

// Construct a dialog box
VVPreferenceWindow::VVPreferenceWindow(FXWindow* owner, vvCanvas* c) :
FXDialogBox(owner,"Preferences", DECOR_TITLE | DECOR_BORDER | DECOR_CLOSE, 50, 50)
{
  _canvas = c;
  _shell = (VVShell*)owner;

  FXVerticalFrame* master=new FXVerticalFrame(this,LAYOUT_FILL_X|LAYOUT_FILL_Y);

  FXGroupBox* algoGroup = new FXGroupBox(master,"Rendering algorithm", FRAME_GROOVE | LAYOUT_FILL_X);
  FXMatrix* algoMatrix = new FXMatrix(algoGroup, 2, MATRIX_BY_COLUMNS | LAYOUT_FILL_X);

  new FXLabel(algoMatrix, "Geometry:", NULL,LABEL_NORMAL);
  _gtCombo=new FXComboBox(algoMatrix,1,this,ID_GEOMTYPE, COMBOBOX_INSERT_LAST | COMBOBOX_STATIC | FRAME_SUNKEN | FRAME_THICK | LAYOUT_FILL_X);

  new FXLabel(algoMatrix, "Voxel type:", NULL,LABEL_NORMAL);
  _vtCombo=new FXComboBox(algoMatrix,1,this,ID_VOXELTYPE, COMBOBOX_INSERT_LAST | COMBOBOX_STATIC | FRAME_SUNKEN | FRAME_THICK | LAYOUT_FILL_X);

  new FXLabel(algoMatrix, "Pixel shader:", NULL, LABEL_NORMAL);
  _psCombo=new FXComboBox(algoMatrix,1,this,ID_PIXEL_SHADER, COMBOBOX_INSERT_LAST | COMBOBOX_STATIC | FRAME_SUNKEN | FRAME_THICK | LAYOUT_FILL_X);

  new FXLabel(algoMatrix, "Brick size:", NULL, LABEL_NORMAL);
  _bsCombo = new FXComboBox(algoMatrix, 1, this, ID_BRICK_SIZE, COMBOBOX_INSERT_LAST | COMBOBOX_STATIC | FRAME_SUNKEN | FRAME_THICK | LAYOUT_FILL_X);

  FXVerticalFrame* checkFrame = new FXVerticalFrame(algoGroup, LAYOUT_FILL_X);
  _linterpButton  = new FXCheckButton(checkFrame,"Linear interpolation",this,VVPreferenceWindow::ID_LINTERP, ICON_BEFORE_TEXT | LAYOUT_FILL_ROW);
  _mipButton      = new FXCheckButton(checkFrame,"Maximum intensity projection (MIP)",this,ID_MIP,ICON_BEFORE_TEXT | LAYOUT_FILL_ROW);
  _suppressButton = new FXCheckButton(checkFrame,"Suppress rendering",this,ID_SUPPRESS,ICON_BEFORE_TEXT | LAYOUT_FILL_ROW);
  _showBricksButton = new FXCheckButton(checkFrame, "Show Bricks", this, VVPreferenceWindow::ID_SHOWBRICKS, ICON_BEFORE_TEXT | LAYOUT_FILL_ROW);
  _computeBrickSizeButton = new FXCheckButton(checkFrame, "Compute Bricksize", this, VVPreferenceWindow::ID_COMPUTE_BRICKSIZE, ICON_BEFORE_TEXT | LAYOUT_FILL_ROW);

  FXHorizontalFrame* texMemoryFrame = new FXHorizontalFrame(algoGroup);
  new FXLabel(texMemoryFrame, "Texture memory size: ");
  _texMemoryField = new FXTextField(texMemoryFrame, 5, this, ID_TEX_MEMORY);

  vvTexRend* texrend = dynamic_cast<vvTexRend*>(_canvas->_renderer);
  if (texrend)
    texrend->setTexMemorySize(FXIntVal(_texMemoryField->getText()));

  FXGroupBox* qualityGroup = new FXGroupBox(master, "Rendering quality", FRAME_GROOVE | LAYOUT_FILL_X);
  FXHorizontalFrame* qualityFrame = new FXHorizontalFrame(qualityGroup, LAYOUT_FILL_X);

  _qualityDial = new FXDial(qualityFrame, this, ID_QUALITY, DIAL_HORIZONTAL | LAYOUT_FIX_WIDTH,0,0,200);
  _qualityDial->setRange(0, 10000);
  _qualityDial->setValue(100);
  _qualityDial->setRevolutionIncrement(200);
  _qualityDial->setNotchSpacing(100);
  _qualityDial->setTipText("0.0 for single slice, 1.0 for one slice per voxel, >1.0 for better reconstruction");

  _qualityTField = new FXTextField(qualityFrame, 5, this, ID_QUALITY_TEXT, TEXTFIELD_REAL | JUSTIFY_RIGHT | TEXTFIELD_NORMAL);
  _qualityTField->setText(FXStringFormat("%.2f", 1.0f));

  FXGroupBox* stereoGroup = new FXGroupBox(master,"Stereo", FRAME_GROOVE | LAYOUT_FILL_X);

  FXHorizontalFrame* stereoFrame = new FXHorizontalFrame(stereoGroup, LAYOUT_FILL_X);
  new FXLabel(stereoFrame, "Stereo mode:", NULL, LABEL_NORMAL);
  _stereoCombo = new FXComboBox(stereoFrame,1,this,ID_STEREO, COMBOBOX_INSERT_LAST | COMBOBOX_STATIC | FRAME_SUNKEN | FRAME_THICK);

  new FXLabel(stereoGroup, "Inter-ocular distance [mm]:", NULL, LABEL_NORMAL);

  FXHorizontalFrame* iodFrame = new FXHorizontalFrame(stereoGroup, LAYOUT_FILL_X);

  _eyeSlider = new FXSlider(iodFrame,this,ID_EYE_DIST, SLIDER_HORIZONTAL | SLIDER_ARROW_DOWN | SLIDER_TICKS_BOTTOM | LAYOUT_FILL_X);
  _eyeSlider->setRange(0,100);
  _eyeSlider->setTickDelta(10);
  _eyeSlider->setValue(63);

  _eyeTField = new FXTextField(iodFrame, 5, NULL, 0, TEXTFIELD_REAL | JUSTIFY_RIGHT | LAYOUT_RIGHT);
  _eyeTField->setEditable(false);
  _eyeTField->setText(FXStringFormat("%d",63));

  _swapEyesButton = new FXCheckButton(stereoGroup,"Swap eyes",this,ID_SWAP_EYES,ICON_BEFORE_TEXT | LAYOUT_FILL_ROW);

  _artoolkitButton = new FXCheckButton(master,"Use ARToolkit tracking",this,ID_ARTOOLKIT,ICON_BEFORE_TEXT);
#ifndef VV_USE_ARTOOLKIT
  _artoolkitButton->disable();
#endif

  FXGroupBox* defGroup = new FXGroupBox(master,"Default data set", FRAME_GROOVE | LAYOUT_FILL_X);
  _defCombo = new FXComboBox(defGroup,1,this, ID_DEF_VOL, COMBOBOX_INSERT_LAST | COMBOBOX_STATIC | FRAME_SUNKEN | FRAME_THICK | LAYOUT_FILL_X);
  _defCombo->appendItem("Multiplicative Arches");
  _defCombo->appendItem("Min-Max Without Bounds");
  _defCombo->appendItem("Four Channel Slabs");
  _defCombo->appendItem("Forest Of Cubes");
  _defCombo->appendItem("Additive Floats");
  _defCombo->setNumVisible(_defCombo->getNumItems());

  new FXButton(master,"Close",NULL,this,ID_ACCEPT,FRAME_RAISED|FRAME_THICK|LAYOUT_CENTER_X|LAYOUT_CENTER_Y,0,0,0,0, 20,20,3,3);
}

// Must delete the menus
VVPreferenceWindow::~VVPreferenceWindow()
{
  delete _eyeSlider;
  delete _eyeTField;
  delete _gtCombo;
  delete _vtCombo;
  delete _psCombo;
  delete _bsCombo;
  delete _linterpButton;
  delete _mipButton;
}

long VVPreferenceWindow::onEyeChange(FXObject*,FXSelector,void*)
{
  float val = (float)_eyeSlider->getValue();
  _canvas->_ov.setIOD(val);
  _eyeTField->setText(FXStringFormat("%d",(FXint)val));
  return 1;
}

long VVPreferenceWindow::onStereoChange(FXObject*,FXSelector,void* ptr)
{
  vvCanvas::StereoType mode = vvCanvas::MONO;

  switch(_stereoCombo->getCurrentItem())
  {
    case 0: mode = vvCanvas::MONO; break;
    case 1: mode = vvCanvas::SIDE_BY_SIDE; break;
    case 2: mode = vvCanvas::RED_BLUE; break;
    case 3: mode = vvCanvas::RED_GREEN; break;
    case 4: mode = vvCanvas::ACTIVE; break;
    default: assert(0); break;
  }
  _canvas->setStereoMode(mode);
  return 1;
}

long VVPreferenceWindow::onARToolkitChange(FXObject*,FXSelector,void* ptr)
{
  int answer = FXMessageBox::question((FXWindow*)this, MBOX_OK_CANCEL, "Warning", "ARToolkit tracking is under construction and may crash DeskVOX. Enable anyway?");
  if (answer == FX::MBOX_CLICKED_CANCEL)
  {
    _artoolkitButton->setCheck(false);
    return 1;
  }

  _canvas->setARToolkit(ptr != NULL);
  if (ptr != NULL)
  {
    _shell->startARToolkitTimer();
    _artoolkitButton->disable();                  // FIXME: once disabling doesn't crash it anymore, delete this line
  }
  else
  {
    _shell->stopARToolkitTimer();
  }
  return 1;
}

long VVPreferenceWindow::onGTChange(FXObject*,FXSelector,void*)
{
  _shell->setCanvasRenderer(NULL, 0, vvTexRend::GeometryType(_gtCombo->getCurrentItem()), _canvas->_currentVoxels);
  return 1;
}

long VVPreferenceWindow::onDefaultVolume(FXObject*,FXSelector,void*)
{
  _shell->loadDefaultVolume(_defCombo->getCurrentItem(), 32, 32, 32);
  return 1;
}

long VVPreferenceWindow::onVTChange(FXObject*,FXSelector,void*)
{
  _shell->setCanvasRenderer(NULL, 0, _canvas->_currentGeom, vvTexRend::VoxelType(_vtCombo->getCurrentItem()));
  return 1;
}

long VVPreferenceWindow::onPSChange(FXObject*,FXSelector,void*)
{
  vvTexRend* texrend = dynamic_cast<vvTexRend*>(_canvas->_renderer);
  if (texrend) texrend->setCurrentShader(_psCombo->getCurrentItem());
  return 1;
}

long VVPreferenceWindow::onBSChange(FXObject*, FXSelector, void*)
{
  int size;

  vvTexRend* texrend = dynamic_cast<vvTexRend*>(_canvas->_renderer);

  if (texrend)
  {
    switch (_bsCombo->getCurrentItem())
    {
      case 0:
        size = 16; break;
      case 1:
        size = 32; break;
      case 2:
        size = 64; break;
      case 3:
        size = 128; break;
      case 4:
        size = 256; break;
    case 5:
      size = 512; break;
      default:
        size = 64; break;
    }
    _shell->_glcanvas->makeCurrent();
    texrend->setBrickSize(size);
    _shell->_glcanvas->makeNonCurrent();

    return 1;
  }
  else
    return 1;
}

long VVPreferenceWindow::onQualityChanging(FXObject*,FXSelector,void*)
{
  _qualityTField->setText(FXStringFormat("%.2f", getQualityDialValue()));
  return 1;
}

float VVPreferenceWindow::getQualityDialValue()
{
  return float(_qualityDial->getValue()) / 100.0f;
}

void VVPreferenceWindow::setQualityDialValue(float val)
{
  _qualityDial->setValue(int(val * 100.0f));
}

long VVPreferenceWindow::onQualityChange(FXObject*,FXSelector,void*)
{
  _canvas->_renderer->_renderState._quality = getQualityDialValue();
  _shell->drawScene();
  return 1;
}

long VVPreferenceWindow::onQualityTextChange(FXObject*,FXSelector,void*)
{
  setQualityDialValue(FXFloatVal(_qualityTField->getText()));
  handle(this, FXSEL(SEL_COMMAND, ID_QUALITY), NULL);
  return 1;
}

long VVPreferenceWindow::onMIPSelect(FXObject*,FXSelector,void* ptr)
{
  _shell->_glcanvas->makeCurrent();
  _canvas->_renderer->_renderState._mipMode = (ptr != NULL) ? 1 : 0;
  _shell->_glcanvas->makeNonCurrent();
  return 1;
}

long VVPreferenceWindow::onInterpolationSelect(FXObject*,FXSelector,void* ptr)
{
  _shell->_glcanvas->makeCurrent();
  _canvas->_renderer->setParameter(vvRenderer::VV_SLICEINT, (ptr != NULL));
  _shell->_glcanvas->makeNonCurrent();
  return 1;
}

long VVPreferenceWindow::onShowBricksSelect(FXObject*, FXSelector, void* ptr)
{
  vvTexRend* texrend = dynamic_cast<vvTexRend*>(_canvas->_renderer);

  if (texrend)
  {
    _shell->_glcanvas->makeCurrent();
    texrend->setShowBricks((_showBricksButton->getCheck()) ? true : false);
    _shell->_glcanvas->makeNonCurrent();
  }
  return 1;
}

long VVPreferenceWindow::onComputeBricksizeSelect(FXObject*, FXSelector, void* ptr)
{
  vvTexRend* texrend = dynamic_cast<vvTexRend*>(_canvas->_renderer);

  if (texrend)
  {
    _shell->_glcanvas->makeCurrent();
    texrend->setComputeBrickSize((_computeBrickSizeButton->getCheck()) ? true : false);
    _shell->_glcanvas->makeNonCurrent();

    if (texrend->getGeomType() == vvTexRend::VV_BRICKS)
    {
      if (_computeBrickSizeButton->getCheck())
      {
        _bsCombo->disable();
        setBSCombo(texrend->getBrickSize());
      }
      else
        _bsCombo->enable();
    }
  }

  return 1;
}

long VVPreferenceWindow::onTexMemoryChange(FXObject*, FXSelector, void*)
{
  vvTexRend* texrend = dynamic_cast<vvTexRend*>(_canvas->_renderer);

  if (texrend)
  {
    _shell->_glcanvas->makeCurrent();
    texrend->setTexMemorySize(FXIntVal(_texMemoryField->getText()));
    if (texrend->getGeomType() == vvTexRend::VV_BRICKS)
      setBSCombo(texrend->getBrickSize());
    _shell->getApp()->reg().writeStringEntry("Settings", "Texture memory size", _texMemoryField->getText().text());
    _shell->_glcanvas->makeCurrent();
  }
  return 1;
}

void VVPreferenceWindow::setBSCombo(int size)
{
  switch (size)
  {
    case 16:
      _bsCombo->setCurrentItem(0); break;
    case 32:
      _bsCombo->setCurrentItem(1); break;
    case 64:
      _bsCombo->setCurrentItem(2); break;
    case 128:
      _bsCombo->setCurrentItem(3); break;
    case 256:
      _bsCombo->setCurrentItem(4); break;
  case 512:
    _bsCombo->setCurrentItem(5); break;
    default:
      _bsCombo->setCurrentItem(1); break;
  }
}

long VVPreferenceWindow::onSuppressRendering(FXObject*,FXSelector,void* ptr)
{
  if (ptr != NULL) _shell->setCanvasRenderer(NULL, -1);
  else _shell->setCanvasRenderer(NULL, 1, _canvas->_currentGeom, _canvas->_currentVoxels);
  return 1;
}

long VVPreferenceWindow::onSwapEyes(FXObject*,FXSelector,void* ptr)
{
  _canvas->setSwapEyes(ptr != NULL);
  return 1;
}

void VVPreferenceWindow::toggleInterpol()
{
  bool newState = !_linterpButton->getCheck();
  _linterpButton->setCheck(newState);
  onInterpolationSelect(this, ID_LINTERP, (void*)newState);
}

void VVPreferenceWindow::toggleMIP()
{
  bool newState = !_mipButton->getCheck();
  _mipButton->setCheck(newState);
  onMIPSelect(this, ID_MIP, (void*)newState);
}

void VVPreferenceWindow::scaleQuality(float factor)
{
  assert(factor>=0.0f);
  float quality = getQualityDialValue();
  quality *= factor;
  setQualityDialValue(quality);
  _qualityTField->setText(FXStringFormat("%.2f", quality));
  _canvas->_renderer->_renderState._quality = quality;
  _shell->drawScene();
}

void VVPreferenceWindow::updateValues()
{
  vvTexRend* texrend = NULL;

  _eyeSlider->setValue(int(_canvas->_ov.getIOD()));
  _eyeTField->setText(FXStringFormat("%d",(FXint)_canvas->_ov.getIOD()));

  setQualityDialValue(_canvas->_renderer->_renderState._quality);
  _qualityTField->setText(FXStringFormat("%.2f",_canvas->_renderer->_renderState._quality));

  _linterpButton->setCheck(_canvas->_renderer->getParameter(vvRenderer::VV_SLICEINT)==1.0f);
  _mipButton->setCheck(_canvas->_renderer->_renderState._mipMode==1.0f);
  _artoolkitButton->setCheck(_canvas->getARToolkit());

  if (_shell->_glcanvas->makeCurrent())
  {
    // Rendering geometry:
    _gtCombo->clearItems();
    _gtCombo->appendItem("Autoselect");
    _gtCombo->appendItem("2D textures (slices)");
    _gtCombo->appendItem("2D textures (cubic)");
                                                  // dynamic_cast requires RTTI to be on
    texrend = dynamic_cast<vvTexRend*>(_canvas->_renderer);

    if (texrend)
    {
      if (texrend->isSupported(vvTexRend::VV_VIEWPORT))
      {
        _gtCombo->appendItem("3D textures (viewport aligned)");
        _gtCombo->appendItem("3D textures (bricked)");
      }
      if (texrend->isSupported(vvTexRend::VV_SPHERICAL)) _gtCombo->appendItem("3D textures (spherical)");
    }
    _gtCombo->setNumVisible(_gtCombo->getNumItems());
    if (texrend)
    {
      _gtCombo->setCurrentItem(int(texrend->getGeomType()));
    }

    // Voxel type:
    _vtCombo->clearItems();
    _vtCombo->appendItem("Autoselect");
    _vtCombo->appendItem("RGBA");
    _vtCombo->appendItem("SGI LUT");
    _vtCombo->appendItem("OpenGL paletted textures");
    _vtCombo->appendItem("Nvidia texture shader");
    _vtCombo->appendItem("Nvidia pixel shader");
    if (texrend)
    {
      if (texrend->isSupported(vvTexRend::VV_FRG_PRG)) _vtCombo->appendItem("ARB fragment program");
    }
    _vtCombo->setNumVisible(_vtCombo->getNumItems());
    if (texrend) _vtCombo->setCurrentItem(int(texrend->getVoxelType()));

    // Pixel shader:
    _psCombo->clearItems();
    _psCombo->appendItem("1 channel");
    _psCombo->appendItem("2 channels");
    _psCombo->appendItem("3 channels");
    _psCombo->appendItem("4 channels");
    _psCombo->appendItem("Grayscale");
    _psCombo->appendItem("2 chan+ow");
    _psCombo->appendItem("3 chan+ow");
    _psCombo->appendItem("4 chan+ow");
    _psCombo->appendItem("2D TF");
    _psCombo->setNumVisible(_psCombo->getNumItems());
    if (texrend) _psCombo->setCurrentItem(texrend->getCurrentShader());

    // Brick size:
    _bsCombo->clearItems();
    _bsCombo->appendItem("16");
    _bsCombo->appendItem("32");
    _bsCombo->appendItem("64");
    _bsCombo->appendItem("128");
    _bsCombo->appendItem("256");
    _bsCombo->appendItem("512");
    _bsCombo->setNumVisible(_bsCombo->getNumItems());

    if (_gtCombo->getCurrentItem() == 4)
    {
      _showBricksButton->enable();
      _computeBrickSizeButton->enable();
    }
    else
    {
      _bsCombo->disable();
      _showBricksButton->disable();
      _computeBrickSizeButton->disable();
    }

    _texMemoryField->setText(getApp()->reg().readStringEntry("Settings", "Texture memory size", ""));

    if (texrend)
      texrend->setTexMemorySize(FXIntVal(_texMemoryField->getText()));

    if ((texrend) && (texrend->getGeomType() == vvTexRend::VV_BRICKS))
    {
      _showBricksButton->setCheck(texrend->getShowBricks());
      if (texrend->getComputeBrickSize())
      {
        _bsCombo->disable();
        _computeBrickSizeButton->setCheck(true);
      }
      else
      {
        _bsCombo->enable();
        _computeBrickSizeButton->setCheck(false);
      }

      setBSCombo(texrend->getBrickSize());
    }
    else
      _bsCombo->setText("");

    // MIP button:
    if (texrend && texrend->isSupported(vvTexRend::VV_MIP)) _mipButton->enable();
    else _mipButton->disable();

    const FXchar* size = getApp()->reg().readStringEntry("Settings", "Texture memory size", NULL);
    if (size != NULL)
      _texMemoryField->setText(size);
    else if (texrend)
      _texMemoryField->setText("512");

    if (texrend) texrend->setTexMemorySize(FXIntVal(_texMemoryField->getText()));

    // Stereo combo box:
    _stereoCombo->clearItems();
    _stereoCombo->appendItem("Off (Mono)");
    _stereoCombo->appendItem("Side-by-side");
    _stereoCombo->appendItem("Anaglyph red-blue");
    _stereoCombo->appendItem("Anaglyph red-green");
    if (_shell->_glvisual->isStereo())
    {
      _stereoCombo->appendItem("Active");
    }
    _stereoCombo->setNumVisible(_stereoCombo->getNumItems());

    // Swap eyes:
    _swapEyesButton->setCheck(_canvas->getSwapEyes());
  }
}
