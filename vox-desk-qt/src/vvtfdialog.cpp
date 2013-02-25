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

#include "vvcanvas.h"
#include "vvtfdialog.h"

#include "ui_vvtfdialog.h"

#include <virvo/vvdebugmsg.h>

#include <QGraphicsScene>

namespace
{
/** Convert canvas x coordinates to data values.
  @param canvas canvas x coordinate [0..1]
  @return data value
*/
float norm2data(const vvVector2f& zoomrange, float canvas)
{
  return canvas * (zoomrange[1] - zoomrange[0]) + zoomrange[0];
}

/** Convert data value to x coordinate in TF canvas.
  @param data data value
  @return canvas x coordinate [0..1]
*/
float data2norm(const vvVector2f& zoomrange, float data)
{
  return (data - zoomrange[0]) / (zoomrange[1] - zoomrange[0]);
}

/** Convert horizontal differences on the canvas to data differences.
*/
float normd2datad(const vvVector2f& zoomrange, float canvas)
{
  return canvas * (zoomrange[1] - zoomrange[0]);
}

/** Convert differences in data to the canvas.
*/
float datad2normd(const vvVector2f& zoomrange, float data)
{
  return data / (zoomrange[1] - zoomrange[0]);
}
}

vvTFDialog::vvTFDialog(vvCanvas* canvas, QWidget* parent)
  : QDialog(parent)
  , ui(new Ui_TFDialog)
  , _scene(new QGraphicsScene)
  , _canvas(canvas)
  , _zoomRange(vvVector2f(0.0f, 1.0f))
{
  vvDebugMsg::msg(1, "vvTFDialog::vvTFDialog()");

  ui->setupUi(this);

  ui->color1DView->setScene(_scene);

  _scene->addRect(QRectF(0, 0, 20, 20));

  connect(ui->colorButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(ui->pyramidButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(ui->gaussianButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(ui->customButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(ui->skipRangeButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(ui->undoButton, SIGNAL(clicked()), this, SLOT(onUndoClicked()));
  connect(ui->presetColorsBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onPresetColorsChanged(int)));
  connect(ui->presetAlphaBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onPresetAlphaChanged(int)));
  connect(ui->applyButton, SIGNAL(clicked()), this, SLOT(onApplyClicked()));
}

vvTFDialog::~vvTFDialog()
{
  vvDebugMsg::msg(1, "vvTFDialog::~vvTFDialog()");

  delete _scene;
}

void vvTFDialog::drawColorTexture()
{
  vvDebugMsg::msg(3, "vvTFDialog::drawColorTexture()");

  int w = 768;
  int h = 3;
  std::vector<uchar> colorBar;
  colorBar.resize(w * h * 4);
  makeColorBar(&colorBar, w);
  QImage img(&colorBar[0], w, h, QImage::Format_ARGB32);
  if (!img.isNull())
  {
    QBrush brush(img);
    ui->color1DView->setBackgroundBrush(brush);
  }
}

void vvTFDialog::drawAlphaTexture()
{
  int w = ui->alpha1DView->width();
  int h = ui->alpha1DView->height();
  std::vector<uchar> alphaTex;
  alphaTex.resize(w * h * 4);
  makeAlphaTexture(&alphaTex, w, h);
  QImage img(&alphaTex[0], w, h, QImage::Format_ARGB32);
  if (!img.isNull())
  {
    QBrush brush(img);
    ui->alpha1DView->setBackgroundBrush(brush);
  }
}

void vvTFDialog::makeColorBar(std::vector<uchar>* colorBar, int width) const
{
  if (/* HDR */ false)
  {

  }
  else // standard iso-range TF mode
  {
    _canvas->getVolDesc()->tf.makeColorBar(width, &(*colorBar)[0], _zoomRange[0], _zoomRange[1], false, vvToolshed::VV_ARGB);
  }
}

void vvTFDialog::makeAlphaTexture(std::vector<uchar>* alphaTex, int width, int height) const
{
  if (/* HDR */ false)
  {

  }
  else // standard iso-range TF mode
  {
    _canvas->getVolDesc()->tf.makeAlphaTexture(width, height, &(*alphaTex)[0], _zoomRange[0], _zoomRange[1]);
  }
}

void vvTFDialog::onUndoClicked()
{
  emit undo();
  emit newTransferFunction();
}

void vvTFDialog::onNewWidget()
{
  vvTFWidget* widget = NULL;

  if (QObject::sender() == ui->colorButton)
  {
    widget = new vvTFColor(vvColor(), norm2data(_zoomRange, 0.5f));
  }
  else if (QObject::sender() == ui->pyramidButton)
  {
    widget = new vvTFPyramid(vvColor(), false, 1.0f, norm2data(_zoomRange, 0.5f), normd2datad(_zoomRange, 0.4f), normd2datad(_zoomRange, 0.2f));
  }
  else if (QObject::sender() == ui->gaussianButton)
  {
    widget = new vvTFBell(vvColor(), false, 1.0f, norm2data(_zoomRange, 0.2f), normd2datad(_zoomRange, 0.2));
  }
  else if (QObject::sender() == ui->customButton)
  {
    widget = new vvTFCustom(norm2data(_zoomRange, 0.5f), norm2data(_zoomRange, 0.5f));
  }
  else if (QObject::sender() == ui->skipRangeButton)
  {
    widget = new vvTFSkip(norm2data(_zoomRange, 0.5f), normd2datad(_zoomRange, 0.2f));
  }
  emit newWidget(widget);
}

void vvTFDialog::onPresetColorsChanged(int index)
{
  vvDebugMsg::msg(3, "vvTFDialog::onPresetColorsChanged()");

  _canvas->getVolDesc()->tf.setDefaultColors(index, _zoomRange[0], _zoomRange[1]);
  emitTransFunc();
}

void vvTFDialog::onPresetAlphaChanged(int index)
{
  vvDebugMsg::msg(3, "vvTFDialog::onPresetAlphaChanged()");

  _canvas->getVolDesc()->tf.setDefaultAlpha(index, _zoomRange[0], _zoomRange[1]);
  emitTransFunc();
}

void vvTFDialog::onApplyClicked()
{
  drawColorTexture();
//  drawAlphaTexture();
}

void vvTFDialog::emitTransFunc()
{
  emit newTransferFunction();
}

