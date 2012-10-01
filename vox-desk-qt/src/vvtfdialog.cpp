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
  std::vector<uchar> colorBar;
  colorBar.resize(w * 3 * 4);
  makeColorBar(&colorBar, w);
  QImage img(&colorBar[0], w, 3, QImage::Format_ARGB32);
  if (!img.isNull())
  {
    QBrush brush(img);
    ui->color1DView->setBackgroundBrush(brush);
  }
}

void vvTFDialog::makeColorBar(std::vector<uchar>* colorBar, const int width) const
{
  vvDebugMsg::msg(3, "vvTFDialog::makeColorBar()");

  if (/* HDR */ false)
  {

  }
  else // standard iso-range TF mode
  {
    _canvas->getVolDesc()->tf.makeColorBar(width, &(*colorBar)[0], _zoomRange[0], _zoomRange[1], true, vvToolshed::VV_ARGB);
  }
}

void vvTFDialog::onPresetColorsChanged(int index)
{
  vvDebugMsg::msg(3, "vvTFDialog::onPresetColorsChanged()");

  _canvas->getVolDesc()->tf.setDefaultColors(index, _zoomRange[0], _zoomRange[1]);
  updateTransFunc();
}

void vvTFDialog::onPresetAlphaChanged(int index)
{
  vvDebugMsg::msg(3, "vvTFDialog::onPresetAlphaChanged()");

  _canvas->getVolDesc()->tf.setDefaultAlpha(index, _zoomRange[0], _zoomRange[1]);
  updateTransFunc();
}

void vvTFDialog::onApplyClicked()
{
  vvDebugMsg::msg(3, "vvTFDialog::onApplyClicked()");

  drawColorTexture();
}

void vvTFDialog::updateTransFunc()
{
  vvDebugMsg::msg(3, "vvTFDialog::updateTransFunc()");

  _canvas->makeCurrent();
  _canvas->getRenderer()->updateTransferFunction();
  _canvas->updateGL();
}

