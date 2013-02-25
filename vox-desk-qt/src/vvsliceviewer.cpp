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

#include "vvsliceviewer.h"

#include "ui_vvsliceviewer.h"

#include <virvo/vvvoldesc.h>

#include <QImage>
#include <QPixmap>

#include <cassert>
#include <iostream>
#include <limits>

#define VV_UNUSED(x) ((void)(x))

namespace
{
void clamp(int* slice, int slices)
{
  *slice = std::min(*slice, slices - 1);
  *slice = std::max(0, *slice);
}

QImage getSlice(vvVolDesc* vd, std::vector<uchar>* texture, int slice, vvVecmath::AxisType axis)
{
  assert(texture != NULL);

  int width;
  int height;
  int slices;
  vd->getVolumeSize(axis, width, height, slices);
  clamp(&slice, slices);
  texture->resize(width * height * 3);
  vd->makeSliceImage(vd->getCurrentFrame(), axis, slice, &(*texture)[0]);
  int bytesPerLine = width * 3 * sizeof(uchar);
  return QImage(&(*texture)[0], width, height, bytesPerLine, QImage::Format_RGB888);
}
}

vvSliceViewer::vvSliceViewer(vvVolDesc* vd, QWidget* parent)
  : QDialog(parent)
  , ui(new Ui_SliceViewer)
  , _vd(vd)
  , _slice(0)
  , _axis(vvVecmath::Z_AXIS)
{
  ui->setupUi(this);

  connect(ui->sliceSlider, SIGNAL(sliderMoved(int)), this, SLOT(setSlice(int)));
  connect(ui->xaxisButton, SIGNAL(clicked(bool)), this, SLOT(updateAxis(bool)));
  connect(ui->yaxisButton, SIGNAL(clicked(bool)), this, SLOT(updateAxis(bool)));
  connect(ui->zaxisButton, SIGNAL(clicked(bool)), this, SLOT(updateAxis(bool)));
  connect(ui->horizontalBox, SIGNAL(clicked(bool)), this, SLOT(updateOrientation(bool)));
  connect(ui->verticalBox, SIGNAL(clicked(bool)), this, SLOT(updateOrientation(bool)));
  connect(ui->fwdButton, SIGNAL(clicked()), this, SLOT(onFwdClicked()));
  connect(ui->fwdFwdButton, SIGNAL(clicked()), this, SLOT(onFwdFwdClicked()));
  connect(ui->backButton, SIGNAL(clicked()), this, SLOT(onBackClicked()));
  connect(ui->backBackButton, SIGNAL(clicked()), this, SLOT(onBackBackClicked()));

  paint();
  updateUi();
}

void vvSliceViewer::paint()
{
  std::vector<uchar> texture;
  QImage img = QImage(getSlice(_vd, &texture, _slice, _axis));
  if (!img.isNull())
  {
    img = img.scaled(ui->frame->width(), ui->frame->height());
    if (ui->horizontalBox->isChecked() || ui->verticalBox->isChecked())
    {
      img = img.mirrored(ui->horizontalBox->isChecked(), ui->verticalBox->isChecked());
    }
    QPixmap pm = QPixmap::fromImage(img);
    QBrush br(pm);
    QPalette pal;
    pal.setBrush(QPalette::Window, br);
    ui->frame->setPalette(pal);
  }
}

void vvSliceViewer::updateUi()
{
  int width;
  int height;
  int slices;
  _vd->getVolumeSize(_axis, width, height, slices);
  clamp(&_slice, slices);

  ui->resolutionLabel->setText(QString::number(width) + " x " + QString::number(height));

  switch (_axis)
  {
  case vvVecmath::X_AXIS:
    ui->xaxisButton->setChecked(true);
    ui->yaxisButton->setChecked(false);
    ui->zaxisButton->setChecked(false);
    break;
  case vvVecmath::Y_AXIS:
    ui->xaxisButton->setChecked(false);
    ui->yaxisButton->setChecked(true);
    ui->zaxisButton->setChecked(false);
    break;
  case vvVecmath::Z_AXIS:
    ui->xaxisButton->setChecked(false);
    ui->yaxisButton->setChecked(false);
    ui->zaxisButton->setChecked(true);
    break;
  default:
    break;
  }

  ui->sliceLabel->setText(QString::number(_slice + 1) + "/" + QString::number(slices));
  ui->sliceSlider->setMinimum(0);
  ui->sliceSlider->setMaximum(slices - 1);
  ui->sliceSlider->setTickInterval(1);
  ui->sliceSlider->setValue(_slice);
}

void vvSliceViewer::onNewVolDesc(vvVolDesc* vd)
{
  _vd = vd;
  _slice = 0;
  _axis = vvVecmath::Z_AXIS;
  paint();
  updateUi();
}

void vvSliceViewer::onNewFrame(int frame)
{
  _vd->setCurrentFrame(frame);
  paint();
  updateUi();
}

void vvSliceViewer::setSlice(int slice)
{
  int width;
  int height;
  int slices;
  _vd->getVolumeSize(_axis, width, height, slices);
  _slice = slice;
  clamp(&_slice, slices);
  paint();
  updateUi();
}

void vvSliceViewer::updateAxis(bool checked)
{
  if (!checked)
  {
    return;
  }

  if (QObject::sender() == ui->xaxisButton)
  {
    _axis = vvVecmath::X_AXIS;
  }
  else if (QObject::sender() == ui->yaxisButton)
  {
    _axis = vvVecmath::Y_AXIS;
  }
  else if (QObject::sender() == ui->zaxisButton)
  {
    _axis = vvVecmath::Z_AXIS;
  }
  paint();
  updateUi();
}

void vvSliceViewer::updateOrientation(bool checked)
{
  VV_UNUSED(checked);
  paint();
  updateUi();
}

void vvSliceViewer::onFwdClicked()
{
  setSlice(_slice + 1);
}

void vvSliceViewer::onFwdFwdClicked()
{
  // setSlice will clamp this to lastslice - 1
  setSlice(std::numeric_limits<int>::max());
}

void vvSliceViewer::onBackClicked()
{
  setSlice(_slice - 1);
}

void vvSliceViewer::onBackBackClicked()
{
  setSlice(0);
}

