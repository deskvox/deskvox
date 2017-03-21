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

#include <virvo/math/math.h>
#include <virvo/vvmacros.h>
#include <virvo/vvvoldesc.h>

#include <QImage>
#include <QPixmap>

#include <cassert>
#include <iostream>
#include <limits>

struct vvSliceViewer::Impl
{
  Impl()
    : ui(new Ui::SliceViewer)
    , slice(0)
    , axis(virvo::cartesian_axis< 3 >::Z)
  {
  }
 
  std::auto_ptr<Ui::SliceViewer> ui;
  size_t slice;
  virvo::cartesian_axis< 3 > axis;

private:

  VV_NOT_COPYABLE(Impl)

};

namespace
{
void clamp(size_t* slice, size_t slices)
{
  *slice = std::min(*slice, slices - 1);
  *slice = std::max(size_t(0), *slice);
}

QImage getSlice(vvVolDesc* vd, std::vector<uchar>* texture, size_t slice, virvo::cartesian_axis< 3 > axis)
{
  assert(texture != NULL);

  size_t width;
  size_t height;
  size_t slices;
  vd->getVolumeSize(axis, width, height, slices);
  clamp(&slice, slices);
  texture->resize(width * height * 3);
  vd->makeSliceImage(vd->getCurrentFrame(), axis, slice, &(*texture)[0]);
  size_t bytesPerLine = width * 3 * sizeof(uchar);
  return QImage(&(*texture)[0], width, height, bytesPerLine, QImage::Format_RGB888);
}
}

vvSliceViewer::vvSliceViewer(vvVolDesc* vd, QWidget* parent)
  : QDialog(parent)
  , impl_(new Impl)
  , _vd(vd)
{
  impl_->ui->setupUi(this);

  connect(impl_->ui->sliceSlider, SIGNAL(sliderMoved(int)), this, SLOT(setSlice(int)));
  connect(impl_->ui->xaxisButton, SIGNAL(clicked(bool)), this, SLOT(updateAxis(bool)));
  connect(impl_->ui->yaxisButton, SIGNAL(clicked(bool)), this, SLOT(updateAxis(bool)));
  connect(impl_->ui->zaxisButton, SIGNAL(clicked(bool)), this, SLOT(updateAxis(bool)));
  connect(impl_->ui->horizontalBox, SIGNAL(clicked(bool)), this, SLOT(updateOrientation(bool)));
  connect(impl_->ui->verticalBox, SIGNAL(clicked(bool)), this, SLOT(updateOrientation(bool)));
  connect(impl_->ui->fwdButton, SIGNAL(clicked()), this, SLOT(onFwdClicked()));
  connect(impl_->ui->fwdFwdButton, SIGNAL(clicked()), this, SLOT(onFwdFwdClicked()));
  connect(impl_->ui->backButton, SIGNAL(clicked()), this, SLOT(onBackClicked()));
  connect(impl_->ui->backBackButton, SIGNAL(clicked()), this, SLOT(onBackBackClicked()));

  paint();
  updateUi();
}

void vvSliceViewer::paint()
{
  if (!_vd)
    return;

  std::vector<uchar> texture;
  QImage img = QImage(getSlice(_vd, &texture, impl_->slice, impl_->axis));
  if (!img.isNull())
  {
    img = img.scaled(impl_->ui->frame->width(), impl_->ui->frame->height());
    if (impl_->ui->horizontalBox->isChecked() || impl_->ui->verticalBox->isChecked())
    {
      img = img.mirrored(impl_->ui->horizontalBox->isChecked(), impl_->ui->verticalBox->isChecked());
    }
    QPixmap pm = QPixmap::fromImage(img);
    QBrush br(pm);
    QPalette pal;
    pal.setBrush(QPalette::Window, br);
    impl_->ui->frame->setPalette(pal);
  }
}

void vvSliceViewer::updateUi()
{
  size_t width;
  size_t height;
  size_t slices;
  if (_vd)
  {
    _vd->getVolumeSize(impl_->axis, width, height, slices);
  }
  else
  {
    width = height = slices = 0;
  }

  clamp(&impl_->slice, slices);

  impl_->ui->resolutionLabel->setText(QString::number(width) + " x " + QString::number(height));

  switch (impl_->axis)
  {
  case virvo::cartesian_axis< 3 >::X:
    impl_->ui->xaxisButton->setChecked(true);
    impl_->ui->yaxisButton->setChecked(false);
    impl_->ui->zaxisButton->setChecked(false);
    break;
  case virvo::cartesian_axis< 3 >::Y:
    impl_->ui->xaxisButton->setChecked(false);
    impl_->ui->yaxisButton->setChecked(true);
    impl_->ui->zaxisButton->setChecked(false);
    break;
  case virvo::cartesian_axis< 3 >::Z:
    impl_->ui->xaxisButton->setChecked(false);
    impl_->ui->yaxisButton->setChecked(false);
    impl_->ui->zaxisButton->setChecked(true);
    break;
  default:
    break;
  }

  impl_->ui->sliceLabel->setText(QString::number(impl_->slice + 1) + "/" + QString::number(slices));
  impl_->ui->sliceSlider->setMinimum(0);
  impl_->ui->sliceSlider->setMaximum(slices - 1);
  impl_->ui->sliceSlider->setTickInterval(1);
  impl_->ui->sliceSlider->setValue(impl_->slice);
}

void vvSliceViewer::onNewVolDesc(vvVolDesc* vd)
{
  _vd = vd;
  impl_->slice = 0;
  impl_->axis = virvo::cartesian_axis< 3 >::Z;
  paint();
  updateUi();
}

void vvSliceViewer::onNewFrame(int frame)
{
  if (_vd)
    _vd->setCurrentFrame(frame);
  paint();
  updateUi();
}

void vvSliceViewer::update()
{
  paint();
}

void vvSliceViewer::setSlice(int slice)
{
  size_t width;
  size_t height;
  size_t slices;
  _vd->getVolumeSize(impl_->axis, width, height, slices);
  impl_->slice = slice;
  clamp(&impl_->slice, slices);
  paint();
  updateUi();
}

void vvSliceViewer::updateAxis(bool checked)
{
  if (!checked)
  {
    return;
  }

  if (QObject::sender() == impl_->ui->xaxisButton)
  {
    impl_->axis = virvo::cartesian_axis< 3 >::X;
  }
  else if (QObject::sender() == impl_->ui->yaxisButton)
  {
    impl_->axis = virvo::cartesian_axis< 3 >::Y;
  }
  else if (QObject::sender() == impl_->ui->zaxisButton)
  {
    impl_->axis = virvo::cartesian_axis< 3 >::Z;
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
  setSlice(impl_->slice + 1);
}

void vvSliceViewer::onFwdFwdClicked()
{
  // setSlice will clamp this to lastslice - 1
  setSlice(std::numeric_limits<int>::max());
}

void vvSliceViewer::onBackClicked()
{
  setSlice(impl_->slice - 1);
}

void vvSliceViewer::onBackBackClicked()
{
  setSlice(0);
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
