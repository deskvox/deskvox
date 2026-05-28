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
#include <QMouseEvent>
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
 
  std::unique_ptr<Ui::SliceViewer> ui;
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
  connect(impl_->ui->screenshotButton, SIGNAL(clicked()), this, SLOT(screenshot()));

  // clipping in main viewport:
  connect(impl_->ui->clipPlaneBox, SIGNAL(clicked(bool)), this, SLOT(emitClipPlane()));
  connect(impl_->ui->sliceSlider, SIGNAL(sliderMoved(int)), this, SLOT(emitClipPlane()));
  connect(impl_->ui->xaxisButton, SIGNAL(clicked(bool)), this, SLOT(emitClipPlane()));
  connect(impl_->ui->yaxisButton, SIGNAL(clicked(bool)), this, SLOT(emitClipPlane()));
  connect(impl_->ui->zaxisButton, SIGNAL(clicked(bool)), this, SLOT(emitClipPlane()));
  connect(impl_->ui->fwdButton, SIGNAL(clicked()), this, SLOT(emitClipPlane()));
  connect(impl_->ui->fwdFwdButton, SIGNAL(clicked()), this, SLOT(emitClipPlane()));
  connect(impl_->ui->backButton, SIGNAL(clicked()), this, SLOT(emitClipPlane()));
  connect(impl_->ui->backBackButton, SIGNAL(clicked()), this, SLOT(emitClipPlane()));

  paint();
  updateUi();
}

vvSliceViewer::~vvSliceViewer()
{
}

void vvSliceViewer::paint()
{
  if (!_vd)
    return;

  std::vector<uchar> texture;
  QImage img = getSlice(_vd, &texture, impl_->slice, impl_->axis);
  if (!img.isNull())
  {
    int s = std::min(impl_->ui->frame->width(), impl_->ui->frame->height());
    img = img.scaled(s, s, Qt::KeepAspectRatioByExpanding);
    if (impl_->ui->horizontalBox->isChecked() || impl_->ui->verticalBox->isChecked())
    {
      img = img.mirrored(impl_->ui->horizontalBox->isChecked(), impl_->ui->verticalBox->isChecked());
    }
    QPixmap pm = QPixmap::fromImage(img);
    impl_->ui->frame->setPixmap(pm);
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

void vvSliceViewer::mouseMoveEvent(QMouseEvent* event)
{
  auto pixmap = impl_->ui->frame->pixmap(Qt::ReturnByValue);
  double xd = (event->pos().x() - impl_->ui->frame->pos().x() - 1)
                / (double)pixmap.width();
  double yd = (event->pos().y() - impl_->ui->frame->pos().y() - 1)
                / (double)pixmap.height();

  int x = 0, y = 0, z = 0;
  int flipped_x = 0, flipped_y = 0, flipped_z = 0;

  if (impl_->axis == virvo::cartesian_axis<3>::X)
  {
    x = impl_->slice;
    y = ts_clamp((ssize_t)(xd * _vd->vox[1]), (ssize_t)0, _vd->vox[1] - 1);
    z = ts_clamp((ssize_t)(yd * _vd->vox[2]), (ssize_t)0, _vd->vox[2] - 1);

    flipped_x = x;
    flipped_y = impl_->ui->horizontalBox->isChecked() ? _vd->vox[1] - y - 1 : y;
    flipped_z = impl_->ui->verticalBox->isChecked() ? _vd->vox[2] - z - 1 : z;
  }
  else if (impl_->axis == virvo::cartesian_axis<3>::Y)
  {
    x = ts_clamp((ssize_t)(xd * _vd->vox[0]), (ssize_t)0, _vd->vox[0] - 1);
    y = impl_->slice;
    z = ts_clamp((ssize_t)(yd * _vd->vox[2]), (ssize_t)0, _vd->vox[2] - 1);

    flipped_x = impl_->ui->horizontalBox->isChecked() ? _vd->vox[0] - x - 1 : x;
    flipped_y = y;
    flipped_z = impl_->ui->verticalBox->isChecked() ? _vd->vox[2] - z - 1 : z;
  }
  else if (impl_->axis == virvo::cartesian_axis<3>::Z)
  {
    x = ts_clamp((ssize_t)(xd * _vd->vox[0]), (ssize_t)0, _vd->vox[0] - 1);
    y = ts_clamp((ssize_t)(yd * _vd->vox[1]), (ssize_t)0, _vd->vox[1] - 1);
    z = impl_->slice;

    flipped_x = impl_->ui->horizontalBox->isChecked() ? _vd->vox[0] - x - 1 : x;
    flipped_y = impl_->ui->verticalBox->isChecked() ? _vd->vox[1] - y - 1 : y;
    flipped_z = z;
  }

  impl_->ui->xLabel->setText("X: " + QString::number(x));
  impl_->ui->yLabel->setText("Y: " + QString::number(y));
  impl_->ui->zLabel->setText("Z: " + QString::number(z));

  float val = _vd->getChannelValue(0, flipped_x, flipped_y, flipped_z, 0);

  impl_->ui->valueLabel->setText("Value: " + QString::number(val));
}

void vvSliceViewer::mousePressEvent(QMouseEvent* event)
{
  mouseMoveEvent(event);
}

void vvSliceViewer::resizeEvent(QResizeEvent* /*event*/)
{
  auto pixmap = impl_->ui->frame->pixmap(Qt::ReturnByValue);
  int width = pixmap.width();
  int height = pixmap.height();
  impl_->ui->viewSizeLabel->setText(QString::number(width) + " x " + QString::number(height));
  paint();
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

void vvSliceViewer::screenshot()
{
  if (!_vd)
    return;

  std::vector<uchar> texture;
  QImage img = getSlice(_vd, &texture, impl_->slice, impl_->axis);
  if (!img.isNull())
  {
    int s = std::min(impl_->ui->frame->width(), impl_->ui->frame->height());
    img = img.scaled(s, s, Qt::KeepAspectRatioByExpanding);
    if (impl_->ui->horizontalBox->isChecked() || impl_->ui->verticalBox->isChecked())
    {
      img = img.mirrored(impl_->ui->horizontalBox->isChecked(), impl_->ui->verticalBox->isChecked());
    }

    std::stringstream str;
    str << _vd->getFilename();
    switch (impl_->axis)
    {
    case virvo::cartesian_axis<3>::X: str << "_X_"; break;
    case virvo::cartesian_axis<3>::Y: str << "_Y_"; break;
    case virvo::cartesian_axis<3>::Z: str << "_Z_"; break;
    default: assert(0); break;
    }
    str << impl_->slice << ".png";
    std::string fn = str.str();
    if (img.save(fn.c_str()))
      std::cout << "Saved slice image to: " << fn << '\n';
    else
      std::cerr << "Error saving image to file: " << fn << '\n';
  }
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

void vvSliceViewer::emitClipPlane() const
{
  if (!impl_->ui->clipPlaneBox->isChecked()) {
    emit(clipPlane(false, {}, {}));
    return;
  }

  size_t width;
  size_t height;
  size_t slices;
  size_t slice = impl_->slice;
  auto axis = impl_->axis;
  _vd->getVolumeSize(axis, width, height, slices);

  float slice01{0.f};
  switch ((int)axis) {
  case 0: slice01 = float(slice)/(width-1); break;
  case 1: slice01 = float(slice)/(height-1); break;
  case 2: slice01 = float(slice)/(slices-1); break;
  default: break;
  }
  slice01 = fmaxf(slice01,0.f);
  slice01 = fminf(slice01,1.f);

  auto size = _vd->getBoundingBox().size();
  float extend = size[(int)axis];
  float offset = slice01*extend-extend*0.5f;

  virvo::vec3f n(0,0,1);
  virvo::vec3f o(offset,0,0);
  switch ((int)axis) {
  case 0: n = {1.f,0.f,0.f}; o = {offset,0.f,0.f}; break;
  case 1: n = {0.f,1.f,0.f}; o = {0.f,-offset,0.f}; break;
  case 2: n = {0.f,0.f,1.f}; o = {0.f,0.f,-offset}; break;
  default: break;
  }

  emit(clipPlane(true, n, o));
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
