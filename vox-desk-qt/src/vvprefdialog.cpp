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

#include "vvprefdialog.h"

#include "ui_vvprefdialog.h"

#include <virvo/vvdebugmsg.h>

#include <cassert>
#include <iostream>
#include <limits>

namespace
{
  int movingDialOldValue = 0;
  int stillDialOldValue = 0;

  int getDialDelta(int oldval, int newval, int minval, int maxval)
  {
    const int median = (minval + maxval) / 2;
//    if (oldval >= minval && oldval < median && newval < maxval
  }
}

vvPrefDialog::vvPrefDialog(QWidget* parent)
  : QDialog(parent)
  , ui(new Ui_PrefDialog)
{
  vvDebugMsg::msg(1, "vvPrefDialog::vvPrefDialog()");

  ui->setupUi(this);

  connect(ui->interpolationCheckBox, SIGNAL(toggled(bool)), this, SLOT(onInterpolationToggled(bool)));
  connect(ui->mipCheckBox, SIGNAL(toggled(bool)), this, SLOT(onMipToggled(bool)));
  connect(ui->movingDial, SIGNAL(valueChanged(int)), this, SLOT(onMovingDialChanged(int)));
  connect(ui->stillDial, SIGNAL(valueChanged(int)), this, SLOT(onStillDialChanged(int)));
}

void vvPrefDialog::toggleInterpolation()
{
  vvDebugMsg::msg(3, "vvPrefDialog::toggletInterpolation()");

  const bool interpolation = ui->interpolationCheckBox->isChecked();
  ui->interpolationCheckBox->setChecked(!interpolation);
  emit parameterChanged(vvRenderer::VV_SLICEINT, !interpolation);
}

void vvPrefDialog::scaleStillQuality(const float s)
{
  vvDebugMsg::msg(3, "vvPrefDialog::scaleStillQuality()");

  assert(s >= 0.0f);

  float quality = static_cast<float>(ui->stillSpinBox->value());
  if (quality <= 0.0f)
  {
    // never let quality drop to or below 0
    quality = std::numeric_limits<float>::epsilon();
  }
  quality *= s;
  
  ui->stillSpinBox->setValue(quality);
  emit parameterChanged(vvRenderer::VV_QUALITY, quality);
}

void vvPrefDialog::onInterpolationToggled(const bool checked)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onInterpolationToggled()");

  emit parameterChanged(vvRenderer::VV_SLICEINT, checked);
}

void vvPrefDialog::onMipToggled(bool checked)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onMipToggled()");

  const int mipMode = checked ? 1 : 0; // don't support mip == 2 (min. intensity) for now
  emit parameterChanged(vvRenderer::VV_MIP_MODE, mipMode);
}

void vvPrefDialog::onMovingDialChanged(int value)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onMovingDialChanged()");

  int d = getDialDelta(movingDialOldValue, value, ui->movingDial->minimum(), ui->movingDial->maximum());
  std::cerr << d << std::endl;
  movingDialOldValue = value;
}

void vvPrefDialog::onStillDialChanged(int value)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onStillDialChanged()");
}

