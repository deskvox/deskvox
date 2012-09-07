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

#include "vvtimestepdialog.h"

#include "ui_vvtimestepdialog.h"

#include <virvo/vvdebugmsg.h>

#include <QSettings>

vvTimeStepDialog::vvTimeStepDialog(QWidget* parent)
  : QDialog(parent)
  , ui(new Ui_TimeStepDialog)
  , _playing(false)
{
  vvDebugMsg::msg(1, "vvTimeStepDialog::vvTimeStepDialog()");

  ui->setupUi(this);

  ui->playButton->setFocus(Qt::OtherFocusReason);

  QSettings settings;
  ui->frameRateBox->setValue(settings.value("timestepdialog/fps").value<double>());

  connect(ui->frameRateBox, SIGNAL(valueChanged(double)), this, SLOT(onFrameRateChanged()));
  connect(ui->playButton, SIGNAL(clicked()), this, SLOT(onPlayClicked()));
  connect(ui->backButton, SIGNAL(clicked()), this, SIGNAL(back()));
  connect(ui->fwdButton, SIGNAL(clicked()), this, SIGNAL(fwd()));
  connect(ui->backBackButton, SIGNAL(clicked()), this, SIGNAL(first()));
  connect(ui->fwdFwdButton, SIGNAL(clicked()), this, SIGNAL(last()));
  connect(ui->timeStepSlider, SIGNAL(sliderMoved(int)), this, SIGNAL(valueChanged(int)));
}

vvTimeStepDialog::~vvTimeStepDialog()
{
  vvDebugMsg::msg(1, "vvTimeStepDialog::~vvTimeStepDialog()");
}

void vvTimeStepDialog::setFrames(const int frames)
{
  vvDebugMsg::msg(3, "vvTimeStepDialog::setFrames()");

  ui->timeStepLabel->setText(QString::number(ui->timeStepSlider->value() + 1) + "/" + QString::number(frames));
  ui->timeStepSlider->setMaximum(frames - 1);
  ui->timeStepSlider->setTickInterval(1);
}

void vvTimeStepDialog::setCurrentFrame(const int frame)
{
  vvDebugMsg::msg(3, "vvTimeStepDialog::setCurrentFrame()");

  ui->timeStepLabel->setText(QString::number(frame + 1) + "/" + QString::number(ui->timeStepSlider->maximum() + 1));
  ui->timeStepSlider->setValue(frame);
}

void vvTimeStepDialog::togglePlayback()
{
  vvDebugMsg::msg(3, "vvTimeStepDialog::togglePlayback()");

  onPlayClicked();
}

void vvTimeStepDialog::onPlayClicked()
{
  vvDebugMsg::msg(3, "vvTimeStepDialog::onPlayClicked()");

  if (!_playing)
  {
    ui->playButton->setText("||");
    emit play(ui->frameRateBox->value());    
  }
  else
  {
    ui->playButton->setText(">");
    emit pause();
  }
  _playing = !_playing;
}

void vvTimeStepDialog::onFrameRateChanged()
{
  vvDebugMsg::msg(3, "vvTimeStepDialog::onFrameRateChanged()");

  QSettings settings;
  settings.setValue("timestepdialog/fps", ui->frameRateBox->value());

  if (_playing)
  {
    // re-emit play signal
    emit play(ui->frameRateBox->value());
  }
}

