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

#include "vvvolinfodialog.h"

#include "ui_vvvolinfodialog.h"

#include <virvo/vvvoldesc.h>

#include <iostream>

vvVolInfoDialog::vvVolInfoDialog(QWidget* parent)
  : QDialog(parent)
  , ui(new Ui_VolInfoDialog)
{
  ui->setupUi(this);

  connect(ui->iconButton, SIGNAL(clicked()), this, SLOT(onUpdateIconClicked()));
}

vvVolInfoDialog::~vvVolInfoDialog()
{
}

void vvVolInfoDialog::onNewVolDesc(vvVolDesc* vd)
{
  ui->filenameEdit->setText(vd->getFilename());
  ui->slicesWidthLabel->setText(QString::number(vd->vox[0]));
  ui->slicesHeightLabel->setText(QString::number(vd->vox[1]));
  ui->slicesDepthLabel->setText(QString::number(vd->vox[2]));
  ui->timeStepsLabel->setText(QString::number(vd->frames));
  ui->bpsLabel->setText(QString::number(vd->bpc));
  ui->channelsLabel->setText(QString::number(vd->chan));
  ui->voxelsLabel->setText(QString::number(vd->getFrameVoxels()));
  ui->bytesLabel->setText(QString::number(vd->getFrameBytes()));
  ui->distXLabel->setText(QString::number(vd->dist[0]));
  ui->distYLabel->setText(QString::number(vd->dist[1]));
  ui->distZLabel->setText(QString::number(vd->dist[2]));
  ui->pminLabel->setText(QString::number(vd->real[0]));
  ui->pmaxLabel->setText(QString::number(vd->real[1]));
  float fmin;
  float fmax;
  vd->findMinMax(0, fmin, fmax);
  ui->minLabel->setText(QString::number(fmin));
  ui->maxLabel->setText(QString::number(fmax));
}

void vvVolInfoDialog::onUpdateIconClicked()
{
}

