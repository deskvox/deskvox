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
#include "vvdimensiondialog.h"

#include "ui_vvdimensiondialog.h"

#include <virvo/vvdebugmsg.h>
#include <virvo/vvtexrend.h>
#include <virvo/vvvoldesc.h>

vvDimensionDialog::vvDimensionDialog(vvCanvas* canvas, QWidget* parent)
  : QDialog(parent)
  , ui(new Ui_DimensionDialog)
  , _canvas(canvas)
{
  vvDebugMsg::msg(1, "vvDimensionDialog::vvDimensionDialog()");

  ui->setupUi(this);

  connect(ui->applyButton, SIGNAL(clicked()), this, SLOT(onApplyClicked()));
}

vvDimensionDialog::~vvDimensionDialog()
{
  vvDebugMsg::msg(1, "vvDimensionDialog::~vvDimensionDialog()");
}

void vvDimensionDialog::setInitialDist(const vvVector3& dist)
{
  vvDebugMsg::msg(3, "vvDimensionDialog::setInitialDist()");

  _initialDist = dist;
}

void vvDimensionDialog::onApplyClicked()
{
  vvDebugMsg::msg(3, "vvDimensionDialog::onApplyClicked()");

  vvVector3 dist(static_cast<float>(ui->distXBox->value()),
                 static_cast<float>(ui->distYBox->value()),
                 static_cast<float>(ui->distZBox->value()));

 _canvas->getVolDesc()->setDist(dist);

  // TODO: hide this implementation detail
  if (vvTexRend* texrend = dynamic_cast<vvTexRend*>(_canvas->getRenderer()))
  {
    if (texrend->getGeomType() == vvTexRend::VV_BRICKS)
    {
      texrend->updateBrickGeom();
    }
  }

  _canvas->updateGL();
}

