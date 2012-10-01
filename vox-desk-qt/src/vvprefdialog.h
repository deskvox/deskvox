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

#ifndef VV_PREFDIALOG_H
#define VV_PREFDIALOG_H

#include "vvparameters.h"

#include <virvo/vvrenderer.h>

#include <QDialog>

class Ui_PrefDialog;

class vvPrefDialog : public QDialog
{
  Q_OBJECT
public:
  vvPrefDialog(QWidget* parent = 0);

  void toggleInterpolation();
  void scaleStillQuality(float s);
private:
  Ui_PrefDialog* ui;
private slots:
  void onInterpolationToggled(bool checked);
  void onMipToggled(bool checked);
  void onMovingSpinBoxChanged(double value);
  void onStillSpinBoxChanged(double value);
  void onMovingDialChanged(int value);
  void onStillDialChanged(int value);
signals:
  void parameterChanged(vvParameters::ParameterType param, const vvParam& value);
  void parameterChanged(vvRenderer::ParameterType param, const vvParam& value);
};

#endif

