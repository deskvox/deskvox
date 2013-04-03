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

#ifndef VV_SLICEVIEWER_H
#define VV_SLICEVIEWER_H

#include <virvo/vvvecmath.h>

#include <QDialog>

class vvVolDesc;
class Ui_SliceViewer;

class vvSliceViewer : public QDialog
{
  Q_OBJECT
public:
  vvSliceViewer(vvVolDesc* vd, QWidget* parent = 0);
private:
  Ui_SliceViewer* ui;

  vvVolDesc* _vd;
  size_t _slice;
  vvVecmath::AxisType _axis;

  void paint();
  void updateUi();
public slots:
  void onNewVolDesc(vvVolDesc* vd);
  void onNewFrame(int frame);
private slots:
  void setSlice(int slice);
  void updateAxis(bool checked);
  void updateOrientation(bool checked);
  void onFwdClicked();
  void onFwdFwdClicked();
  void onBackClicked();
  void onBackBackClicked();
};

#endif

