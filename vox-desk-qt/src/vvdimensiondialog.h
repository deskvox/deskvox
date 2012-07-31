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

#ifndef VV_DIMENSIONDIALOG_H
#define VV_DIMENSIONDIALOG_H

class vvCanvas;
class Ui_DimensionDialog;

#include <virvo/vvvecmath.h>

#include <QDialog>

class vvDimensionDialog : public QDialog
{
  Q_OBJECT
public:
  vvDimensionDialog(vvCanvas* canvas, QWidget* parent = 0);
  ~vvDimensionDialog();

  /*! initial dist should be set whenever a new volume is loaded
   */
  void setInitialDist(const vvVector3& dist);
private:
  Ui_DimensionDialog* ui;

  vvCanvas* _canvas;

  vvVector3 _initialDist; ///< should be assigned when a new file is loaded
private slots:
  void onApplyClicked();
};

#endif

