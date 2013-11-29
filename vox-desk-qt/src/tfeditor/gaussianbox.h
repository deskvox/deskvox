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

#pragma once

#include <virvo/vvcolor.h>
#include <virvo/vvvecmath.h>

#include <QGroupBox>

#include <memory>

namespace tf
{
class GaussianBox : public QGroupBox
{

  Q_OBJECT
  Q_DISABLE_COPY(GaussianBox)

public:
  GaussianBox(QWidget* parent = 0);
  ~GaussianBox();

  void setHasColor(bool hascolor);
  void setColor(const vvColor& color);
  void setSize(const vvVector3& size);
  void setOpacity(float opacity);
private:

  struct Impl;
  std::auto_ptr<Impl> impl;

private slots:
  void getColor();
  void emitSize(int sliderval);
  void emitOpacity(int sliderval);
signals:
  void hasColor(bool hascolor);
  void color(const QColor& color);
  void size(const vvVector3& size);
  void opacity(float opacity);
};
}

