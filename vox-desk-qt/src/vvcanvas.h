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

#ifndef VV_CANVAS_H
#define VV_CANVAS_H

#include "vvobjview.h"
#include "vvplugin.h"

#include <virvo/vvrendererfactory.h>
#include <virvo/vvvecmath.h>

#include <QGLWidget>
#include <QList>
#include <QMouseEvent>

class vvPlugin;
class QTimer;

class vvCanvas : public QGLWidget
{
  Q_OBJECT
public:
  enum ParameterType
  {
    VV_BG_COLOR = 0,
    VV_DOUBLEBUFFERING,
    VV_SUPERSAMPLES,
    VV_PROJECTIONTYPE
  };

  vvCanvas(const QGLFormat& format, const QString& filename = "", QWidget* parent = 0);
  ~vvCanvas();

  void setVolDesc(vvVolDesc* vd);
  void setPlugins(const QList<vvPlugin*>& plugins);

  vvVolDesc* getVolDesc() const;
  vvRenderer* getRenderer() const;

  void loadCamera(const QString& filename);
  void saveCamera(const QString& filename);
protected:
  void initializeGL();
  void paintGL();
  void resizeGL(int w, int h);

  void mouseMoveEvent(QMouseEvent* event);
  void mousePressEvent(QMouseEvent* event);
  void mouseReleaseEvent(QMouseEvent* event);
private:
  vvVolDesc* _vd;
  vvRenderer* _renderer;
  std::string _currentRenderer;
  vvRendererFactory::Options _currentOptions;

  QList<vvPlugin*> _plugins;

  vox::vvObjView _ov;
  vox::vvObjView::ProjectionType _projectionType;
  vvColor _bgColor;
  bool _doubleBuffering;
  int _superSamples;

  Qt::MouseButton _mouseButton;
  QPoint _lastMousePos;

  QTimer* _animTimer;

  void init();
  void createRenderer();
  void updateProjection();
  void setCurrentFrame(int frame);
public slots:
  void setParameter(ParameterType param, const vvParam& value);
  void setParameter(vvRenderer::ParameterType param, const vvParam& value);
  vvParam getParameter(ParameterType param) const;
  vvParam getParameter(vvRenderer::ParameterType param) const;

  void startAnimation(double fps);
  void stopAnimation();
  void setTimeStep(int step);
  void incTimeStep();
  void decTimeStep();
  void firstTimeStep();
  void lastTimeStep();
  void setQuality(float quality);
signals:
  void newVolDesc(vvVolDesc* vd);
  void currentFrame(int frame);
  void resized(const QSize& size);
};

#endif

