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

#include <GL/glew.h>

#include "vvcanvas.h"

#include <virvo/vvdebugmsg.h>
#include <virvo/vvfileio.h>
#include <virvo/vvgltools.h>

#include <QSettings>

#include <iostream>

using vox::vvObjView;

vvCanvas::vvCanvas(QWidget* parent)
  : QGLWidget(parent)
  , _vd(NULL)
  , _renderer(NULL)
  , _projectionType(vox::vvObjView::PERSPECTIVE)
  , _doubleBuffering(true)
  , _superSamples(0)
{
  vvDebugMsg::msg(1, "vvCanvas::vvCanvas()");

  // init ui
  setMouseTracking(true);
  setFocusPolicy(Qt::StrongFocus);

  // init visual
  QGLFormat format;
  format.setDoubleBuffer(_doubleBuffering);
  format.setDepth(true);
  format.setRgba(true);
  format.setAlpha(true);
  format.setAccum(true);
  format.setStencil(false);
  if (_superSamples > 0)
  {
    format.setSampleBuffers(true);
    format.setSamples(_superSamples);
  }
  setFormat(format);

  // read persistent settings
  QSettings settings;
  QColor qcolor = settings.value("canvas/bgcolor").value<QColor>();
  _bgColor = vvColor(qcolor.redF(), qcolor.greenF(), qcolor.blueF());
}

vvCanvas::~vvCanvas()
{
  vvDebugMsg::msg(1, "vvCanvas::~vvCanvas()");

  delete _renderer;
  delete _vd;
}

void vvCanvas::setParameter(ParameterType param, const vvParam& value)
{
  vvDebugMsg::msg(3, "vvCanvas::setParameter()");

  switch (param)
  {
  case VV_BG_COLOR:
    _bgColor = value;
    break;
  case VV_DOUBLEBUFFERING:
    _doubleBuffering = value;
    break;
  case VV_SUPERSAMPLES:
    _superSamples = value;
    break;
  default:
    break;
  }
}

void vvCanvas::setVolDesc(vvVolDesc* vd)
{
  vvDebugMsg::msg(3, "vvCanvas::setVolDesc()");

  delete _vd;
  _vd = vd;

  if (_vd != NULL)
  {
    createRenderer();
  }
}

vvVolDesc* vvCanvas::getVolDesc() const
{
  vvDebugMsg::msg(3, "vvCanvas::getVolDesc()");

  return _vd;
}

vvRenderer* vvCanvas::getRenderer() const
{
  vvDebugMsg::msg(3, "vvCanvas::getRenderer()");

  return _renderer;
}

void vvCanvas::loadCamera(const QString& filename)
{
  vvDebugMsg::msg(3, "vvCanvas::loadCamera()");

  QByteArray ba = filename.toLatin1();
  _ov.loadCamera(ba.data());
}

void vvCanvas::saveCamera(const QString& filename)
{
  vvDebugMsg::msg(3, "vvCanvas::saveCamera()");

  QByteArray ba = filename.toLatin1();
  _ov.saveCamera(ba.data());
}

void vvCanvas::initializeGL()
{
  vvDebugMsg::msg(1, "vvCanvas::initializeGL()");

  glewInit();
  init();
}

void vvCanvas::paintGL()
{
  vvDebugMsg::msg(3, "vvCanvas::paintGL()");

  if (_renderer == NULL)
  {
    return;
  }

  if (_doubleBuffering)
  {
    glDrawBuffer(GL_BACK);
  }
  else
  {
    glDrawBuffer(GL_FRONT);
  }

  glClearColor(_bgColor[0], _bgColor[1], _bgColor[2], 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  _ov.setModelviewMatrix(vvObjView::CENTER);
  _renderer->renderVolumeGL();
}

void vvCanvas::resizeGL(const int w, const int h)
{
  vvDebugMsg::msg(3, "vvCanvas::resizeGL()");

  glViewport(0, 0, w, h);
  if (h > 0)
  {
    _ov.setAspectRatio(static_cast<float>(w) / static_cast<float>(h));
  }
  updateGL();

  emit resized(QSize(w, h));
}

void vvCanvas::mouseMoveEvent(QMouseEvent* event)
{
  vvDebugMsg::msg(3, "vvCanvas::mouseMoveEvent()");

  switch (_mouseButton)
  {
  case Qt::LeftButton:
  {
    _ov._camera.trackballRotation(width(), height(),
      _lastMousePos.x(), _lastMousePos.y(),
      event->pos().x(), event->pos().y());
    break;
  }
  case Qt::MiddleButton:
  {
    const float pixelInWorld = _ov.getViewportWidth() / static_cast<float>(width());
    const float dx = static_cast<float>(event->pos().x() - _lastMousePos.x());
    const float dy = static_cast<float>(event->pos().y() - _lastMousePos.y());
    vvVector2f pan(pixelInWorld * dx, pixelInWorld * dy);
    _ov._camera.translate(pan[0], -pan[1], 0.0f);
    break;
  }
  case Qt::RightButton:
  {
    const float factor = event->pos().y() - _lastMousePos.y();
    _ov._camera.translate(0.0f, 0.0f, factor);
    break;
  }
  default:
    break;
  }
  _lastMousePos = event->pos();
  updateGL();
}

void vvCanvas::mousePressEvent(QMouseEvent* event)
{
  vvDebugMsg::msg(3, "vvCanvas::mousePressEvent()");

  _mouseButton = event->button();
  _lastMousePos = event->pos();
}

void vvCanvas::mouseReleaseEvent(QMouseEvent*)
{
  vvDebugMsg::msg(3, "vvCanvas::mouseReleaseEvent()");

  _mouseButton = Qt::NoButton;
}

void vvCanvas::init()
{
  vvDebugMsg::msg(3, "vvCanvas::init()");

  // load default volume
  _vd = new vvVolDesc;
  _vd->vox[0] = 32;
  _vd->vox[1] = 32;
  _vd->vox[2] = 32;
  _vd->frames = 0;
  vvFileIO* fio = new vvFileIO;
  fio->loadVolumeData(_vd, vvFileIO::ALL_DATA);
  delete fio;

  // default transfer function
  if (_vd->tf.isEmpty())
  {
    _vd->tf.setDefaultAlpha(0, _vd->real[0], _vd->real[1]);
    _vd->tf.setDefaultColors((_vd->chan == 1) ? 0 : 3, _vd->real[0], _vd->real[1]);
  }

  // init renderer
  if (_vd != NULL)
  {
    _currentRenderer = "viewport";
    _currentOptions["voxeltype"] = "arb";
    createRenderer();
  }

  updateProjection();
}

void vvCanvas::createRenderer()
{
  vvDebugMsg::msg(3, "vvCanvas::createRenderer()");

  vvRenderState state;
  if (_renderer)
  {
    state = *_renderer;
    delete _renderer;
  }

  vvRendererFactory::Options opt(_currentOptions);
  _renderer = vvRendererFactory::create(_vd, state, _currentRenderer.c_str(), opt);
}

void vvCanvas::updateProjection()
{
  vvDebugMsg::msg(3, "vvCanvas::updateProjection()");

  if (_projectionType == vvObjView::PERSPECTIVE)
  {
    _ov.setProjection(vvObjView::PERSPECTIVE, vvObjView::DEF_FOV, vvObjView::DEF_CLIP_NEAR, vvObjView::DEF_CLIP_FAR);
  }
  else if (_projectionType == vvObjView::ORTHO)
  {
    _ov.setProjection(vvObjView::ORTHO, vvObjView::DEF_VIEWPORT_WIDTH, vvObjView::DEF_CLIP_NEAR, vvObjView::DEF_CLIP_FAR);
  }
}

