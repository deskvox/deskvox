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
#include "vvlightinteractor.h"

#include <virvo/vvdebugmsg.h>
#include <virvo/vvfileio.h>
#include <virvo/vvoffscreenbuffer.h>

#include <virvo/private/vvgltools.h>

#include <QSettings>
#include <QTimer>
#include <QVector3D>

#include <iostream>

using vox::vvObjView;

namespace
{
void updateStencilBuffer(int w, int h, vox::StereoMode mode)
{
  static const GLubyte patternlines[] =
  {
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00
  };

  static const GLubyte patternchecker[] =
  {
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
    0xAA, 0xAA, 0xAA, 0xAA,
    0x55, 0x55, 0x55, 0x55,
  };

  glPushAttrib(GL_ALL_ATTRIB_BITS);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_SCISSOR_TEST);
  glDisable(GL_LIGHTING);

  glViewport(0, 0, w, h);
  
  glClearStencil(0);
  glClear(GL_STENCIL_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glEnable(GL_POLYGON_STIPPLE);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  switch (mode)
  {
  case vox::InterlacedLines:
    glPolygonStipple(patternlines);
    break;
  case vox::InterlacedCheckerboard:
    glPolygonStipple(patternchecker);
    break;
  default:
    break;
  }

  glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
  glDepthMask(GL_FALSE);

  glEnable(GL_STENCIL_TEST);
  glStencilMask(0xFFFFFFFF);
  glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);
  glStencilFunc(GL_ALWAYS, 1, 0xFFFFFFFF);

  glRectf(0.0f, 0.0f, 1.0f, 1.0f);

  glPopAttrib(); 
}
}

vvCanvas::vvCanvas(const QGLFormat& format, const QString& filename, QWidget* parent)
  : QGLWidget(format, parent)
  , _vd(NULL)
  , _renderer(NULL)
  , _projectionType(vox::vvObjView::PERSPECTIVE)
  , _doubleBuffering(format.doubleBuffer())
  , _superSamples(format.samples())
  , _stillQuality(1.0f)
  , _movingQuality(1.0f)
  , _spinAnimation(false)
  , _lightVisible(false)
  , _stereoMode(vox::Mono)
  , _swapEyes(false)
  , _mouseButton(Qt::NoButton)
  , _updateStencilBuffer(true)
{
  vvDebugMsg::msg(1, "vvCanvas::vvCanvas()");

  if (filename != "")
  {
    _vd = new vvVolDesc(filename.toStdString().c_str());
  }
  else
  {
    // load default volume
    _vd = new vvVolDesc;
    _vd->vox[0] = 32;
    _vd->vox[1] = 32;
    _vd->vox[2] = 32;
    _vd->frames = 0;
  }

  // init ui
  setMouseTracking(true);
  setFocusPolicy(Qt::StrongFocus);

  // read persistent settings
  QSettings settings;
  QColor qcolor = settings.value("canvas/bgcolor").value<QColor>();
  _bgColor = vvColor(qcolor.redF(), qcolor.greenF(), qcolor.blueF());

  _lighting = settings.value("canvas/lighting").toBool();

  // note: Qt 4.6 introduced QVector3D
  QVector3D qlightpos = settings.value("canvas/lightpos").value<QVector3D>();
  _lightPos = vvVector3(qlightpos.x(), qlightpos.y(), qlightpos.z());

  QVector3D qlightatt = settings.value("canvas/lightattenuation").value<QVector3D>();
  _lightAtt = vvVector3(qlightatt.x(), qlightatt.y(), qlightatt.z());

  _animTimer = new QTimer(this);
  connect(_animTimer, SIGNAL(timeout()), this, SLOT(incTimeStep()));

  _spinTimer = new QTimer(this);
  connect(_spinTimer, SIGNAL(timeout()), this, SLOT(repeatLastRotation()));
}

vvCanvas::~vvCanvas()
{
  vvDebugMsg::msg(1, "vvCanvas::~vvCanvas()");

  delete _renderer;
  delete _vd;
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

  foreach (vvPlugin* plugin, _plugins)
  {
    plugin->setVolDesc(_vd);
  }

  std::string str;
  _vd->makeInfoString(&str);
  emit statusMessage(str);
  emit newVolDesc(_vd);
}

void vvCanvas::setPlugins(const QList<vvPlugin*>& plugins)
{
  vvDebugMsg::msg(3, "vvCanvas::setPlugins()");

  _plugins = plugins;
}

void vvCanvas::setInteractors(const QList<vvInteractor*>& interactors)
{
  _interactors = interactors;
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

const QList<vvInteractor*>& vvCanvas::getInteractors() const
{
  return _interactors;
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

  glPushAttrib(GL_ALL_ATTRIB_BITS);

  if (_renderer == NULL)
  {
    return;
  }

  if (_lighting)
  {
    glEnable(GL_LIGHTING);
    float lv[4] = { _lightPos[0], _lightPos[1], _lightPos[2], 1.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, lv);
    glLightfv(GL_LIGHT0, GL_CONSTANT_ATTENUATION, &_lightAtt[0]);
    glLightfv(GL_LIGHT0, GL_LINEAR_ATTENUATION, &_lightAtt[1]);
    glLightfv(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, &_lightAtt[2]);
  }

  glClearColor(_bgColor[0], _bgColor[1], _bgColor[2], 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (_stereoMode == vox::Mono)
  {
    if (_doubleBuffering)
    {
      glDrawBuffer(GL_BACK);
    }
    else
    {
      glDrawBuffer(GL_FRONT);
    }

    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_MODELVIEW);
    _ov.setModelviewMatrix(vvObjView::CENTER);

    render();
  }
  else if (_stereoMode == vox::InterlacedLines || _stereoMode == vox::InterlacedCheckerboard)
  {
    if (_updateStencilBuffer)
    {
      glDrawBuffer(GL_BACK);
      ::updateStencilBuffer(width(), height(), _stereoMode);
      updateProjection();

      _updateStencilBuffer = false;
    }

    std::vector<vvObjView::EyeType> eyes;
    eyes.push_back(vvObjView::LEFT_EYE);
    eyes.push_back(vvObjView::RIGHT_EYE);

    for (std::vector<vvObjView::EyeType>::const_iterator it = eyes.begin();
         it != eyes.end(); ++it)
    {
      glEnable(GL_DEPTH_TEST);

      glEnable(GL_STENCIL_TEST);
      GLint ref = _swapEyes ? *it == vvObjView::LEFT_EYE : *it == vvObjView::RIGHT_EYE;
      glStencilFunc(GL_EQUAL, ref, 0xFF);

      glMatrixMode(GL_MODELVIEW);
      _ov.setModelviewMatrix(*it);

      render();
    }
  }
  else if (_stereoMode == vox::SideBySide)
  {
    std::vector<vvObjView::EyeType> eyes;
    eyes.push_back(vvObjView::LEFT_EYE);
    eyes.push_back(vvObjView::RIGHT_EYE);

    _ov.setAspectRatio((float)width() / 2.0f / (float)height());
    for (std::vector<vvObjView::EyeType>::const_iterator it = eyes.begin();
         it != eyes.end(); ++it)
    {
      glMatrixMode(GL_MODELVIEW);
      _ov.setModelviewMatrix(*it);

      if ((*it == vvObjView::LEFT_EYE && !_swapEyes) || (*it == vvObjView::RIGHT_EYE && _swapEyes))
      {
        glViewport(0, 0, width() / 2, height());
      }
      else
      {
        glViewport(width() / 2, 0, width() / 2, height());
      }

      render();
    }
    glViewport(0, 0, width(), height());
    _ov.setAspectRatio((float)width() / (float)height());
  }
  else if (_stereoMode == vox::RedCyan)
  {
    GLboolean cm[4];
    glGetBooleanv(GL_COLOR_WRITEMASK, cm);

    std::vector<vvObjView::EyeType> eyes;
    eyes.push_back(vvObjView::LEFT_EYE);
    eyes.push_back(vvObjView::RIGHT_EYE);

    for (std::vector<vvObjView::EyeType>::const_iterator it = eyes.begin();
         it != eyes.end(); ++it)
    {
      glMatrixMode(GL_MODELVIEW);
      _ov.setModelviewMatrix(*it);

      if ((*it == vvObjView::LEFT_EYE && !_swapEyes) || (*it == vvObjView::RIGHT_EYE && _swapEyes))
      {
        glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
      }
      else
      {
        glColorMask(GL_FALSE, GL_TRUE, GL_TRUE, GL_FALSE);
      }

      render();
    }

    glColorMask(cm[0], cm[1], cm[2], cm[3]);
  }

  glPopAttrib();
}

void vvCanvas::resizeGL(int w, int h)
{
  vvDebugMsg::msg(3, "vvCanvas::resizeGL()");

  glViewport(0, 0, w, h);
  if (h > 0)
  {
    _ov.setAspectRatio(static_cast<float>(w) / static_cast<float>(h));
  }
  _updateStencilBuffer = true;
  updateGL();

  emit resized(QSize(w, h));
}

void vvCanvas::mouseMoveEvent(QMouseEvent* event)
{
  vvDebugMsg::msg(3, "vvCanvas::mouseMoveEvent()");

  // interactors
  foreach (vvInteractor* interactor, _interactors)
  {
    // first interactor with focus gets event
    if (interactor->enabled() && interactor->hasFocus())
    {
      interactor->mouseMoveEvent(event);
      // no more mouse processing
      return;
    }
  }

  // default mouse move event
  switch (_mouseButton)
  {
  case Qt::LeftButton:
  {
    _lastRotation = _ov._camera.trackballRotation(width(), height(),
      _lastMousePos.x(), _lastMousePos.y(),
      event->pos().x(), event->pos().y());
    if (_spinAnimation)
    {
      repeatLastRotation();
    }
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

  // interactors
  foreach (vvInteractor* interactor, _interactors)
  {
    // first interactor with focus gets event
    if (interactor->enabled() && interactor->hasFocus())
    {
      interactor->mousePressEvent(event);
      // no more mouse processing
      return;
    }
  }

  // default mouse press event
  _stillQuality = _renderer->getParameter(vvRenderer::VV_QUALITY);
  _renderer->setParameter(vvRenderer::VV_QUALITY, _movingQuality);
  _mouseButton = event->button();
  _lastMousePos = event->pos();
  _lastRotation.identity();
  if (_spinAnimation)
  {
    _spinTimer->stop();
  }
}

void vvCanvas::mouseReleaseEvent(QMouseEvent* event)
{
  vvDebugMsg::msg(3, "vvCanvas::mouseReleaseEvent()");

  // interactors
  foreach (vvInteractor* interactor, _interactors)
  {
    // first interactor with focus gets event
    if (interactor->enabled() && interactor->hasFocus())
    {
      interactor->mouseReleaseEvent(event);
      // no more mouse processing
      return;
    }
  }

  // default mouse release event
  _mouseButton = Qt::NoButton;
  _renderer->setParameter(vvRenderer::VV_QUALITY, _stillQuality);
  updateGL();
}

void vvCanvas::init()
{
  vvDebugMsg::msg(3, "vvCanvas::init()");
  
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
    _currentRenderer = "planar";
    _currentOptions["voxeltype"] = "arb";
    createRenderer();
  }

  updateProjection();

  foreach (vvPlugin* plugin, _plugins)
  {
    plugin->setVolDesc(_vd);
  }

  emit newVolDesc(_vd);
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

  const float DEFAULT_OBJ_SIZE = 0.6f;
  _vd->resizeEdgeMax(_ov.getViewportWidth() * DEFAULT_OBJ_SIZE);

  _renderer = vvRendererFactory::create(_vd, state, _currentRenderer.c_str(), _currentOptions);

  // set boundary color to inverse of background
  vvColor invColor;
  if (_bgColor[0] + _bgColor[1] + _bgColor[2] > 1.5f)
  {
    invColor = vvColor(0.0f, 0.0f, 0.0f);
  }
  else
  {
    invColor = vvColor(1.0f, 1.0f, 1.0f);
  }
  _renderer->setParameter(vvRenderState::VV_BOUND_COLOR, invColor);
  _renderer->setParameter(vvRenderState::VV_CLIP_COLOR, invColor);
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

void vvCanvas::setCurrentFrame(const int frame)
{
  vvDebugMsg::msg(3, "vvCanvas::setCurrentFrame()");

  _renderer->setCurrentFrame(frame);
  emit currentFrame(frame);

  // inform plugins of new frame
  foreach (vvPlugin* plugin, _plugins)
  {
    plugin->timestep();
  }

  updateGL();
}

void vvCanvas::render()
{
  foreach (vvPlugin* plugin, _plugins)
  {
    if (plugin->isActive())
    {
      plugin->prerender();
    }
  }

  _renderer->renderVolumeGL();

  foreach (vvPlugin* plugin, _plugins)
  {
    if (plugin->isActive())
    {
      plugin->postrender();
    }
  }

  foreach (vvInteractor* interactor, _interactors)
  {
    if (interactor->enabled() && interactor->visible())
    {
      interactor->render();
    }
  }
}

void vvCanvas::setRenderer(const std::string& name, const vvRendererFactory::Options& options)
{
  vvDebugMsg::msg(3, "vvCanvas::setRenderer()");

  _currentRenderer = name;
  _currentOptions = options;
  createRenderer();
  updateGL();
}

void vvCanvas::setParameter(vvParameters::ParameterType param, const vvParam& value)
{
  vvDebugMsg::msg(3, "vvCanvas::setParameter()");

  switch (param)
  {
  case vvParameters::VV_BG_COLOR:
    _bgColor = value;
    break;
  case vvParameters::VV_DOUBLEBUFFERING:
    _doubleBuffering = value;
    break;
  case vvParameters::VV_LIGHTING:
    {
      _lighting = value;
      foreach (vvInteractor* interactor, _interactors)
      {
        vvLightInteractor* li = dynamic_cast<vvLightInteractor*>(interactor);
        if (li != NULL)
        {
          li->setLightingEnabled(_lighting);
        }
      }
    } 
    break;
  case vvParameters::VV_MOVING_QUALITY:
    _movingQuality = value;
    break;
  case vvParameters::VV_SUPERSAMPLES:
    _superSamples = value;
    break;
  case vvParameters::VV_PROJECTIONTYPE:
    _projectionType = static_cast<vvObjView::ProjectionType>(value.asInt());
    updateProjection();
    break;
  case vvParameters::VV_SPIN_ANIMATION:
    _spinAnimation = value;
    break;
  case vvParameters::VV_STEREO_MODE:
    _stereoMode = static_cast<vox::StereoMode>(value.asInt());
    if (_stereoMode == vox::InterlacedLines || _stereoMode == vox::InterlacedCheckerboard)
    {
      _updateStencilBuffer = true;
    }
    break;
  case vvParameters::VV_SWAP_EYES:
    _swapEyes = value;
    break;
  default:
    break;
  }
  updateGL();
}

void vvCanvas::setParameter(vvRenderer::ParameterType param, const vvParam& value)
{
  vvDebugMsg::msg(3, "vvCanvas::setParameter()");

  if (_renderer != NULL)
  {
    _renderer->setParameter(param, value);
    updateGL();
  }
}

vvParam vvCanvas::getParameter(vvParameters::ParameterType param) const
{
  switch (param)
  {
  case vvParameters::VV_BG_COLOR:
    return _bgColor;
  case vvParameters::VV_DOUBLEBUFFERING:
    return _doubleBuffering;
  case vvParameters::VV_LIGHTING:
    return _lighting;
  case vvParameters::VV_SUPERSAMPLES:
    return _superSamples;
  case vvParameters::VV_PROJECTIONTYPE:
    return static_cast<int>(_projectionType);
  case vvParameters::VV_SPIN_ANIMATION:
    return _spinAnimation;
  case vvParameters::VV_STEREO_MODE:
    return _stereoMode;
  case vvParameters::VV_SWAP_EYES:
    return _swapEyes;
  default:
    return vvParam();
  }
}

vvParam vvCanvas::getParameter(vvRenderer::ParameterType param) const
{
  if (_renderer != NULL)
  {
    return _renderer->getParameter(param);
  }
  return vvParam();
}

void vvCanvas::startAnimation(const double fps)
{
  vvDebugMsg::msg(3, "vvCanvas::startAnimation()");

  _vd->dt = 1.0f / static_cast<float>(fps);
  const float delay = std::abs(_vd->dt * 1000.0f);
  _animTimer->start(static_cast<int>(delay));
}

void vvCanvas::stopAnimation()
{
  vvDebugMsg::msg(3, "vvCanvas::stopAnimation()");

  _animTimer->stop();
}

void vvCanvas::setTimeStep(const int step)
{
  vvDebugMsg::msg(3, "vvCanvas::setTimeStep()");

  int f = step;
  while (f < 0)
  {
    f += _vd->frames; 
  }

  while (f >= _vd->frames)
  {
    f -= _vd->frames;
  }

  setCurrentFrame(f);
}

void vvCanvas::incTimeStep()
{
  vvDebugMsg::msg(3, "vvCanvas::incTimeStep()");

  int f = _renderer->getCurrentFrame();
  f = (f >= _vd->frames - 1) ? 0 : f + 1;
  setCurrentFrame(f);
}

void vvCanvas::decTimeStep()
{
  vvDebugMsg::msg(3, "vvCanvas::decTimeStep()");

  int f = _renderer->getCurrentFrame();
  f = (f <= 0) ? _vd->frames - 1 : f - 1;
  setCurrentFrame(f);
}

void vvCanvas::firstTimeStep()
{
  vvDebugMsg::msg(3, "vvCanvas::firstTimeStep()");

  setCurrentFrame(0);
}

void vvCanvas::lastTimeStep()
{
  vvDebugMsg::msg(3, "vvCanvas::lastTimeStep()");

  setCurrentFrame(_vd->frames - 1);
}

void vvCanvas::enableLighting(bool enabled)
{
  setParameter(vvRenderState::VV_LIGHTING, enabled);
  setParameter(vvParameters::VV_LIGHTING, enabled);
  updateGL();
}

void vvCanvas::showLightSource(bool show)
{
  _lightVisible = show;

  // canvas lazily uses the light source interactor to visualize the light source
  vvLightInteractor* li = NULL;
  foreach (vvInteractor* interactor, _interactors)
  {
    if (dynamic_cast<vvLightInteractor*>(interactor) != NULL)
    {
      li = static_cast<vvLightInteractor*>(interactor);
    }
  }

  if (li == NULL)
  {
    li = new vvLightInteractor;
    _interactors.append(li);
  }

  li->setEnabled(true);
  li->setLightingEnabled(getParameter(vvParameters::VV_LIGHTING));
  li->setPos(_lightPos);

  if (!li->hasFocus())
  {
    li->setVisible(_lightVisible);
  }

  updateGL();
}

void vvCanvas::editLightPosition(bool edit)
{
  if (edit)
  {
    vvLightInteractor* li = NULL;
    foreach (vvInteractor* interactor, _interactors)
    {
      if (dynamic_cast<vvLightInteractor*>(interactor) != NULL)
      {
        li = static_cast<vvLightInteractor*>(interactor);
      }
    }

    if (li == NULL)
    {
      li = new vvLightInteractor;
      _interactors.append(li);
    }

    li->setFocus();
    li->setVisible(true);
    li->setLightingEnabled(getParameter(vvParameters::VV_LIGHTING));
    li->setPos(_lightPos);
    connect(li, SIGNAL(lightPos(const vvVector3&)), this, SLOT(setLightPos(const vvVector3&)));
  }
  else
  {
    foreach (vvInteractor* interactor, _interactors)
    {
      if (dynamic_cast<vvLightInteractor*>(interactor) != NULL)
      {
        interactor->clearFocus();
        interactor->setVisible(_lightVisible);
      }
    }
  }
  updateGL();
}

void vvCanvas::setLightAttenuation(const vvVector3& att)
{
  _lightAtt = att;

  QVector3D qatt(att[0], att[1], att[2]);
  QSettings settings;
  settings.setValue("canvas/lightattenuation", qatt);

  updateGL();
}

void vvCanvas::repeatLastRotation()
{
  vvDebugMsg::msg(3, "vvCanvas::repeatLastRotation()");

  _ov._camera.multiplyRight(_lastRotation);
  updateGL();

  const float spindelay = 0.05f;
  const float delay = std::abs(spindelay * 1000.0f);
  _spinTimer->start(static_cast<int>(delay));
}

void vvCanvas::setLightPos(const vvVector3& pos)
{
  _lightPos = pos;

  QVector3D qpos(pos[0], pos[1], pos[2]);
  QSettings settings;
  settings.setValue("canvas/lightpos", qpos);

  updateGL();
}

