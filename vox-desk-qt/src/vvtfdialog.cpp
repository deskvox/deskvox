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
#include "vvtfdialog.h"

#include "tfeditor/colorbox.h"
#include "tfeditor/gaussianbox.h"
#include "tfeditor/graphicsscene.h"
#include "tfeditor/pyramidbox.h"
#include "tfeditor/skipbox.h"

#include "ui_vvtfdialog.h"

#include <virvo/vvdebugmsg.h>
#include <virvo/vvvoldesc.h>

#include <QFileDialog>
#include <QGraphicsPixmapItem>
#include <QGraphicsRectItem>

#include <map>

namespace
{

float PIN_WIDTH = 2.0f;
float SELECTED_WIDTH;
size_t COLORBAR_HEIGHT = 30;
size_t TF_WIDTH = 768;
size_t TF_HEIGHT = 256;

/** Convert canvas x coordinates to data values.
  @param canvas canvas x coordinate [0..1]
  @return data value
*/
float norm2data(const vvVector2f& zoomrange, float canvas)
{
  return canvas * (zoomrange[1] - zoomrange[0]) + zoomrange[0];
}

/** Convert data value to x coordinate in TF canvas.
  @param data data value
  @return canvas x coordinate [0..1]
*/
float data2norm(const vvVector2f& zoomrange, float data)
{
  return (data - zoomrange[0]) / (zoomrange[1] - zoomrange[0]);
}

/** Convert horizontal differences on the canvas to data differences.
*/
float normd2datad(const vvVector2f& zoomrange, float canvas)
{
  return canvas * (zoomrange[1] - zoomrange[0]);
}

/** Convert differences in data to the canvas.
*/
float datad2normd(const vvVector2f& zoomrange, float data)
{
  return data / (zoomrange[1] - zoomrange[0]);
}
}

typedef QGraphicsRectItem Pin;

struct vvTFDialog::QData
{
  QData()
    : colorscene(new MouseGraphicsScene)
    , colortex(new QGraphicsPixmapItem)
    , alphascene(new MouseGraphicsScene)
    , alphatex(new QGraphicsPixmapItem)
    , selected(NULL)
    , moving(NULL)
  {
    colorscene->addItem(colortex);
    alphascene->addItem(alphatex);
  }

  ~QData()
  {
    delete colortex;
    delete alphatex;
    delete colorscene;
    delete alphascene;
  }

  MouseGraphicsScene* colorscene;
  QGraphicsPixmapItem* colortex;
  std::vector<Pin*> colorpins;

  MouseGraphicsScene* alphascene;
  QGraphicsPixmapItem* alphatex;
  std::vector<Pin*> alphapins;

  Pin* selected;
  Pin* moving;

  std::map<Pin*, vvTFWidget*> pin2widgetmap;
  std::map<vvTFWidget*, Pin*> widget2pinmap;
};

vvTFDialog::vvTFDialog(vvCanvas* canvas, QWidget* parent)
  : QDialog(parent)
  , ui(new Ui_TFDialog)
  , _qdata(new QData)
  , _canvas(canvas)
  , _zoomRange(vvVector2f(0.0f, 1.0f))
{
  vvDebugMsg::msg(1, "vvTFDialog::vvTFDialog()");

  ui->setupUi(this);

  ui->color1DView->setScene(_qdata->colorscene);
  ui->alpha1DView->setScene(_qdata->alphascene);

  connect(ui->colorButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(ui->pyramidButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(ui->gaussianButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(ui->customButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(ui->skipRangeButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(ui->deleteButton, SIGNAL(clicked()), this, SLOT(onDeleteClicked()));
  connect(ui->undoButton, SIGNAL(clicked()), this, SLOT(onUndoClicked()));
  connect(ui->presetColorsBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onPresetColorsChanged(int)));
  connect(ui->presetAlphaBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onPresetAlphaChanged(int)));
  connect(ui->applyButton, SIGNAL(clicked()), this, SLOT(onApplyClicked()));
  connect(_canvas, SIGNAL(newVolDesc(vvVolDesc*)), this, SLOT(onNewVolDesc(vvVolDesc*)));
  connect(ui->saveButton, SIGNAL(clicked()), this, SLOT(saveTF()));
  connect(ui->loadButton, SIGNAL(clicked()), this, SLOT(loadTF()));
  connect(_qdata->colorscene, SIGNAL(mouseMoved(QPointF, Qt::MouseButton)), this, SLOT(onTFMouseMove(QPointF, Qt::MouseButton)));
  connect(_qdata->colorscene, SIGNAL(mousePressed(QPointF, Qt::MouseButton)), this, SLOT(onTFMousePress(QPointF, Qt::MouseButton)));
  connect(_qdata->colorscene, SIGNAL(mouseReleased(QPointF, Qt::MouseButton)), this, SLOT(onTFMouseRelease(QPointF, Qt::MouseButton)));
  connect(_qdata->alphascene, SIGNAL(mouseMoved(QPointF, Qt::MouseButton)), this, SLOT(onTFMouseMove(QPointF, Qt::MouseButton)));
  connect(_qdata->alphascene, SIGNAL(mousePressed(QPointF, Qt::MouseButton)), this, SLOT(onTFMousePress(QPointF, Qt::MouseButton)));
  connect(_qdata->alphascene, SIGNAL(mouseReleased(QPointF, Qt::MouseButton)), this, SLOT(onTFMouseRelease(QPointF, Qt::MouseButton)));
}

vvTFDialog::~vvTFDialog()
{
  vvDebugMsg::msg(1, "vvTFDialog::~vvTFDialog()");

  delete _qdata;
}

void vvTFDialog::drawTF()
{
  drawColorTexture();
  drawAlphaTexture();
  _qdata->colorscene->invalidate();
  _qdata->alphascene->invalidate();
}

void vvTFDialog::drawColorTexture()
{
  vvDebugMsg::msg(3, "vvTFDialog::drawColorTexture()");

  int w = 768;
  int h = 3;
  std::vector<uchar> colorBar;
  colorBar.resize(w * h * 4);
  makeColorBar(&colorBar, w);
  QImage img(&colorBar[0], w, 3, QImage::Format_ARGB32);
  img = img.scaled(QSize(w, h * COLORBAR_HEIGHT / 3));
  if (!img.isNull())
  {
    QPixmap colorpm = QPixmap::fromImage(img);
    _qdata->colortex->setPixmap(colorpm);
  }
}

void vvTFDialog::drawAlphaTexture()
{
  int w = ui->alpha1DView->width();
  int h = ui->alpha1DView->height();
  std::vector<uchar> alphaTex;
  alphaTex.resize(w * h * 4);
  makeAlphaTexture(&alphaTex, w, h);
  QImage img(&alphaTex[0], w, h, QImage::Format_ARGB32);
  if (!img.isNull())
  {
    QPixmap alphapm = QPixmap::fromImage(img);
    _qdata->alphatex->setPixmap(alphapm);
  }
}

void vvTFDialog::clearPins()
{
  for (std::vector<Pin*>::const_iterator it = _qdata->colorpins.begin();
       it != _qdata->colorpins.end(); ++it)
  {
    _qdata->colorscene->removeItem(*it);
    delete *it;
  }
  _qdata->colorpins.clear();

  for (std::vector<Pin*>::const_iterator it = _qdata->alphapins.begin();
       it != _qdata->alphapins.end(); ++it)
  {
    _qdata->alphascene->removeItem(*it);
    delete *it;
  }
  _qdata->alphapins.clear();

  _qdata->pin2widgetmap.clear();
  _qdata->widget2pinmap.clear();
  _qdata->selected = NULL;
  _qdata->moving = NULL;
}

void vvTFDialog::createPins()
{
  for (std::vector<vvTFWidget*>::const_iterator it = _canvas->getVolDesc()->tf._widgets.begin();
       it != _canvas->getVolDesc()->tf._widgets.end(); ++it)
  {
    createPin(*it);
  }
}

void vvTFDialog::createPin(vvTFWidget* w)
{
  bool selected = false; // TODO
//  bool mouseover = false;
  float rectw = selected ? SELECTED_WIDTH : PIN_WIDTH;
  float xpos = data2norm(_zoomRange, w->_pos[0]) * static_cast<float>(TF_WIDTH);

  Pin* pin = NULL;
  if (dynamic_cast<vvTFColor*>(w) != NULL) // draw color pin
  {
    pin = _qdata->colorscene->addRect(-rectw * 0.5f, 0, rectw, COLORBAR_HEIGHT - 1, QPen(), QBrush(Qt::SolidPattern));
    pin->setPos(xpos, 0);
    _qdata->colorpins.push_back(pin);
  }
  else if ((dynamic_cast<vvTFPyramid*>(w) != NULL) ||
           (dynamic_cast<vvTFBell*>(w) != NULL) ||
           (dynamic_cast<vvTFSkip*>(w) != NULL) ||
           (dynamic_cast<vvTFCustom*>(w) != NULL)) // draw alpha pin
  {
    pin = _qdata->alphascene->addRect(-rectw * 0.5f, 0, rectw, TF_HEIGHT - COLORBAR_HEIGHT, QPen(), QBrush(Qt::SolidPattern));
    pin->setPos(xpos, 0);
    _qdata->alphapins.push_back(pin);

  }
  _qdata->pin2widgetmap[pin] = w;
  _qdata->widget2pinmap[w] = pin;
  assert(_qdata->pin2widgetmap.size() == _qdata->widget2pinmap.size());
}

void vvTFDialog::makeColorBar(std::vector<uchar>* colorBar, int width) const
{
  if (/* HDR */ false)
  {

  }
  else // standard iso-range TF mode
  {
    // BGRA to fit QImage's little endian ARGB32 format
    _canvas->getVolDesc()->tf.makeColorBar(width, &(*colorBar)[0], _zoomRange[0], _zoomRange[1], false, vvToolshed::VV_BGRA);
  }
}

void vvTFDialog::makeAlphaTexture(std::vector<uchar>* alphaTex, int width, int height) const
{
  if (/* HDR */ false)
  {

  }
  else // standard iso-range TF mode
  {
    // BGRA to fit QImage's little endian ARGB32 format
    _canvas->getVolDesc()->tf.makeAlphaTexture(width, height, &(*alphaTex)[0], _zoomRange[0], _zoomRange[1], vvToolshed::VV_BGRA);
  }
}

void vvTFDialog::onUndoClicked()
{
  emit undo();
  emit newTransferFunction();
  clearPins();
  createPins();
  drawTF();
}

void vvTFDialog::onNewWidget()
{
  vvTFWidget* widget = NULL;

  if (QObject::sender() == ui->colorButton)
  {
    widget = new vvTFColor(vvColor(), norm2data(_zoomRange, 0.5f));
  }
  else if (QObject::sender() == ui->pyramidButton)
  {
    widget = new vvTFPyramid(vvColor(), false, 1.0f, norm2data(_zoomRange, 0.5f), normd2datad(_zoomRange, 0.4f), normd2datad(_zoomRange, 0.2f));
  }
  else if (QObject::sender() == ui->gaussianButton)
  {
    widget = new vvTFBell(vvColor(), false, 1.0f, norm2data(_zoomRange, 0.5f), normd2datad(_zoomRange, 0.2f));
  }
  else if (QObject::sender() == ui->customButton)
  {
    widget = new vvTFCustom(norm2data(_zoomRange, 0.5f), norm2data(_zoomRange, 0.5f));
  }
  else if (QObject::sender() == ui->skipRangeButton)
  {
    widget = new vvTFSkip(norm2data(_zoomRange, 0.5f), normd2datad(_zoomRange, 0.2f));
  }

  emit newWidget(widget);
  createPin(widget);
  _qdata->selected = _qdata->widget2pinmap[widget];
  updateSettingsBox();
  drawTF();
}

void vvTFDialog::onDeleteClicked()
{
  if(_canvas->getVolDesc()->tf._widgets.size() == 0 || _qdata->selected == NULL)
  {
    return;
  }
  _canvas->getVolDesc()->tf.putUndoBuffer();
  _canvas->getVolDesc()->tf._widgets.erase(
    std::find(_canvas->getVolDesc()->tf._widgets.begin(), _canvas->getVolDesc()->tf._widgets.end(),
      _qdata->pin2widgetmap[_qdata->selected]));
  _qdata->selected = NULL;
  emitTransFunc();
  clearPins();
  createPins();
  updateSettingsBox();
  drawTF();
}

void vvTFDialog::onPresetColorsChanged(int index)
{
  vvDebugMsg::msg(3, "vvTFDialog::onPresetColorsChanged()");

  _canvas->getVolDesc()->tf.setDefaultColors(index, _zoomRange[0], _zoomRange[1]);
  emitTransFunc();
  clearPins();
  createPins();
  drawTF();
}

void vvTFDialog::onPresetAlphaChanged(int index)
{
  vvDebugMsg::msg(3, "vvTFDialog::onPresetAlphaChanged()");

  _canvas->getVolDesc()->tf.setDefaultAlpha(index, _zoomRange[0], _zoomRange[1]);
  emitTransFunc();
  clearPins();
  createPins();
  drawTF();
}

void vvTFDialog::onApplyClicked()
{
  drawTF();
}

void vvTFDialog::onNewVolDesc(vvVolDesc*)
{
  clearPins();
  createPins();
  drawTF();
}

void vvTFDialog::saveTF()
{
  QString caption = tr("Save Transfer Function");
  QString dir;
  QString filter = tr("Transfer function files (*.vtf);;"
    "All Files (*.*)");
  QString filename = QFileDialog::getSaveFileName(this, caption, dir, filter);
  if (!filename.isEmpty())
  {
    std::string strfn = filename.toStdString();
    _canvas->getVolDesc()->tf.save(strfn.c_str());
  }
}

void vvTFDialog::loadTF()
{
  QString caption = tr("Load Transfer Function");
  QString dir;
  QString filter = tr("Transfer function files (*.vtf);;"
    "All Files (*.*)");
  QString filename = QFileDialog::getOpenFileName(this, caption, dir, filter);
  if (!filename.isEmpty())
  {
    std::string strfn = filename.toStdString();
    _canvas->getVolDesc()->tf.load(strfn.c_str());
  }
  emitTransFunc();
  clearPins();
  createPins();
  drawTF();
}

void vvTFDialog::onColor(const QColor& color)
{
  assert(_qdata->selected != NULL);
  vvTFWidget* wid = _qdata->pin2widgetmap[_qdata->selected];
  if (vvTFColor* c = dynamic_cast<vvTFColor*>(wid))
  {
    c->setColor(vvColor(color.redF(), color.greenF(), color.blueF()));
  }
  else if (vvTFBell* b = dynamic_cast<vvTFBell*>(wid))
  {
    b->setColor(vvColor(color.redF(), color.greenF(), color.blueF()));
  }
  else if (vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(wid))
  {
    p->setColor(vvColor(color.redF(), color.greenF(), color.blueF()));
  }
  else
  {
    assert(false);
  }
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onHasOwnColor(bool hascolor)
{
  assert(_qdata->selected != NULL);
  vvTFWidget* wid = _qdata->pin2widgetmap[_qdata->selected];
  if (vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(wid))
  {
    p->setOwnColor(hascolor);
  }
  else if (vvTFBell* b = dynamic_cast<vvTFBell*>(wid))
  {
    b->setOwnColor(hascolor);
  }
  else
  {
    assert(false);
  }
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onOpacity(float opacity)
{
  assert(_qdata->selected != NULL);
  vvTFWidget* wid = _qdata->pin2widgetmap[_qdata->selected];
  if (vvTFBell* b = dynamic_cast<vvTFBell*>(wid))
  {
    b->setOpacity(opacity);
  }
  else if (vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(wid))
  {
    p->setOpacity(opacity);
  }
  else
  {
    assert(false);
  }
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onSize(const vvVector3& size)
{
  assert(_qdata->selected != NULL);
  vvTFWidget* wid = _qdata->pin2widgetmap[_qdata->selected];
  if (vvTFBell* b = dynamic_cast<vvTFBell*>(wid))
  {
    b->setSize(size);
  }
  else if (vvTFSkip* s = dynamic_cast<vvTFSkip*>(wid))
  {
    s->setSize(size);
  }
  else
  {
    assert(false);
  }
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onTop(const vvVector3& top)
{
  assert(_qdata->selected != NULL);
  vvTFWidget* wid = _qdata->pin2widgetmap[_qdata->selected];
  vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(wid);
  assert(p != NULL);
  p->setTop(top);
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onBottom(const vvVector3& bottom)
{
  assert(_qdata->selected != NULL);
  vvTFWidget* wid = _qdata->pin2widgetmap[_qdata->selected];
  vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(wid);
  assert(p != NULL);
  p->setBottom(bottom);
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onTFMouseMove(QPointF pos, Qt::MouseButton /* button */)
{
  std::vector<Pin*>& pins = QObject::sender() == _qdata->colorscene ? _qdata->colorpins : _qdata->alphapins;

  if (_qdata->moving != NULL)
  {
    std::vector<Pin*>::iterator it = std::find(pins.begin(), pins.end(), _qdata->moving);
    if (it == pins.end())
    {
      return;
    }

    float x = static_cast<float>(pos.x());
    x = ts_clamp(x, 0.0f, static_cast<float>(TF_WIDTH));
    QPointF pinpos = (*it)->pos();
    (*it)->setPos(x, pinpos.y());
    vvTFWidget* w = _qdata->pin2widgetmap[*it];
    if (w != NULL)
    {
      x /= static_cast<float>(TF_WIDTH);
      x = norm2data(_zoomRange, x);
      vvVector3 oldpos = w->pos();
      w->setPos(x, oldpos[1], oldpos[2]);
    }
    emitTransFunc();
    drawTF();
  }
  else
  {
    for (std::vector<Pin*>::const_iterator it = pins.begin();
         it != pins.end(); ++it)
    {
      float minx = (*it)->pos().x() - PIN_WIDTH * 0.5f - 1;
      float maxx = (*it)->pos().x() + PIN_WIDTH * 0.5f + 1;
      if (pos.x() >= minx && pos.x() <= maxx)
      {
        // TODO: add highlight
      }
      else
      {
        // TODO: remove highlight
      }
    }
  }
}

void vvTFDialog::onTFMousePress(QPointF pos, Qt::MouseButton button)
{
  const std::vector<Pin*>& pins = QObject::sender() == _qdata->colorscene ? _qdata->colorpins : _qdata->alphapins;

  if (button == Qt::LeftButton)
  {
    _qdata->selected = NULL;
    for (std::vector<Pin*>::const_iterator it = pins.begin();
         it != pins.end(); ++it)
    {
      float minx = (*it)->pos().x() - PIN_WIDTH * 0.5f - 1;
      float maxx = (*it)->pos().x() + PIN_WIDTH * 0.5f + 1;
      if (pos.x() >= minx && pos.x() <= maxx)
      {
        _qdata->moving = *it;
        if (_qdata->selected != *it)
        {
          _qdata->selected = *it;
        }
        break;
      }
    }
    updateSettingsBox();
  }
}

void vvTFDialog::onTFMouseRelease(QPointF /* pos */, Qt::MouseButton /* button */)
{
  _qdata->moving = NULL;
}

void vvTFDialog::emitTransFunc()
{
  emit newTransferFunction();
}

void vvTFDialog::updateSettingsBox()
{
  // clear settings layout
  if (QLayoutItem* item = ui->settingsLayout->takeAt(0))
  {
    QWidget* widget = item->widget();
    delete widget;
  }

  Pin* selected = _qdata->selected;
  if (selected == NULL)
  {
    return;
  }

  // new settings box
  vvTFWidget* w = _qdata->pin2widgetmap[selected];
  if (vvTFColor* c = dynamic_cast<vvTFColor*>(w))
  {
    tf::ColorBox* cb = new tf::ColorBox(this);
    ui->settingsLayout->addWidget(cb);
    cb->setColor(c->color());
    connect(cb, SIGNAL(color(const QColor&)), this, SLOT(onColor(const QColor&)));
  }
  else if (vvTFBell* b = dynamic_cast<vvTFBell*>(w))
  {
    tf::GaussianBox* gb = new tf::GaussianBox(this);
    ui->settingsLayout->addWidget(gb);
    gb->setHasColor(b->hasOwnColor());
    gb->setColor(b->color());
    gb->setSize(b->size());
    gb->setOpacity(b->opacity());
    connect(gb, SIGNAL(color(const QColor&)), this, SLOT(onColor(const QColor&)));
    connect(gb, SIGNAL(hasColor(bool)), this, SLOT(onHasOwnColor(bool)));
    connect(gb, SIGNAL(size(const vvVector3&)), this, SLOT(onSize(const vvVector3&)));
    connect(gb, SIGNAL(opacity(float)), this, SLOT(onOpacity(float)));
  }
  else if (vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(w))
  {
    tf::PyramidBox* pb = new tf::PyramidBox(this);
    ui->settingsLayout->addWidget(pb);
    pb->setHasColor(p->hasOwnColor());
    pb->setColor(p->color());
    pb->setTop(p->top());
    pb->setBottom(p->bottom());
    pb->setOpacity(p->opacity());
    connect(pb, SIGNAL(color(const QColor&)), this, SLOT(onColor(const QColor&)));
    connect(pb, SIGNAL(hasColor(bool)), this, SLOT(onHasOwnColor(bool)));
    connect(pb, SIGNAL(top(const vvVector3&)), this, SLOT(onTop(const vvVector3&)));
    connect(pb, SIGNAL(bottom(const vvVector3&)), this, SLOT(onBottom(const vvVector3&)));
    connect(pb, SIGNAL(opacity(float)), this, SLOT(onOpacity(float)));
  }
  else if (vvTFSkip* s = dynamic_cast<vvTFSkip*>(w))
  {
    tf::SkipBox* sb = new tf::SkipBox(this);
    ui->settingsLayout->addWidget(sb);
    sb->setSize(s->size());
    connect(sb, SIGNAL(size(const vvVector3&)), this, SLOT(onSize(const vvVector3&)));
  }
}

