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

#include <boost/bimap.hpp>


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

struct vvTFDialog::Impl
{
  Impl()
    : ui(new Ui::TFDialog)
    , colorscene(new MouseGraphicsScene)
    , colortex(new QGraphicsPixmapItem)
    , alphascene(new MouseGraphicsScene)
    , alphatex(new QGraphicsPixmapItem)
    , selected(NULL)
    , moving(NULL)
    , zoomRange(virvo::Vec2f(0.0f, 1.0f))
  {
    colorscene->addItem(colortex);
    alphascene->addItem(alphatex);
  }

  ~Impl()
  {
    delete colortex;
    delete alphatex;
    delete colorscene;
    delete alphascene;
  }

  std::auto_ptr<Ui::TFDialog> ui;

  MouseGraphicsScene* colorscene;
  QGraphicsPixmapItem* colortex;
  std::vector<Pin*> colorpins;

  MouseGraphicsScene* alphascene;
  QGraphicsPixmapItem* alphatex;
  std::vector<Pin*> alphapins;

  Pin* selected;
  Pin* moving;

  typedef boost::bimap< Pin*, vvTFWidget*> bm_type;
  bm_type pin2widget;

  vvVector2f zoomRange; ///< min/max for zoom area on data range

private:

  Impl(Impl const& rhs);
  Impl& operator=(Impl const& rhs);

};

vvTFDialog::vvTFDialog(vvCanvas* canvas, QWidget* parent)
  : QDialog(parent)
  , impl_(new Impl)
  , _canvas(canvas)
{
  vvDebugMsg::msg(1, "vvTFDialog::vvTFDialog()");

  impl_->ui->setupUi(this);

  impl_->ui->color1DView->setScene(impl_->colorscene);
  impl_->ui->alpha1DView->setScene(impl_->alphascene);

  connect(impl_->ui->colorButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(impl_->ui->pyramidButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(impl_->ui->gaussianButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(impl_->ui->customButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(impl_->ui->skipRangeButton, SIGNAL(clicked()), this, SLOT(onNewWidget()));
  connect(impl_->ui->deleteButton, SIGNAL(clicked()), this, SLOT(onDeleteClicked()));
  connect(impl_->ui->undoButton, SIGNAL(clicked()), this, SLOT(onUndoClicked()));
  connect(impl_->ui->presetColorsBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onPresetColorsChanged(int)));
  connect(impl_->ui->presetAlphaBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onPresetAlphaChanged(int)));
  connect(impl_->ui->discrSlider, SIGNAL(valueChanged(int)), this, SLOT(onDiscrChanged(int)));
  connect(impl_->ui->applyButton, SIGNAL(clicked()), this, SLOT(onApplyClicked()));
  connect(_canvas, SIGNAL(newVolDesc(vvVolDesc*)), this, SLOT(onNewVolDesc(vvVolDesc*)));
  connect(impl_->ui->saveButton, SIGNAL(clicked()), this, SLOT(saveTF()));
  connect(impl_->ui->loadButton, SIGNAL(clicked()), this, SLOT(loadTF()));
  connect(impl_->colorscene, SIGNAL(mouseMoved(QPointF, Qt::MouseButton)), this, SLOT(onTFMouseMove(QPointF, Qt::MouseButton)));
  connect(impl_->colorscene, SIGNAL(mousePressed(QPointF, Qt::MouseButton)), this, SLOT(onTFMousePress(QPointF, Qt::MouseButton)));
  connect(impl_->colorscene, SIGNAL(mouseReleased(QPointF, Qt::MouseButton)), this, SLOT(onTFMouseRelease(QPointF, Qt::MouseButton)));
  connect(impl_->alphascene, SIGNAL(mouseMoved(QPointF, Qt::MouseButton)), this, SLOT(onTFMouseMove(QPointF, Qt::MouseButton)));
  connect(impl_->alphascene, SIGNAL(mousePressed(QPointF, Qt::MouseButton)), this, SLOT(onTFMousePress(QPointF, Qt::MouseButton)));
  connect(impl_->alphascene, SIGNAL(mouseReleased(QPointF, Qt::MouseButton)), this, SLOT(onTFMouseRelease(QPointF, Qt::MouseButton)));
}

vvTFDialog::~vvTFDialog()
{
  vvDebugMsg::msg(1, "vvTFDialog::~vvTFDialog()");
}

void vvTFDialog::drawTF()
{
  drawColorTexture();
  drawAlphaTexture();
  impl_->colorscene->invalidate();
  impl_->alphascene->invalidate();
}

void vvTFDialog::drawColorTexture()
{
  vvDebugMsg::msg(3, "vvTFDialog::drawColorTexture()");

  size_t w = 768;
  size_t h = 3;
  std::vector<uchar> colorBar;
  colorBar.resize(w * h * 4);
  makeColorBar(&colorBar, w);
  QImage img(&colorBar[0], w, 3, QImage::Format_ARGB32);
  img = img.scaled(QSize(w, h * COLORBAR_HEIGHT / 3));
  if (!img.isNull())
  {
    QPixmap colorpm = QPixmap::fromImage(img);
    impl_->colortex->setPixmap(colorpm);
  }
}

void vvTFDialog::drawAlphaTexture()
{
  int w = impl_->ui->alpha1DView->width();
  int h = impl_->ui->alpha1DView->height();
  assert(w >= 0 && h >= 0);

  std::vector<uchar> alphaTex;
  alphaTex.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
  makeAlphaTexture(&alphaTex, w, h);
  QImage img(&alphaTex[0], w, h, QImage::Format_ARGB32);
  if (!img.isNull())
  {
    QPixmap alphapm = QPixmap::fromImage(img);
    impl_->alphatex->setPixmap(alphapm);
  }
}

void vvTFDialog::clearPins()
{
  for (std::vector<Pin*>::const_iterator it = impl_->colorpins.begin();
       it != impl_->colorpins.end(); ++it)
  {
    impl_->colorscene->removeItem(*it);
    delete *it;
  }
  impl_->colorpins.clear();

  for (std::vector<Pin*>::const_iterator it = impl_->alphapins.begin();
       it != impl_->alphapins.end(); ++it)
  {
    impl_->alphascene->removeItem(*it);
    delete *it;
  }
  impl_->alphapins.clear();

  impl_->pin2widget.clear();
  impl_->selected = NULL;
  impl_->moving = NULL;
}

void vvTFDialog::createPins()
{
  if (!_canvas->getVolDesc())
    return;

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
  float xpos = data2norm(impl_->zoomRange, w->_pos[0]) * static_cast<float>(TF_WIDTH);

  Pin* pin = NULL;
  if (dynamic_cast<vvTFColor*>(w) != NULL) // draw color pin
  {
    pin = impl_->colorscene->addRect(-rectw * 0.5f, 0, rectw, COLORBAR_HEIGHT - 1, QPen(), QBrush(Qt::SolidPattern));
    pin->setPos(xpos, 0);
    impl_->colorpins.push_back(pin);
  }
  else if ((dynamic_cast<vvTFPyramid*>(w) != NULL) ||
           (dynamic_cast<vvTFBell*>(w) != NULL) ||
           (dynamic_cast<vvTFSkip*>(w) != NULL) ||
           (dynamic_cast<vvTFCustom*>(w) != NULL)) // draw alpha pin
  {
    pin = impl_->alphascene->addRect(-rectw * 0.5f, 0, rectw, TF_HEIGHT - COLORBAR_HEIGHT, QPen(), QBrush(Qt::SolidPattern));
    pin->setPos(xpos, 0);
    impl_->alphapins.push_back(pin);

  }
  impl_->pin2widget.insert(Impl::bm_type::value_type(pin, w));
}

void vvTFDialog::makeColorBar(std::vector<uchar>* colorBar, int width) const
{
  if (!_canvas->getVolDesc())
  {
    memset(&(*colorBar)[0], 0, colorBar->size());
    return;
  }

  if (/* HDR */ false)
  {

  }
  else // standard iso-range TF mode
  {
    // BGRA to fit QImage's little endian ARGB32 format
    _canvas->getVolDesc()->tf.makeColorBar(width, &(*colorBar)[0], impl_->zoomRange[0], impl_->zoomRange[1], false, vvToolshed::VV_BGRA);
  }
}

void vvTFDialog::makeAlphaTexture(std::vector<uchar>* alphaTex, int width, int height) const
{
  if (!_canvas->getVolDesc())
  {
    memset(&(*alphaTex)[0], 0, alphaTex->size());
    return;
  }

  if (/* HDR */ false)
  {

  }
  else // standard iso-range TF mode
  {
    // BGRA to fit QImage's little endian ARGB32 format
    _canvas->getVolDesc()->tf.makeAlphaTexture(width, height, &(*alphaTex)[0], impl_->zoomRange[0], impl_->zoomRange[1], vvToolshed::VV_BGRA);
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

  if (QObject::sender() == impl_->ui->colorButton)
  {
    widget = new vvTFColor(vvColor(), norm2data(impl_->zoomRange, 0.5f));
  }
  else if (QObject::sender() == impl_->ui->pyramidButton)
  {
    widget = new vvTFPyramid(vvColor(), false, 1.0f, norm2data(impl_->zoomRange, 0.5f), normd2datad(impl_->zoomRange, 0.4f), normd2datad(impl_->zoomRange, 0.2f));
  }
  else if (QObject::sender() == impl_->ui->gaussianButton)
  {
    widget = new vvTFBell(vvColor(), false, 1.0f, norm2data(impl_->zoomRange, 0.5f), normd2datad(impl_->zoomRange, 0.2f));
  }
  else if (QObject::sender() == impl_->ui->customButton)
  {
    widget = new vvTFCustom(norm2data(impl_->zoomRange, 0.5f), norm2data(impl_->zoomRange, 0.5f));
  }
  else if (QObject::sender() == impl_->ui->skipRangeButton)
  {
    widget = new vvTFSkip(norm2data(impl_->zoomRange, 0.5f), normd2datad(impl_->zoomRange, 0.2f));
  }

  emit newWidget(widget);
  createPin(widget);
  Impl::bm_type::right_const_iterator rit = impl_->pin2widget.right.find(widget);
  impl_->selected = rit->second;
  updateSettingsBox();
  drawTF();
}

void vvTFDialog::onDeleteClicked()
{
  if(_canvas->getVolDesc()->tf._widgets.size() == 0 || impl_->selected == NULL)
  {
    return;
  }
  _canvas->getVolDesc()->tf.putUndoBuffer();

  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(impl_->selected);
  _canvas->getVolDesc()->tf._widgets.erase
  (
    std::find(_canvas->getVolDesc()->tf._widgets.begin(), _canvas->getVolDesc()->tf._widgets.end(), lit->second)
  );
  impl_->selected = NULL;

  emitTransFunc();
  clearPins();
  createPins();
  updateSettingsBox();
  drawTF();
}

void vvTFDialog::onPresetColorsChanged(int index)
{
  vvDebugMsg::msg(3, "vvTFDialog::onPresetColorsChanged()");

  _canvas->getVolDesc()->tf.setDefaultColors(index, impl_->zoomRange[0], impl_->zoomRange[1]);
  emitTransFunc();
  clearPins();
  createPins();
  drawTF();
}

void vvTFDialog::onPresetAlphaChanged(int index)
{
  vvDebugMsg::msg(3, "vvTFDialog::onPresetAlphaChanged()");

  _canvas->getVolDesc()->tf.setDefaultAlpha(index, impl_->zoomRange[0], impl_->zoomRange[1]);
  emitTransFunc();
  clearPins();
  createPins();
  drawTF();
}

void vvTFDialog::onDiscrChanged(int num)
{
  impl_->ui->discrLabel->setText(QString::number(num));
  _canvas->getVolDesc()->tf.setDiscreteColors(num);
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onApplyClicked()
{
  drawTF();
}

void vvTFDialog::onNewVolDesc(vvVolDesc *vd)
{
  clearPins();
  createPins();
  if (vd != NULL)
  {
    impl_->ui->discrSlider->setValue(vd->tf.getDiscreteColors());
  }
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
  impl_->ui->discrSlider->setValue(_canvas->getVolDesc()->tf.getDiscreteColors());
  drawTF();
}

void vvTFDialog::onColor(const QColor& color)
{
  assert(impl_->selected != NULL);
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(impl_->selected);
  vvTFWidget* wid = lit->second;
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
  assert(impl_->selected != NULL);
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(impl_->selected);
  vvTFWidget* wid = lit->second;
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
  assert(impl_->selected != NULL);
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(impl_->selected);
  vvTFWidget* wid = lit->second;
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
  assert(impl_->selected != NULL);
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(impl_->selected);
  vvTFWidget* wid = lit->second;
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
  assert(impl_->selected != NULL);
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(impl_->selected);
  vvTFWidget* wid = lit->second;
  vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(wid);
  assert(p != NULL);
  p->setTop(top);
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onBottom(const vvVector3& bottom)
{
  assert(impl_->selected != NULL);
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(impl_->selected);
  vvTFWidget* wid = lit->second;
  vvTFPyramid* p = dynamic_cast<vvTFPyramid*>(wid);
  assert(p != NULL);
  p->setBottom(bottom);
  emitTransFunc();
  drawTF();
}

void vvTFDialog::onTFMouseMove(QPointF pos, Qt::MouseButton /* button */)
{
  std::vector<Pin*>& pins = QObject::sender() == impl_->colorscene ? impl_->colorpins : impl_->alphapins;

  if (impl_->moving != NULL)
  {
    std::vector<Pin*>::iterator it = std::find(pins.begin(), pins.end(), impl_->moving);
    if (it == pins.end())
    {
      return;
    }

    float x = static_cast<float>(pos.x());
    x = ts_clamp(x, 0.0f, static_cast<float>(TF_WIDTH));
    QPointF pinpos = (*it)->pos();
    (*it)->setPos(x, pinpos.y());
    Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(*it);
    vvTFWidget* w = lit->second;
    if (w != NULL)
    {
      x /= static_cast<float>(TF_WIDTH);
      x = norm2data(impl_->zoomRange, x);
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
  const std::vector<Pin*>& pins = QObject::sender() == impl_->colorscene ? impl_->colorpins : impl_->alphapins;

  if (button == Qt::LeftButton)
  {
    impl_->selected = NULL;
    for (std::vector<Pin*>::const_iterator it = pins.begin();
         it != pins.end(); ++it)
    {
      float minx = (*it)->pos().x() - PIN_WIDTH * 0.5f - 1;
      float maxx = (*it)->pos().x() + PIN_WIDTH * 0.5f + 1;
      if (pos.x() >= minx && pos.x() <= maxx)
      {
        impl_->moving = *it;
        if (impl_->selected != *it)
        {
          impl_->selected = *it;
        }
        break;
      }
    }
    updateSettingsBox();
  }
}

void vvTFDialog::onTFMouseRelease(QPointF /* pos */, Qt::MouseButton /* button */)
{
  impl_->moving = NULL;
}

void vvTFDialog::emitTransFunc()
{
  emit newTransferFunction();
}

void vvTFDialog::updateSettingsBox()
{
  // clear settings layout
  if (QLayoutItem* item = impl_->ui->settingsLayout->takeAt(0))
  {
    QWidget* widget = item->widget();
    delete widget;
  }

  Pin* selected = impl_->selected;
  if (selected == NULL)
  {
    return;
  }

  // new settings box
  Impl::bm_type::left_const_iterator lit = impl_->pin2widget.left.find(selected);
  vvTFWidget* w = lit->second;
  if (vvTFColor* c = dynamic_cast<vvTFColor*>(w))
  {
    tf::ColorBox* cb = new tf::ColorBox(this);
    impl_->ui->settingsLayout->addWidget(cb);
    cb->setColor(c->color());
    connect(cb, SIGNAL(color(const QColor&)), this, SLOT(onColor(const QColor&)));
  }
  else if (vvTFBell* b = dynamic_cast<vvTFBell*>(w))
  {
    tf::GaussianBox* gb = new tf::GaussianBox(this);
    impl_->ui->settingsLayout->addWidget(gb);
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
    impl_->ui->settingsLayout->addWidget(pb);
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
    impl_->ui->settingsLayout->addWidget(sb);
    sb->setSize(s->size());
    connect(sb, SIGNAL(size(const vvVector3&)), this, SLOT(onSize(const vvVector3&)));
  }
}

