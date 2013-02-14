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
#include <QUrl>

#include "vvcanvas.h"
#include "vvprefdialog.h"
#include "vvstereomode.h"

#include "ui_vvprefdialog.h"

#include <virvo/vvbonjour/vvbonjour.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/vvremoteevents.h>
#include <virvo/vvshaderfactory.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvsocketmap.h>
#include <virvo/vvtoolshed.h>
#include <virvo/vvvirvo.h>

#include <QMessageBox>
#include <QSettings>
#include <QValidator>

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <utility>

#define VV_UNUSED(x) ((void)(x))

namespace
{
  std::map<int, vvRenderer::RendererType> rendererMap;
  std::map<int, std::string> texRendTypeMap;
  std::map<int, std::string> voxTypeMap;
  std::map<int, int> fboPrecisionMap;
  std::map<int, vox::StereoMode> stereoModeMap;
  std::map<std::string, std::string> rendererDescriptions;
  std::map<std::string, std::string> algoDescriptions;

  // e.g. ibr or image
  std::string remoterend = "";
  vvTcpSocket* sock;

  double movingSpinBoxOldValue = 1.0;
  double stillSpinBoxOldValue = 1.0;
  int movingDialOldValue = 0;
  int stillDialOldValue = 0;

  /* make qt dials behave as if they had an unlimited range
   */
  int getDialDelta(int oldval, int newval, int minval, int maxval)
  {
    const int eps = 10; // largest possible step from a single user action
    const int mineps = minval + eps;
    const int maxeps = maxval - eps;

    if (oldval < mineps && newval > maxeps)
    {
      return -(oldval + maxval + 1 - newval);
    }
    else if (oldval > maxeps && newval < mineps)
    {
      return maxval + 1 - oldval + newval;
    }
    else
    {
      return newval - oldval;
    }
  }
}

vvPrefDialog::vvPrefDialog(vvCanvas* canvas, QWidget* parent)
  : QDialog(parent)
  , ui(new Ui_PrefDialog)
  , _canvas(canvas)
{
  vvDebugMsg::msg(1, "vvPrefDialog::vvPrefDialog()");

  ui->setupUi(this);

  // can't be done in designer unfortunately
  QIcon ic = style()->standardIcon(QStyle::SP_MessageBoxInformation);
  ui->texInfoIconLabel->setPixmap(ic.pixmap(32, 32));

  _canvas->makeCurrent();
  glewInit(); // we need glCreateProgram etc. when checking for glsl support

  rendererDescriptions.insert(std::pair<std::string, std::string>("slices", "OpenGL textures"));
  rendererDescriptions.insert(std::pair<std::string, std::string>("cubic2d", "OpenGL textures"));
  rendererDescriptions.insert(std::pair<std::string, std::string>("planar", "OpenGL textures"));
  rendererDescriptions.insert(std::pair<std::string, std::string>("spherical", "OpenGL textures"));
  rendererDescriptions.insert(std::pair<std::string, std::string>("rayrend", "CUDA ray casting"));
  rendererDescriptions.insert(std::pair<std::string, std::string>("softrayrend", "Software ray casting"));

  algoDescriptions.insert(std::pair<std::string, std::string>("default", "Autoselect"));
  algoDescriptions.insert(std::pair<std::string, std::string>("slices", "2D textures (slices)"));
  algoDescriptions.insert(std::pair<std::string, std::string>("cubic2d", "2D textures (cubic)"));
  algoDescriptions.insert(std::pair<std::string, std::string>("planar", "3D textures (viewport aligned)"));
  algoDescriptions.insert(std::pair<std::string, std::string>("spherical", "3D textures (spherical)"));

  // renderer combo box
  int idx = 0;
  if (vvRendererFactory::hasRenderer(vvRenderer::TEXREND))
  {
    ui->rendererBox->addItem(rendererDescriptions["slices"].c_str());
    rendererMap.insert(std::pair<int, vvRenderer::RendererType>(idx, vvRenderer::TEXREND));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer(vvRenderer::RAYREND))
  {
    ui->rendererBox->addItem(rendererDescriptions["rayrend"].c_str());
    rendererMap.insert(std::pair<int, vvRenderer::RendererType>(idx, vvRenderer::RAYREND));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer(vvRenderer::SOFTRAYREND))
  {
    ui->rendererBox->addItem(rendererDescriptions["softrayrend"].c_str());
    rendererMap.insert(std::pair<int, vvRenderer::RendererType>(idx, vvRenderer::SOFTRAYREND));
    ++idx;
  }

  if (ui->rendererBox->count() <= 0)
  {
    ui->rendererBox->setEnabled(false);
  }

  // texrend geometry combo box
  idx = 0;

  if (vvRendererFactory::hasRenderer(vvRenderer::TEXREND))
  {
    ui->geometryBox->addItem(algoDescriptions["default"].c_str());
    texRendTypeMap.insert(std::pair<int, std::string>(idx, "default"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer("slices"))
  {
    ui->geometryBox->addItem(algoDescriptions["slices"].c_str());
    texRendTypeMap.insert(std::pair<int, std::string>(idx, "slices"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer("cubic2d"))
  {
    ui->geometryBox->addItem(algoDescriptions["cubic2d"].c_str());
    texRendTypeMap.insert(std::pair<int, std::string>(idx, "cubic2d"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer("planar"))
  {
    ui->geometryBox->addItem(algoDescriptions["planar"].c_str());
    texRendTypeMap.insert(std::pair<int, std::string>(idx, "planar"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer("spherical"))
  {
    ui->geometryBox->addItem(algoDescriptions["spherical"].c_str());
    texRendTypeMap.insert(std::pair<int, std::string>(idx, "spherical"));
    ++idx;
  }

  // voxel type combo box
  idx = 0;

  if (vvRendererFactory::hasRenderer(vvRenderer::TEXREND))
  {
    ui->voxTypeBox->addItem("Autoselect");
    voxTypeMap.insert(std::pair<int, std::string>(idx, "default"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer(vvRenderer::TEXREND))
  {
    ui->voxTypeBox->addItem("RGBA");
    voxTypeMap.insert(std::pair<int, std::string>(idx, "rgba"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer(vvRenderer::TEXREND))
  {
    ui->voxTypeBox->addItem("ARB fragment program");
    voxTypeMap.insert(std::pair<int, std::string>(idx, "arb"));
    ++idx;
  }

  if (vvRendererFactory::hasRenderer(vvRenderer::TEXREND) && vvShaderFactory::isSupported("glsl"))
  {
    ui->voxTypeBox->addItem("GLSL fragment program");
    voxTypeMap.insert(std::pair<int, std::string>(idx, "shader"));
    ++idx;
  }

  // fbo combo box
  idx = 0;

  ui->fboBox->addItem("None");
  fboPrecisionMap.insert(std::pair<int, int>(idx, 0));
  ++idx;

  ui->fboBox->addItem("8 bit precision");
  fboPrecisionMap.insert(std::pair<int, int>(idx, 8));
  ++idx;

  ui->fboBox->addItem("16 bit precision");
  fboPrecisionMap.insert(std::pair<int, int>(idx, 16));
  ++idx;

  ui->fboBox->addItem("32 bit precision");
  fboPrecisionMap.insert(std::pair<int, int>(idx, 32));
  ++idx;

  // stereo mode combo box
  idx = 0;

  ui->stereoModeBox->addItem("Off (Mono)");
  stereoModeMap.insert(std::pair<int, vox::StereoMode>(idx, vox::Mono));
  ++idx;

  if (_canvas->format().stencil())
  {
    ui->stereoModeBox->addItem("Interlaced (Lines)");
    stereoModeMap.insert(std::pair<int, vox::StereoMode>(idx, vox::InterlacedLines));
    ++idx;
  }

  if (_canvas->format().stencil())
  {
    ui->stereoModeBox->addItem("Interlaced (Checkerboard)");
    stereoModeMap.insert(std::pair<int, vox::StereoMode>(idx, vox::InterlacedCheckerboard));
    ++idx;
  }

  ui->stereoModeBox->addItem("Red cyan");
  stereoModeMap.insert(std::pair<int, vox::StereoMode>(idx, vox::RedCyan));
  ++idx;

  ui->stereoModeBox->addItem("Side by side");
  stereoModeMap.insert(std::pair<int, vox::StereoMode>(idx, vox::SideBySide));
  ++idx;


  // remote rendering page
  if (virvo::hasFeature("bonjour"))
  {
    ui->browseButton->setEnabled(true);
  }

  QIntValidator* val = new QIntValidator(this);
  val->setRange(ui->stereoDistSlider->minimum(), ui->stereoDistSlider->maximum());
  ui->stereoDistEdit->setValidator(val);
  ui->stereoDistEdit->setText(QString::number(ui->stereoDistSlider->value()));

  connect(ui->rendererBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onRendererChanged(int)));
  connect(ui->geometryBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onTexRendOptionChanged(int)));
  connect(ui->fboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onFboChanged(int)));
  connect(ui->voxTypeBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onTexRendOptionChanged(int)));
  connect(ui->pixShdBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onTexRendOptionChanged(int)));
  connect(ui->hostEdit, SIGNAL(textChanged(const QString&)), this, SLOT(onHostChanged(const QString&)));
  connect(ui->portBox, SIGNAL(valueChanged(int)), this, SLOT(onPortChanged(int)));
  connect(ui->getInfoButton, SIGNAL(clicked()), this, SLOT(onGetInfoClicked()));
  connect(ui->browseButton, SIGNAL(clicked()), this, SLOT(onBrowseClicked()));
  connect(ui->connectButton, SIGNAL(clicked()), this, SLOT(onConnectClicked()));
  connect(ui->ibrBox, SIGNAL(toggled(bool)), this, SLOT(onIbrToggled(bool)));
  connect(ui->interpolationCheckBox, SIGNAL(toggled(bool)), this, SLOT(onInterpolationToggled(bool)));
  connect(ui->mipCheckBox, SIGNAL(toggled(bool)), this, SLOT(onMipToggled(bool)));
  connect(ui->stereoModeBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onStereoModeChanged(int)));
  connect(ui->stereoDistEdit, SIGNAL(textEdited(const QString&)), this, SLOT(onStereoDistEdited(const QString&)));
  connect(ui->stereoDistSlider, SIGNAL(sliderMoved(int)), this, SLOT(onStereoDistSliderMoved(int)));
  connect(ui->stereoDistSlider, SIGNAL(valueChanged(int)), this, SLOT(onStereoDistChanged(int)));
  connect(ui->swapEyesBox, SIGNAL(toggled(bool)), this, SLOT(onSwapEyesToggled(bool)));
  connect(ui->movingSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onMovingSpinBoxChanged(double)));
  connect(ui->stillSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onStillSpinBoxChanged(double)));
  connect(ui->movingDial, SIGNAL(valueChanged(int)), this, SLOT(onMovingDialChanged(int)));
  connect(ui->stillDial, SIGNAL(valueChanged(int)), this, SLOT(onStillDialChanged(int)));

  // apply settings after signals/slots are connected
  QSettings settings;
  ui->hostEdit->setText(settings.value("remote/host").toString());
  if (settings.value("remote/port").toString() != "")
  {
    int port = settings.value("remote/port").toInt();
    ui->portBox->setValue(port);
  }
  ui->ibrBox->setChecked(settings.value("remote/ibr").toBool());

  if (!settings.value("stereo/distance").isNull())
  {
    int dist = settings.value("stereo/distance").toInt();
    ui->stereoDistEdit->setText(QString::number(dist));
    ui->stereoDistSlider->setValue(dist);
  }

  if (!settings.value("stereo/swap").isNull())
  {
    ui->swapEyesBox->setChecked(settings.value("stereo/swap").toBool());
  }
}

vvPrefDialog::~vvPrefDialog()
{
  if (::sock != NULL)
  {
    vvSocketMap::remove(vvSocketMap::getIndex(::sock));
  }
  delete ::sock;
}

void vvPrefDialog::toggleInterpolation()
{
  vvDebugMsg::msg(3, "vvPrefDialog::toggletInterpolation()");

  const bool interpolation = ui->interpolationCheckBox->isChecked();
  ui->interpolationCheckBox->setChecked(!interpolation);
  emit parameterChanged(vvRenderer::VV_SLICEINT, !interpolation);
}

void vvPrefDialog::scaleStillQuality(const float s)
{
  vvDebugMsg::msg(3, "vvPrefDialog::scaleStillQuality()");

  assert(s >= 0.0f);

  float quality = static_cast<float>(ui->stillSpinBox->value());
  if (quality <= 0.0f)
  {
    // never let quality drop to or below 0
    quality = std::numeric_limits<float>::epsilon();
  }
  quality *= s;

  ui->stillSpinBox->setValue(quality);
}

void vvPrefDialog::emitRenderer()
{
  vvDebugMsg::msg(3, "vvPrefDialog::emitRenderer()");

  ui->texInfoLabel->setText("");
  std::string name = "";

  // indices to activate appropriate options tool box pages
  const int texidx = 0;
  const int rayidx = 1;

  vvRendererFactory::Options options;

  if (::remoterend == "ibr" || ::remoterend == "image")
  {
    int s = vvSocketMap::add(::sock);
    std::stringstream sockstr;
    sockstr << s;
    if (sockstr.str() != "")
    {
      name = ::remoterend;
      options["sockets"] = sockstr.str();
    }
  }
  else
  {
    switch (rendererMap[ui->rendererBox->currentIndex()])
    {
    case vvRenderer::RAYREND:
      ui->optionsToolBox->setCurrentIndex(rayidx);
      name = "rayrend";
      break;
    case vvRenderer::SOFTRAYREND:
      ui->optionsToolBox->setCurrentIndex(rayidx);
      name = "softrayrend";
      break;
    case vvRenderer::TEXREND:
      ui->optionsToolBox->setCurrentIndex(texidx);
      name = texRendTypeMap[ui->geometryBox->currentIndex()];
      options["voxeltype"] = voxTypeMap[ui->voxTypeBox->currentIndex()];
      break;
    default:
      name = "default";
      break;
    }

    if (options["voxeltype"] == "rgba")
    {
      ui->texInfoLabel->setText(ui->texInfoLabel->text() + "<html><b>Voxel type RGBA</b><br />"
        "Pre-interpolative transfer function,"
        " is applied by assigning each voxel an RGBA color before rendering.</html>");
    }
    else if (options["voxeltype"] == "arb")
    {
      ui->texInfoLabel->setText(ui->texInfoLabel->text() + "<html><b>Voxel type ARB fragment program</b><br />"
        "Post-interpolative transfer function,"
        " is applied after sampling the volume texture.</html>");
    }
    else if (options["voxeltype"] == "shader")
    {
      ui->texInfoLabel->setText(ui->texInfoLabel->text() + "<html><b>Voxel type GLSL fragment program</b><br />"
        "Post-interpolative transfer function,"
        " is applied after sampling the volume texture.</html>");
    }
  }

  if (name != "")
  {
    emit rendererChanged(name, options);
  }
}

bool vvPrefDialog::validateRemoteHost(const QString& host, const ushort port)
{
  int parsedPort = vvToolshed::parsePort(host.toStdString());
  if (parsedPort >= 0 && parsedPort <= std::numeric_limits<ushort>::max()
   && static_cast<ushort>(parsedPort) != port)
  {
    ui->portBox->setValue(parsedPort);
  }

  std::string h = (parsedPort == -1)
    ? host.toStdString()
    : vvToolshed::stripPort(host.toStdString());
  ushort p = static_cast<ushort>(ui->portBox->value());

  if (h == "")
  {
    return false;
  }

  QUrl url(h.c_str());
  url.setPort(p);
  return url.isValid();
}

void vvPrefDialog::onRendererChanged(const int index)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onRendererChanged()");

  VV_UNUSED(index);
  assert(index == ui->rendererBox->currentIndex());
  emitRenderer();
}

void vvPrefDialog::onTexRendOptionChanged(const int index)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onTexRendOptionChanged()");

  VV_UNUSED(index);

  if (rendererMap[ui->rendererBox->currentIndex()] == vvRenderer::TEXREND)
  {
    emitRenderer();
  }
}

void vvPrefDialog::onFboChanged(int index)
{
  ui->texInfoLabel->setText("");
  if (fboPrecisionMap[index] >= 8)
  {
    ui->texInfoLabel->setText("<html><b>" + QString::number(fboPrecisionMap[index]) + " bit fbo rendering</b><br />"
      "An fbo is bound during slice compositing. A higher precision can help to avoid rounding errors but will result "
      "in an increased rendering time.</html>");
  }

  switch (fboPrecisionMap[index])
  {
  case 8:
    emit parameterChanged(vvRenderer::VV_OFFSCREENBUFFER, true);
    emit parameterChanged(vvRenderer::VV_IMG_PRECISION, 8);
    break;
  case 16:
    emit parameterChanged(vvRenderer::VV_OFFSCREENBUFFER, true);
    emit parameterChanged(vvRenderer::VV_IMG_PRECISION, 16);
    break;
  case 32:
    emit parameterChanged(vvRenderer::VV_OFFSCREENBUFFER, true);
    emit parameterChanged(vvRenderer::VV_IMG_PRECISION, 32);
    break;
  default:
    emit parameterChanged(vvRenderer::VV_OFFSCREENBUFFER, false);
    break;
  }
}

void vvPrefDialog::onHostChanged(const QString& text)
{
  const ushort port = static_cast<ushort>(ui->portBox->value());
  if (validateRemoteHost(text, port))
  {
    ui->getInfoButton->setEnabled(true);
    ui->connectButton->setEnabled(true);
  }
  else
  {
    ui->getInfoButton->setEnabled(false);
    ui->connectButton->setEnabled(false);
  }
}

void vvPrefDialog::onPortChanged(const int i)
{
  const ushort port = static_cast<ushort>(i);
  if (validateRemoteHost(ui->hostEdit->text(), port))
  {
    ui->getInfoButton->setEnabled(true);
    ui->connectButton->setEnabled(true);
  }
  else
  {
    ui->getInfoButton->setEnabled(false);
    ui->connectButton->setEnabled(false);
  }
}

void vvPrefDialog::onGetInfoClicked()
{
  if (validateRemoteHost(ui->hostEdit->text(), static_cast<ushort>(ui->portBox->value())))
  {
    vvTcpSocket* sock = new vvTcpSocket;
    if (sock->connectToHost(ui->hostEdit->text().toStdString(),
      static_cast<ushort>(static_cast<ushort>(ui->portBox->value()))) == vvSocket::VV_OK)
    {
      sock->setParameter(vvSocket::VV_NO_NAGLE, true);
      vvSocketIO io(sock);

      vvServerInfo info;
      io.putEvent(virvo::ServerInfo);
      io.getServerInfo(info);
      QString qrenderers;
      std::vector<std::string> renderers = vvToolshed::split(info.renderers, ",");
      for (std::vector<std::string>::const_iterator it = renderers.begin();
           it != renderers.end(); ++it)
      {
        std::string rend = rendererDescriptions[*it];
        std::string algo = algoDescriptions[*it];
        qrenderers += "<tr><td>" + tr(rend.c_str()) + "</td><td>" + tr(algo.c_str()) + "</td></tr>";
      }
      QMessageBox::information(this, tr("Server info"), tr("Remote server supports the following rendering algorithms<br /><br />")
        + tr("<table>") + qrenderers + tr("</table>"), QMessageBox::Ok);
      io.putEvent(virvo::Disconnect);

      // store to registry because connection was successful
      QSettings settings;
      settings.setValue("remote/host", ui->hostEdit->text());
      settings.setValue("remote/port", ui->portBox->value());
    }
    else
    {
      QMessageBox::warning(this, tr("Failed to connect"), tr("Could not connect to host \"") + ui->hostEdit->text()
        + tr("\" on port \"") + QString::number(ui->portBox->value()) + tr("\""), QMessageBox::Ok);
    }
    delete sock;
  }
}

void vvPrefDialog::onBrowseClicked()
{
  if (virvo::hasFeature("bonjour"))
  {
    vvBonjour bonjour;
    std::vector<std::string> servers = bonjour.getConnectionStringsFor("_vserver._tcp");
    for (std::vector<std::string>::const_iterator it = servers.begin();
         it != servers.end(); ++it)
    {
      std::cerr << *it << std::endl;
    }
  }
}

void vvPrefDialog::onConnectClicked()
{
  if (::remoterend == "")
  {
    if (validateRemoteHost(ui->hostEdit->text(), static_cast<ushort>(ui->portBox->value())))
    {
      delete ::sock;
      ::sock = new vvTcpSocket;
      if (sock->connectToHost(ui->hostEdit->text().toStdString(),
        static_cast<ushort>(static_cast<ushort>(ui->portBox->value()))) == vvSocket::VV_OK)
      {
        ::sock->setParameter(vvSocket::VV_NO_NAGLE, true);

        vvSocketIO io(::sock);

        ui->connectButton->setText(tr("Disconnect"));

        if (!ui->ibrBox->isChecked())
        {
          ::remoterend = "image";
        }
        else
        {
          ::remoterend = "ibr";
          if (io.putEvent(virvo::RemoteServerType) == vvSocket::VV_OK)
          {
            io.putRendererType(vvRenderer::REMOTE_IBR);
          }
        }

        // store to registry because connection was successful
        QSettings settings;
        settings.setValue("remote/host", ui->hostEdit->text());
        settings.setValue("remote/port", ui->portBox->value());

        emitRenderer();
      }
      else
      {
        ::remoterend = "";
        QMessageBox::warning(this, tr("Failed to connect"), tr("Could not connect to host \"") + ui->hostEdit->text()
          + tr("\" on port \"") + QString::number(ui->portBox->value()) + tr("\""), QMessageBox::Ok);
        delete ::sock;
        ::sock = NULL;
      }
    }
  }
  else
  {
    ::remoterend = "";

    if (::sock != NULL)
    {
      vvSocketMap::remove(vvSocketMap::getIndex(sock));
      delete ::sock;
      ::sock = NULL;
    }

    emitRenderer();
  }
}

void vvPrefDialog::onIbrToggled(const bool checked)
{
  QSettings settings;
  settings.setValue("remote/ibr", checked);

  if (::sock != NULL && ::remoterend != "")
  {
    vvSocketIO io(::sock);
    if (io.putEvent(virvo::RemoteServerType) == vvSocket::VV_OK)
    {
      if (checked)
      {
        ::remoterend = "ibr";
        io.putRendererType(vvRenderer::REMOTE_IBR);
      }
      else
      {
        ::remoterend = "image";
        io.putRendererType(vvRenderer::REMOTE_IMAGE);
      }

      emitRenderer();
    }
  }
}

void vvPrefDialog::onInterpolationToggled(const bool checked)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onInterpolationToggled()");

  emit parameterChanged(vvRenderer::VV_SLICEINT, checked);
}

void vvPrefDialog::onMipToggled(bool checked)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onMipToggled()");

  const int mipMode = checked ? 1 : 0; // don't support mip == 2 (min. intensity) for now
  emit parameterChanged(vvRenderer::VV_MIP_MODE, mipMode);
}

void vvPrefDialog::onStereoModeChanged(int index)
{
  emit parameterChanged(vvParameters::VV_STEREO_MODE, static_cast<int>(stereoModeMap[index]));
}

void vvPrefDialog::onStereoDistEdited(const QString& text)
{
  ui->stereoDistSlider->setValue(text.toInt());
}

void vvPrefDialog::onStereoDistSliderMoved(int value)
{
  ui->stereoDistEdit->setText(QString::number(value));
}

void vvPrefDialog::onStereoDistChanged(int value)
{
  QSettings settings;
  settings.setValue("stereo/distance", value);
  emit parameterChanged(vvParameters::VV_EYE_DIST, static_cast<float>(value));
}

void vvPrefDialog::onSwapEyesToggled(bool checked)
{
  QSettings settings;
  settings.setValue("stereo/swap", checked);
  emit parameterChanged(vvParameters::VV_SWAP_EYES, checked);
}

void vvPrefDialog::onMovingSpinBoxChanged(double value)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onMovingSpinBoxChanged()");

  disconnect(ui->movingDial, SIGNAL(valueChanged(int)), this, SLOT(onMovingDialChanged(int)));
  const int upper = ui->movingDial->maximum() + 1;
  double d = value - movingSpinBoxOldValue;
  int di = vvToolshed::round(d * upper);
  int dialval = ui->movingDial->value();
  dialval += di;
  dialval %= upper;
  while (dialval < ui->movingDial->minimum())
  {
    dialval += upper;
  }
  movingDialOldValue = dialval;
  ui->movingDial->setValue(dialval);
  movingSpinBoxOldValue = value;
  emit parameterChanged(vvParameters::VV_MOVING_QUALITY, static_cast<float>(ui->movingSpinBox->value()));
  connect(ui->movingDial, SIGNAL(valueChanged(int)), this, SLOT(onMovingDialChanged(int)));
}

void vvPrefDialog::onStillSpinBoxChanged(double value)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onStillSpinBoxChanged()");

  disconnect(ui->stillDial, SIGNAL(valueChanged(int)), this, SLOT(onStillDialChanged(int)));
  const int upper = ui->stillDial->maximum() + 1;
  double d = value - stillSpinBoxOldValue;
  int di = vvToolshed::round(d * upper);
  int dialval = ui->stillDial->value();
  dialval += di;
  dialval %= upper;
  while (dialval < ui->stillDial->minimum())
  {
    dialval += upper;
  }
  stillDialOldValue = dialval;
  ui->stillDial->setValue(dialval);
  stillSpinBoxOldValue = value;
  emit parameterChanged(vvRenderer::VV_QUALITY, static_cast<float>(ui->stillSpinBox->value()));
  connect(ui->stillDial, SIGNAL(valueChanged(int)), this, SLOT(onStillDialChanged(int)));
}

void vvPrefDialog::onMovingDialChanged(int value)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onMovingDialChanged()");

  const int d = getDialDelta(movingDialOldValue, value, ui->movingDial->minimum(), ui->movingDial->maximum());
  const double dd = static_cast<double>(d) / static_cast<double>(ui->movingDial->maximum() + 1);
  disconnect(ui->movingSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onMovingSpinBoxChanged(double)));
  ui->movingSpinBox->setValue(ui->movingSpinBox->value() + dd);
  movingSpinBoxOldValue = ui->movingSpinBox->value();
  movingDialOldValue = value;
  emit parameterChanged(vvParameters::VV_MOVING_QUALITY, static_cast<float>(ui->movingSpinBox->value()));
  connect(ui->movingSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onMovingSpinBoxChanged(double)));
}

void vvPrefDialog::onStillDialChanged(int value)
{
  vvDebugMsg::msg(3, "vvPrefDialog::onStillDialChanged()");

  const int d = getDialDelta(stillDialOldValue, value, ui->stillDial->minimum(), ui->stillDial->maximum());
  const double dd = static_cast<double>(d) / static_cast<double>(ui->stillDial->maximum() + 1);
  disconnect(ui->stillSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onStillSpinBoxChanged(double)));
  ui->stillSpinBox->setValue(ui->stillSpinBox->value() + dd);
  stillSpinBoxOldValue = ui->stillSpinBox->value();
  stillDialOldValue = value;
  emit parameterChanged(vvRenderer::VV_QUALITY, static_cast<float>(ui->stillSpinBox->value()));
  connect(ui->stillSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onStillSpinBoxChanged(double)));
}

