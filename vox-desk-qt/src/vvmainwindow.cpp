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
#include "vvdimensiondialog.h"
#include "vvlightdialog.h"
#include "vvmainwindow.h"
#include "vvmergedialog.h"
#include "vvobjview.h"
#include "vvplugin.h"
#include "vvpluginutil.h"
#include "vvprefdialog.h"
#include "vvscreenshotdialog.h"
#include "vvshortcutdialog.h"
#include "vvtfdialog.h"
#include "vvtimestepdialog.h"
#include "vvvolinfodialog.h"

#include "ui_vvmainwindow.h"

#include <virvo/vvdebugmsg.h>
#include <virvo/vvfileio.h>
#include <virvo/vvvoldesc.h>

#include <QApplication>
#include <QByteArray>
#include <QColorDialog>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QSettings>
#include <QShortcut>
#include <QStringList>

using vox::vvObjView;

vvMainWindow::vvMainWindow(const QString& filename, QWidget* parent)
  : QMainWindow(parent)
  , ui(new Ui_MainWindow)
{
  vvDebugMsg::msg(1, "vvMainWindow::vvMainWindow()");

  ui->setupUi(this);

  // plugins
  _plugins = vvPluginUtil::getAll();
  foreach (vvPlugin* plugin, _plugins)
  {
    if (QDialog* dialog = plugin->dialog(this))
    {
      ui->menuPlugins->setEnabled(true);
      QAction* dialogAction = new QAction(plugin->name(), ui->menuPlugins);
      ui->menuPlugins->addAction(dialogAction);
      connect(dialogAction, SIGNAL(triggered()), dialog, SLOT(show()));
    }
  }

  // widgets and dialogs
  const int superSamples = 0;
  QGLFormat format;
  format.setDoubleBuffer(true);
  format.setDepth(true);
  format.setRgba(true);
  format.setAlpha(true);
  format.setAccum(true);
  format.setStencil(false);
  if (superSamples > 0)
  {
    format.setSampleBuffers(true);
    format.setSamples(superSamples);
  }

  QString fn = filename;
  if (fn == "")
  {
    QSettings settings;
    fn = settings.value("canvas/recentfile").toString();
  }

  _canvas = new vvCanvas(format, fn, this);
  _canvas->setPlugins(_plugins);
  setCentralWidget(_canvas);

  _prefDialog = new vvPrefDialog(_canvas, this);

  _tfDialog = new vvTFDialog(_canvas, this);
  _lightDialog = new vvLightDialog(this);

  _dimensionDialog = new vvDimensionDialog(_canvas, this);
  _mergeDialog = new vvMergeDialog(this);
  _screenshotDialog = new vvScreenshotDialog(_canvas, this);
  _shortcutDialog = new vvShortcutDialog(this);
  _timeStepDialog = new vvTimeStepDialog(this);
  _volInfoDialog = new vvVolInfoDialog(this);

  // file menu
  connect(ui->actionLoadVolume, SIGNAL(triggered()), this, SLOT(onLoadVolumeTriggered()));
  connect(ui->actionReloadVolume, SIGNAL(triggered()), this, SLOT(onReloadVolumeTriggered()));
  connect(ui->actionSaveVolumeAs, SIGNAL(triggered()), this, SLOT(onSaveVolumeAsTriggered()));
  connect(ui->actionMergeFiles, SIGNAL(triggered()), this, SLOT(onMergeFilesTriggered()));
  connect(ui->actionLoadCamera, SIGNAL(triggered()), this, SLOT(onLoadCameraTriggered()));
  connect(ui->actionSaveCameraAs, SIGNAL(triggered()), this, SLOT(onSaveCameraAsTriggered()));
  connect(ui->actionScreenshot, SIGNAL(triggered()), this, SLOT(onScreenshotTriggered()));
  connect(ui->actionPreferences, SIGNAL(triggered()), this, SLOT(onPreferencesTriggered()));

  // settings menu
  connect(ui->actionTransferFunction, SIGNAL(triggered()), this, SLOT(onTransferFunctionTriggered()));
  connect(ui->actionLightSource, SIGNAL(triggered()), this, SLOT(onLightSourceTriggered()));
  connect(ui->actionBackgroundColor, SIGNAL(triggered()), this, SLOT(onBackgroundColorTriggered()));

  // edit menu
  connect(ui->actionSampleDistances, SIGNAL(triggered()), this, SLOT(onSampleDistancesTriggered()));

  // view menu
  connect(ui->actionShowOrientation, SIGNAL(triggered(bool)), this, SLOT(onShowOrientationTriggered(bool)));
  connect(ui->actionShowBoundaries, SIGNAL(triggered(bool)), this, SLOT(onShowBoundariesTriggered(bool)));
  connect(ui->actionShowPalette, SIGNAL(triggered(bool)), this, SLOT(onShowPaletteTriggered(bool)));
  connect(ui->actionShowNumTextures, SIGNAL(triggered(bool)), this, SLOT(onShowNumTexturesTriggered(bool)));
  connect(ui->actionShowFrameRate, SIGNAL(triggered(bool)), this, SLOT(onShowFrameRateTriggered(bool)));
  connect(ui->actionAutoRotation, SIGNAL(triggered(bool)), this, SLOT(onAutoRotationTriggered(bool)));
  connect(ui->actionVolumeInformation, SIGNAL(triggered(bool)), this, SLOT(onVolumeInformationTriggered()));
  connect(ui->actionTimeSteps, SIGNAL(triggered()), this, SLOT(onTimeStepsTriggered()));

  // help menu
  connect(ui->actionKeyboardCommands, SIGNAL(triggered()), this, SLOT(onKeyboardCommandsClicked()));

  // misc.
  connect(_canvas, SIGNAL(newVolDesc(vvVolDesc*)), this, SLOT(onNewVolDesc(vvVolDesc*)));
  connect(_canvas, SIGNAL(statusMessage(const std::string&)), this, SLOT(onStatusMessage(const std::string&)));

  connect(_prefDialog, SIGNAL(rendererChanged(const std::string&, const vvRendererFactory::Options&)),
    _canvas, SLOT(setRenderer(const std::string&, const vvRendererFactory::Options&)));
  connect(_prefDialog, SIGNAL(parameterChanged(vvParameters::ParameterType, const vvParam&)),
    _canvas, SLOT(setParameter(vvParameters::ParameterType, const vvParam&)));
  connect(_prefDialog, SIGNAL(parameterChanged(vvRenderer::ParameterType, const vvParam&)),
    _canvas, SLOT(setParameter(vvRenderer::ParameterType, const vvParam&)));

  connect(_lightDialog, SIGNAL(enabled(bool)), _canvas, SLOT(enableLighting(bool)));
  connect(_lightDialog, SIGNAL(showLightSource(bool)), _canvas, SLOT(showLightSource(bool)));
  connect(_lightDialog, SIGNAL(editPositionToggled(bool)), _canvas, SLOT(editLightPosition(bool)));
  connect(_lightDialog, SIGNAL(attenuationChanged(const vvVector3&)), _canvas, SLOT(setLightAttenuation(const vvVector3&)));

  connect(_canvas, SIGNAL(newVolDesc(vvVolDesc*)), _volInfoDialog, SLOT(onNewVolDesc(vvVolDesc*)));

  connect(_timeStepDialog, SIGNAL(valueChanged(int)), _canvas, SLOT(setTimeStep(int)));
  connect(_timeStepDialog, SIGNAL(play(double)), _canvas, SLOT(startAnimation(double)));
  connect(_timeStepDialog, SIGNAL(pause()), _canvas, SLOT(stopAnimation()));
  connect(_timeStepDialog, SIGNAL(back()), _canvas, SLOT(decTimeStep()));
  connect(_timeStepDialog, SIGNAL(fwd()), _canvas, SLOT(incTimeStep()));
  connect(_timeStepDialog, SIGNAL(first()), _canvas, SLOT(firstTimeStep()));
  connect(_timeStepDialog, SIGNAL(last()), _canvas, SLOT(lastTimeStep()));
  connect(_canvas, SIGNAL(currentFrame(int)), _timeStepDialog, SLOT(setCurrentFrame(int)));

  // shortcuts

  QShortcut* sc; // reassign for each shortcut, objects are ref-counted by Qt, anyway

  // rendering quality
  sc = new QShortcut(tr("+"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(incQuality()));

  sc = new QShortcut(tr("="), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(incQuality()));

  sc = new QShortcut(tr("-"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(decQuality()));

  // rendering
  sc = new QShortcut(tr("o"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(toggleOrientation()));

  sc = new QShortcut(tr("b"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(toggleBoundaries()));

  sc = new QShortcut(tr("c"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(togglePalette()));

  sc = new QShortcut(tr("f"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(toggleFrameRate()));

  sc = new QShortcut(tr("t"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(toggleNumTextures()));

  sc = new QShortcut(tr("i"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(toggleInterpolation()));

  sc = new QShortcut(tr("p"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(toggleProjectionType()));

  // animation
  sc = new QShortcut(tr("a"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), _timeStepDialog, SLOT(togglePlayback()));

  sc = new QShortcut(tr("n"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), _timeStepDialog, SLOT(stepFwd()));

  sc = new QShortcut(tr("Shift+n"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), _timeStepDialog, SLOT(stepBack()));

  // misc.
  sc = new QShortcut(tr("q"), this);
  sc->setContext(Qt::ApplicationShortcut);
  connect(sc, SIGNAL(activated()), this, SLOT(close()));

  statusBar()->showMessage(tr("Welcome to DeskVOX!"));
}

vvMainWindow::~vvMainWindow()
{
  vvDebugMsg::msg(1, "vvMainWindow::~vvMainWindow()");
}

void vvMainWindow::loadVolumeFile(const QString& filename)
{
  QByteArray ba = filename.toLatin1();
  vvVolDesc* vd = new vvVolDesc(ba.data());
  vvFileIO* fio = new vvFileIO;
  switch (fio->loadVolumeData(vd, vvFileIO::ALL_DATA))
  {
  case vvFileIO::OK:
  {
    vvDebugMsg::msg(2, "Loaded file: ", ba.data());
    // use default TF if none stored
    if (vd->tf.isEmpty())
    {
      vd->tf.setDefaultAlpha(0, vd->real[0], vd->real[1]);
      vd->tf.setDefaultColors((vd->chan == 1) ? 0 : 2, vd->real[0], vd->real[1]);
    }
    if (vd->bpc == 4 && vd->real[0] == 0.0f && vd->real[1] == 1.0f)
    {
      vd->setDefaultRealMinMax();
    }
    _canvas->setVolDesc(vd);
    _dimensionDialog->setInitialDist(vd->dist);

    QSettings settings;
    settings.setValue("canvas/recentfile", filename);
    break;
  }
  case vvFileIO::FILE_NOT_FOUND:
    vvDebugMsg::msg(2, "File not found: ", ba.data());
    delete vd;
    QMessageBox::warning(this, tr("Error loading file"), tr("File not found: ") + filename, QMessageBox::Ok);
    break;
  default:
    vvDebugMsg::msg(2, "Cannot load file: ", ba.data());
    delete vd;
    QMessageBox::warning(this, tr("Error loading file"), tr("Cannot load file: ") + filename, QMessageBox::Ok);
    break;
  }
}

void vvMainWindow::mergeFiles(const QString& firstFile, const int num, const int increment, vvVolDesc::MergeType mergeType)
{
  vvDebugMsg::msg(1, "vvMainWindow::mergeFiles()");

  QByteArray ba = firstFile.toLatin1();
  vvVolDesc* vd = new vvVolDesc(ba.data());
  vvFileIO* fio = new vvFileIO;
  switch (fio->mergeFiles(vd, num, increment, mergeType))
  {
  case vvFileIO::OK:
    vvDebugMsg::msg(2, "Loaded slice sequence: ", vd->getFilename());
    // use default TF if non stored
    if (vd->tf.isEmpty())
    {
      vd->tf.setDefaultAlpha(0, vd->real[0], vd->real[1]);
      vd->tf.setDefaultColors((vd->chan == 1) ? 0 : 2, vd->real[0], vd->real[1]);
    }
    if (vd->bpc == 4 && vd->real[0] == 0.0f && vd->real[1] == 1.0f)
    {
      vd->setDefaultRealMinMax();
    }
    _canvas->setVolDesc(vd);
    break;
  case vvFileIO::FILE_NOT_FOUND:
    vvDebugMsg::msg(2, "File not found: ", ba.data());
    delete vd;
    QMessageBox::warning(this, tr("Error merging file"), tr("File not found: ") + firstFile, QMessageBox::Ok);
    break;
  default:
    vvDebugMsg::msg(2, "Cannot merge file: ", ba.data());
    delete vd;
    QMessageBox::warning(this, tr("Error merging file"), tr("Cannot merge file: ") + firstFile, QMessageBox::Ok);
    break;
  }
}

void vvMainWindow::toggleOrientation()
{
  ui->actionShowOrientation->trigger();
}

void vvMainWindow::toggleBoundaries()
{
  ui->actionShowBoundaries->trigger();
}

void vvMainWindow::togglePalette()
{
  ui->actionShowPalette->trigger();
}

void vvMainWindow::toggleFrameRate()
{
  ui->actionShowFrameRate->trigger();
}

void vvMainWindow::toggleNumTextures()
{
  ui->actionShowNumTextures->trigger();
}

void vvMainWindow::toggleInterpolation()
{
  _prefDialog->toggleInterpolation();
}

void vvMainWindow::toggleProjectionType()
{
  vvObjView::ProjectionType type = static_cast<vvObjView::ProjectionType>(_canvas->getParameter(vvParameters::VV_PROJECTIONTYPE).asInt());

  if (type == vvObjView::PERSPECTIVE)
  {
    type = vvObjView::ORTHO;
  }
  else
  {
    type = vvObjView::PERSPECTIVE;
  }
  _canvas->setParameter(vvParameters::VV_PROJECTIONTYPE, static_cast<int>(type));
}

void vvMainWindow::incQuality()
{
  _prefDialog->scaleStillQuality(1.05f);
}

void vvMainWindow::decQuality()
{
  _prefDialog->scaleStillQuality(0.95f);
}

void vvMainWindow::onLoadVolumeTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onLoadVolumeTriggered()");

  QSettings settings;

  QString caption = tr("Load Volume File");
  QString dir = settings.value("canvas/voldir").value<QString>();
  QString filter = tr("All Volume Files (*.rvf *.xvf *.avf *.tif *.tiff *.hdr *.volb);;"
    "3D TIF Files (*.tif,*.tiff);;"
    "ASCII Volume Files (*.avf);;"
    "Extended Volume Files (*.xvf);;"
    "Raw Volume Files (*.rvf);;"
    "All Files (*.*)");
  QString filename = QFileDialog::getOpenFileName(this, caption, dir, filter);
  if (!filename.isEmpty())
  {
    loadVolumeFile(filename);

    QDir dir = QFileInfo(filename).absoluteDir();
    settings.setValue("canvas/voldir", dir.path());
  }
  else
  {
    QMessageBox::warning(this, tr("Error loading file"), tr("File name is empty"), QMessageBox::Ok);
  }
}

void vvMainWindow::onReloadVolumeTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onReloadVolumeTriggered()");

  vvVolDesc* vd = _canvas->getVolDesc();
  if (vd != NULL)
  {
    loadVolumeFile(vd->getFilename());
  }
}

void vvMainWindow::onSaveVolumeAsTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onSaveVolumeTriggered()");

  QString caption = tr("Save Volume");
  QString dir;
  QString filter = tr("All Volume Files (*.xvf *.rvf *.avf);;"
    "Extended Volume Files (*.xvf);;"
    "Raw Volume Files (*.rvf);;"
    "ASCII Volume Files (*.avf);;"
    "All Files (*.*)");
  QString filename = QFileDialog::getSaveFileName(this, caption, dir, filter);
  if (!filename.isEmpty())
  {
    vvFileIO* fio = new vvFileIO;
    QByteArray ba = filename.toLatin1();
    _canvas->getVolDesc()->setFilename(ba.data());
    switch (fio->saveVolumeData(_canvas->getVolDesc(), true))
    {
    case vvFileIO::OK:
      vvDebugMsg::msg(2, "Volume saved as ", _canvas->getVolDesc()->getFilename());
      break;
    default:
      vvDebugMsg::msg(0, "Unhandled error saving ", _canvas->getVolDesc()->getFilename());
      break;
    }
  }
}

void vvMainWindow::onMergeFilesTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onMergeFilesTriggered()");

  if (_mergeDialog->exec() == QDialog::Accepted)
  {
    const QString filename = _mergeDialog->getFilename();

    int numFiles = 0;
    if (_mergeDialog->numFilesLimited())
    {
      numFiles = _mergeDialog->getNumFiles();
    }

    int increment = 0;
    if (_mergeDialog->filesNumbered())
    {
      increment = _mergeDialog->getFileIncrement();
    }

    const vvVolDesc::MergeType mergeType = _mergeDialog->getMergeType();

    mergeFiles(filename, numFiles, increment, mergeType);
  }
}

void vvMainWindow::onLoadCameraTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onLoadCameraTriggered()");

  QString caption = tr("Load Camera File");
  QString dir;
  QString filter = tr("Camera Files (*cam);;"
    "All Files (*.*)");
  QString filename = QFileDialog::getOpenFileName(this, caption, dir, filter);
  if (!filename.isEmpty())
  {
    _canvas->loadCamera(filename);
  }
}

void vvMainWindow::onSaveCameraAsTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onSaveCameraAsTriggered()");

  QString caption = tr("Save Camera to File");
  QString dir = "camera.cam";
  QString filter = tr("Camera Files (*cam);;"
    "All Files (*.*)");
  QString filename = QFileDialog::getSaveFileName(this, caption, dir, filter);
  if (!filename.isEmpty())
  {
    _canvas->saveCamera(filename);
  }
}

void vvMainWindow::onScreenshotTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onScreenshotTriggered()");

  _screenshotDialog->raise();
  _screenshotDialog->show();
}

void vvMainWindow::onPreferencesTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onPreferencesTriggered()");

  _prefDialog->raise();
  _prefDialog->show();
}

void vvMainWindow::onTransferFunctionTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onTransferFunctionTriggered()");

  _tfDialog->raise();
  _tfDialog->show();
}

void vvMainWindow::onLightSourceTriggered()
{
  _lightDialog->raise();
  _lightDialog->show();
}

void vvMainWindow::onBackgroundColorTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onBackgroundColorTriggered()");

  QColor qcolor = QColorDialog::getColor();
  vvColor color(qcolor.redF(), qcolor.greenF(), qcolor.blueF());
  _canvas->setParameter(vvParameters::VV_BG_COLOR, color);
  QSettings settings;
  settings.setValue("canvas/bgcolor", qcolor);
}

void vvMainWindow::onSampleDistancesTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onSampleDistancesTriggered()");

  _dimensionDialog->raise();
  _dimensionDialog->show();
}

void vvMainWindow::onShowOrientationTriggered(bool checked)
{
  vvDebugMsg::msg(3, "vvMainWindow::onShowOrientationTriggered()");

  _canvas->setParameter(vvRenderState::VV_ORIENTATION, checked);
}

void vvMainWindow::onShowBoundariesTriggered(bool checked)
{
  vvDebugMsg::msg(3, "vvMainWindow::onShowBoundariesTriggered()");

  _canvas->setParameter(vvRenderState::VV_BOUNDARIES, checked);
}

void vvMainWindow::onShowPaletteTriggered(bool checked)
{
  vvDebugMsg::msg(3, "vvMainWindow::onShowPaletteTriggered()");

  _canvas->setParameter(vvRenderState::VV_PALETTE, checked);
}

void vvMainWindow::onShowNumTexturesTriggered(bool checked)
{
  vvDebugMsg::msg(3, "vvMainWindow::onShowNumTexturesTriggered()");

  _canvas->setParameter(vvRenderState::VV_QUALITY_DISPLAY, checked);
}

void vvMainWindow::onShowFrameRateTriggered(bool checked)
{
  vvDebugMsg::msg(3, "vvMainWindow::onShowFrameRateTriggered()");

  _canvas->setParameter(vvRenderState::VV_FPS_DISPLAY, checked);
}

void vvMainWindow::onAutoRotationTriggered(bool checked)
{
  vvDebugMsg::msg(3, "vvMainWindow::onAutoRotationTriggered()");

  _canvas->setParameter(vvParameters::VV_SPIN_ANIMATION, checked);
}

void vvMainWindow::onVolumeInformationTriggered()
{
  _volInfoDialog->raise();
  _volInfoDialog->show();
}

void vvMainWindow::onTimeStepsTriggered()
{
  vvDebugMsg::msg(3, "vvMainWindow::onTimeStepsTriggered()");

  _timeStepDialog->raise();
  _timeStepDialog->show();
}

void vvMainWindow::onKeyboardCommandsClicked()
{
  vvDebugMsg::msg(3, "vvMainWindow::onKeyboardCommandsClicked()");

  _shortcutDialog->raise();
  _shortcutDialog->show();
}

void vvMainWindow::onNewVolDesc(vvVolDesc* vd)
{
  vvDebugMsg::msg(3, "vvMainWindow::onNewVolDesc()");

  _timeStepDialog->setFrames(vd->frames);
}

void vvMainWindow::onStatusMessage(const std::string& str)
{
  vvDebugMsg::msg(3, "vvMainWindow::onStatusMessage()");

  statusBar()->showMessage(str.c_str());
}

int main(int argc, char** argv)
{
  vvDebugMsg::setDebugLevel(0);

  QApplication a(argc, argv);

  // parse command line
  QString filename;
  QSize size(600, 600);
  QStringList arglist = a.arguments();
  for (QStringList::iterator it = arglist.begin();
       it != arglist.end(); ++it)
  {
    QString str = *it;
    if (str == arglist.first())
    {
      continue;
    }

    if (str == "-size")
    {
      ++it;
      int w = 0;
      int h = 0;;
      if (it != arglist.end())
      {
        str = *it;
        w = str.toInt();
      }
      else
      {
        vvDebugMsg::msg(0, "Warning: -size followed by no arguments");
        break;
      }

      ++it;
      if (it != arglist.end())
      {
        str = *it;
        h = str.toInt();
        size = QSize(w, h);
      }
      else
      {
        vvDebugMsg::msg(0, "Warning: -size followed by only one argument");
        break;
      }
    }
    else if (str[0] == '-')
    {
      vvDebugMsg::msg(0, "Warning: invalid command line option");
      break;
    }
    else
    {
      filename = str;
    }
  }

  // create main window
  vvMainWindow win(filename);
  win.resize(size);
  win.show();
  return a.exec();
}

