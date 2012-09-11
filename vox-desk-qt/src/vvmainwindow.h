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

#ifndef VV_MAINWINDOW_H
#define VV_MAINWINDOW_H

#include <virvo/vvvoldesc.h>

#include <QMainWindow>

class vvCanvas;
class vvDimensionDialog;
class vvMergeDialog;
class vvPlugin;
class vvPrefDialog;
class vvScreenshotDialog;
class vvShortcutDialog;
class vvTFDialog;
class vvTimeStepDialog;
class Ui_MainWindow;

class vvMainWindow : public QMainWindow
{
  Q_OBJECT
public:
  explicit vvMainWindow(const QString& filename, QWidget* parent = 0);
  ~vvMainWindow();
private:
  Ui_MainWindow* ui;

  QList<vvPlugin*> _plugins;

  vvCanvas* _canvas;
  vvPrefDialog* _prefDialog;
  vvDimensionDialog* _dimensionDialog;
  vvMergeDialog* _mergeDialog;
  vvScreenshotDialog* _screenshotDialog;
  vvShortcutDialog* _shortcutDialog;
  vvTFDialog* _tfDialog;
  vvTimeStepDialog* _timeStepDialog;

  void loadVolumeFile(const QString& filename);
  void mergeFiles(const QString& firstFile, int num, int increment, vvVolDesc::MergeType mergeType);
public slots:
  void toggleOrientation();
  void toggleBoundaries();
  void togglePalette();
  void toggleFrameRate();
  void toggleNumTextures();
  void toggleInterpolation();
  void toggleProjectionType();
  void incQuality();
  void decQuality();
private slots:
  // menus
  void onLoadVolumeTriggered();
  void onReloadVolumeTriggered();
  void onSaveVolumeAsTriggered();
  void onMergeFilesTriggered();
  void onLoadCameraTriggered();
  void onSaveCameraAsTriggered();
  void onScreenshotTriggered();
  void onPreferencesTriggered();

  void onTransferFunctionTriggered();
  void onBackgroundColorTriggered();

  void onSampleDistancesTriggered();

  void onShowOrientationTriggered(bool checked);
  void onShowBoundariesTriggered(bool checked);
  void onShowPaletteTriggered(bool checked);
  void onShowNumTexturesTriggered(bool checked);
  void onShowFrameRateTriggered(bool checked);
  void onAutoRotationTriggered(bool checked);
  void onTimeStepsTriggered();

  void onKeyboardCommandsClicked();

  // misc.
  void onNewVolDesc(vvVolDesc* vd);
};

#endif

