find_package(Qt4 REQUIRED QtCore QtGui QtOpenGL)
find_package(GLEW REQUIRED)

deskvox_use_package(Qt
  INCDIRS ${QT_INCLUDES}
  LIBS ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY} ${QT_QTOPENGL_LIBRARY}
)

deskvox_use_package(GLEW)

deskvox_link_libraries(virvo)

set(VOXDESK_QT_INC
  tfeditor/colorbox.h
  tfeditor/gaussianbox.h
  tfeditor/graphicsscene.h
  tfeditor/pyramidbox.h
  tfeditor/skipbox.h

  vvcanvas.h
  vvdimensiondialog.h
  vvinteractor.h
  vvlightdialog.h
  vvlightinteractor.h
  vvmainwindow.h
  vvmergedialog.h
  vvobjview.h
  vvparameters.h
  vvpluginutil.h
  vvprefdialog.h
  vvscreenshotdialog.h
  vvshortcutdialog.h
  vvsliceviewer.h
  vvtfdialog.h
  vvtimestepdialog.h
  vvvolinfodialog.h
)

SET(VOXDESK_QT_SRC
  tfeditor/colorbox.cpp
  tfeditor/gaussianbox.cpp
  tfeditor/graphicsscene.cpp
  tfeditor/pyramidbox.cpp
  tfeditor/skipbox.cpp

  vvcanvas.cpp
  vvdimensiondialog.cpp
  vvinteractor.cpp
  vvlightdialog.cpp
  vvlightinteractor.cpp
  vvmainwindow.cpp
  vvmergedialog.cpp
  vvobjview.cpp
  vvpluginutil.cpp
  vvprefdialog.cpp
  vvscreenshotdialog.cpp
  vvshortcutdialog.cpp
  vvsliceviewer.cpp
  vvtfdialog.cpp
  vvtimestepdialog.cpp
  vvvolinfodialog.cpp
)

SET(VOXDESK_QT_UI
  tfeditor/colorbox.ui
  tfeditor/gaussianbox.ui
  tfeditor/pyramidbox.ui
  tfeditor/skipbox.ui

  vvdimensiondialog.ui
  vvlightdialog.ui
  vvmainwindow.ui
  vvmergedialog.ui
  vvprefdialog.ui
  vvscreenshotdialog.ui
  vvshortcutdialog.ui
  vvsliceviewer.ui
  vvtfdialog.ui
  vvtimestepdialog.ui
  vvvolinfodialog.ui
)

IF(QT_FOUND)
  QT4_WRAP_CPP(VOXDESK_QT_MOC_INC ${VOXDESK_QT_INC})
  QT4_WRAP_UI(VOXDESK_QT_UI_INC ${VOXDESK_QT_UI})
ENDIF(QT_FOUND)

# moc'd headers reside here
include_directories(${CMAKE_CURRENT_BINARY_DIR})

deskvox_add_tool(vox-desk-qt ${VOXDESK_QT_INC} ${VOXDESK_QT_SRC} ${VOXDESK_QT_MOC_INC} ${VOXDESK_QT_UI_INC})
