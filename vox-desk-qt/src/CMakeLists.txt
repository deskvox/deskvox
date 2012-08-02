LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

FIND_PACKAGE(Qt)
IF(NOT QT_FOUND)
  RETURN()
ENDIF(NOT QT_FOUND)

SET(VOXDESK_QT_INC
  vvcanvas.h
  vvdimensiondialog.h
  vvmainwindow.h
  vvmergedialog.h
  vvobjview.h
  vvpluginutil.h
  vvscreenshotdialog.h
  vvtfdialog.h
)

SET(VOXDESK_QT_SRC
  vvcanvas.cpp
  vvdimensiondialog.cpp
  vvmainwindow.cpp
  vvmergedialog.cpp
  vvobjview.cpp
  vvpluginutil.cpp
  vvscreenshotdialog.cpp
  vvtfdialog.cpp
)

SET(VOXDESK_QT_UI
  vvdimensiondialog.ui
  vvmainwindow.ui
  vvmergedialog.ui
  vvscreenshotdialog.ui
  vvtfdialog.ui
)

QT4_WRAP_CPP(VOXDESK_QT_MOC_INC ${VOXDESK_QT_INC})
QT4_WRAP_UI(VOXDESK_QT_UI_INC ${VOXDESK_QT_UI})

# moc'd headers reside here
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

ADD_EXECUTABLE(vox-desk-qt ${VOXDESK_QT_SRC} ${VOXDESK_QT_MOC_INC} ${VOXDESK_QT_UI_INC})

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/virvo)
INCLUDE_DIRECTORIES(${QT_INCLUDE_DIR} ${QT_QTCORE_INCLUDE_DIR} ${QT_QTGUI_INCLUDE_DIR} ${QT_QTOPENGL_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(vox-desk-qt virvo ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY} ${QT_QTOPENGL_LIBRARY})
