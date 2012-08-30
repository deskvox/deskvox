find_package(Qt REQUIRED QtCore QtGui QtOpenGL)
find_package(GLEW REQUIRED)

deskvox_use_package(Qt
  INCDIRS ${QT_INCLUDES}
  LIBS ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY} ${QT_QTOPENGL_LIBRARY}
)

deskvox_link_libraries(virvo)

add_definitions(-DVV_APPLICATION_BUILD)

set(VOXDESK_QT_INC
  vvcanvas.h
  vvdimensiondialog.h
  vvmainwindow.h
  vvmergedialog.h
  vvobjview.h
  vvpluginutil.h
  vvscreenshotdialog.h
  vvtfdialog.h
  vvtimestepdialog.h
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
  vvtimestepdialog.cpp
)

SET(VOXDESK_QT_UI
  vvdimensiondialog.ui
  vvmainwindow.ui
  vvmergedialog.ui
  vvscreenshotdialog.ui
  vvtfdialog.ui
  vvtimestepdialog.ui
)

QT4_WRAP_CPP(VOXDESK_QT_MOC_INC ${VOXDESK_QT_INC})
QT4_WRAP_UI(VOXDESK_QT_UI_INC ${VOXDESK_QT_UI})

# moc'd headers reside here
include_directories(${CMAKE_CURRENT_BINARY_DIR})

deskvox_add_tool(vox-desk-qt ${VOXDESK_QT_SRC} ${VOXDESK_QT_MOC_INC} ${VOXDESK_QT_UI_INC})
