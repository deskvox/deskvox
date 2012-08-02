find_package(Qt REQUIRED)

deskvox_use_package(Qt)

add_definitions(-DVV_APPLICATION_BUILD)

include_directories(
  ${VIRVO_CONFIG_DIR}
  ${VIRVO_INCLUDE_DIR}
  ${VIRVO_INCLUDE_DIR}/virvo
)

set(VOXDESK_QT_INC
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

INCLUDE_DIRECTORIES(${QT_INCLUDE_DIR} ${QT_QTCORE_INCLUDE_DIR} ${QT_QTGUI_INCLUDE_DIR} ${QT_QTOPENGL_INCLUDE_DIR})

deskvox_link_libraries(virvo)
deskvox_add_tool(vox-desk-qt ${VOXDESK_QT_SRC} ${VOXDESK_QT_MOC_INC} ${VOXDESK_QT_UI_INC})
TARGET_LINK_LIBRARIES(vox-desk-qt virvo ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY} ${QT_QTOPENGL_LIBRARY})
