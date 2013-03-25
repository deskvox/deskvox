find_package(Qt4 REQUIRED QtCore QtGui QtOpenGL)

deskvox_use_package(Qt
  INCDIRS ${QT_INCLUDES}
  LIBS ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY} ${QT_QTOPENGL_LIBRARY}
)

deskvox_link_libraries(virvo)

add_definitions(-DQT_PLUGIN)

add_subdirectory(servermon)
