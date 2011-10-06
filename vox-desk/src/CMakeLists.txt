FIND_PACKAGE(FOX REQUIRED)
IF(NOT FOX_FOUND)
  RETURN()
ENDIF(NOT FOX_FOUND)

SET(VOXDESK_INC
  vvartracking.h
  vvcanvas.h
  vvclient.h
  vvdialogs.h
  vvmovie.h
  vvobjview.h
  vvprefwindow.h
  vvshell.h
  vvsliceviewer.h
  vvtranswindow.h
)

SET(VOXDESK_SRC
  vvartracking.cpp
  vvcanvas.cpp
  vvclient.cpp
  vvdialogs.cpp
  vvmovie.cpp
  vvobjview.cpp
  vvprefwindow.cpp
  vvshell.cpp
  vvsliceviewer.cpp
  vvtranswindow.cpp
)

SET(VCONV_INC
  vvconv.h
)

SET(VCONV_SRC
  vvconv.cpp
)

ADD_EXECUTABLE(vox-desk ${VOXDESK_INC} ${VOXDESK_SRC})
ADD_EXECUTABLE(vconv ${VCONV_INC} ${VCONV_SRC})

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

INCLUDE_DIRECTORIES(${GLEW_INCLUDE_DIR})

IF(FOX_FOUND)
  INCLUDE_DIRECTORIES(${FOX_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(vox-desk ${FOX_LIBRARIES})
ENDIF(FOX_FOUND)

TARGET_LINK_LIBRARIES(vox-desk virvo)
TARGET_LINK_LIBRARIES(vconv virvo)

INSTALL(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/vox-desk
  DESTINATION ${INSTALL_DIR_BIN}
  RENAME deskvox
)

INSTALL(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/vconv
  DESTINATION ${INSTALL_DIR_BIN}
)
