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

FIND_PACKAGE(FOX REQUIRED)
IF(FOX_FOUND)
  INCLUDE_DIRECTORIES(${FOX_INCLUDE_DIR})
  TARGET_LINK_LIBRARIES(vox-desk ${FOX_LIBRARIES})
ENDIF(FOX_FOUND)

TARGET_LINK_LIBRARIES(vox-desk virvo)
TARGET_LINK_LIBRARIES(vconv virvo)