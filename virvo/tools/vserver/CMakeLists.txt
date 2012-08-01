SET(VSERVER_HEADERS
  vvresourcemanager.h
  vvserver.h
  vvsimpleserver.h
)

SET(VSERVER_SOURCES
  vvresourcemanager.cpp
  vvserver.cpp
  vvsimpleserver.cpp
)

INCLUDE_DIRECTORIES(../common)
INCLUDE_DIRECTORIES(../..)
INCLUDE_DIRECTORIES(${PTHREAD_INCLUDE_DIR})

ADD_EXECUTABLE(vserver ${VSERVER_HEADERS} ${VSERVER_SOURCES})

TARGET_LINK_LIBRARIES(vserver virvo)

INSTALL(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/vserver
  DESTINATION ${INSTALL_DIR_BIN}
  RENAME voxserver
)
