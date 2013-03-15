find_package(FOX)
if (NOT FOX_FOUND)
  return()
endif()
find_package(OpenGL REQUIRED)

deskvox_use_package(FOX)
deskvox_use_package(OpenGL)

add_definitions(-DVV_APPLICATION_BUILD)
if(NOT BUILD_SHARED_LIBS)
  add_definitions(-DVIRVO_STATIC)
endif()

include_directories(
  ${VIRVO_CONFIG_DIR}
  ${VIRVO_INCLUDE_DIR}
  ${VIRVO_INCLUDE_DIR}/virvo
)

##
## TODO:
## public virvo headers should not include third-party library headers...
##

if(CUDA_FOUND)
  include_directories(SYSTEM ${CUDA_TOOLKIT_INCLUDE})
endif()

deskvox_link_libraries(virvo)

deskvox_add_tool(vox-desk
  vvartracking.cpp
  vvartracking.h
  vvcanvas.cpp
  vvcanvas.h
  vvclient.cpp
  vvclient.h
  vvdialogs.cpp
  vvdialogs.h
  vvmovie.cpp
  vvmovie.h
  vvobjview.cpp
  vvobjview.h
  vvprefwindow.cpp
  vvprefwindow.h
  vvshell.cpp
  vvshell.h
  vvsliceviewer.cpp
  vvsliceviewer.h
# vvtranslines.cpp
# vvtranslines.h
  vvtranswindow.cpp
  vvtranswindow.h
)
